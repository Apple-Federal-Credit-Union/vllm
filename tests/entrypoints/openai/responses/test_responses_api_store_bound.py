# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound the in-process Responses API store.

With ``VLLM_ENABLE_RESPONSES_API_STORE`` enabled, ``POST /v1/responses``
keeps every stored response. These tests are the contract for closing that
hole:

* ``response_store``, ``msg_store``, and ``event_store`` together stay within
  ``MAX_STORED_RESPONSES`` entries and ``MAX_STORED_BYTES`` of caller payload.
* The bound is global. Distinct ``user`` values, ``safety_identifier`` values,
  and ``Authorization`` bearer tokens do not each get a fresh copy of the cap.
* Explicit ``store=true`` still keeps the newest response and drops the
  oldest. Omitting ``store`` must not retain an unbounded history either
  (a default of false, or the same cap, both satisfy this).
* Reading or cancelling a response does not leave every previous payload
  resident.

Harmony and ``ParsableContext`` are not given their own cases. Both reach the
same three dictionaries through the same ``request.store`` writes in
``OpenAIServingResponses``.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.api_router import (
    cancel_responses,
    create_responses,
    retrieve_responses,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.inputs.engine import tokens_input
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.online_renderer import OnlineRenderer

pytestmark = pytest.mark.skip_global_cleanup

MODEL_NAME = "store-bound-test"
# A few large requests must already exceed both limits.
MAX_STORED_RESPONSES = 4
MAX_STORED_BYTES = 64 * 1024
PAYLOAD_BYTES = 32 * 1024
REQUESTS_PER_CLIENT = 4
# Two clients, so a per-caller cap of MAX_STORED_RESPONSES still overflows
# the process if each caller is given that full cap.
CLIENTS = (
    {
        "user": "client-a",
        "safety_identifier": "safety-a",
        "token": "placeholder-token-a",
    },
    {
        "user": "client-b",
        "safety_identifier": "safety-b",
        "token": "placeholder-token-b",
    },
)


@dataclass
class _HFConfig:
    model_type: str = "test"


@dataclass
class _ModelConfig:
    model: str = MODEL_NAME
    max_model_len: int = 1024
    hf_config: _HFConfig = None  # type: ignore[assignment]
    generation_config: str = "auto"
    override_generation_config: dict = None  # type: ignore[assignment]
    is_encoder_decoder: bool = False

    def __post_init__(self) -> None:
        if self.hf_config is None:
            self.hf_config = _HFConfig()
        if self.override_generation_config is None:
            self.override_generation_config = {}

    def get_diff_sampling_param(self) -> dict:
        return {}


class _Engine:
    """Stand-in for the model engine.

    The serving layer only needs one finished ``RequestOutput``. Tokenization
    and weight execution do not decide whether the caller payload is retained.
    """

    def __init__(self, model_config: _ModelConfig) -> None:
        self.model_config = model_config
        self.renderer = self
        self.input_processor = None
        self.vllm_config = None
        self.errored = False
        self.tokenizer = None

    def get_tokenizer(self):
        return None

    def check_admission(self, n: int = 1) -> None:
        return None

    async def generate(self, engine_input, sampling_params, request_id, **kwargs):
        yield RequestOutput(
            request_id=request_id,
            prompt="",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="ok",
                    token_ids=(1,),
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            finished=True,
        )


class _ResponsesRenderer(OnlineRenderer):
    async def preprocess_chat(self, request, messages, *args, **kwargs):
        # ``render_responses`` already built ``messages`` with the caller
        # input. Tokenization must not replace that list.
        del request, messages, args, kwargs
        return [], [tokens_input([1])]


def _payload(client_user: str, index: int) -> str:
    marker = f"{client_user}-{index}-"
    return marker + ("x" * (PAYLOAD_BYTES - len(marker)))


def _iter_strings(obj: object, seen: set[int]) -> Iterator[str]:
    ident = id(obj)
    if ident in seen:
        return
    seen.add(ident)
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _iter_strings(key, seen)
            yield from _iter_strings(value, seen)
        return
    if isinstance(obj, (list, tuple, set, frozenset, deque)):
        for value in obj:
            yield from _iter_strings(value, seen)
        return
    if isinstance(obj, (int, float, bool, type(None), bytes, bytearray)):
        return
    module = type(obj).__module__
    if module.startswith(("torch", "numpy", "logging", "asyncio")):
        return
    attributes = getattr(obj, "__dict__", None)
    if isinstance(attributes, dict):
        yield from _iter_strings(attributes, seen)


def _retained_payload_bytes(
    serving: OpenAIServingResponses, payloads: list[str]
) -> int:
    blobs = list(_iter_strings(serving.__dict__, set()))
    retained = 0
    for payload in payloads:
        if any(payload in blob for blob in blobs):
            retained += len(payload.encode())
    return retained


def _stored_ids(serving: OpenAIServingResponses) -> set[str]:
    return (
        set(serving.response_store) | set(serving.msg_store) | set(serving.event_store)
    )


async def _never_disconnect():
    await asyncio.Future()


def _raw_request(app: FastAPI, token: str) -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 80),
            "app": app,
        },
        receive=_never_disconnect,
    )


def _serving(monkeypatch: pytest.MonkeyPatch, store_enabled: bool) -> tuple:
    monkeypatch.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1" if store_enabled else "0")
    model_config = _ModelConfig()
    engine = _Engine(model_config)
    models = OpenAIServingModels(
        engine,
        [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)],
    )
    renderer = _ResponsesRenderer(
        model_config=model_config,
        renderer=engine,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    serving = OpenAIServingResponses(
        engine_client=engine,
        models=models,
        online_renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    app = FastAPI()
    app.state.openai_serving_responses = serving
    return serving, app


def _request_body(
    payload: str,
    client: dict[str, str],
    mode: str,
    store_field: str,
) -> dict:
    body: dict = {
        "model": MODEL_NAME,
        "input": payload,
        "user": client["user"],
        "safety_identifier": client["safety_identifier"],
        "max_output_tokens": 1,
        "background": mode in ("background", "background_stream"),
        "stream": mode in ("stream", "background_stream"),
    }
    if store_field == "true":
        body["store"] = True
    elif store_field == "false":
        body["store"] = False
    return body


async def _post(
    app: FastAPI,
    serving: OpenAIServingResponses,
    body: dict,
    token: str,
) -> str:
    request = ResponsesRequest.model_validate(body)
    request_id = request.request_id
    response = await create_responses(request, _raw_request(app, token))
    if isinstance(response, StreamingResponse):
        if request_id in serving.background_tasks:
            await serving.background_tasks[request_id]
        chunks = [chunk async for chunk in response.body_iterator]
        if response.status_code != 200:
            pytest.fail(f"stream status {response.status_code}: {chunks!r}")
        return request_id
    assert isinstance(response, Response), type(response)
    payload = response.body.decode()
    if response.status_code != 200:
        pytest.fail(f"status {response.status_code}: {payload}")
    if request_id in serving.background_tasks:
        await serving.background_tasks[request_id]
    parsed = json.loads(payload)
    assert parsed["id"] == request_id
    return request_id


async def _status(response: Response) -> int:
    return response.status_code


def _assert_bounded(
    serving: OpenAIServingResponses,
    ids: list[str],
    payloads: list[str],
    *,
    keep_latest: bool,
) -> None:
    retained_bytes = _retained_payload_bytes(serving, payloads)
    stored = _stored_ids(serving)
    counts = (
        len(serving.response_store),
        len(serving.msg_store),
        len(serving.event_store),
    )
    if max(counts) > MAX_STORED_RESPONSES or retained_bytes > MAX_STORED_BYTES:
        raise AssertionError(
            "store kept "
            f"responses={counts[0]} messages={counts[1]} events={counts[2]} "
            f"payload_bytes={retained_bytes} "
            f"limits=({MAX_STORED_RESPONSES}, {MAX_STORED_BYTES})"
        )
    if ids[0] in stored:
        raise AssertionError("oldest response is still stored")
    if keep_latest and (
        ids[-1] not in serving.response_store or ids[-1] not in serving.msg_store
    ):
        raise AssertionError("newest opted-in response was not retained")
    if stored and not keep_latest and ids[-1] not in stored:
        raise AssertionError("store kept history but dropped the newest response")
    # Events for a response that has been evicted must be gone too.
    if not set(serving.event_store) <= set(serving.response_store):
        raise AssertionError("event_store holds responses missing from response_store")


@pytest.mark.asyncio
@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "mode",
    ["unary", "stream", "background", "background_stream"],
)
@pytest.mark.parametrize("store_field", ["omitted", "true"])
async def test_enabled_store_cannot_grow_without_bound(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    store_field: str,
):
    serving, app = _serving(monkeypatch, store_enabled=True)
    ids: list[str] = []
    payloads: list[str] = []
    for client in CLIENTS:
        for index in range(REQUESTS_PER_CLIENT):
            payload = _payload(client["user"], index)
            body = _request_body(payload, client, mode, store_field)
            request_id = await _post(app, serving, body, client["token"])
            ids.append(request_id)
            payloads.append(payload)

    _assert_bounded(
        serving,
        ids,
        payloads,
        keep_latest=store_field == "true",
    )

    missing = await retrieve_responses(
        ids[0],
        _raw_request(app, CLIENTS[0]["token"]),
        starting_after=None,
        stream=False,
    )
    assert await _status(missing) == 404

    latest = await retrieve_responses(
        ids[-1],
        _raw_request(app, CLIENTS[-1]["token"]),
        starting_after=None,
        stream=False,
    )
    if store_field == "true":
        assert await _status(latest) == 200
        assert json.loads(latest.body)["id"] == ids[-1]
        # A later read still leaves the history bounded.
        _assert_bounded(serving, ids, payloads, keep_latest=True)
    else:
        assert await _status(latest) in (200, 404)


@pytest.mark.asyncio
@pytest.mark.cpu_test
@pytest.mark.parametrize("mode", ["unary", "stream"])
async def test_store_false_retains_nothing(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    serving, app = _serving(monkeypatch, store_enabled=True)
    client = CLIENTS[0]
    payload = _payload(client["user"], 0)
    body = _request_body(payload, client, mode, "false")
    await _post(app, serving, body, client["token"])
    assert serving.response_store == {}
    assert serving.msg_store == {}
    assert serving.event_store == {}
    assert _retained_payload_bytes(serving, [payload]) == 0


@pytest.mark.asyncio
@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "mode",
    ["unary", "stream", "background", "background_stream"],
)
async def test_disabled_store_flag_retains_nothing(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    serving, app = _serving(monkeypatch, store_enabled=False)
    client = CLIENTS[0]
    payload = _payload(client["user"], 0)
    body = _request_body(payload, client, mode, "omitted")
    request = ResponsesRequest.model_validate(body)
    response = await create_responses(request, _raw_request(app, client["token"]))
    # Background asks for the default store, which this flag rejects.
    # Non-background requests are still served, with storage forced off.
    if mode in ("background", "background_stream"):
        assert isinstance(response, Response)
        assert response.status_code == 400
    else:
        if isinstance(response, StreamingResponse):
            _ = [chunk async for chunk in response.body_iterator]
            assert response.status_code == 200
        else:
            assert response.status_code == 200
    assert serving.response_store == {}
    assert serving.msg_store == {}
    assert serving.event_store == {}
    assert _retained_payload_bytes(serving, [payload]) == 0


@pytest.mark.asyncio
@pytest.mark.cpu_test
async def test_cancel_does_not_keep_an_unbounded_history(
    monkeypatch: pytest.MonkeyPatch,
):
    serving, app = _serving(monkeypatch, store_enabled=True)
    ids: list[str] = []
    payloads: list[str] = []
    client = CLIENTS[0]
    for index in range(REQUESTS_PER_CLIENT * len(CLIENTS)):
        payload = _payload(client["user"], index)
        body = _request_body(payload, client, "background", "true")
        request_id = await _post(app, serving, body, client["token"])
        cancelled = await cancel_responses(
            request_id, _raw_request(app, client["token"])
        )
        # Completed responses cannot be cancelled. That must not be required
        # in order to drop them; the cap still applies after the call.
        assert cancelled.status_code in (200, 400)
        ids.append(request_id)
        payloads.append(payload)

    _assert_bounded(serving, ids, payloads, keep_latest=True)
