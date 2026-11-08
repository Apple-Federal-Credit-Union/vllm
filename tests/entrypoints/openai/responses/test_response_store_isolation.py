# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Responses store must not be a process-wide capability keyed by request id.

These tests exercise ``POST /v1/responses``, ``GET /v1/responses/{id}``,
``GET /v1/responses/{id}?stream=true``, ``previous_response_id``, and
``POST /v1/responses/{id}/cancel`` on the real router and serving handler.

They fail while stored responses are keyed by a caller-supplied ``request_id``
and while retrieve, continue, and cancel ignore which bearer token created
the record. A later change should make them pass without weakening the
assertions below.

Same bearer token (or no token, when the server has no API keys) is one
principal: that principal may retrieve and continue by the id the server
returned. A different bearer token is a different principal. The text the
engine sees for a continuation must be the history captured for the
authorized response, not a later ``msg_store`` / ``response_store`` lookup.

The gpt-oss harmony branch (``msg_store[id]`` plus
``OnlineRenderer._render_responses_with_harmony``) is not executed here.
``HarmonyContext`` requires a ``HarmonyParser``, which these tests do not
construct. The non-harmony ``msg_store.get`` path is the one that runs.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport

from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.openai.responses.api_router import attach_router
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware
from vllm.inputs import tokens_input
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.online_renderer import OnlineRenderer

pytestmark = pytest.mark.skip_global_cleanup

_MODEL_NAME = "test-model"
_VICTIM_TOKEN = "victim-placeholder"
_ATTACKER_TOKEN = "attacker-placeholder"
_INSTRUCTION = "INSTR_VICTIM_ONLY"
_PROMPT = "PROMPT_VICTIM_ONLY"
_TOOL_NAME = "lookup_secret_tool"
_OUTPUT = "OUTPUT_VICTIM_ONLY"
_INJECTED = "INJECTED_OTHER_CLIENT"
_FRESH = "FRESH_GENERATION"

_TOOL = {
    "type": "function",
    "name": _TOOL_NAME,
    "description": "Looks up a private value",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


@dataclass
class _HFConfig:
    model_type: str = "llama"


@dataclass
class _ModelConfig:
    task: str = "generate"
    runner_type: str = "generate"
    model: str = _MODEL_NAME
    tokenizer: str = _MODEL_NAME
    trust_remote_code: bool = False
    tokenizer_mode: str = "auto"
    max_model_len: int = 256
    tokenizer_revision: str | None = None
    multimodal_config: Any = None
    hf_config: _HFConfig = field(default_factory=_HFConfig)
    hf_text_config: _HFConfig = field(default_factory=_HFConfig)
    logits_processors: list[str] | None = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False

    def get_diff_sampling_param(self) -> dict[str, Any]:
        return {}


class _Tokenizer:
    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del token_ids, kwargs
        return ""


class _JoiningRenderer:
    """Render chat messages by JSON-encoding the conversation.

    Message construction stays in ``construct_input_messages`` /
    ``OnlineRenderer.render_responses``. This only turns that list into the
    prompt string the engine echoes back, so ``input_messages`` shows the
    history the serving layer actually rendered.
    """

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    def get_tokenizer(self) -> _Tokenizer:
        return self.tokenizer

    async def render_chat_async(
        self,
        conversations: list[list[Any]],
        chat_params: Any,
        tok_params: Any = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ) -> tuple[list[list[Any]], list[Any]]:
        del chat_params, tok_params, prompt_extras, skip_mm_cache
        rendered: list[list[Any]] = []
        engines: list[Any] = []
        for conversation in conversations:
            text = json.dumps(conversation, default=str, sort_keys=True)
            rendered.append(conversation)
            engines.append(tokens_input([1], prompt=text))
        return rendered, engines


class _Engine:
    """Engine stand-in whose generate() echoes the rendered prompt.

    Blocking requests wait until cancelled or released, so background cancel
    can be observed on the request that was actually waiting.
    """

    def __init__(self, model_config: _ModelConfig, renderer: _JoiningRenderer) -> None:
        self.model_config = model_config
        self.renderer = renderer
        self.input_processor = None
        self.vllm_config = None
        self.block = False
        self.prompts: list[str] = []
        self.started_ids: list[str] = []
        self.cancelled_prompts: list[str] = []
        self._release = asyncio.Event()

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        return RuntimeError("engine stopped")

    def check_admission(self, n: int = 1, request_id: str | None = None) -> None:
        del n, request_id

    async def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        **kwargs: Any,
    ):
        del sampling_params, kwargs
        text = prompt.get("prompt") if isinstance(prompt, dict) else ""
        text = text or ""
        self.prompts.append(text)
        self.started_ids.append(request_id)
        if self.block:
            try:
                await self._release.wait()
            except asyncio.CancelledError:
                self.cancelled_prompts.append(text)
                raise
        yield _completion(request_id, text)

    def release(self) -> None:
        self._release.set()


def _completion(request_id: str, prompt: str) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt=prompt,
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=_OUTPUT,
                token_ids=(3,),
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        finished=True,
    )


class _SideEffectOnGet(dict):
    """dict.get that runs ``on_get`` after the first hit, before returning."""

    def __init__(self, source: dict, on_get: Callable[[str], None]) -> None:
        super().__init__(source)
        self._on_get = on_get
        self._fired = False

    def get(self, key: str, default: Any = None) -> Any:
        value = super().get(key, default)
        if value is not None and not self._fired:
            self._fired = True
            self._on_get(key)
        return value


class _SecondGet(dict):
    """First ``get`` returns the stored value; later gets return ``replacement``."""

    def __init__(self, source: dict, replacement: dict) -> None:
        super().__init__(source)
        self._replacement = replacement
        self._gets = 0

    def get(self, key: str, default: Any = None) -> Any:
        self._gets += 1
        if self._gets >= 2 and key in self._replacement:
            return self._replacement[key]
        return super().get(key, default)


@dataclass
class _App:
    client: httpx.AsyncClient
    handler: OpenAIServingResponses
    engine: _Engine


def _headers(token: str | None) -> dict[str, str]:
    if token is None:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _build_handler(
    engine: _Engine, renderer: _JoiningRenderer
) -> OpenAIServingResponses:
    online = OnlineRenderer(
        model_config=engine.model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    models = OpenAIModelRegistry(
        model_config=engine.model_config,
        base_model_paths=[BaseModelPath(name=_MODEL_NAME, model_path=_MODEL_NAME)],
    )
    return OpenAIServingResponses(
        engine_client=engine,
        models=models,
        online_renderer=online,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


def _build_app(handler: OpenAIServingResponses, api_keys: list[str] | None) -> FastAPI:
    app = FastAPI()
    app.state.openai_serving_responses = handler
    attach_router(app)
    if api_keys:
        app.add_middleware(AuthenticationMiddleware, tokens=api_keys)
    return app


@pytest_asyncio.fixture
async def isolated(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
    monkeypatch.setenv("VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", "0")
    model_config = _ModelConfig()
    renderer = _JoiningRenderer()
    engine = _Engine(model_config, renderer)
    handler = _build_handler(engine, renderer)
    app = _build_app(handler, [_VICTIM_TOKEN, _ATTACKER_TOKEN])
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://responses.test"
    ) as client:
        yield _App(client=client, handler=handler, engine=engine)


@pytest_asyncio.fixture
async def anonymous(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
    monkeypatch.setenv("VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT", "0")
    model_config = _ModelConfig()
    renderer = _JoiningRenderer()
    engine = _Engine(model_config, renderer)
    handler = _build_handler(engine, renderer)
    app = _build_app(handler, None)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://responses.test"
    ) as client:
        yield _App(client=client, handler=handler, engine=engine)


def _victim_body(**extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": _MODEL_NAME,
        "input": _PROMPT,
        "instructions": _INSTRUCTION,
        "tools": [_TOOL],
        "store": True,
        "enable_response_messages": True,
    }
    body.update(extra)
    return body


async def _create(app: _App, token: str | None, body: dict[str, Any]) -> httpx.Response:
    return await app.client.post(
        "/v1/responses",
        json=body,
        headers=_headers(token),
    )


async def _wait_for(predicate: Callable[[], bool]) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for the background request")
        await asyncio.sleep(0)


def _denied(response: httpx.Response) -> None:
    assert response.status_code in (401, 403, 404), response.text
    assert _PROMPT not in response.text
    assert _INSTRUCTION not in response.text
    assert _TOOL_NAME not in response.text
    assert _OUTPUT not in response.text
    assert _INJECTED not in response.text


@pytest.mark.asyncio
async def test_client_supplied_request_id_is_not_the_store_key(anonymous: _App) -> None:
    """A caller-chosen request_id must not be the id used to read the store."""
    chosen = "resp_chosen_by_caller"
    created = await _create(
        anonymous,
        None,
        _victim_body(request_id=chosen),
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["id"] != chosen
    assert body["instructions"] == _INSTRUCTION

    by_chosen = await anonymous.client.get(f"/v1/responses/{chosen}")
    _denied(by_chosen)

    by_server_id = await anonymous.client.get(f"/v1/responses/{body['id']}")
    assert by_server_id.status_code == 200, by_server_id.text
    assert by_server_id.json()["instructions"] == _INSTRUCTION
    assert _PROMPT in by_server_id.text
    assert _OUTPUT in by_server_id.text
    assert _TOOL_NAME in by_server_id.text


@pytest.mark.asyncio
async def test_other_bearer_cannot_retrieve_stored_response(isolated: _App) -> None:
    created = await _create(isolated, _VICTIM_TOKEN, _victim_body())
    assert created.status_code == 200, created.text
    response_id = created.json()["id"]

    stolen = await isolated.client.get(
        f"/v1/responses/{response_id}",
        headers=_headers(_ATTACKER_TOKEN),
    )
    _denied(stolen)

    owner = await isolated.client.get(
        f"/v1/responses/{response_id}",
        headers=_headers(_VICTIM_TOKEN),
    )
    assert owner.status_code == 200, owner.text
    payload = owner.json()
    assert payload["instructions"] == _INSTRUCTION
    assert _TOOL_NAME in owner.text
    assert _OUTPUT in owner.text
    assert _PROMPT in owner.text


@pytest.mark.asyncio
async def test_retrieve_returns_authorized_object_not_a_later_lookup(
    isolated: _App,
) -> None:
    """A second store lookup between the check and the response must not win."""
    created = await _create(isolated, _VICTIM_TOKEN, _victim_body())
    assert created.status_code == 200, created.text
    response_id = created.json()["id"]
    stored = isolated.handler.response_store[response_id]
    foreign = stored.model_copy(deep=True)
    foreign.instructions = _INJECTED
    foreign.output = []
    isolated.handler.response_store = _SecondGet(
        isolated.handler.response_store,
        {response_id: foreign},
    )

    owner = await isolated.client.get(
        f"/v1/responses/{response_id}",
        headers=_headers(_VICTIM_TOKEN),
    )
    assert owner.status_code == 200, owner.text
    assert owner.json()["instructions"] == _INSTRUCTION
    assert _INJECTED not in owner.text


@pytest.mark.asyncio
async def test_other_bearer_cannot_read_background_event_stream(
    isolated: _App,
) -> None:
    created = await _create(
        isolated,
        _VICTIM_TOKEN,
        _victim_body(background=True, stream=True),
    )
    assert created.status_code == 200, created.text
    assert _INSTRUCTION in created.text
    response_id = _response_id_from_sse(created.text)

    stolen = await isolated.client.get(
        f"/v1/responses/{response_id}",
        params={"stream": "true"},
        headers=_headers(_ATTACKER_TOKEN),
    )
    _denied(stolen)


@pytest.mark.asyncio
async def test_other_bearer_previous_response_id_does_not_load_history(
    isolated: _App,
) -> None:
    created = await _create(isolated, _VICTIM_TOKEN, _victim_body())
    assert created.status_code == 200, created.text
    response_id = created.json()["id"]
    before = len(isolated.engine.prompts)

    continued = await _create(
        isolated,
        _ATTACKER_TOKEN,
        {
            "model": _MODEL_NAME,
            "input": _FRESH,
            "previous_response_id": response_id,
            "store": False,
            "enable_response_messages": True,
        },
    )
    _denied(continued)
    new_prompts = isolated.engine.prompts[before:]
    assert all(_PROMPT not in prompt for prompt in new_prompts)
    assert all(_OUTPUT not in prompt for prompt in new_prompts)
    assert all(_INSTRUCTION not in prompt for prompt in new_prompts)


@pytest.mark.asyncio
async def test_continuation_uses_history_captured_for_the_authorized_response(
    isolated: _App,
) -> None:
    """History is fixed when the previous response is authorized.

    Rebinding ``msg_store`` inside the response-store lookup must not change
    the messages rendered for that response.
    """
    created = await _create(isolated, _VICTIM_TOKEN, _victim_body())
    assert created.status_code == 200, created.text
    response_id = created.json()["id"]

    def _rebind(key: str) -> None:
        isolated.handler.msg_store[key] = [
            {"role": "user", "content": _INJECTED},
        ]

    isolated.handler.response_store = _SideEffectOnGet(
        isolated.handler.response_store,
        _rebind,
    )
    before = len(isolated.engine.prompts)
    continued = await _create(
        isolated,
        _VICTIM_TOKEN,
        {
            "model": _MODEL_NAME,
            "input": _FRESH,
            "previous_response_id": response_id,
            "store": False,
            "enable_response_messages": True,
        },
    )
    assert continued.status_code == 200, continued.text
    assert _INJECTED not in continued.text
    rendered = isolated.engine.prompts[before:]
    assert rendered
    assert any(_PROMPT in prompt for prompt in rendered)
    assert all(_INJECTED not in prompt for prompt in rendered)


@pytest.mark.asyncio
async def test_other_bearer_cannot_cancel_background_response(isolated: _App) -> None:
    isolated.engine.block = True
    created = await _create(
        isolated,
        _VICTIM_TOKEN,
        _victim_body(background=True, stream=False, request_id="resp_hold_victim"),
    )
    assert created.status_code == 200, created.text
    response_id = created.json()["id"]
    await _wait_for(lambda: len(isolated.engine.started_ids) >= 1)
    stored = isolated.handler.response_store[response_id]
    assert stored.status in ("queued", "in_progress")

    cancelled = await isolated.client.post(
        f"/v1/responses/{response_id}/cancel",
        headers=_headers(_ATTACKER_TOKEN),
    )
    _denied(cancelled)
    assert isolated.handler.response_store[response_id].status in (
        "queued",
        "in_progress",
    )
    assert isolated.engine.cancelled_prompts == []
    isolated.engine.release()


@pytest.mark.asyncio
async def test_cancel_aborts_the_task_bound_to_the_authorized_response(
    isolated: _App,
) -> None:
    """Cancel must abort the task captured with the authorized response.

    Swapping ``background_tasks`` during the response-store lookup is the
    window between authorizing one response and acting on another.
    """
    isolated.engine.block = True
    first = await _create(
        isolated,
        _VICTIM_TOKEN,
        _victim_body(
            background=True,
            stream=False,
            input="PROMPT_HOLD_FIRST",
            request_id="resp_hold_first",
        ),
    )
    second = await _create(
        isolated,
        _VICTIM_TOKEN,
        _victim_body(
            background=True,
            stream=False,
            input="PROMPT_HOLD_SECOND",
            instructions="INSTR_HOLD_SECOND",
            request_id="resp_hold_second",
        ),
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_id = first.json()["id"]
    second_id = second.json()["id"]
    await _wait_for(lambda: len(isolated.engine.started_ids) >= 2)
    second_task = isolated.handler.background_tasks[second_id]

    def _swap(key: str) -> None:
        if key == first_id:
            isolated.handler.background_tasks[key] = second_task

    isolated.handler.response_store = _SideEffectOnGet(
        isolated.handler.response_store,
        _swap,
    )
    cancelled = await isolated.client.post(
        f"/v1/responses/{first_id}/cancel",
        headers=_headers(_VICTIM_TOKEN),
    )
    assert cancelled.status_code == 200, cancelled.text
    assert isolated.handler.response_store[second_id].status in (
        "queued",
        "in_progress",
    )
    cancelled_prompts = isolated.engine.cancelled_prompts
    assert not any("PROMPT_HOLD_SECOND" in prompt for prompt in cancelled_prompts)
    assert any("PROMPT_HOLD_FIRST" in prompt for prompt in cancelled_prompts)
    isolated.engine.release()


def _response_id_from_sse(payload: str) -> str:
    for line in payload.splitlines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line.removeprefix("data: "))
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            return response["id"]
    raise AssertionError("background stream did not include a response id")
