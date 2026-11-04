# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API key must cover every route the server actually dispatches.

``AuthenticationMiddleware`` currently authenticates only paths under
``/v1``, ``/v2``, ``/inference``, and ``/cohere``. These tests build the
production FastAPI app and require a configured key before any other
inference or control handler runs. ``/health`` and ``/ping`` stay public.

The assertions fail while unprefixed routes still run their handlers
without a key.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import threading
from argparse import Namespace
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from starlette._utils import get_route_path

from vllm.config import ProfilerConfig
from vllm.entrypoints.generate.generative_scoring.serving import (
    GenerativeScoringItemResult,
    GenerativeScoringResponse,
)
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponse,
    CompletionResponseChoice,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeResponse,
    TokenizeResponse,
    TokenizerInfoResponse,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Obvious placeholder. Not a deployed credential.
_API_KEY = "placeholder-api-key"
_AUTH = {"Authorization": f"Bearer {_API_KEY}"}
_WRONG = {"Authorization": "Bearer wrong-placeholder"}

_CHAT_BODY = {"messages": [{"role": "user", "content": "probe-chat"}]}
_COMPLETION_BODY = {"prompt": "probe-completion"}
_CLASSIFY_BODY = {"input": "probe-classify"}
_POOL_BODY = {"input": "probe-pool"}
_SCORE_BODY = {"queries": "probe-query", "documents": "probe-doc"}
_RERANK_BODY = {"query": "probe-query", "documents": ["probe-doc"]}
_TOKENIZE_BODY = {"prompt": "probe-tokenize"}
_DETOKENIZE_BODY = {"tokens": [3, 1, 4]}
_SCORE_GEN_BODY = {
    "query": "probe-query",
    "items": ["probe-item"],
    "label_token_ids": [7],
}
_SCALE_BODY = {"new_data_parallel_size": 2, "drain_timeout": 5}
_FAULT_BODY = {"instruction": "retry", "params": {"rank": 0}}


class CallLog:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, name: str, **details: Any) -> None:
        self.events.append({"name": name, **details})

    def clear(self) -> None:
        self.events.clear()

    def names(self) -> list[str]:
        return [event["name"] for event in self.events]


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return message.content


class _ParallelConfig:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    @property
    def world_size_across_dp(self) -> int:
        self._log.record("world_size")
        return 4

    @property
    def world_size(self) -> int:
        self._log.record("world_size")
        return 2


class _ServerInfoConfig:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    def __str__(self) -> str:
        self._log.record("server_info")
        return "server-info-sentinel"


class _Engine:
    def __init__(self, log: CallLog) -> None:
        self.log = log
        self.vllm_config = Namespace(parallel_config=_ParallelConfig(log))

    async def check_health(self) -> None:
        self.log.record("check_health")

    async def scale_elastic_ep(self, new_size: int, drain_timeout: int) -> None:
        self.log.record(
            "scale_elastic_ep",
            marker=new_size,
            drain_timeout=drain_timeout,
        )

    async def get_external_elastic_ep_status(self) -> dict[str, Any]:
        self.log.record("elastic_status")
        return {
            "phase": "idle",
            "epoch": None,
            "error": None,
            "requested_data_parallel_size": None,
        }

    async def handle_fault(self, request: Any) -> Namespace:
        self.log.record("fault_apply", marker=request.instruction)
        return Namespace(success=True, reason=None)

    async def get_status(self) -> dict[str, str]:
        self.log.record("fault_status")
        return {"phase": "idle"}

    async def get_kv_event_sources(self) -> dict[str, str]:
        self.log.record("kv_event_sources")
        return {"source": "probe"}

    def abort(self, request_ids: list[str], internal: bool = False) -> Any:
        # Record when the route invokes abort. The tokens-only route schedules
        # the returned coroutine without awaiting it.
        self.log.record("abort", marker=list(request_ids), internal=internal)

        async def _finish() -> None:
            return None

        return _finish()

    async def pause_generation(self, mode: str, clear_cache: bool) -> None:
        self.log.record("pause", marker=mode, clear_cache=clear_cache)

    async def resume_generation(self) -> None:
        self.log.record("resume")

    async def is_paused(self) -> bool:
        self.log.record("is_paused")
        return False

    async def sleep(self, level: int, mode: str) -> None:
        self.log.record("sleep", marker=level, mode=mode)

    async def release_kv_cache_memory(self) -> None:
        self.log.record("release_kv")

    async def wake_up(self, tags: list[str] | None) -> bool:
        self.log.record("wake", marker=tags)
        return True

    async def is_sleeping(self) -> bool:
        self.log.record("is_sleeping")
        return False

    async def reset_prefix_cache(
        self, reset_running_requests: bool, reset_external: bool
    ) -> bool:
        self.log.record(
            "reset_prefix",
            running=reset_running_requests,
            external=reset_external,
        )
        return True

    async def reset_mm_cache(self) -> None:
        self.log.record("reset_mm")

    async def reset_encoder_cache(self) -> None:
        self.log.record("reset_encoder")

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None,
        args: tuple[str, ...],
        kwargs: dict[str, str],
    ) -> list[str]:
        self.log.record("collective_rpc", marker=method)
        return ["probe"]

    async def start_profile(self, **overrides: Any) -> None:
        self.log.record("start_profile", marker=sorted(overrides))

    async def stop_profile(self) -> None:
        self.log.record("stop_profile")

    async def init_weight_transfer_engine(self, request: Any) -> None:
        self.log.record("init_weights", marker=request.init_info)

    async def shutdown_weight_transfer_engine(self) -> None:
        self.log.record("shutdown_weights")

    async def start_weight_update(self) -> None:
        self.log.record("start_weights")

    async def start_draft_weight_update(self) -> None:
        self.log.record("start_draft_weights")

    async def update_weights(self, request: Any) -> None:
        self.log.record("update_weights", marker=request.update_info)

    async def finish_weight_update(self, weight_version: str | None) -> None:
        self.log.record("finish_weights", marker=weight_version)

    async def update_weight_version(self, version: str) -> None:
        self.log.record("update_version", marker=version)

    async def get_weight_version(self) -> str:
        self.log.record("weight_info")
        return "probe-version"

    async def compute_weight_checksums(self) -> list[dict[str, str]]:
        self.log.record("weight_checksum")
        return [{"probe": "1"}]


class _Chat:
    def __init__(self, log: CallLog) -> None:
        self.log = log

    async def create_chat_completion(self, request: Any, raw_request: Any) -> Any:
        self.log.record(
            "chat",
            route=get_route_path(raw_request.scope),
            scope_path=raw_request.scope["path"],
            marker=_message_content(request.messages[0]),
        )
        return ChatCompletionResponse(
            model="probe-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="probe-reply"),
                )
            ],
            usage=UsageInfo(),
        )


class _Completion:
    def __init__(self, log: CallLog) -> None:
        self.log = log

    async def create_completion(self, request: Any, raw_request: Any) -> Any:
        self.log.record(
            "completion",
            route=get_route_path(raw_request.scope),
            marker=request.prompt,
        )
        return CompletionResponse(
            model="probe-model",
            choices=[CompletionResponseChoice(index=0, text="probe-reply")],
            usage=UsageInfo(),
        )


class _Tokenization:
    def __init__(self, log: CallLog) -> None:
        self.log = log

    async def create_tokenize(self, request: Any, raw_request: Any) -> Any:
        prompt = getattr(request, "prompt", None)
        self.log.record(
            "tokenize",
            route=get_route_path(raw_request.scope),
            marker=prompt,
        )
        return TokenizeResponse(count=1, max_model_len=8, tokens=[3])

    async def create_detokenize(self, request: Any, raw_request: Any) -> Any:
        self.log.record(
            "detokenize",
            route=get_route_path(raw_request.scope),
            marker=list(request.tokens),
        )
        return DetokenizeResponse(prompt="probe-text")

    async def get_tokenizer_info(self) -> TokenizerInfoResponse:
        self.log.record("tokenizer_info")
        return TokenizerInfoResponse(tokenizer_class="ProbeTokenizer")


class _GenerativeScoring:
    def __init__(self, log: CallLog) -> None:
        self.log = log

    async def create_generative_scoring(self, request: Any, raw_request: Any) -> Any:
        self.log.record(
            "generative_scoring",
            route=get_route_path(raw_request.scope),
            marker=request.query,
        )
        return GenerativeScoringResponse(
            model="probe-model",
            data=[GenerativeScoringItemResult(index=0, score=0.25)],
            usage=UsageInfo(),
        )


class _PoolingHandler:
    def __init__(self, log: CallLog, name: str) -> None:
        self.log = log
        self.name = name

    async def __call__(self, request: Any, raw_request: Any) -> Any:
        from fastapi.responses import JSONResponse

        marker = getattr(request, "input", None)
        if marker is None:
            marker = getattr(request, "query", None)
        if marker is None:
            marker = getattr(request, "queries", None)
        self.log.record(
            self.name,
            route=get_route_path(raw_request.scope),
            marker=marker,
        )
        return JSONResponse({"handler": self.name, "ok": True})


class _PoolingModelConfig:
    def __init__(self) -> None:
        self.hf_config = Namespace(num_labels=1)
        self.architectures: list[str] = []

    def get_pooling_task(self, supported_tasks: tuple[str, ...]) -> str | None:
        if "classify" in supported_tasks:
            return "classify"
        return None


def _args(**overrides: Any) -> Namespace:
    args = Namespace(
        disable_fastapi_docs=False,
        enable_offline_docs=False,
        root_path=None,
        allowed_origins=[],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=[_API_KEY],
        enable_request_id_headers=False,
        enable_fault_tolerance=False,
        middleware=[],
        log_error_stack=False,
        enable_tokenizer_info_endpoint=False,
        profiler_config=None,
        tokens_only=False,
        enable_scale_out=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_common(app: Any, log: CallLog) -> _Engine:
    engine = _Engine(log)
    app.state.log_stats = False
    app.state.server_load_metrics = 17
    app.state.engine_client = engine
    app.state.openai_serving_chat = _Chat(log)
    app.state.openai_serving_completion = _Completion(log)
    app.state.serving_tokenization = _Tokenization(log)
    app.state.serving_generative_scoring = _GenerativeScoring(log)
    app.state.serving_classification = _PoolingHandler(log, "classify")
    app.state.serving_pooling = _PoolingHandler(log, "pooling")
    app.state.serving_scores = _PoolingHandler(log, "scores")
    app.state.vllm_config = _ServerInfoConfig(log)
    return engine


def _build(
    log: CallLog,
    tasks: tuple[str, ...],
    *,
    model_config: Any = None,
    **arg_overrides: Any,
) -> Any:
    app = build_app(_args(**arg_overrides), tasks, model_config)
    _install_common(app, log)
    return app


def _send(
    client: TestClient | httpx.Client,
    method: str,
    path: str,
    body: Any,
    headers: dict[str, str] | None,
) -> httpx.Response:
    kwargs: dict[str, Any] = {"headers": headers or {}}
    if body is not None:
        kwargs["json"] = body
    return client.request(method, path, **kwargs)


def _assert_key_gates_handler(
    client: TestClient,
    log: CallLog,
    method: str,
    path: str,
    body: Any,
    effect: str,
    ok_status: int,
    marker: Any = None,
) -> None:
    """A missing or wrong key must not run ``effect``. The real key must."""
    log.clear()
    denied = _send(client, method, path, body, None)
    assert denied.status_code == 401, f"{method} {path} returned {denied.status_code}"
    assert log.events == [], f"{method} {path} ran {log.names()} without a key"

    log.clear()
    wrong = _send(client, method, path, body, _WRONG)
    assert wrong.status_code == 401, f"{method} {path} accepted a wrong key"
    assert log.events == [], f"{method} {path} ran {log.names()} with a wrong key"

    log.clear()
    allowed = _send(client, method, path, body, _AUTH)
    assert allowed.status_code == ok_status, (
        f"{method} {path} with the configured key returned {allowed.status_code}"
    )
    assert log.names() == [effect], (
        f"{method} {path} ran {log.names()}, expected only {effect}"
    )
    event = log.events[0]
    if "route" in event:
        # The handler must observe the path the router dispatched, not a
        # different path that happened to be presented to the auth check.
        assert (
            event["route"] == path
            or event["scope_path"] == path
            or (event["route"] == path.split("?", 1)[0])
        ), event
    if marker is not None:
        assert event.get("marker") == marker, event
    log.clear()


def _client(app: Any) -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def generate_app() -> tuple[TestClient, CallLog]:
    log = CallLog()
    app = _build(
        log,
        ("generate",),
        enable_fault_tolerance=True,
        enable_tokenizer_info_endpoint=True,
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=tempfile.mkdtemp(prefix="vllm-probe-profiler-"),
        ),
    )
    with _client(app) as client:
        yield client, log


@pytest.fixture(scope="module")
def pooling_app() -> tuple[TestClient, CallLog]:
    log = CallLog()
    app = _build(log, ("classify",), model_config=_PoolingModelConfig())
    with _client(app) as client:
        yield client, log


@pytest.fixture(scope="module")
def dev_app() -> tuple[TestClient, CallLog]:
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    log = CallLog()
    try:
        app = _build(log, ("generate",))
    finally:
        os.environ.pop("VLLM_SERVER_DEV_MODE", None)
    with _client(app) as client:
        yield client, log


@pytest.fixture(scope="module")
def tokens_only_app() -> tuple[TestClient, CallLog]:
    log = CallLog()
    app = _build(log, ("generate",), tokens_only=True)
    with _client(app) as client:
        yield client, log


# method, path, body, effect, ok status, marker
_GENERATE_CASES = [
    ("POST", "/v1/chat/completions", _CHAT_BODY, "chat", 200, "probe-chat"),
    (
        "POST",
        "/v1/completions",
        _COMPLETION_BODY,
        "completion",
        200,
        "probe-completion",
    ),
    ("POST", "/invocations", _CHAT_BODY, "chat", 200, "probe-chat"),
    (
        "POST",
        "/invocations",
        _COMPLETION_BODY,
        "completion",
        200,
        "probe-completion",
    ),
    ("POST", "/tokenize", _TOKENIZE_BODY, "tokenize", 200, "probe-tokenize"),
    ("POST", "/detokenize", _DETOKENIZE_BODY, "detokenize", 200, [3, 1, 4]),
    ("GET", "/tokenizer_info", None, "tokenizer_info", 200, None),
    (
        "POST",
        "/generative_scoring",
        _SCORE_GEN_BODY,
        "generative_scoring",
        200,
        "probe-query",
    ),
    ("POST", "/scale_elastic_ep", _SCALE_BODY, "scale_elastic_ep", 200, 2),
    ("POST", "/is_scaling_elastic_ep", None, "elastic_status", 200, None),
    ("POST", "/fault_tolerance/apply", _FAULT_BODY, "fault_apply", 202, "retry"),
    ("GET", "/fault_tolerance/status", None, "fault_status", 200, None),
    ("GET", "/kv_event_sources", None, "kv_event_sources", 200, None),
    ("POST", "/start_profile", None, "start_profile", 200, None),
    ("POST", "/stop_profile", None, "stop_profile", 200, None),
]

_POOL_CASES = [
    ("POST", "/classify", _CLASSIFY_BODY, "classify", 200, "probe-classify"),
    ("POST", "/pooling", _POOL_BODY, "pooling", 200, "probe-pool"),
    ("POST", "/score", _SCORE_BODY, "scores", 200, "probe-query"),
    ("POST", "/rerank", _RERANK_BODY, "scores", 200, "probe-query"),
    ("POST", "/v1/score", _SCORE_BODY, "scores", 200, "probe-query"),
    ("POST", "/invocations", _CLASSIFY_BODY, "classify", 200, "probe-classify"),
    ("POST", "/invocations", _SCORE_BODY, "scores", 200, "probe-query"),
    ("POST", "/invocations", _RERANK_BODY, "scores", 200, "probe-query"),
]

_DEV_CASES = [
    ("POST", "/pause", None, "pause", 200, "abort"),
    ("POST", "/resume", None, "resume", 200, None),
    (
        "POST",
        "/abort_requests",
        {"request_ids": ["probe-req"]},
        "abort",
        200,
        ["probe-req"],
    ),
    ("GET", "/is_paused", None, "is_paused", 200, None),
    ("POST", "/sleep", None, "sleep", 200, 1),
    ("POST", "/release_kv_cache_memory", None, "release_kv", 200, None),
    ("POST", "/wake_up", None, "wake", 200, None),
    ("GET", "/is_sleeping", None, "is_sleeping", 200, None),
    ("POST", "/reset_prefix_cache", None, "reset_prefix", 200, None),
    ("POST", "/reset_mm_cache", None, "reset_mm", 200, None),
    ("POST", "/reset_encoder_cache", None, "reset_encoder", 200, None),
    (
        "POST",
        "/collective_rpc",
        {"method": "probe_method"},
        "collective_rpc",
        200,
        "probe_method",
    ),
    ("GET", "/server_info", None, "server_info", 200, None),
    ("POST", "/init_weight_transfer_engine", {}, "init_weights", 200, {}),
    ("POST", "/shutdown_weight_transfer_engine", None, "shutdown_weights", 200, None),
    ("POST", "/start_weight_update", None, "start_weights", 200, None),
    ("POST", "/start_draft_weight_update", None, "start_draft_weights", 200, None),
    ("POST", "/update_weights", {}, "update_weights", 200, {}),
    ("POST", "/finish_weight_update", None, "finish_weights", 200, None),
    (
        "POST",
        "/update_weight_version",
        {"new_version": "probe-version"},
        "update_version",
        200,
        "probe-version",
    ),
    ("GET", "/weight_info", None, "weight_info", 200, None),
    ("POST", "/weight_checker", {"action": "checksum"}, "weight_checksum", 200, None),
    ("GET", "/get_world_size", None, "world_size", 200, None),
]


@pytest.mark.parametrize(
    ("method", "path", "body", "effect", "ok_status", "marker"),
    _GENERATE_CASES,
    ids=[f"{index}:{item[3]}" for index, item in enumerate(_GENERATE_CASES)],
)
def test_generate_routes_require_api_key(
    generate_app: tuple[TestClient, CallLog],
    method: str,
    path: str,
    body: Any,
    effect: str,
    ok_status: int,
    marker: Any,
) -> None:
    client, log = generate_app
    _assert_key_gates_handler(
        client, log, method, path, body, effect, ok_status, marker
    )


@pytest.mark.parametrize(
    ("method", "path", "body", "effect", "ok_status", "marker"),
    _POOL_CASES,
    ids=[f"{index}:{item[3]}" for index, item in enumerate(_POOL_CASES)],
)
def test_pooling_routes_require_api_key(
    pooling_app: tuple[TestClient, CallLog],
    method: str,
    path: str,
    body: Any,
    effect: str,
    ok_status: int,
    marker: Any,
) -> None:
    client, log = pooling_app
    _assert_key_gates_handler(
        client, log, method, path, body, effect, ok_status, marker
    )


@pytest.mark.parametrize(
    ("method", "path", "body", "effect", "ok_status", "marker"),
    _DEV_CASES,
    ids=[f"{index}:{item[3]}" for index, item in enumerate(_DEV_CASES)],
)
def test_dev_routes_require_api_key(
    dev_app: tuple[TestClient, CallLog],
    method: str,
    path: str,
    body: Any,
    effect: str,
    ok_status: int,
    marker: Any,
) -> None:
    client, log = dev_app
    _assert_key_gates_handler(
        client, log, method, path, body, effect, ok_status, marker
    )


def test_tokens_only_abort_requires_api_key(
    tokens_only_app: tuple[TestClient, CallLog],
) -> None:
    client, log = tokens_only_app
    _assert_key_gates_handler(
        client,
        log,
        "POST",
        "/abort_requests",
        {"request_ids": ["probe-req"]},
        "abort",
        200,
        ["probe-req"],
    )


def test_prefixed_abort_stays_bound_to_the_same_request_ids(
    tokens_only_app: tuple[TestClient, CallLog],
) -> None:
    """``/inference/v1/abort_requests`` is already behind the prefix check.

    The guarded route has to abort the ids in that request, which keeps a
    later default-deny change from swapping in a different target.
    """
    client, log = tokens_only_app
    _assert_key_gates_handler(
        client,
        log,
        "POST",
        "/inference/v1/abort_requests",
        {"request_ids": ["probe-req"]},
        "abort",
        200,
        ["probe-req"],
    )


def test_health_and_ping_remain_public_and_run_the_health_handler(
    generate_app: tuple[TestClient, CallLog],
) -> None:
    client, log = generate_app
    for method, path in (
        ("GET", "/health"),
        ("GET", "/ping"),
        ("POST", "/ping"),
    ):
        log.clear()
        response = client.request(method, path)
        assert response.status_code == 200, f"{method} {path} -> {response.status_code}"
        assert log.names() == ["check_health"], log.names()


def test_utility_routes_require_api_key(
    generate_app: tuple[TestClient, CallLog],
) -> None:
    client, log = generate_app
    for method, path in (
        ("GET", "/load"),
        ("GET", "/version"),
        ("GET", "/metrics"),
        ("GET", "/openapi.json"),
        ("GET", "/docs"),
    ):
        log.clear()
        denied = client.request(method, path)
        assert denied.status_code == 401, f"{method} {path} -> {denied.status_code}"
        assert log.events == []
        allowed = client.request(method, path, headers=_AUTH)
        assert allowed.status_code == 200, f"{method} {path} -> {allowed.status_code}"
        log.clear()


def test_options_does_not_run_the_invocation_handler(
    generate_app: tuple[TestClient, CallLog],
) -> None:
    client, log = generate_app
    log.clear()
    response = client.options("/invocations")
    assert log.events == [], log.names()
    assert response.status_code != 200 or log.events == []


def test_root_path_under_a_guarded_prefix_still_reaches_invocations() -> None:
    """Router strips ``root_path`` before matching. Auth must use that path.

    Uvicorn stores ``scope["root_path"]`` and prefixes ``scope["path"]`` with
    it. Starlette then matches routes on the stripped path. A check of the
    unstripped ``/v1/invocations`` string is not the path that runs.
    """
    log = CallLog()
    app = _build(log, ("generate",))
    with TestClient(app, root_path="/v1") as client:
        denied = client.post("/v1/invocations", json=_CHAT_BODY)
        assert denied.status_code == 401, denied.status_code
        assert log.events == [], log.names()

        log.clear()
        allowed = client.post("/v1/invocations", json=_CHAT_BODY, headers=_AUTH)
        assert allowed.status_code == 200, allowed.status_code
        assert log.names() == ["chat"]
        assert log.events[0]["route"] == "/invocations"
        assert log.events[0]["scope_path"] == "/v1/invocations"
        assert log.events[0]["marker"] == "probe-chat"


def _raw_http(
    port: int,
    target: str,
    body: bytes,
    headers: list[tuple[str, str]],
) -> int:
    lines = [
        f"POST {target} HTTP/1.1",
        f"Host: 127.0.0.1:{port}",
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
        "Connection: close",
    ]
    lines.extend(f"{name}: {value}" for name, value in headers)
    payload = ("\r\n".join(lines) + "\r\n\r\n").encode() + body
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(payload)
        chunks: list[bytes] = []
        while True:
            piece = sock.recv(65536)
            if not piece:
                break
            chunks.append(piece)
    head = b"".join(chunks).split(b"\r\n", 1)[0]
    return int(head.split(b" ", 2)[1])


def _serve_until(app: Any, http: str) -> tuple[uvicorn.Server, int, threading.Thread]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        app,
        log_level="error",
        access_log=False,
        http=http,
        lifespan="on",
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        asyncio.run(server.serve(sockets=[sock]))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            return server, port, thread
        thread.join(0.05)
    raise RuntimeError(f"uvicorn {http} did not start")


def _stop(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(10)


@pytest.mark.parametrize("http", ["httptools", "h11"])
def test_parsed_http_request_cannot_invoke_without_api_key(http: str) -> None:
    """Same app behind the HTTP parsers vLLM's uvicorn server uses.

    ``http='auto'`` selects httptools when that package is installed, which
    is the lockfile used here. h11 is the other parser uvicorn ships and the
    one ``serve_http`` configures header limits for.
    """
    import json

    log = CallLog()
    app = _build(log, ("generate",))
    server, port, thread = _serve_until(app, http)
    body = json.dumps(_CHAT_BODY).encode()
    try:
        log.clear()
        status = _raw_http(port, "/invocations", body, [])
        assert status == 401, status
        assert log.events == [], log.names()

        log.clear()
        status = _raw_http(port, "/%69nvocations", body, [])
        assert status == 401, status
        assert log.events == [], log.names()

        log.clear()
        status = _raw_http(port, "/v1/chat/completions", body, [])
        assert status == 401, status
        assert log.events == [], log.names()

        log.clear()
        status = _raw_http(
            port,
            "/invocations",
            body,
            [("Authorization", f"Bearer {_API_KEY}")],
        )
        assert status == 200, status
        assert log.names() == ["chat"]
        assert log.events[0]["route"] == "/invocations"
        assert log.events[0]["marker"] == "probe-chat"

        log.clear()
        status = _raw_http(
            port,
            "/%69nvocations",
            body,
            [("Authorization", f"Bearer {_API_KEY}")],
        )
        assert status == 200, status
        assert log.events[0]["route"] == "/invocations"
        assert log.events[0]["scope_path"] == "/invocations"
        assert log.events[0]["marker"] == "probe-chat"
    finally:
        _stop(server, thread)
