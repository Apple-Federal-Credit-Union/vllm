# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API-key checks must use the path the router matches.

``AuthenticationMiddleware`` drops ``scope["root_path"]`` with
``str.removeprefix`` and then decides whether to call ``verify_token``.
Starlette matches routes with ``get_route_path``, which only drops a
``root_path`` on a path-segment boundary. These tests send the original
URL through FastAPI, so a fix that strips the path the same way again
still lets the guarded handler run.
"""

import importlib.util
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

# Obvious placeholder. Not a credential.
PLACEHOLDER_API_KEY = "placeholder-token"
HANDLER_MARKER = "guarded-handler-ran"

# Real guarded routes. Each prefix in GUARDED_PREFIX has at least one.
GUARDED_ROUTES = (
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/responses"),
    ("GET", "/v1/models"),
    ("POST", "/v1/audio/transcriptions"),
    ("POST", "/v1/score"),
    ("POST", "/v2/embed"),
    ("POST", "/v2/rerank"),
    ("POST", "/inference/v1/generate"),
    ("POST", "/inference/v1/abort_requests"),
    ("POST", "/cohere/v2/chat"),
    ("POST", "/cohere/v2/chat/render"),
)

# root_path values that removeprefix eats without being a segment boundary.
# The router still matches the guarded route.
NON_BOUNDARY_ROOTS = (
    ("/", "/v1/chat/completions", "POST"),
    ("/v", "/v1/chat/completions", "POST"),
    ("/v1/", "/v1/chat/completions", "POST"),
    ("/v1/chat/", "/v1/chat/completions", "POST"),
    ("/", "/v1/completions", "POST"),
    ("/", "/v1/embeddings", "POST"),
    ("/", "/v1/messages", "POST"),
    ("/", "/v1/responses", "POST"),
    ("/", "/v1/models", "GET"),
    ("/", "/v1/audio/transcriptions", "POST"),
    ("/", "/v1/score", "POST"),
    ("/", "/v2/embed", "POST"),
    ("/v2/", "/v2/embed", "POST"),
    ("/", "/v2/rerank", "POST"),
    ("/", "/inference/v1/generate", "POST"),
    ("/inference/", "/inference/v1/generate", "POST"),
    ("/", "/inference/v1/abort_requests", "POST"),
    ("/", "/cohere/v2/chat", "POST"),
    ("/cohere/", "/cohere/v2/chat", "POST"),
    ("/", "/cohere/v2/chat/render", "POST"),
)


def _load_auth_module():
    """Load the middleware file without importing the vLLM package.

    ``import vllm`` imports torch via ``vllm.env_override``. This file does
    not. The loaded source is the repository middleware.
    """
    path = (
        Path(__file__).resolve().parents[4]
        / "vllm/entrypoints/serve/middleware/authenticate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_root_path_auth_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_AUTH = _load_auth_module()
AuthenticationMiddleware = _AUTH.AuthenticationMiddleware
GUARDED_PREFIX = _AUTH.GUARDED_PREFIX


def _build_app(root_path: str | None, routes=GUARDED_ROUTES):
    """Wire root_path and auth the way the server does.

    ``build_app`` assigns ``app.root_path``. ``init_entrypoints_middleware``
    passes the configured keys to ``AuthenticationMiddleware``. Uvicorn's
    own root_path stays empty; FastAPI copies ``app.root_path`` into the
    scope before middleware runs.
    """
    calls: list[str] = []
    app = FastAPI()
    app.root_path = root_path

    for method, route_path in routes:
        def handler(request: Request, route_path=route_path):
            calls.append(route_path)
            return JSONResponse(
                {
                    HANDLER_MARKER: True,
                    "route": route_path,
                    "scope_path": request.scope["path"],
                }
            )

        app.add_api_route(route_path, handler, methods=[method])

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket):
        calls.append("/v1/realtime")
        await websocket.accept()
        await websocket.send_json({HANDLER_MARKER: True, "route": "/v1/realtime"})

    app.add_middleware(AuthenticationMiddleware, tokens=[PLACEHOLDER_API_KEY])
    return app, calls


def _assert_handler_not_run(response, calls: list[str], route: str) -> None:
    assert response.status_code == 401, response.text
    assert calls == [], calls
    assert HANDLER_MARKER not in response.text
    assert route not in calls


def test_every_guarded_prefix_has_a_route():
    covered = set()
    for _, path in GUARDED_ROUTES:
        for prefix in GUARDED_PREFIX:
            if path.startswith(prefix):
                covered.add(prefix)
    assert covered == set(GUARDED_PREFIX)


@pytest.mark.parametrize(
    ("root_path", "route", "method"),
    NON_BOUNDARY_ROOTS,
)
def test_non_boundary_root_path_requires_token(root_path, route, method):
    app, calls = _build_app(root_path)
    with TestClient(app) as client:
        response = client.request(method, route)
    _assert_handler_not_run(response, calls, route)


@pytest.mark.parametrize(
    ("root_path", "route", "method"),
    (
        ("/", "/v1/chat/completions", "POST"),
        ("/v1/", "/v1/chat/completions", "POST"),
        ("/v", "/v1/chat/completions", "POST"),
    ),
)
def test_wrong_token_does_not_run_handler(root_path, route, method):
    app, calls = _build_app(root_path)
    headers = {"Authorization": "Bearer placeholder-wrong"}
    with TestClient(app) as client:
        response = client.request(method, route, headers=headers)
    _assert_handler_not_run(response, calls, route)


def test_segment_root_path_authorizes_the_route_that_runs():
    """A proxy prefix on a segment boundary still guards the matched route.

    The client sends ``/api/v1/chat/completions``. The router strips ``/api``
    and runs the ``/v1/chat/completions`` handler. That handler is the
    object that must have been authorized.
    """
    app, calls = _build_app("/api")
    with TestClient(app) as client:
        rejected = client.post("/api/v1/chat/completions")
        accepted = client.post(
            "/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"},
        )
    assert rejected.status_code == 401, rejected.text
    assert HANDLER_MARKER not in rejected.text
    assert accepted.status_code == 200, accepted.text
    assert calls == ["/v1/chat/completions"]
    assert accepted.json()["route"] == "/v1/chat/completions"
    assert accepted.json()["scope_path"] == "/api/v1/chat/completions"


def test_root_path_equal_to_guarded_prefix_does_not_run_handler():
    """``--root-path /v1`` with a request for ``/v1/chat/completions``.

    The router and the middleware both see a remainder of
    ``/chat/completions``, so the guarded handler must not run.
    """
    app, calls = _build_app("/v1")
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions")
    assert calls == []
    assert HANDLER_MARKER not in response.text
    assert response.status_code != 200


def test_unrelated_root_path_still_requires_token():
    app, calls = _build_app("/llm")
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions")
    _assert_handler_not_run(response, calls, "/v1/chat/completions")


def test_unset_root_path_still_requires_token():
    app, calls = _build_app(None)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions")
    _assert_handler_not_run(response, calls, "/v1/chat/completions")


def test_placeholder_token_reaches_original_chat_route():
    app, calls = _build_app("/")
    headers = {"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"}
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", headers=headers)
    assert response.status_code == 200, response.text
    assert calls == ["/v1/chat/completions"]
    assert response.json()["scope_path"] == "/v1/chat/completions"


def test_realtime_websocket_requires_token_when_root_path_is_slash():
    app, calls = _build_app("/")
    received = None
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/v1/realtime") as websocket:
                received = websocket.receive_json()
        except Exception:
            received = None
    assert calls == []
    assert received is None


def test_realtime_websocket_accepts_placeholder_token():
    app, calls = _build_app("/")
    headers = {"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"}
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers=headers) as websocket:
            received = websocket.receive_json()
    assert calls == ["/v1/realtime"]
    assert received[HANDLER_MARKER] is True


def _serve(app: FastAPI) -> tuple[uvicorn.Server, int, threading.Thread]:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        http="h11",
        log_level="error",
        access_log=False,
        timeout_keep_alive=1,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    sockets = server.servers[0].sockets
    assert sockets is not None
    port = sockets[0].getsockname()[1]
    assert isinstance(port, int)
    return server, port, thread


def _stop(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)


def test_h11_server_chat_requires_token_for_non_boundary_root_path():
    cases = ("/", "/v1/", "/v")
    for root_path in cases:
        app, calls = _build_app(root_path)
        server, port, thread = _serve(app)
        try:
            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            with httpx.Client(timeout=5) as client:
                rejected = client.post(url)
                wrong = client.post(
                    url,
                    headers={"Authorization": "Bearer placeholder-wrong"},
                )
                accepted = client.post(
                    url,
                    headers={"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"},
                )
        finally:
            _stop(server, thread)
        assert rejected.status_code == 401, (root_path, rejected.text)
        assert wrong.status_code == 401, (root_path, wrong.text)
        assert HANDLER_MARKER not in rejected.text
        assert HANDLER_MARKER not in wrong.text
        assert accepted.status_code == 200, (root_path, accepted.text)
        assert accepted.json()["route"] == "/v1/chat/completions"
        assert calls == ["/v1/chat/completions"]


def test_h11_server_root_path_v1_does_not_run_chat_handler():
    app, calls = _build_app("/v1")
    server, port, thread = _serve(app)
    try:
        with httpx.Client(timeout=5) as client:
            response = client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions"
            )
    finally:
        _stop(server, thread)
    assert calls == []
    assert HANDLER_MARKER not in response.text
    assert response.status_code != 200
