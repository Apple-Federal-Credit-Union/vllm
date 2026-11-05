# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec for VLLM_HTTP_MAX_REQUEST_BODY_SIZE on the Python HTTP server.

serve_http must reject a body larger than that limit before the ASGI app
buffers it. The Rust frontend already does. These tests fail while the Python
server ignores the setting.

The app below is the body-reading behavior of the routes served by serve_http:
JSON models (chat completions), request.body() (completions), and request.json()
(SageMaker invocations). Auth is off, which is the default when no API key is
configured.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI, Request
from pydantic import BaseModel

from vllm.entrypoints.launchers.launcher import serve_http

_ENV = "VLLM_HTTP_MAX_REQUEST_BODY_SIZE"
_DEFAULT_LIMIT = 32 * 1024 * 1024
_H11_INCOMPLETE_EVENT_LIMIT = 4 * 1024 * 1024
_REJECTION_STATUSES = {400, 413}
_PROTOCOLS = ("auto", "h11")
_PATHS = ("/v1/chat/completions", "/v1/completions", "/invocations")


class _ChatBody(BaseModel):
    model: str
    messages: list


class _Observed:
    def __init__(self) -> None:
        self.bytes = 0
        self.completed: list[str] = []


class _ByteCounter:
    """Counts body bytes that actually reach the application."""

    def __init__(self, app, observed: _Observed) -> None:
        self.app = app
        self.observed = observed

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def receive_counted():
            message = await receive()
            if message["type"] == "http.request":
                self.observed.bytes += len(message.get("body") or b"")
            return message

        await self.app(scope, receive_counted, send)


def _json_body(content_len: int) -> bytes:
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "a" * content_len}],
    }
    return json.dumps(payload).encode()


def _body_of_length(length: int) -> bytes:
    prefix = b'{"model":"m","messages":[{"role":"user","content":"'
    suffix = b'"}]}'
    filler = length - len(prefix) - len(suffix)
    if filler < 0:
        raise ValueError(f"body length {length} is shorter than the JSON envelope")
    return prefix + b"a" * filler + suffix


def _headers(path: str, *, content_length: int | None, chunked: bool) -> bytes:
    lines = [
        f"POST {path} HTTP/1.1",
        "Host: 127.0.0.1",
        "Content-Type: application/json",
        "Connection: close",
    ]
    if chunked:
        lines.append("Transfer-Encoding: chunked")
    else:
        lines.append(f"Content-Length: {content_length}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def _status(raw: bytes) -> int | None:
    if not raw.startswith(b"HTTP/"):
        return None
    parts = raw.split(b"\r\n", 1)[0].split(b" ")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _read(sock: socket.socket, timeout: float) -> bytes:
    sock.settimeout(timeout)
    data = bytearray()
    try:
        while b"\r\n\r\n" not in data:
            piece = sock.recv(65536)
            if not piece:
                break
            data.extend(piece)
    except TimeoutError:
        return bytes(data)
    return bytes(data)


def _connect(port: int) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


async def _connect_async(port: int) -> socket.socket:
    # Blocking socket calls on the server loop deadlock the response.
    return await asyncio.to_thread(_connect, port)


async def _send_async(sock: socket.socket, data: bytes) -> None:
    await asyncio.to_thread(sock.sendall, data)


async def _read_async(sock: socket.socket, timeout: float) -> bytes:
    return await asyncio.to_thread(_read, sock, timeout)


def _assert_rejected(raw: bytes, observed: _Observed, limit: int) -> None:
    status = _status(raw)
    assert status in _REJECTION_STATUSES, (
        f"expected the server to reject a body above {limit} bytes, got {raw[:300]!r}"
    )
    assert b"Invalid HTTP request received." not in raw
    assert observed.completed == []
    assert observed.bytes <= limit


@contextlib.asynccontextmanager
async def _server(
    limit: int | None, http_protocol: str
) -> AsyncIterator[tuple[int, _Observed]]:
    previous = os.environ.get(_ENV)
    if limit is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = str(limit)

    observed = _Observed()
    app = FastAPI()
    app.add_middleware(_ByteCounter, observed=observed)

    @app.post("/v1/chat/completions")
    async def chat(body: _ChatBody) -> dict[str, bool]:
        observed.completed.append("/v1/chat/completions")
        return {"ok": bool(body.model)}

    @app.post("/v1/completions")
    async def completions(request: Request) -> dict[str, int]:
        raw = await request.body()
        observed.completed.append("/v1/completions")
        return {"n": len(raw)}

    @app.post("/invocations")
    async def invocations(request: Request) -> dict[str, int]:
        payload = await request.json()
        observed.completed.append("/invocations")
        return {"n": len(json.dumps(payload))}

    app.state.engine_client = None
    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind(("127.0.0.1", 0))
    listen.listen(16)
    port = listen.getsockname()[1]
    task = asyncio.create_task(
        serve_http(
            app,
            sock=listen,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
            timeout_keep_alive=2,
            http=http_protocol,
        )
    )
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while True:
            server = getattr(app.state, "server", None)
            if server is not None and server.started:
                break
            if task.done():
                raise RuntimeError("HTTP server exited during startup") from (
                    task.exception()
                )
            if loop.time() >= deadline:
                raise TimeoutError("HTTP server did not start")
            await asyncio.sleep(0.01)
        yield port, observed
    finally:
        if previous is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = previous
        server = getattr(app.state, "server", None)
        if server is not None:
            server.should_exit = True
        if not task.done():
            try:
                result = await asyncio.wait_for(task, timeout=5)
            except TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            else:
                if asyncio.iscoroutine(result):
                    await result
        else:
            with contextlib.suppress(asyncio.CancelledError):
                result = task.exception()
                if result is None and asyncio.iscoroutine(task.result()):
                    await task.result()


@pytest.mark.asyncio
@pytest.mark.parametrize("http_protocol", _PROTOCOLS)
@pytest.mark.parametrize("path", _PATHS)
async def test_content_length_over_limit_is_rejected_before_the_body_is_sent(
    http_protocol: str, path: str
) -> None:
    limit = 1024
    declared = _H11_INCOMPLETE_EVENT_LIMIT + 1
    async with _server(limit, http_protocol) as (port, observed):
        sock = await _connect_async(port)
        try:
            await _send_async(
                sock, _headers(path, content_length=declared, chunked=False)
            )
            raw = await _read_async(sock, timeout=1)
        finally:
            sock.close()
        _assert_rejected(raw, observed, limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("http_protocol", _PROTOCOLS)
async def test_default_limit_rejects_a_declared_body_before_it_is_sent(
    http_protocol: str,
) -> None:
    async with _server(None, http_protocol) as (port, observed):
        sock = await _connect_async(port)
        try:
            await _send_async(
                sock,
                _headers(
                    "/v1/chat/completions",
                    content_length=_DEFAULT_LIMIT + 1,
                    chunked=False,
                ),
            )
            raw = await _read_async(sock, timeout=1)
        finally:
            sock.close()
        _assert_rejected(raw, observed, _DEFAULT_LIMIT)


@pytest.mark.asyncio
@pytest.mark.parametrize("http_protocol", _PROTOCOLS)
@pytest.mark.parametrize("path", _PATHS)
async def test_chunked_body_over_limit_is_rejected_before_end_of_message(
    http_protocol: str, path: str
) -> None:
    limit = 1024
    async with _server(limit, http_protocol) as (port, observed):
        sock = await _connect_async(port)
        try:
            await _send_async(sock, _headers(path, content_length=None, chunked=True))
            await _send_async(sock, _chunk(b"a" * limit))
            early = await _read_async(sock, timeout=0.2)
            assert _status(early) is None, early[:300]
            await _send_async(sock, _chunk(b"b" * limit))
            raw = early + await _read_async(sock, timeout=1)
        finally:
            sock.close()
        _assert_rejected(raw, observed, limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("http_protocol", _PROTOCOLS)
@pytest.mark.parametrize("path", _PATHS)
async def test_complete_json_over_limit_is_rejected_without_buffering_it(
    http_protocol: str, path: str
) -> None:
    limit = 1024
    body = _json_body(4096)
    assert len(body) > limit
    async with _server(limit, http_protocol) as (port, observed):
        sock = await _connect_async(port)
        try:
            await _send_async(
                sock, _headers(path, content_length=len(body), chunked=False)
            )
            await _send_async(sock, body)
            raw = await _read_async(sock, timeout=5)
        finally:
            sock.close()
        _assert_rejected(raw, observed, limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("http_protocol", _PROTOCOLS)
@pytest.mark.parametrize("path", _PATHS)
async def test_body_exactly_at_the_limit_is_accepted(
    http_protocol: str, path: str
) -> None:
    limit = 256
    body = _body_of_length(limit)
    assert len(body) == limit
    async with _server(limit, http_protocol) as (port, observed):
        sock = await _connect_async(port)
        try:
            await _send_async(
                sock, _headers(path, content_length=len(body), chunked=False)
            )
            await _send_async(sock, body)
            raw = await _read_async(sock, timeout=5)
        finally:
            sock.close()
    assert _status(raw) == 200, raw[:300]
    assert observed.bytes == limit
    assert observed.completed == [path]


@pytest.mark.asyncio
@pytest.mark.parametrize("http_protocol", _PROTOCOLS)
async def test_zero_removes_the_limit(http_protocol: str) -> None:
    body = b"a" * (_DEFAULT_LIMIT + 1)
    async with _server(0, http_protocol) as (port, observed):
        sock = await _connect_async(port)
        try:
            await _send_async(
                sock,
                _headers("/v1/completions", content_length=len(body), chunked=False),
            )
            await _send_async(sock, body)
            raw = await _read_async(sock, timeout=30)
        finally:
            sock.close()
    assert _status(raw) == 200, raw[:300]
    assert observed.bytes == len(body)
    assert observed.completed == ["/v1/completions"]


@pytest.mark.asyncio
async def test_h11_accepts_a_body_larger_than_the_incomplete_header_limit() -> None:
    body = _json_body(_H11_INCOMPLETE_EVENT_LIMIT + 1024)
    assert len(body) > _H11_INCOMPLETE_EVENT_LIMIT
    assert len(body) < _DEFAULT_LIMIT
    async with _server(None, "h11") as (port, observed):
        sock = await _connect_async(port)
        try:
            await _send_async(
                sock,
                _headers(
                    "/v1/chat/completions",
                    content_length=len(body),
                    chunked=False,
                ),
            )
            await _send_async(sock, body)
            raw = await _read_async(sock, timeout=30)
        finally:
            sock.close()
    assert _status(raw) == 200, raw[:300]
    assert observed.bytes == len(body)
    assert observed.completed == ["/v1/chat/completions"]
