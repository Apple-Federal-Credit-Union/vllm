# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec for the realtime append buffer cap.

``/v1/realtime`` must apply ``VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`` mebibytes to
the base64-decoded PCM16 byte length of each ``input_audio_buffer.append``,
and to the running total of PCM still buffered on that connection. The
comparison happens before any float32 copy of a chunk that would exceed
either limit. The socket is closed when the limit is hit. Appends are
subject to the cap without ``session.update`` and without
``input_audio_buffer.commit``. Each connection has its own total.

These tests fail while a connection keeps every accepted float32 chunk in
an unbounded queue.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pybase64
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketState

import vllm.entrypoints.speech_to_text.realtime.api_router as api_router
import vllm.entrypoints.speech_to_text.realtime.connection as connection_mod
import vllm.envs as envs
from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware
from vllm.entrypoints.speech_to_text.factories import (
    register_speech_to_text_api_routers,
)
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection
from vllm.utils.mem_constants import MiB_bytes

# Obvious placeholder. Not a credential.
_API_KEY_PLACEHOLDER = "placeholder"

_LIMIT_MB = 1
_MAX_PCM_BYTES = _LIMIT_MB * MiB_bytes

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _UnusedRealtimeServing:
    """Stored on app.state. Append handling must not call into it."""

    def _is_model_supported(self, model: str | None) -> bool:
        raise AssertionError(model)


class RecordingConnection(RealtimeConnection):
    """Real connection that records when each client event has been handled."""

    opened: list[RecordingConnection] = []

    def __init__(self, websocket: WebSocket, serving: _UnusedRealtimeServing):
        super().__init__(websocket, serving)
        self.handled = 0
        self._cond = threading.Condition()
        RecordingConnection.opened.append(self)

    async def handle_event(self, event: dict):
        try:
            await super().handle_event(event)
        finally:
            with self._cond:
                self.handled += 1
                self._cond.notify_all()

    def wait_until_handled(self, count: int) -> None:
        deadline = time.monotonic() + 5
        with self._cond:
            while self.handled < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"handled {self.handled} events, wanted {count}"
                    )
                self._cond.wait(remaining)


def _pcm(nbytes: int) -> bytes:
    assert nbytes % 2 == 0
    return np.arange(nbytes // 2, dtype=np.int16).tobytes()


def _append_event(pcm: bytes) -> dict[str, str]:
    return {
        "type": "input_audio_buffer.append",
        "audio": pybase64.b64encode(pcm).decode("ascii"),
    }


def _expected_audio(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _queued(conn: RecordingConnection) -> list[np.ndarray]:
    return [item for item in conn.audio_queue._queue if item is not None]


def _assert_queued_pcm(conn: RecordingConnection, pcms: list[bytes]) -> None:
    queued = _queued(conn)
    assert len(queued) == len(pcms)
    for got, pcm in zip(queued, pcms, strict=True):
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, _expected_audio(pcm))
        assert got.nbytes // 2 == len(pcm)


def _outcome(
    conn: RecordingConnection,
    copied: list[int],
    *,
    queued_pcms: list[bytes],
    closed: bool,
) -> str:
    """Return a description of every cap violation, or an empty string."""
    problems: list[str] = []
    retained = sum(item.nbytes // 2 for item in _queued(conn))
    if sum(copied) > _MAX_PCM_BYTES or any(size > _MAX_PCM_BYTES for size in copied):
        problems.append(f"float32 copy of pcm bytes {copied}")
    if retained > _MAX_PCM_BYTES or len(_queued(conn)) != len(queued_pcms):
        problems.append(
            f"queue holds {len(_queued(conn))} chunks, {retained} pcm bytes"
        )
    else:
        try:
            _assert_queued_pcm(conn, queued_pcms)
        except AssertionError as exc:
            problems.append(str(exc))
    disconnected = conn.websocket.application_state is WebSocketState.DISCONNECTED
    if closed and not disconnected:
        problems.append(f"socket stayed {conn.websocket.application_state}")
    if not closed and disconnected:
        problems.append("socket closed while still under the cap")
    if conn.generation_task is not None or conn.request_id is not None:
        problems.append("append started a transcription")
    return "; ".join(problems)


def _build_app(api_key: str | None) -> FastAPI:
    app = FastAPI()
    register_speech_to_text_api_routers(app, ("realtime",))
    app.state.openai_serving_realtime = _UnusedRealtimeServing()
    if api_key is not None:
        app.add_middleware(AuthenticationMiddleware, tokens=[api_key])
    return app


def _headers(api_key: str | None) -> dict[str, str]:
    if api_key is None:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def float32_copies(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record PCM byte lengths passed to the float32 copy in append."""
    copied: list[int] = []
    real_frombuffer = connection_mod.np.frombuffer

    class _TrackedPCM(np.ndarray):
        def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
            if np.dtype(dtype) == np.float32:
                copied.append(int(self.nbytes))
            return super().astype(
                dtype, order=order, casting=casting, subok=subok, copy=copy
            )

    def _tracking_frombuffer(buffer, dtype=float, count=-1, offset=0, **kwargs):
        viewed = real_frombuffer(
            buffer, dtype=dtype, count=count, offset=offset, **kwargs
        )
        if np.dtype(dtype) == np.int16:
            return viewed.view(_TrackedPCM)
        return viewed

    monkeypatch.setattr(connection_mod.np, "frombuffer", _tracking_frombuffer)
    return copied


@pytest.fixture
def realtime_client(monkeypatch: pytest.MonkeyPatch, api_key: str | None):
    monkeypatch.setenv("VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", str(_LIMIT_MB))
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", _LIMIT_MB)
    RecordingConnection.opened = []
    monkeypatch.setattr(api_router, "RealtimeConnection", RecordingConnection)
    with TestClient(_build_app(api_key)) as client:
        yield client


def _send_append(ws, conn: RecordingConnection, pcm: bytes) -> None:
    handled = conn.handled
    ws.send_json(_append_event(pcm))
    conn.wait_until_handled(handled + 1)


@contextmanager
def _session(
    client: TestClient, api_key: str | None
) -> Iterator[tuple[object, RecordingConnection]]:
    with client.websocket_connect("/v1/realtime", headers=_headers(api_key)) as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        conn = RecordingConnection.opened[-1]
        assert conn._max_audio_filesize_mb == _LIMIT_MB
        assert conn.generation_task is None
        yield ws, conn


@pytest.mark.parametrize("api_key", [None, _API_KEY_PLACEHOLDER])
def test_decoded_pcm_over_filesize_limit_is_not_buffered(
    realtime_client: TestClient,
    float32_copies: list[int],
    api_key: str | None,
):
    """A chunk whose PCM bytes exceed the limit is refused before float32.

    Its sample count is still under the mebibyte figure, which is the
    comparison the unbounded path uses.
    """
    pcm = _pcm(_MAX_PCM_BYTES + 2)
    assert (len(pcm) // 2) / MiB_bytes <= _LIMIT_MB
    assert len(pcm) > _MAX_PCM_BYTES

    with _session(realtime_client, api_key) as (ws, conn):
        _send_append(ws, conn, pcm)
        assert not (
            outcome := _outcome(conn, float32_copies, queued_pcms=[], closed=True)
        ), outcome


@pytest.mark.parametrize("api_key", [None, _API_KEY_PLACEHOLDER])
def test_oversized_chunk_is_refused_before_float32_copy(
    realtime_client: TestClient,
    float32_copies: list[int],
    api_key: str | None,
):
    """A chunk the sample-count check rejects must not be float32-copied."""
    pcm = _pcm((_LIMIT_MB * MiB_bytes + 1) * 2)
    assert (len(pcm) // 2) / MiB_bytes > _LIMIT_MB

    with _session(realtime_client, api_key) as (ws, conn):
        _send_append(ws, conn, pcm)
        assert not (
            outcome := _outcome(conn, float32_copies, queued_pcms=[], closed=True)
        ), outcome


@pytest.mark.parametrize("api_key", [None, _API_KEY_PLACEHOLDER])
def test_uncommitted_appends_cannot_exceed_connection_cap(
    realtime_client: TestClient,
    float32_copies: list[int],
    api_key: str | None,
):
    """Several under-limit appends, with no commit, stop at the connection cap.

    A second connection can still buffer its own under-limit chunk.
    """
    chunk = _pcm(400 * 1024)
    assert len(chunk) < _MAX_PCM_BYTES
    assert 2 * len(chunk) <= _MAX_PCM_BYTES
    assert 3 * len(chunk) > _MAX_PCM_BYTES

    with _session(realtime_client, api_key) as (first, first_conn):
        _send_append(first, first_conn, chunk)
        _send_append(first, first_conn, chunk)
        _send_append(first, first_conn, chunk)
        assert not (
            outcome := _outcome(
                first_conn,
                float32_copies,
                queued_pcms=[chunk, chunk],
                closed=True,
            )
        ), outcome

    copied_after_first = len(float32_copies)
    with _session(realtime_client, api_key) as (second, second_conn):
        _send_append(second, second_conn, chunk)
        assert not (
            outcome := _outcome(
                second_conn,
                float32_copies[copied_after_first:],
                queued_pcms=[chunk],
                closed=False,
            )
        ), outcome
