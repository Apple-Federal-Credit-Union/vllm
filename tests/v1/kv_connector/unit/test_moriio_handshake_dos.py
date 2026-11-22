# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO handshake and notify listeners must survive a bad peer frame.

Workers bind both listeners on every interface with no peer authentication.
One unexpected frame currently leaves the thread, and the worker does not
start another. Later peers can no longer exchange KV registration or
completion messages. These tests fail until that frame is rejected and the
following valid message is still served.
"""

import socket
import threading
import time
from collections.abc import Iterator

import msgpack
import msgspec
import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    get_role,
    set_role,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)

_RECV_TIMEOUT_MS = 1000
_METADATA = MoRIIOAgentMetadata(
    engine_id="engine-under-test",
    agent_metadata=b"agent-metadata",
    kv_caches_base_addr=[0x1000, 0x2000],
    num_blocks=4,
    block_len=16,
    attn_backend_name="test-backend",
)
_LAYERS: dict[str, list[bytes]] = {"layer0": [b"region-0", b"region-1"]}

pytestmark = pytest.mark.skip_global_cleanup


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("0.0.0.0", 0))
        return sock.getsockname()[1]


def _wildcard_listeners(port: int) -> list[str]:
    """Return local addresses of TCP listeners on ``port``.

    ``/proc/net/tcp`` stores the address as little-endian hex. ``00000000``
    and the all-zero IPv6 form are wildcards, so any routed peer can connect.
    """
    suffix = f":{port:04X}"
    found: list[str] = []
    for proc_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        with open(proc_path, encoding="ascii") as proc_file:
            next(proc_file)
            for line in proc_file:
                local, _remote, state = line.split()[1:4]
                if state == "0A" and local.endswith(suffix):
                    found.append(local.split(":", 1)[0])
    return found


def _assert_bound_on_wildcard(port: int) -> None:
    deadline = time.monotonic() + 5
    found: list[str] = []
    while time.monotonic() < deadline:
        found = _wildcard_listeners(port)
        if found:
            break
        time.sleep(0.01)
    wildcards = {"00000000", "00000000000000000000000000000000"}
    assert any(address in wildcards for address in found), found


class _Dealer:
    def __init__(self, port: int) -> None:
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, _RECV_TIMEOUT_MS)
        self._sock.setsockopt(zmq.SNDTIMEO, _RECV_TIMEOUT_MS)
        self._sock.connect(f"tcp://127.0.0.1:{port}")

    def send(self, payload: bytes) -> None:
        self._sock.send(payload)

    def send_multipart(self, frames: list[bytes]) -> None:
        self._sock.send_multipart(frames)

    def recv_multipart(self) -> list[bytes]:
        return self._sock.recv_multipart()

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctx.term()


def _start_handshake_listener() -> tuple[threading.Thread, int]:
    port = _free_port()
    ready = threading.Event()
    thread = threading.Thread(
        target=MoRIIOConnectorWorker._moriio_handshake_listener,
        args=(_METADATA, ready, port, 0, 0, _LAYERS),
        daemon=True,
        name="moriio_handshake_listener",
    )
    thread.start()
    assert ready.wait(5), "handshake listener did not bind"
    _assert_bound_on_wildcard(port)
    return thread, port


def _assert_metadata_served(dealer: _Dealer, thread: threading.Thread) -> None:
    """A metadata request must still return the registered KV description."""
    dealer.send(MoRIIOConstants.GET_META_MSG)
    try:
        agent_frames = dealer.recv_multipart()
        layer_frames = dealer.recv_multipart()
    except zmq.ZMQError as exc:
        raise AssertionError(
            "handshake listener did not answer a metadata request; "
            f"thread_alive={thread.is_alive()}"
        ) from exc
    assert agent_frames[0] == b""
    decoded = msgspec.msgpack.Decoder(MoRIIOAgentMetadata).decode(agent_frames[1])
    assert decoded.engine_id == _METADATA.engine_id
    assert decoded.agent_metadata == _METADATA.agent_metadata
    assert decoded.kv_caches_base_addr == _METADATA.kv_caches_base_addr
    assert decoded.num_blocks == _METADATA.num_blocks
    assert decoded.block_len == _METADATA.block_len
    assert layer_frames[0] == b""
    assert msgpack.loads(layer_frames[1]) == _LAYERS
    assert thread.is_alive()


def _start_notify_listener() -> tuple[MoRIIOWrapper, int]:
    port = _free_port()
    wrapper = MoRIIOWrapper()
    wrapper.notify_port = port
    wrapper.async_wait_reqid()
    _assert_bound_on_wildcard(port)
    return wrapper, port


def _wait_for_completion(wrapper: MoRIIOWrapper, transfer_id: str) -> None:
    deadline = time.monotonic() + (_RECV_TIMEOUT_MS / 1000)
    seen: set[str] = set()
    while time.monotonic() < deadline:
        seen |= wrapper.pop_finished_write_req_ids()
        if transfer_id in seen:
            assert wrapper.notify_thread is not None
            assert wrapper.notify_thread.is_alive()
            return
        time.sleep(0.01)
    alive = wrapper.notify_thread.is_alive() if wrapper.notify_thread else False
    raise AssertionError(
        f"notify listener did not record {transfer_id!r}; "
        f"thread_alive={alive} seen={seen!r}"
    )


@pytest.fixture
def consumer_role() -> Iterator[None]:
    previous = get_role()
    set_role(ROLE.CONSUMER)
    yield
    set_role(previous)


@pytest.mark.parametrize(
    "payload",
    [b"not-a-handshake", b""],
    ids=["unknown-type", "empty-frame"],
)
def test_unexpected_handshake_frame_still_serves_metadata(payload: bytes) -> None:
    """A frame that is neither metadata nor done must not drop the listener."""
    thread, port = _start_handshake_listener()
    dealer = _Dealer(port)
    try:
        _assert_metadata_served(dealer, thread)
        dealer.send(payload)
        _assert_metadata_served(dealer, thread)
        peer = _Dealer(port)
        try:
            _assert_metadata_served(peer, thread)
        finally:
            peer.close()
    finally:
        dealer.close()


def test_extra_handshake_frame_still_serves_metadata() -> None:
    """A metadata request with an unexpected extra frame must not be fatal."""
    thread, port = _start_handshake_listener()
    dealer = _Dealer(port)
    try:
        dealer.send_multipart([MoRIIOConstants.GET_META_MSG, b"extra"])
        _assert_metadata_served(dealer, thread)
    finally:
        dealer.close()


def test_non_text_done_request_id_still_serves_metadata() -> None:
    """The done path must not decode an arbitrary frame as text and exit."""
    thread, port = _start_handshake_listener()
    dealer = _Dealer(port)
    try:
        dealer.send(MoRIIOConstants.POP_DONE_RECV)
        dealer.send(b"\xff")
        _assert_metadata_served(dealer, thread)
    finally:
        dealer.close()


def test_done_request_with_extra_frame_still_serves_metadata() -> None:
    """A done message with the wrong frame count must not kill the listener."""
    thread, port = _start_handshake_listener()
    dealer = _Dealer(port)
    try:
        dealer.send(MoRIIOConstants.POP_DONE_RECV)
        dealer.send_multipart([b"req", b"extra"])
        _assert_metadata_served(dealer, thread)
    finally:
        dealer.close()


@pytest.mark.usefixtures("consumer_role")
@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        msgpack.dumps({"type": "not-a-real-type", "transfer_id": "ignored"}),
    ],
    ids=["non-text", "unknown-message-type"],
)
def test_bad_notify_frame_still_records_later_completion(payload: bytes) -> None:
    """A bad notify frame must not stop later completion messages."""
    assert get_role() == ROLE.CONSUMER
    wrapper, port = _start_notify_listener()
    dealer = _Dealer(port)
    try:
        dealer.send(b"before-bad-frame")
        _wait_for_completion(wrapper, "before-bad-frame")
        dealer.send(payload)
        dealer.send(b"after-bad-frame")
        _wait_for_completion(wrapper, "after-bad-frame")
        peer = _Dealer(port)
        try:
            peer.send(b"later-peer")
            _wait_for_completion(wrapper, "later-peer")
        finally:
            peer.close()
    finally:
        dealer.close()


@pytest.mark.usefixtures("consumer_role")
def test_extra_notify_frame_still_records_later_completion() -> None:
    """A notify message with the wrong frame count must not kill the listener."""
    assert get_role() == ROLE.CONSUMER
    wrapper, port = _start_notify_listener()
    dealer = _Dealer(port)
    try:
        dealer.send_multipart([b"bad", b"frame"])
        dealer.send(b"after-extra-frame")
        _wait_for_completion(wrapper, "after-extra-frame")
    finally:
        dealer.close()
