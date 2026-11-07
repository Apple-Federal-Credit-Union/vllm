# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The control port must not dial a peer-supplied ZMQ identity.

An inbound ROUTER frame's identity is chosen by the sender. Handling that
frame must not open a TCP connection to the host:port in the identity, and
must not write the KV connect message (data-plane registration and block
base address) to that address. These tests fail while the transport parses
the identity as a dial address.
"""

from __future__ import annotations

import importlib.util
import select
import socket
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import msgspec
import pytest
import zmq
import zmq.utils.monitor

from vllm.v1.kv_offload.tiering.p2p.control import zmq as zmq_control
from vllm.v1.kv_offload.tiering.p2p.control.zmq import ZmqTransport
from vllm.v1.kv_offload.tiering.p2p.session.protocol import TYPE_KEY, ConnectMsg
from vllm.v1.kv_offload.tiering.p2p.session.session import P2PSession

# Load the data-plane base module by path. Importing the package runs
# data/__init__.py, which imports NixlTransport and the distributed stack.
# DataTransport itself only computes the block-region address this test sends.
_DATA_BASE_NAME = "vllm.v1.kv_offload.tiering.p2p.data.base"
if _DATA_BASE_NAME not in sys.modules:
    _data_base_path = (
        Path(zmq_control.__file__).resolve().parents[1] / "data" / "base.py"
    )
    _spec = importlib.util.spec_from_file_location(_DATA_BASE_NAME, _data_base_path)
    assert _spec is not None and _spec.loader is not None
    _data_base = importlib.util.module_from_spec(_spec)
    sys.modules[_DATA_BASE_NAME] = _data_base
    _spec.loader.exec_module(_data_base)
else:
    _data_base = sys.modules[_DATA_BASE_NAME]

CancelMode = _data_base.CancelMode
DataTransport = _data_base.DataTransport
PollResult = _data_base.PollResult

# Control socket and the attacker both live on this address. The identity
# names a different loopback address, so a connection there was dialed from
# the identity string rather than taken from the accepted peer.
_CONTROL_HOST = "127.0.0.1"
_DIAL_HOST = "127.0.0.2"
_WAIT_S = 2.0

# Bytes get_agent_metadata() returns. NixlTransport returns the NIXL agent's
# metadata unchanged; this stands in for that blob.
_REGISTRATION = b"kv-registration"


def _free_bind(host: str) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, 0))
    listener.listen(8)
    listener.setblocking(False)
    return listener


def _make_transport(host: str = _CONTROL_HOST, attempts: int = 8):
    last_err: Exception | None = None
    for _ in range(attempts):
        probe = _free_bind(host)
        port = probe.getsockname()[1]
        probe.close()
        try:
            return ZmqTransport(f"{host}:{port}", host, port), port
        except (zmq.ZMQError, OSError) as exc:
            last_err = exc
    assert last_err is not None
    raise last_err


def _wait_monitor(mon: zmq.Socket, timeout_s: float = _WAIT_S) -> bool:
    """Return True once the monitor socket has an event."""
    poller = zmq.Poller()
    poller.register(mon, zmq.POLLIN)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if poller.poll(timeout=max(remaining_ms, 0)):
            zmq.utils.monitor.recv_monitor_message(mon)
            return True
    return False


class _KvRegion(DataTransport):
    """Real block-region transport with a fixed registration blob."""

    def __init__(self, metadata: bytes) -> None:
        self._buffer = bytearray(4 * 32)
        view = memoryview(self._buffer).cast("B", (4, 32))
        super().__init__(view)
        self._metadata = metadata

    def get_agent_metadata(self) -> bytes:
        return self._metadata

    def add_remote_peer(
        self,
        peer_id: str,
        agent_metadata: bytes,
        base_addr: int,
        num_blocks: int,
        block_len: int,
    ) -> None:
        del peer_id, agent_metadata, base_addr, num_blocks, block_len

    def remove_remote_peer(self, peer_id: str) -> None:
        del peer_id

    def write_blocks(
        self,
        peer_id: str,
        local_idxs: list[int],
        remote_idxs: list[int],
    ) -> int | None:
        del peer_id, local_idxs, remote_idxs
        return None

    def poll(self, peer_id: str | None = None) -> PollResult:
        del peer_id
        return PollResult(done=[], failed=[])

    def cancel(
        self,
        transfer_ids: Iterable[int],
        mode: CancelMode = "immediate",
    ) -> list[int]:
        del transfer_ids, mode
        return []

    def close(self) -> None:
        return None


def _open_attacker(
    ctx: zmq.Context,
    control_port: int,
    identity: str,
    body: dict | int,
) -> tuple[zmq.Socket, zmq.Socket]:
    """Connect a DEALER whose routing id is ``identity`` and send ``body``."""
    attacker = ctx.socket(zmq.DEALER)
    attacker.setsockopt(zmq.LINGER, 0)
    attacker.setsockopt(zmq.IDENTITY, identity.encode())
    mon = ctx.socket(zmq.PAIR)
    mon.setsockopt(zmq.LINGER, 0)
    safe_identity = identity.replace(":", "-").replace(".", "-")
    mon_addr = f"inproc://attacker-hs-{control_port}-{safe_identity}"
    attacker.monitor(mon_addr, zmq.EVENT_HANDSHAKE_SUCCEEDED)
    mon.connect(mon_addr)
    attacker.connect(f"tcp://{_CONTROL_HOST}:{control_port}")
    if not _wait_monitor(mon):
        attacker.close()
        mon.close()
        raise AssertionError("control port did not complete the ZMQ handshake")
    attacker.send(msgspec.msgpack.encode(body))
    return attacker, mon


@pytest.mark.parametrize("body", [{"unused": 1}, 1], ids=["map", "int"])
def test_inbound_identity_does_not_open_tcp_connection(body: dict | int) -> None:
    """A decoded inbound frame must not dial the identity host:port.

    The sender's TCP peer is the control host. The identity names a
    different loopback address. Any accepted connection on that address
    means the transport used the peer-supplied identity as a dial target.
    The body is valid msgpack and is not a handshake message.
    """
    listener = _free_bind(_DIAL_HOST)
    dial_port = listener.getsockname()[1]
    identity = f"{_DIAL_HOST}:{dial_port}"
    transport, control_port = _make_transport()
    ctx = zmq.Context()
    attacker: zmq.Socket | None = None
    mon: zmq.Socket | None = None
    accepted: tuple[socket.socket, tuple] | None = None
    try:
        attacker, mon = _open_attacker(ctx, control_port, identity, body)
        deadline = time.monotonic() + _WAIT_S
        while time.monotonic() < deadline and accepted is None:
            transport.poll()
            readable, _, _ = select.select([listener], [], [], 0.05)
            if readable:
                accepted = listener.accept()
        peer = accepted[1] if accepted is not None else None
        assert accepted is None, (
            f"control port dialed {identity} from an inbound ZMQ identity; "
            f"accepted peer {peer}"
        )
    finally:
        if accepted is not None:
            accepted[0].close()
        listener.close()
        if attacker is not None:
            attacker.close()
        if mon is not None:
            mon.close()
        ctx.term()
        transport.close()


@pytest.mark.parametrize("body", [{"unused": 1}, 1], ids=["map", "int"])
def test_inbound_identity_does_not_receive_kv_connect(body: dict | int) -> None:
    """The KV connect message must not be written to the identity address.

    ``_accept_new_peers`` builds a ``P2PSession`` on the connection
    ``poll()`` returns, and that constructor sends ``ConnectMsg``. The
    identity address must not accept that connection and must not observe
    the registration blob or the block-region base address.
    """
    ctx = zmq.Context()
    target = ctx.socket(zmq.ROUTER)
    target.setsockopt(zmq.LINGER, 0)
    target.bind(f"tcp://{_DIAL_HOST}:0")
    endpoint = target.getsockopt(zmq.LAST_ENDPOINT).decode()
    dial_port = int(endpoint.rsplit(":", 1)[1])
    identity = f"{_DIAL_HOST}:{dial_port}"

    target_mon = ctx.socket(zmq.PAIR)
    target_mon.setsockopt(zmq.LINGER, 0)
    target_mon_addr = f"inproc://target-accept-{dial_port}"
    target.monitor(
        target_mon_addr,
        zmq.EVENT_ACCEPTED | zmq.EVENT_HANDSHAKE_SUCCEEDED,
    )
    target_mon.connect(target_mon_addr)

    transport, control_port = _make_transport()
    region = _KvRegion(_REGISTRATION)
    attacker: zmq.Socket | None = None
    attacker_mon: zmq.Socket | None = None
    sessions: list[P2PSession] = []
    contacted = False
    received: list[bytes] | None = None
    try:
        attacker, attacker_mon = _open_attacker(ctx, control_port, identity, body)
        deadline = time.monotonic() + _WAIT_S
        while time.monotonic() < deadline and received is None and not contacted:
            for conn in transport.poll():
                # Same construction as P2PSecondaryTierManager._accept_new_peers
                # once the peer is under the session cap and not in backoff.
                sessions.append(
                    P2PSession(
                        peer_id=conn.peer_id,
                        local_id=f"{_CONTROL_HOST}:{control_port}",
                        transport=region,
                        local_block_len=region.block_len,
                        local_hash_seed="0",
                        conn=conn,
                    )
                )
            if _wait_monitor(target_mon, timeout_s=0.05):
                contacted = True
            if target.poll(0, zmq.POLLIN):
                received = target.recv_multipart(zmq.NOBLOCK)
        if received is None and target.poll(200, zmq.POLLIN):
            received = target.recv_multipart(zmq.NOBLOCK)
        detail = ""
        if received is not None:
            decoded = msgspec.msgpack.decode(received[-1])
            if isinstance(decoded, dict):
                detail = (
                    f" message type={decoded.get(TYPE_KEY)!r}"
                    f" base_addr={decoded.get(ConnectMsg.BASE_ADDR)!r}"
                    f" (region {region.base_addr})"
                    f" agent_metadata={decoded.get(ConnectMsg.AGENT_METADATA)!r}"
                )
            else:
                detail = f" payload={decoded!r}"
        assert not contacted and received is None, (
            f"identity address {identity} was contacted by the control port.{detail}"
        )
    finally:
        for session in sessions:
            session.close()
        if attacker is not None:
            attacker.close()
        if attacker_mon is not None:
            attacker_mon.close()
        target_mon.close()
        target.close()
        ctx.term()
        transport.close()
        region.close()
