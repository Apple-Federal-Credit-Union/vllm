# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unauthenticated TCP clients must not drive the DP coordinator.

These tests fail while a peer that only knows the bound TCP endpoints can
change the wave or kill coordination. They are the spec for the fix.
"""

import multiprocessing
from multiprocessing.connection import Connection

import msgspec
import pytest
import zmq

from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.coordinator import DPCoordinatorProc
from vllm.v1.serial_utils import MsgpackEncoder
from vllm.v1.utils import get_engine_client_zmq_addr

_ENGINE_COUNT = 2
_HOST = "127.0.0.1"


def _run_coordinator(
    engine_count: int,
    front: str,
    back_out: str,
    back_pub: str,
    pipe: Connection,
) -> None:
    DPCoordinatorProc.run_coordinator(
        engine_count=engine_count,
        front_publish_address=front,
        back_output_address=back_out,
        back_publish_address=back_pub,
        zmq_addr_pipe=pipe,
        min_stats_update_interval_ms=50,
        enable_wave_coordination=True,
    )


def _start_coordinator() -> tuple[
    multiprocessing.Process, zmq.Context, list[zmq.Socket], zmq.Socket, zmq.Socket
]:
    """Bind the real coordinator on TCP and finish the engine subscribe handshake."""
    ctx = multiprocessing.get_context("fork")
    requested = [
        get_engine_client_zmq_addr(False, _HOST) for _ in range(_ENGINE_COUNT + 1)
    ]
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_run_coordinator,
        args=(_ENGINE_COUNT, *requested, child),
        daemon=True,
    )
    proc.start()
    child.close()
    if not parent.poll(15):
        proc.kill()
        pytest.fail("DP coordinator did not publish its TCP endpoints")
    front, back_out, back_pub = parent.recv()
    assert front.startswith("tcp://")
    assert back_out.startswith("tcp://")
    assert back_pub.startswith("tcp://")

    zctx = zmq.Context()
    engines: list[zmq.Socket] = []
    for _ in range(_ENGINE_COUNT):
        sock = make_zmq_socket(zctx, back_pub, zmq.XSUB, bind=False, linger=0)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.send(b"\x01")
        engines.append(sock)
    for sock in engines:
        assert sock.recv() == b"READY"

    front_sub = make_zmq_socket(zctx, front, zmq.XSUB, bind=False, linger=0)
    front_sub.setsockopt(zmq.RCVTIMEO, 5000)
    front_sub.send(b"\x01")
    baseline = msgspec.msgpack.decode(front_sub.recv())
    assert baseline[1:] == [0, False]
    assert len(baseline[0]) == _ENGINE_COUNT

    push = make_zmq_socket(zctx, back_out, zmq.PUSH, bind=False, linger=0)
    # NULL mechanism: the bound socket did not require CURVE or ZAP.
    assert push.getsockopt(zmq.MECHANISM) == zmq.NULL
    return proc, zctx, engines, front_sub, push


def _stop(proc: multiprocessing.Process, zctx: zmq.Context) -> None:
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
    zctx.term()


def test_unauthenticated_start_wave_does_not_move_the_cluster() -> None:
    """A plain TCP PUSH of start_wave must not broadcast START_DP_WAVE.

    The frame is the same msgpack body an engine sends on its coordinator
    PUSH socket. No credential is attached.
    """
    proc, zctx, engines, front_sub, push = _start_coordinator()
    try:
        encoded = MsgpackEncoder().encode(
            EngineCoreOutputs(start_wave=4, engine_index=1)
        )
        push.send_multipart(list(encoded))

        for sock in engines:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
        received: list[list[bytes]] = []
        for sock in engines:
            try:
                received.append(sock.recv_multipart())
            except zmq.Again:
                pass
        try:
            published = msgspec.msgpack.decode(front_sub.recv())
        except zmq.Again:
            published = None

        assert received == []
        assert published is None or published[1:] == [0, False]
        assert proc.is_alive()
    finally:
        _stop(proc, zctx)


def test_unauthenticated_wave_complete_does_not_advance_the_wave() -> None:
    """A plain TCP PUSH of wave_complete must leave the published wave at 0."""
    proc, zctx, engines, front_sub, push = _start_coordinator()
    try:
        encoded = MsgpackEncoder().encode(
            EngineCoreOutputs(wave_complete=3, engine_index=0)
        )
        push.send_multipart(list(encoded))
        front_sub.setsockopt(zmq.RCVTIMEO, 1000)
        try:
            published = msgspec.msgpack.decode(front_sub.recv())
        except zmq.Again:
            published = None
        assert published is None or published[1:] == [0, False]
        assert proc.is_alive()
    finally:
        _stop(proc, zctx)
        del engines
