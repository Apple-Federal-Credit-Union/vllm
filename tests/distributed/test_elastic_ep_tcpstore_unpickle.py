# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Elastic expert-parallel stores must not execute bytes from other clients.

Stateless data-parallel, expert-parallel, and EPLB groups publish their
TCPStore ports on an unauthenticated coordination store and then
``pickle.loads`` whatever is stored under the next broadcast or send key.
These tests fail while a client that is not a member of the group can make
a waiting rank run code. They pass once those bytes are not executed,
including when a legitimate rank wrote the key first and the other client
replaced it before the receiver read it.

The CUDA collective path does not read these keys. ``GroupCoordinator``'s
gloo object broadcast is a different channel and is not covered here.
"""

import pickle
import socket
import struct
import time
from datetime import timedelta
from multiprocessing import get_context
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest
import torch
from torch.distributed import TCPStore

from vllm.distributed.stateless_coordinator import _PORTS_FMT
from vllm.distributed.utils import StatelessProcessGroup, create_tcp_store

_HOST = "127.0.0.1"
_START_TIMEOUT_S = 60.0
_EFFECT_TIMEOUT_S = 5.0
_STORE_OPS = (
    "broadcast_obj",
    "recv_obj",
    "broadcast",
    "recv",
    "all_gather_obj",
    "all_reduce",
    "barrier",
    "in_the_same_node_as",
)
_OVERWRITE_OPS = ("broadcast_obj", "recv_obj", "broadcast", "recv")
_COORD_OPS = (
    "broadcast_object",
    "broadcast_object_list",
    "broadcast_tensor_dict",
    "recv_object",
    "recv_tensor_dict",
    "broadcast",
    "barrier",
    "broadcast_expert_mapping",
)


def _payload(marker: Path) -> bytes:
    # exec is a builtin, so the receiver does not need this module to import
    # the reducer. The directory is created before the exception, which aborts
    # the caller before later store polls.
    code = (
        "import os\n"
        f"os.makedirs({str(marker)!r})\n"
        "raise RuntimeError('untrusted store bytes')\n"
    )

    class _Marker:
        def __reduce__(self):
            return (exec, (code,))

    return pickle.dumps(_Marker())


def _key_for(op: str) -> str:
    if op in {
        "broadcast_obj",
        "all_gather_obj",
        "all_reduce",
        "barrier",
        "in_the_same_node_as",
        "broadcast_object",
        "broadcast_object_list",
        "broadcast_tensor_dict",
    }:
        return "broadcast_from/0/0"
    if op in {"recv_obj", "recv_object", "recv_tensor_dict"}:
        return "send_to/1/0"
    if op in {"broadcast", "broadcast_expert_mapping"}:
        return "broadcast_tensor/0/0"
    if op == "recv":
        return "send_tensor/1/0"
    raise AssertionError(op)


def _run_stateless(pg: StatelessProcessGroup, op: str) -> None:
    if op == "broadcast_obj":
        pg.broadcast_obj(None, src=0)
    elif op == "recv_obj":
        pg.recv_obj(0)
    elif op == "broadcast":
        pg.broadcast(torch.zeros(4), src=0)
    elif op == "recv":
        pg.recv(torch.zeros(4), src=0)
    elif op == "all_gather_obj":
        pg.all_gather_obj(1)
    elif op == "all_reduce":
        pg.all_reduce(torch.zeros(4))
    elif op == "barrier":
        # The unsafe load is the first broadcast. Cap the later wait so a
        # broken store client cannot retry for the default half minute.
        pg.barrier(timeout=1.0)
    elif op == "in_the_same_node_as":
        from vllm.distributed.parallel_state import in_the_same_node_as

        in_the_same_node_as(pg, source_rank=0)
    else:
        raise AssertionError(op)


def _send_benign(pg: StatelessProcessGroup, op: str) -> None:
    if op == "broadcast_obj":
        pg.broadcast_obj("ok", src=0)
    elif op == "recv_obj":
        pg.send_obj("ok", dst=1)
    elif op == "broadcast":
        pg.broadcast(torch.ones(4), src=0)
    elif op == "recv":
        pg.send(torch.ones(4), dst=1)
    else:
        raise AssertionError(op)


def _stateless_rank0(
    host: str,
    port: int,
    op: str,
    overwrite: bool,
    started: Event,
    sent: Event,
    done: Event,
) -> None:
    group = StatelessProcessGroup.create(
        host, port, rank=0, world_size=2, store_timeout=30
    )
    started.set()
    if overwrite:
        _send_benign(group, op)
        sent.set()
    done.wait(_START_TIMEOUT_S)


def _stateless_rank1(
    host: str,
    port: int,
    op: str,
    started: Event,
    go: Event,
) -> None:
    group = StatelessProcessGroup.create(
        host, port, rank=1, world_size=2, store_timeout=30
    )
    started.set()
    if not go.wait(_START_TIMEOUT_S):
        return
    try:
        _run_stateless(group, op)
    except Exception:
        return


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _stop(processes: list[BaseProcess], done: Event | None = None) -> None:
    if done is not None:
        done.set()
    deadline = time.monotonic() + 5
    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        proc.join(remaining)
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(2)
    for proc in processes:
        if proc.is_alive():
            proc.kill()
            proc.join(2)


def _wait_for_store(store: TCPStore, key: str, expected: bytes) -> bool:
    # get blocks until the key exists. Repeating it observes a later set
    # from this same client, whose requests the server applies in order.
    deadline = time.monotonic() + _EFFECT_TIMEOUT_S
    while time.monotonic() < deadline:
        current = bytes(store.get(key))
        if current == expected:
            return True
        time.sleep(0.02)
    return False


def _marker_appeared(marker: Path) -> bool:
    deadline = time.monotonic() + _EFFECT_TIMEOUT_S
    while time.monotonic() < deadline:
        if marker.exists():
            return True
        time.sleep(0.02)
    return marker.exists()


def _connect(host: str, port: int) -> TCPStore:
    return TCPStore(
        host,
        port,
        is_master=False,
        wait_for_workers=False,
        timeout=timedelta(seconds=10),
    )


def _listen_addresses(port: int) -> set[str]:
    addresses: set[str] = set()
    with open("/proc/net/tcp", encoding="ascii") as proc_tcp:
        next(proc_tcp)
        for line in proc_tcp:
            fields = line.split()
            local, state = fields[1], fields[3]
            if state != "0A":
                continue
            hex_addr, hex_port = local.split(":")
            if int(hex_port, 16) != port:
                continue
            raw = bytes.fromhex(hex_addr)
            addresses.add(socket.inet_ntoa(raw[::-1]))
    return addresses


def _outsider_delivers(store: TCPStore, key: str, payload: bytes) -> bool:
    try:
        store.set(key, payload)
    except Exception:
        return False
    try:
        return _wait_for_store(store, key, payload)
    except Exception:
        return False


def _assert_no_execution(marker: Path, delivered: bool, op: str) -> None:
    appeared = _marker_appeared(marker) if delivered else marker.exists()
    assert not appeared, (
        f"{op} executed bytes written by a TCPStore client that is not a "
        "member of the group"
    )


def _events(ctx: BaseContext, count: int) -> list[Event]:
    return [ctx.Event() for _ in range(count)]


@pytest.mark.parametrize("op", _STORE_OPS)
@pytest.mark.parametrize("host", [_HOST, "0.0.0.0"])
def test_stateless_receiver_does_not_run_outsider_bytes(
    tmp_path: Path, op: str, host: str
) -> None:
    """A client that joins after both ranks must not be unpickled."""
    marker = tmp_path / op
    payload = _payload(marker)
    port = _free_port(host)
    ctx = get_context("spawn")
    started0, started1, go, done = _events(ctx, 4)
    processes = [
        ctx.Process(
            target=_stateless_rank0,
            args=(host, port, op, False, started0, go, done),
        ),
        ctx.Process(
            target=_stateless_rank1,
            args=(host, port, op, started1, go),
        ),
    ]
    for proc in processes:
        proc.start()
    try:
        assert started0.wait(_START_TIMEOUT_S), processes[0].exitcode
        assert started1.wait(_START_TIMEOUT_S), processes[1].exitcode
        connect_host = _HOST if host == "0.0.0.0" else host
        if host == "0.0.0.0":
            assert "0.0.0.0" in _listen_addresses(port)
        elif host == _HOST:
            assert _listen_addresses(port) == {_HOST}
        outsider = _connect(connect_host, port)
        delivered = _outsider_delivers(outsider, _key_for(op), payload)
        go.set()
        _assert_no_execution(marker, delivered, op)
    finally:
        _stop(processes, done)


@pytest.mark.parametrize("op", _OVERWRITE_OPS)
def test_overwrite_after_legitimate_set_does_not_run(
    tmp_path: Path, op: str
) -> None:
    """Replacing a key after the sender's set still must not be executed."""
    marker = tmp_path / op
    payload = _payload(marker)
    port = _free_port(_HOST)
    ctx = get_context("spawn")
    started0, started1, sent, go, done = _events(ctx, 5)
    processes = [
        ctx.Process(
            target=_stateless_rank0,
            args=(_HOST, port, op, True, started0, sent, done),
        ),
        ctx.Process(
            target=_stateless_rank1,
            args=(_HOST, port, op, started1, go),
        ),
    ]
    for proc in processes:
        proc.start()
    try:
        assert started0.wait(_START_TIMEOUT_S), processes[0].exitcode
        assert started1.wait(_START_TIMEOUT_S), processes[1].exitcode
        assert sent.wait(_START_TIMEOUT_S)
        outsider = _connect(_HOST, port)
        benign = bytes(outsider.get(_key_for(op)))
        assert benign != payload
        delivered = _outsider_delivers(outsider, _key_for(op), payload)
        go.set()
        _assert_no_execution(marker, delivered, op)
    finally:
        _stop(processes, done)


def _run_coordinator(group, op: str) -> None:
    if op == "broadcast_object":
        group.broadcast_object(None, src=0)
    elif op == "broadcast_object_list":
        group.broadcast_object_list([None], src=0)
    elif op == "broadcast_tensor_dict":
        group.broadcast_tensor_dict(None, src=0)
    elif op == "recv_object":
        group.recv_object(0)
    elif op == "recv_tensor_dict":
        group.recv_tensor_dict(0)
    elif op == "broadcast":
        group.broadcast(torch.zeros(4), src=0)
    elif op == "barrier":
        group.barrier()
    elif op == "broadcast_expert_mapping":
        from vllm.distributed.elastic_ep.elastic_execute import (
            broadcast_expert_mapping,
        )

        broadcast_expert_mapping(
            None, group, torch.device("cpu"), src_rank=0
        )
    else:
        raise AssertionError(op)


def _coordinator_rank(
    rank: int,
    coord_port: int,
    group_name: str,
    op: str,
    overwrite: bool,
    started: Event,
    sent: Event,
    go: Event,
    done: Event,
) -> None:
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

    coord = _connect(_HOST, coord_port)
    group = StatelessGroupCoordinator(
        group_ranks=[[0, 1]],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        coord_store=coord,
        group_name=group_name,
        host=_HOST,
        global_rank=rank,
        global_world_size=2,
    )
    started.set()
    if rank == 0:
        if overwrite:
            group.broadcast_object("ok", src=0)
            sent.set()
        done.wait(_START_TIMEOUT_S)
        return
    if not go.wait(_START_TIMEOUT_S):
        return
    try:
        _run_coordinator(group, op)
    except Exception:
        return


def _group_tcp_port(coord: TCPStore, group_name: str) -> int:
    client = _connect(_HOST, coord.port)
    _device, _cpu, tcp_port = struct.unpack(
        _PORTS_FMT, client.get(f"{group_name}_0")
    )
    return tcp_port


@pytest.mark.parametrize(
    "group_name,op",
    [("dp", op) for op in _COORD_OPS]
    + [("ep", "broadcast_object"), ("eplb", "broadcast_object")],
)
def test_coordinator_receiver_does_not_run_outsider_bytes(
    tmp_path: Path, op: str, group_name: str
) -> None:
    """Ports read from the coordination store must not yield code execution."""
    marker = tmp_path / f"{group_name}-{op}"
    payload = _payload(marker)
    # Same construction as launch_core_engines for elastic expert parallelism.
    coord = create_tcp_store(
        _HOST,
        0,
        is_master=True,
        world_size=-1,
        wait_for_workers=False,
    )
    ctx = get_context("spawn")
    started0, started1, sent, go, done = _events(ctx, 5)
    processes = [
        ctx.Process(
            target=_coordinator_rank,
            args=(
                rank,
                coord.port,
                group_name,
                op,
                False,
                started0 if rank == 0 else started1,
                sent,
                go,
                done,
            ),
        )
        for rank in (0, 1)
    ]
    for proc in processes:
        proc.start()
    try:
        assert started0.wait(_START_TIMEOUT_S), processes[0].exitcode
        assert started1.wait(_START_TIMEOUT_S), processes[1].exitcode
        tcp_port = _group_tcp_port(coord, group_name)
        outsider = _connect(_HOST, tcp_port)
        delivered = _outsider_delivers(outsider, _key_for(op), payload)
        go.set()
        _assert_no_execution(marker, delivered, f"{group_name}.{op}")
    finally:
        _stop(processes, done)


def test_coordinator_overwrite_after_broadcast_object_does_not_run(
    tmp_path: Path,
) -> None:
    """A later set of the same broadcast key must not be unpickled either."""
    marker = tmp_path / "overwrite"
    payload = _payload(marker)
    coord = create_tcp_store(
        _HOST,
        0,
        is_master=True,
        world_size=-1,
        wait_for_workers=False,
    )
    ctx = get_context("spawn")
    started0, started1, sent, go, done = _events(ctx, 5)
    processes = [
        ctx.Process(
            target=_coordinator_rank,
            args=(
                rank,
                coord.port,
                "dp",
                "broadcast_object",
                True,
                started0 if rank == 0 else started1,
                sent,
                go,
                done,
            ),
        )
        for rank in (0, 1)
    ]
    for proc in processes:
        proc.start()
    try:
        assert started0.wait(_START_TIMEOUT_S), processes[0].exitcode
        assert started1.wait(_START_TIMEOUT_S), processes[1].exitcode
        assert sent.wait(_START_TIMEOUT_S)
        outsider = _connect(_HOST, _group_tcp_port(coord, "dp"))
        benign = bytes(outsider.get("broadcast_from/0/0"))
        assert benign != payload
        delivered = _outsider_delivers(
            outsider, "broadcast_from/0/0", payload
        )
        go.set()
        _assert_no_execution(marker, delivered, "dp.broadcast_object")
    finally:
        _stop(processes, done)
