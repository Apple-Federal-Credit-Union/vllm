# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL side-channel handshake must survive a malformed request.

These tests fail while one bad ZeroMQ message tears down the scheduler's
only handshake thread. They pass once that thread keeps serving registered
metadata to a later peer, without a second call to
``set_xfer_handshake_metadata``.
"""

import socket
from types import SimpleNamespace

import msgspec
import pytest
import torch
import zmq

from vllm.config import CacheConfig, KVTransferConfig, ParallelConfig, SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.utils.network_utils import get_open_port, make_zmq_path
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

_REPLY_TIMEOUT_MS = 2000
_SCHEDULERS = [NixlPullConnectorScheduler, NixlPushConnectorScheduler]
# Frames a DEALER sends. The ROUTER prepends the peer identity.
_POISONS: list[tuple[str, list[bytes]]] = [
    ("one_frame", [b"not-a-handshake"]),
    ("too_many_frames", [b"", b"a", b"b", b"c"]),
    ("malformed_msgpack", [b"", b"\xff\xff"]),
    ("short_list", [b"", msgspec.msgpack.encode((GET_META_MSG, 0))]),
    ("long_list", [b"", msgspec.msgpack.encode((GET_META_MSG, 0, 0, 1))]),
    ("non_sequence", [b"", msgspec.msgpack.encode(7)]),
    ("unregistered_pp_rank", [b"", msgspec.msgpack.encode((GET_META_MSG, 1, 0))]),
    ("unregistered_tp_rank", [b"", msgspec.msgpack.encode((GET_META_MSG, 0, 4))]),
    ("unhashable_rank", [b"", msgspec.msgpack.encode((GET_META_MSG, [0], 0))]),
]


def _reserve_side_channel_base(dp_index: int) -> int:
    for _ in range(32):
        base = get_open_port()
        candidate = base + dp_index
        if candidate > 65535:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                continue
        return base
    raise RuntimeError("Could not reserve a NIXL side-channel port")


def _scheduler_config(dp_index: int) -> tuple[SimpleNamespace, KVCacheConfig]:
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    # The scheduler constructor reads these four configs. A full VllmConfig
    # also requires a model, which this path does not use.
    vllm_config = SimpleNamespace(
        parallel_config=ParallelConfig(
            data_parallel_size=dp_index + 1,
            data_parallel_rank=dp_index,
        ),
        cache_config=CacheConfig(block_size=16),
        scheduler_config=SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=True,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_producer",
            engine_id="nixl-handshake-dos",
        ),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], spec)],
    )
    return vllm_config, kv_cache_config


def _start_scheduler(monkeypatch, scheduler_cls, dp_index: int = 1):
    base = _reserve_side_channel_base(dp_index)
    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "127.0.0.1")
    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_PORT", str(base))
    vllm_config, kv_cache_config = _scheduler_config(dp_index)
    scheduler = scheduler_cls(vllm_config, "nixl-handshake-dos", kv_cache_config)
    assert scheduler.side_channel_host == "127.0.0.1"
    assert scheduler.side_channel_port == base + dp_index
    payload = NixlHandshakePayload(
        compatibility_hash="hash",
        agent_metadata_bytes=b"agent-metadata",
    )
    scheduler.set_xfer_handshake_metadata({(0, 0): payload})
    thread = scheduler._nixl_handshake_listener_t
    assert thread is not None and thread.is_alive()
    expected = scheduler._handshake_data[(0, 0)]
    path = make_zmq_path(
        "tcp", scheduler.side_channel_host, scheduler.side_channel_port
    )
    return scheduler, path, expected


def _open_dealer(path: str) -> tuple[zmq.Context, zmq.Socket]:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, _REPLY_TIMEOUT_MS)
    sock.connect(path)
    return ctx, sock


def _count_registered_replies(sock: zmq.Socket, expected: bytes, count: int) -> int:
    """Count DEALER replies whose metadata frame is the registered payload."""
    found = 0
    while found < count:
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            break
        if len(parts) >= 2 and parts[1] == expected:
            found += 1
    return found


def _fetch_as_worker(path: str, expected: bytes) -> None:
    """Same REQ shape the NIXL worker uses for a metadata handshake."""
    valid = msgspec.msgpack.encode((GET_META_MSG, 0, 0))
    with zmq_ctx(zmq.REQ, path) as peer:
        peer.setsockopt(zmq.RCVTIMEO, _REPLY_TIMEOUT_MS)
        peer.setsockopt(zmq.LINGER, 0)
        peer.send(valid)
        try:
            parts = peer.recv_multipart()
        except zmq.Again as exc:
            raise AssertionError(
                "a later peer could not fetch registered handshake metadata"
            ) from exc
    assert parts[0] == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("scheduler_cls", _SCHEDULERS)
@pytest.mark.parametrize(
    ("poison_name", "poison"),
    _POISONS,
    ids=[name for name, _ in _POISONS],
)
def test_malformed_handshake_keeps_serving_metadata(
    monkeypatch, scheduler_cls, poison_name: str, poison: list[bytes]
):
    """One bad side-channel message must not stop later metadata fetches.

    Pull and push schedulers both start ``_nixl_handshake_listener`` and do
    not replace it. The listener has to keep answering ``GET_META_MSG`` for a
    rank it published.
    """
    scheduler, path, expected = _start_scheduler(monkeypatch, scheduler_cls)
    valid = msgspec.msgpack.encode((GET_META_MSG, 0, 0))
    ctx, sock = _open_dealer(path)
    try:
        sock.send_multipart([b"", valid])
        assert _count_registered_replies(sock, expected, 1) == 1, (
            "listener did not serve the registered rank before the bad message"
        )

        sock.send_multipart(poison)
        sock.send_multipart([b"", valid])
        sock.send_multipart([b"", valid])
        served = _count_registered_replies(sock, expected, 2)
        assert served == 2, (
            f"{scheduler_cls.__name__} stopped serving handshake metadata "
            f"after {poison_name}"
        )
        thread = scheduler._nixl_handshake_listener_t
        assert thread is not None and thread.is_alive(), (
            "handshake thread exited and the scheduler will not start another"
        )
        _fetch_as_worker(path, expected)
    finally:
        sock.close()
        ctx.destroy(linger=0)
        scheduler.shutdown()
