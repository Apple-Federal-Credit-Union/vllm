# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake prefill side channel must not serve KV blocks to an arbitrary peer.

The producer binds a ZeroMQ ROUTER and publishes that address through the
bootstrap registry. A pull is a ``MooncakeXferMetadata`` message. These tests
drive that socket, not ``send_kv_to_decode`` directly.

A pull may copy blocks only for a transfer id the scheduler has recorded, and
only to the decode peer named on that request:

* ``kv_transfer_params["decode_block_host"]``
* ``kv_transfer_params["decode_block_port"]``

Those values have to survive ``build_connector_meta`` and ``start_load_kv``.
The session passed to ``TransferEngine.batch_transfer_sync_write`` must be
``"{decode_block_host}:{decode_block_port}"`` for that transfer, and the source
pointers must be that transfer's blocks. Any other session, including one that
only matches the transfer id, must be rejected with
``MooncakeXferResponseStatus.ERROR`` without waiting
``VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`` and without a transfer call.

``kv_both`` uses the same listener and is not started separately. The consumer
pull path does not serve blocks. A pull that arrives before blocks are ready
uses this same listener and is covered by the unknown-id and ready-block tests.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from itertools import count

import httpx
import msgspec
import pytest
import torch
import zmq

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.platforms import current_platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
)
from vllm.v1.request import RequestStatus

from .utils import create_request, create_vllm_config, maybe_update_block_size

_LAYER = "model.layers.0.self_attn"
_ABORT_TIMEOUT_S = 30
_FAST_S = 1.5
_AUTH_HOST = "198.51.100.10"
_AUTH_PORT = 40000
_OTHER_HOST = "198.51.100.20"
_OTHER_PORT = 40001
_ATTACK_HOST = "203.0.113.10"
_ATTACK_PORT = 50000
_REQUEST_IDS = count(1)


class _RecordingTransferEngine:
    """Records ``batch_transfer_sync_write`` arguments and performs no I/O.

    ``mooncake.engine.TransferEngine.batch_transfer_sync_write`` (package
    mooncake-transfer-engine 0.3.14, the release satisfying
    ``requirements/kv_connectors.txt``) opens ``target_hostname`` as a segment
    and submits a WRITE of each local buffer to that peer. Installing that
    engine here would require an RDMA device, so the worker is given this
    recorder. The assertions are on the session and pointers the connector
    passes into that call.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, list[int], list[int], list[int]]] = []
        self.protocol: str | None = None

    def initialize(
        self,
        local_hostname: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ) -> int:
        self.protocol = protocol
        return 0

    def get_rpc_port(self) -> int:
        return 1

    def batch_register_memory(
        self, buffer_addresses: list[int], capacities: list[int]
    ) -> int:
        return 0

    def batch_unregister_memory(self, buffer_addresses: list[int]) -> int:
        return 0

    def batch_transfer_sync_write(
        self,
        target_hostname: str,
        buffers: list[int],
        peer_buffer_addresses: list[int],
        lengths: list[int],
    ) -> int:
        self.writes.append(
            (
                target_hostname,
                list(buffers),
                list(peer_buffer_addresses),
                list(lengths),
            )
        )
        return 0


@dataclass
class _Producer:
    worker_connector: MooncakeConnector
    scheduler: MooncakeConnector
    cache: torch.Tensor
    bootstrap: MooncakeBootstrapServer
    side_channel: str

    @property
    def worker(self):
        worker = self.worker_connector.connector_worker
        assert worker is not None
        return worker

    @property
    def engine(self) -> _RecordingTransferEngine:
        engine = self.worker.engine
        assert isinstance(engine, _RecordingTransferEngine)
        return engine


def _kv_cache_config() -> tuple[KVCacheConfig, FullAttentionSpec]:
    block_size = maybe_update_block_size(16)
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
    )
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([_LAYER], spec)],
    )
    return config, spec


def _pp_group() -> GroupCoordinator:
    group = object.__new__(GroupCoordinator)
    group.rank_in_group = 0
    return group


@pytest.fixture(autouse=True)
def _cpu_kernel_block_size(monkeypatch: pytest.MonkeyPatch):
    """CPU attention declares a larger kernel block than these tests use."""
    if not current_platform.is_cpu():
        return
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    monkeypatch.setattr(
        CPUAttentionBackend,
        "get_supported_kernel_block_sizes",
        staticmethod(lambda kv_cache_spec=None: [MultipleOf(16)]),
    )


@pytest.fixture
def producer(monkeypatch: pytest.MonkeyPatch):
    """Start a real prefiller listener and an unauthenticated bootstrap."""
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv(
        "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", str(_ABORT_TIMEOUT_S)
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.TransferEngine",
        _RecordingTransferEngine,
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(current_platform, "set_device", lambda device: None)
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.get_pp_group",
        _pp_group,
    )

    bootstrap = MooncakeBootstrapServer("127.0.0.1", 0)
    bootstrap.start()
    kv_config, spec = _kv_cache_config()
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        block_size=spec.block_size,
        kv_connector_extra_config={
            "num_workers": 1,
            "bootstrap_server_address": f"127.0.0.1:{bootstrap.port}",
        },
    )
    raw = torch.zeros(8 * spec.page_size_bytes, dtype=torch.int8)
    raw[spec.page_size_bytes * 3 : spec.page_size_bytes * 4] = 0x5A
    cache = dense_kv_cache_views(raw, spec, 1, 8, KVCacheLayout.LBHNC)[0]
    connector: MooncakeConnector | None = None
    try:
        with set_current_vllm_config(vllm_config):
            connector = MooncakeConnector(
                vllm_config, KVConnectorRole.WORKER, kv_config
            )
            scheduler = MooncakeConnector(
                vllm_config, KVConnectorRole.SCHEDULER, kv_config
            )
        connector.register_kv_caches({_LAYER: cache})
        side_channel = _discover_side_channel(bootstrap.port)
        assert side_channel.startswith("tcp://127.0.0.1:")
        yield _Producer(connector, scheduler, cache, bootstrap, side_channel)
    finally:
        # Worker.shutdown() and Worker.__del__ call Context.term() while the
        # listener loop still polls the ROUTER, and term() waits for that
        # socket. Drop the destructor and stop the daemon loop instead.
        if connector is not None and connector.connector_worker is not None:
            worker = connector.connector_worker
            worker.shutdown = lambda: None  # type: ignore[method-assign]
            if worker.sender_loop.is_running():
                worker.sender_loop.call_soon_threadsafe(worker.sender_loop.stop)
                worker._sender_listener_t.join(timeout=2)
            worker._sender_executor.shutdown(wait=False)
        bootstrap.shutdown()


def _discover_side_channel(bootstrap_port: int) -> str:
    """Read the published worker address with no credentials."""
    response = httpx.get(
        f"http://127.0.0.1:{bootstrap_port}/query", timeout=5.0
    )
    response.raise_for_status()
    found: list[str] = []
    for entry in response.json().values():
        for ranks in entry["worker_addr"].values():
            found.extend(ranks.values())
    assert found, response.text
    return found[0]


def _block_sources(worker, block_id: int) -> set[int]:
    return {
        base + block_id * block_len
        for base, block_len in zip(
            worker.kv_caches_base_addr, worker.block_len_per_layer
        )
    }


def _pull(
    producer: _Producer,
    transfer_blocks: dict[str, list[int]],
    host: str,
    port: int,
    *,
    layout_id: int = 0,
    include_layout: bool = True,
) -> MooncakeXferMetadata:
    """Build a pull the way ``_build_pull_metadata`` does, for this peer."""
    worker = producer.worker
    req_blocks = {
        f"decode-{transfer_id}": (transfer_id, [block_ids])
        for transfer_id, block_ids in transfer_blocks.items()
    }
    remote_bases = [
        0xA0000000 + i * 0x100000
        for i in range(len(worker.kv_caches_base_addr))
    ]
    if not include_layout:
        return MooncakeXferMetadata(
            remote_hostname=host,
            remote_port=port,
            remote_tp_size=worker.tp_size,
            remote_tp_rank=worker.tp_rank,
            remote_pp_size=worker.pp_size,
            layout_id=layout_id,
            req_blocks=req_blocks,
            kv_caches_base_addr=[],
            block_lens=[],
            kv_block_lens=[],
        )
    return MooncakeXferMetadata(
        remote_hostname=host,
        remote_port=port,
        remote_tp_size=worker.tp_size,
        remote_tp_rank=worker.tp_rank,
        remote_pp_size=worker.pp_size,
        layout_id=layout_id,
        req_blocks=req_blocks,
        kv_caches_base_addr=remote_bases,
        block_lens=list(worker.block_len_per_layer),
        kv_block_lens=list(worker.kv_block_len_per_layer),
        registered_layer_names=list(worker.registered_layer_names),
        registered_layer_indices=list(worker.registered_layer_indices),
        registered_group_indices=list(worker.registered_group_indices),
        registered_shared_group_ids=[
            list(groups) for groups in worker.region_shared_groups
        ],
        registered_row_offsets=list(worker.region_row_offsets),
    )


_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder(MooncakeXferResponse)


def _send(addr: str, meta: MooncakeXferMetadata, timeout_s: float = _FAST_S):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    started = time.monotonic()
    try:
        sock.connect(addr)
        sock.send(_ENCODER.encode(meta))
        try:
            raw = sock.recv()
        except zmq.Again:
            return None, time.monotonic() - started
        return _DECODER.decode(raw), time.monotonic() - started
    finally:
        sock.close()
        ctx.term()


def _record(
    producer: _Producer,
    transfer_id: str,
    block_ids: list[int],
    host: str | None,
    port: int | None,
    *,
    ready: bool,
) -> None:
    """Drive the scheduler path that arms a producer send."""
    request = create_request(
        request_id=next(_REQUEST_IDS),
        num_tokens=16,
        do_remote_decode=True,
        block_size=producer.worker.block_size,
    )
    assert request.kv_transfer_params is not None
    request.kv_transfer_params["transfer_id"] = transfer_id
    if host is not None and port is not None:
        request.kv_transfer_params["decode_block_host"] = host
        request.kv_transfer_params["decode_block_port"] = port
    scheduler = producer.scheduler
    scheduler.update_state_after_alloc(request, None, 0)
    meta = scheduler.build_connector_meta(None)
    producer.worker_connector.bind_connector_metadata(meta)
    producer.worker_connector.start_load_kv(None)
    _wait_until(
        lambda: transfer_id in producer.worker.reqs_need_send,
        "scheduler did not record the transfer",
    )
    if not ready:
        return
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    scheduler.request_finished(request, block_ids)
    meta = scheduler.build_connector_meta(None)
    producer.worker_connector.bind_connector_metadata(meta)
    producer.worker_connector.start_load_kv(None)
    _wait_until(
        lambda: (
            (send := producer.worker.reqs_need_send.get(transfer_id)) is not None
            and send.ready.is_set()
            and send.local_block_ids == [block_ids]
        ),
        "producer blocks were not marked ready",
    )


def _wait_until(predicate, message: str, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def _assert_fast_error(reply, elapsed: float) -> None:
    assert reply is not None, (
        "side channel did not answer before the abort timeout"
    )
    assert elapsed < _FAST_S
    assert reply.status == MooncakeXferResponseStatus.ERROR
    assert not reply.ok_reqs


def _new_writes(producer: _Producer, mark: int):
    return producer.engine.writes[mark:]


def test_bootstrap_query_publishes_the_side_channel_without_credentials(
    producer,
):
    """The listener address is the one the worker registered."""
    again = _discover_side_channel(producer.bootstrap.port)
    assert again == producer.side_channel


def test_unknown_transfer_does_not_occupy_sender_until_abort(producer):
    """Two unknown pulls must not pin both sender tasks (num_workers=1)."""
    _record(producer, "xfer-ready", [3], _AUTH_HOST, _AUTH_PORT, ready=True)
    mark = len(producer.engine.writes)
    unknown_replies: list[tuple[object, float]] = []

    def _unknown(transfer_id: str) -> None:
        meta = _pull(producer, {transfer_id: [0]}, _ATTACK_HOST, _ATTACK_PORT)
        unknown_replies.append(_send(producer.side_channel, meta))

    threads = [
        threading.Thread(target=_unknown, args=(f"unknown-{i}",))
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    _wait_until(
        lambda: len(unknown_replies) == 2
        or all(f"unknown-{i}" in producer.worker.reqs_need_send for i in range(2)),
        "unknown pulls were neither rejected nor parked",
    )
    auth = _pull(producer, {"xfer-ready": [0]}, _AUTH_HOST, _AUTH_PORT)
    reply, elapsed = _send(producer.side_channel, auth)
    for thread in threads:
        thread.join(timeout=_FAST_S + 1)
    writes = _new_writes(producer, mark)
    assert any(session == f"{_AUTH_HOST}:{_AUTH_PORT}" for session, *_ in writes), (
        writes
    )
    assert reply is not None and elapsed < _FAST_S
    assert reply.ok_reqs == ["decode-xfer-ready"]
    for unknown_reply, unknown_elapsed in unknown_replies:
        _assert_fast_error(unknown_reply, unknown_elapsed)
    assert all(f"unknown-{i}" not in producer.worker.reqs_need_send for i in range(2))


def test_ready_blocks_are_not_copied_to_an_unregistered_session(producer):
    """A peer that knows the transfer id still must not receive the blocks."""
    _record(producer, "xfer-victim", [3], _AUTH_HOST, _AUTH_PORT, ready=True)
    mark = len(producer.engine.writes)
    meta = _pull(producer, {"xfer-victim": [0]}, _ATTACK_HOST, _ATTACK_PORT)
    reply, elapsed = _send(producer.side_channel, meta)
    _assert_fast_error(reply, elapsed)
    assert _new_writes(producer, mark) == []
    victim = _block_sources(producer.worker, 3)
    assert all(victim.isdisjoint(srcs) for _, srcs, _, _ in producer.engine.writes)


def test_transfer_without_a_registered_peer_is_refused(producer):
    """A recorded transfer with no decode peer is not an open pull."""
    _record(producer, "xfer-open", [3], None, None, ready=True)
    mark = len(producer.engine.writes)
    meta = _pull(producer, {"xfer-open": [0]}, _ATTACK_HOST, _ATTACK_PORT)
    reply, elapsed = _send(producer.side_channel, meta)
    _assert_fast_error(reply, elapsed)
    assert _new_writes(producer, mark) == []


def test_one_pull_cannot_take_another_transfers_blocks(producer):
    """Authorizing session A must not send transfer B's blocks to A."""
    _record(producer, "xfer-a", [1], _AUTH_HOST, _AUTH_PORT, ready=True)
    _record(producer, "xfer-b", [4], _OTHER_HOST, _OTHER_PORT, ready=True)
    mark = len(producer.engine.writes)
    meta = _pull(
        producer,
        {"xfer-a": [0], "xfer-b": [1]},
        _AUTH_HOST,
        _AUTH_PORT,
    )
    reply, elapsed = _send(producer.side_channel, meta)
    assert reply is not None and elapsed < _FAST_S
    writes = _new_writes(producer, mark)
    stolen = _block_sources(producer.worker, 4)
    allowed = _block_sources(producer.worker, 1)
    for session, srcs, _, lengths in writes:
        assert session == f"{_AUTH_HOST}:{_AUTH_PORT}"
        assert stolen.isdisjoint(srcs)
        assert set(srcs) <= allowed
        assert lengths
        assert all(length > 0 for length in lengths)
    assert "decode-xfer-b" not in (reply.ok_reqs or [])


def test_repeated_layout_pull_cannot_skip_the_peer_check(producer):
    """A second pull that omits the layout still has to match the peer."""
    _record(producer, "xfer-layout", [3], _AUTH_HOST, _AUTH_PORT, ready=True)
    mark = len(producer.engine.writes)
    first = _pull(
        producer,
        {"xfer-layout": [0]},
        _ATTACK_HOST,
        _ATTACK_PORT,
        layout_id=0x11,
    )
    first_reply, first_elapsed = _send(producer.side_channel, first)
    _assert_fast_error(first_reply, first_elapsed)
    second = _pull(
        producer,
        {"xfer-layout": [0]},
        _ATTACK_HOST,
        _ATTACK_PORT,
        layout_id=0x11,
        include_layout=False,
    )
    second_reply, second_elapsed = _send(producer.side_channel, second)
    _assert_fast_error(second_reply, second_elapsed)
    assert _new_writes(producer, mark) == []


def test_registered_peer_receives_its_ready_blocks(producer):
    """The decode peer named on the request still receives that request."""
    _record(producer, "xfer-ok", [3], _AUTH_HOST, _AUTH_PORT, ready=True)
    meta = _pull(producer, {"xfer-ok": [7]}, _AUTH_HOST, _AUTH_PORT)
    reply, elapsed = _send(producer.side_channel, meta)
    assert reply is not None and elapsed < _FAST_S
    assert reply.status == MooncakeXferResponseStatus.FINISH
    assert reply.ok_reqs == ["decode-xfer-ok"]
    assert len(producer.engine.writes) == 1
    session, srcs, dsts, lengths = producer.engine.writes[0]
    assert session == f"{_AUTH_HOST}:{_AUTH_PORT}"
    assert set(srcs) == _block_sources(producer.worker, 3)
    remote_base = 0xA0000000
    remote_block = 7
    expected_dst = {
        remote_base + remote_block * block_len
        for block_len in producer.worker.block_len_per_layer
    }
    assert set(dsts) == expected_dst
    assert lengths == list(producer.worker.kv_block_len_per_layer)
