# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Elastic expert parallelism must not hand engine traffic to an impostor.

Enabling elastic EP turns on ZMQ ROUTER handover for the engine input socket
and binds that socket to the data-parallel master address. Engine identities
are the two-byte rank. These tests connect a raw DEALER that presents only
that rank and require that later engine requests still reach the engine that
already registered, including prompt token ids and multimodal fields.
"""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import msgspec.msgpack
import pytest
import torch
import zmq
from zmq.utils.monitor import parse_monitor_message

from vllm.config.cache import CacheConfig
from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.distributed.elastic_ep.external_elastic_ep import (
    ExternalElasticEPScaleUpHandshakeServer,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    get_open_port,
    get_tcp_uri,
    make_zmq_socket,
    split_zmq_path,
)
from vllm.v1.engine import (
    EngineCoreRequest,
    EngineCoreRequestType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.core_client import AsyncMPClient, MPClient, SyncMPClient
from vllm.v1.engine.utils import (
    CoreEngine,
    EngineZmqAddresses,
    bind_engine_zmq_listeners,
    get_engine_zmq_bind_addresses,
)
from vllm.v1.serial_utils import MsgpackDecoder

pytestmark = pytest.mark.skip_global_cleanup

_PROMPT_TOKEN_IDS = [11, 22, 33, 44]
_MM_IDENTIFIER = "mm-payload-marker"
_HANDSHAKE_EVENTS = (
    zmq.EVENT_HANDSHAKE_SUCCEEDED
    | zmq.EVENT_HANDSHAKE_FAILED_NO_DETAIL
    | zmq.EVENT_HANDSHAKE_FAILED_PROTOCOL
    | zmq.EVENT_HANDSHAKE_FAILED_AUTH
)


@dataclass
class _FrontendConfig:
    parallel_config: ParallelConfig
    cache_config: CacheConfig
    model_config: None = None


def _elastic_config(master_ip: str) -> ParallelConfig:
    # The config validator rejects elastic EP on hosts that are not CUDA-like.
    # This process has no GPU; the predicate is not part of the socket path.
    original = current_platform.is_cuda_alike
    current_platform.is_cuda_alike = lambda: True  # type: ignore[method-assign]
    try:
        return ParallelConfig(
            data_parallel_size=2,
            data_parallel_size_local=2,
            data_parallel_rank=0,
            enable_elastic_ep=True,
            enable_eplb=True,
            enable_expert_parallel=True,
            eplb_config=EPLBConfig(use_async=False),
            data_parallel_master_ip=master_ip,
        )
    finally:
        current_platform.is_cuda_alike = original


def _frontend_config(master_ip: str) -> _FrontendConfig:
    return _FrontendConfig(
        parallel_config=_elastic_config(master_ip),
        cache_config=CacheConfig(),
    )


def _rank_identity(rank: int) -> bytes:
    """Identity published by EngineCoreProc and CoreEngine: the numeric rank."""
    produced = CoreEngine(index=rank).identity
    encoded = rank.to_bytes(length=2, byteorder="little")
    assert produced == encoded
    return encoded


def _concrete_address(template: str) -> str:
    """Use the host the production bind selector chose, with a known port."""
    if template.startswith(("ipc://", "inproc://")):
        return template
    scheme, host, _port = split_zmq_path(template)
    assert scheme == "tcp"
    return get_tcp_uri(host, get_open_port())


def _request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="req-takeover",
        prompt_token_ids=list(_PROMPT_TOKEN_IDS),
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=_MM_IDENTIFIER,
                mm_position=PlaceholderRange(offset=1, length=2),
            )
        ],
        sampling_params=None,
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=0,
        prompt_embeds=torch.tensor([[1.5, 2.5]], dtype=torch.float32),
    )


def _connect_engine(ctx: zmq.Context, address: str, rank: int) -> zmq.Socket:
    """Connect the way EngineCoreProc.process_input_sockets connects."""
    sock = make_zmq_socket(
        ctx,
        address,
        zmq.DEALER,
        identity=_rank_identity(rank),
        bind=False,
        linger=0,
    )
    sock.send(b"")
    return sock


def _open_raw_peer(address: str, rank: int) -> tuple[zmq.Context, zmq.Socket]:
    """Peer that knows the rank and the bound address, and nothing else."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.IDENTITY, _rank_identity(rank))
    monitor = sock.get_monitor_socket(events=_HANDSHAKE_EVENTS)
    try:
        sock.connect(address)
        if not monitor.poll(2000):
            raise AssertionError("raw peer connection did not settle")
        parse_monitor_message(monitor.recv_multipart())
    finally:
        monitor.close(linger=0)
    return ctx, sock


def _recv_frames(sock: zmq.Socket, count: int, first_wait_ms: int) -> list[list[bytes]]:
    frames: list[list[bytes]] = []
    wait_ms = first_wait_ms
    while len(frames) < count and sock.poll(wait_ms):
        frames.append([bytes(frame) for frame in sock.recv_multipart()])
        wait_ms = 100
    return frames


def _describe(frames: list[list[bytes]]) -> list[object]:
    described: list[object] = []
    for frame in frames:
        kind = frame[0]
        if kind == EngineCoreRequestType.ADD.value:
            request = MsgpackDecoder(EngineCoreRequest).decode(frame[1:])
            identifier = None
            if request.mm_features:
                identifier = request.mm_features[0].identifier
            embeds = None
            if request.prompt_embeds is not None:
                embeds = request.prompt_embeds.tolist()
            described.append(("ADD", request.prompt_token_ids, identifier, embeds))
        else:
            described.append((kind, msgspec.msgpack.decode(frame[1])))
    return described


def _assert_impostor_got_nothing(frames: list[list[bytes]]) -> None:
    assert frames == [], "unauthenticated peer received engine traffic: " + repr(
        _describe(frames)
    )


def _assert_engine_got_add(frames: list[list[bytes]]) -> None:
    described = _describe(frames)
    assert ("ADD", _PROMPT_TOKEN_IDS, _MM_IDENTIFIER, [[1.5, 2.5]]) in described


@contextmanager
def _running_client(
    master_ip: str,
    *,
    asyncio_mode: bool,
    adopt_listener: bool,
) -> Iterator[tuple[MPClient, dict[int, zmq.Socket], str]]:
    config = _frontend_config(master_ip)
    ctx = zmq.Context()
    engines: dict[int, zmq.Socket] = {}
    client: MPClient | None = None
    listeners = None
    try:
        if adopt_listener:
            listeners = bind_engine_zmq_listeners(config)
            input_address = listeners.inputs[0].address
            output_address = listeners.outputs[0].address
            client_addresses = {
                "input_address": input_address,
                "output_address": output_address,
                "input_listener": listeners.inputs[0].socket,
                "output_listener": listeners.outputs[0].socket,
            }
        else:
            templates = get_engine_zmq_bind_addresses(config)
            input_address = _concrete_address(templates.inputs[0])
            output_address = _concrete_address(templates.outputs[0])
            client_addresses = {
                "input_address": input_address,
                "output_address": output_address,
            }
        for rank in (0, 1):
            engines[rank] = _connect_engine(ctx, input_address, rank)
        if asyncio_mode:
            client = AsyncMPClient(
                vllm_config=config,
                executor_class=object,
                log_stats=False,
                client_addresses=client_addresses,
            )
        else:
            client = MPClient(
                asyncio_mode=False,
                vllm_config=config,
                executor_class=object,
                log_stats=False,
                client_addresses=client_addresses,
            )
        yield client, engines, client.input_address
    finally:
        if client is not None:
            client.shutdown()
        for sock in engines.values():
            sock.close(linger=0)
        ctx.destroy(linger=0)
        if listeners is not None:
            listeners.cleanup()


@pytest.mark.parametrize("master_ip", ["127.0.0.1", "172.30.0.2"])
def test_sync_router_does_not_deliver_requests_to_rank_impostor(master_ip: str):
    """Rank 0's add, abort, and utility frames stay with the registered engine."""
    with _running_client(master_ip, asyncio_mode=False, adopt_listener=False) as (
        client,
        engines,
        address,
    ):
        _ctx, impostor = _open_raw_peer(address, 0)
        try:
            request = _request()
            SyncMPClient._send_input(client, EngineCoreRequestType.ADD, request)
            SyncMPClient._send_input(
                client, EngineCoreRequestType.ABORT, ["req-takeover"]
            )
            SyncMPClient._send_input(
                client,
                EngineCoreRequestType.UTILITY,
                (0, 7, "get_supported_tasks", ()),
            )
            stolen = _recv_frames(impostor, 3, 1000)
            delivered = _recv_frames(engines[0], 3, 1000)
        finally:
            impostor.close(linger=0)
            _ctx.destroy(linger=0)

    _assert_impostor_got_nothing(stolen)
    _assert_engine_got_add(delivered)
    assert (EngineCoreRequestType.ABORT.value, ["req-takeover"]) in _describe(delivered)
    assert (
        EngineCoreRequestType.UTILITY.value,
        [0, 7, "get_supported_tasks", []],
    ) in _describe(delivered)


def test_async_router_does_not_deliver_nonzero_rank_requests():
    """Async sends addressed to rank 1 stay with that engine."""

    async def _run() -> tuple[list[list[bytes]], list[list[bytes]]]:
        with _running_client("127.0.0.1", asyncio_mode=True, adopt_listener=False) as (
            client,
            engines,
            address,
        ):
            assert isinstance(client, AsyncMPClient)
            _ctx, impostor = _open_raw_peer(address, 1)
            try:
                await client._send_input(
                    EngineCoreRequestType.ADD,
                    _request(),
                    _rank_identity(1),
                )
                stolen = _recv_frames(impostor, 1, 1000)
                delivered = _recv_frames(engines[1], 1, 1000)
            finally:
                impostor.close(linger=0)
                _ctx.destroy(linger=0)
        return stolen, delivered

    stolen, delivered = asyncio.run(_run())
    _assert_impostor_got_nothing(stolen)
    _assert_engine_got_add(delivered)


def test_inherited_listener_does_not_deliver_requests_to_rank_impostor():
    """The supervisor-bound input listener uses the same unprotected router."""
    with _running_client("127.0.0.1", asyncio_mode=False, adopt_listener=True) as (
        client,
        engines,
        address,
    ):
        _ctx, impostor = _open_raw_peer(address, 0)
        try:
            SyncMPClient._send_input(client, EngineCoreRequestType.ADD, _request())
            stolen = _recv_frames(impostor, 1, 1000)
            delivered = _recv_frames(engines[0], 1, 1000)
        finally:
            impostor.close(linger=0)
            _ctx.destroy(linger=0)

    _assert_impostor_got_nothing(stolen)
    _assert_engine_got_add(delivered)


def test_scale_up_handshake_does_not_give_metadata_to_rank_impostor():
    """A second peer with a new rank's identity must not receive scale-up metadata."""
    rank = 2
    marker = "tcp://203.0.113.10:9"
    address = get_tcp_uri("127.0.0.1", get_open_port())
    server = ExternalElasticEPScaleUpHandshakeServer(
        handshake_address=address,
        expected_new_ranks=[rank],
        addresses=EngineZmqAddresses(
            inputs=[marker],
            outputs=["tcp://203.0.113.10:10"],
        ),
        bootstrap=ReconfigureDistributedRequest(
            new_data_parallel_size=3,
            new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
            new_data_parallel_master_ip="127.0.0.1",
            new_data_parallel_master_port=29500,
            new_data_parallel_master_port_list=[29501],
            coord_store_port=29502,
        ),
        num_redundant_experts=0,
        elastic_ep_max_dp_size=4,
    )
    ctx = zmq.Context()
    engine = impostor = None
    impostor_ctx = None
    try:
        server.start()
        engine = make_zmq_socket(
            ctx,
            address,
            zmq.DEALER,
            identity=_rank_identity(rank),
            bind=False,
            linger=0,
        )
        hello = msgspec.msgpack.encode({"status": "HELLO", "local": False})
        engine.send(hello)
        first = engine.recv()
        assert marker.encode() in first

        impostor_ctx, impostor = _open_raw_peer(address, rank)
        impostor.send(hello)
        stolen = _recv_frames(impostor, 1, 1000)
        engine.send(hello)
        follow_up = _recv_frames(engine, 1, 1000)
    finally:
        if impostor is not None:
            impostor.close(linger=0)
        if impostor_ctx is not None:
            impostor_ctx.destroy(linger=0)
        if engine is not None:
            engine.close(linger=0)
        ctx.destroy(linger=0)
        server.stop()

    assert stolen == [], (
        "unauthenticated peer received scale-up handshake metadata: " + repr(stolen)
    )
    assert follow_up, "registered engine did not receive the scale-up handshake"
    assert marker.encode() in follow_up[0][0]
