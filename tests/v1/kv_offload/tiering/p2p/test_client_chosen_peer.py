# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client-supplied P2P peers must not be dialed.

An API caller can put ``remote_prefiller`` or ``remote_kv_source`` in
``kv_transfer_params``. With the P2P secondary tier enabled, that address
is opened as a ZMQ control connection and the local KV layout is sent in
``ConnectMsg`` before any peer authentication. These tests drive the
public request objects into ``P2PSecondaryTierManager`` and require that
an untrusted address is never contacted and is never treated as a hit.

``NixlTransport`` is replaced with ``_MemoryLayoutTransport``. NIXL is an
external library; the stand-in uses the real ``DataTransport`` memory
layout (base address, block count, block length, config fingerprint).
``P2PSession._send_connect`` sends ``get_agent_metadata()`` in the same
field ``NixlTransport.get_agent_metadata`` fills from the NIXL agent.
"""

from __future__ import annotations

import time
from typing import Any

import msgspec
import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    _create_req_context,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingSpec,
    OffloadPolicy,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.p2p import manager as manager_module
from vllm.v1.kv_offload.tiering.p2p.data.base import DataTransport, PollResult
from vllm.v1.kv_offload.tiering.p2p.manager import P2PSecondaryTierManager
from vllm.v1.request import Request

pytestmark = pytest.mark.skip_global_cleanup

_AGENT_METADATA = b"p2p-agent-metadata"


class _Spec(OffloadingSpec):
    def get_manager(self):
        raise NotImplementedError

    def get_worker(self, kv_caches):
        raise NotImplementedError


class _MemoryLayoutTransport(DataTransport):
    """Data-plane stand-in with the real ``DataTransport`` layout fields.

    Constructor signature matches ``NixlTransport`` so the P2P manager can
    build it. ``get_agent_metadata`` returns opaque bytes, which is the
    contract ``NixlTransport.get_agent_metadata`` implements by returning
    the NIXL agent's metadata.
    """

    def __init__(
        self,
        agent_name: str,
        view: memoryview,
        config_fields: dict | None = None,
        backends: list[str] | None = None,
        num_threads: int = 4,
    ) -> None:
        super().__init__(view, config_fields)
        self._agent_name = agent_name
        self._metadata = _AGENT_METADATA

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
        return None

    def remove_remote_peer(self, peer_id: str) -> None:
        return None

    def write_blocks(
        self,
        peer_id: str,
        local_idxs: list[int],
        remote_idxs: list[int],
    ) -> int | None:
        return None

    def poll(self, peer_id: str | None = None) -> PollResult:
        return PollResult(done=(), failed=())

    def cancel(self, transfer_ids, mode="immediate") -> list[int]:
        return []

    def close(self) -> None:
        return None


class _SecondRead:
    """Stringifies to ``first`` once, then to ``later``.

    ``_peer_id_from_params`` interpolates host and port once. A check that
    reads them again and dials the later value still contacts the client.
    """

    def __init__(self, first: str, later: str) -> None:
        self._first = first
        self._later = later
        self.reads = 0

    def __str__(self) -> str:
        self.reads += 1
        return self._first if self.reads == 1 else self._later

    def __bool__(self) -> bool:
        return True


def _offloading_config() -> OffloadingConfig:
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=16,
                layer_names=("layer.0",),
                group_id=0,
            ),
        ),
        worker_kv_bytes_per_block=16,
        enable_kv_cache_events=False,
        extra_config={},
        engine_id="p2p-client-peer-test",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=True,
        ),
    )


def _request_from_sampling(sampling: SamplingParams) -> Request:
    engine_request = EngineCoreRequest(
        request_id="client-req",
        prompt_token_ids=[10, 11, 12, 13],
        mm_features=None,
        sampling_params=sampling,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return Request.from_engine_core_request(engine_request, None)


def _bind_router() -> tuple[zmq.Context, zmq.Socket, zmq.Socket, int]:
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    endpoint = f"inproc://p2p-client-peer-{id(router)}"
    events = zmq.EVENT_ACCEPTED
    handshake = getattr(zmq, "EVENT_HANDSHAKE_SUCCEEDED", 0)
    router.monitor(endpoint, events | handshake)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    monitor = ctx.socket(zmq.PAIR)
    monitor.setsockopt(zmq.LINGER, 0)
    monitor.connect(endpoint)
    return ctx, router, monitor, port


def _peer_contact(
    router: zmq.Socket,
    monitor: zmq.Socket,
    timeout_s: float,
) -> tuple[bool, dict | None]:
    """Return whether a TCP peer was accepted and any decoded control message."""
    accepted = False
    payload: dict | None = None
    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)
    poller.register(monitor, zmq.POLLIN)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not (accepted and payload is not None):
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        events = dict(poller.poll(remaining_ms))
        if monitor in events:
            # An accepted TCP connection is a side effect even with no payload.
            monitor.recv_multipart(zmq.NOBLOCK)
            accepted = True
        if router in events:
            frames = router.recv_multipart(zmq.NOBLOCK)
            payload = msgspec.msgpack.decode(frames[-1])
            accepted = True
    return accepted, payload


def _close_listener(ctx: zmq.Context, router: zmq.Socket, monitor: zmq.Socket) -> None:
    router.close(linger=0)
    monitor.close(linger=0)
    ctx.term()


@pytest.fixture
def p2p_manager(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(manager_module, "NixlTransport", _MemoryLayoutTransport)
    spec = _Spec(_offloading_config())
    storage = bytearray(4 * 16)
    view = memoryview(storage).cast("B", (4, 16))
    manager = P2PSecondaryTierManager(
        offloading_spec=spec,
        primary_kv_view=view,
        host="127.0.0.1",
        port=0,
    )
    try:
        yield manager
    finally:
        manager.shutdown()


def _assert_not_contacted(
    accepted: bool,
    payload: dict | None,
    lookup: LookupResult,
) -> None:
    problems: list[str] = []
    if accepted or payload is not None:
        problems.append(
            f"client-chosen peer was contacted (accepted={accepted}, message={payload})"
        )
    if lookup is not LookupResult.MISS:
        problems.append(f"lookup of an untrusted peer returned {lookup}")
    assert not problems, "; ".join(problems)


def _chat_sampling(params: dict[str, Any]) -> SamplingParams:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "You are a helpful assistant."}],
            "max_tokens": 8,
            "kv_transfer_params": params,
        }
    )
    return request.to_sampling_params(8, {})


def _completion_sampling(params: dict[str, Any]) -> SamplingParams:
    request = CompletionRequest.model_validate(
        {
            "model": "test-model",
            "prompt": "You are a helpful assistant.",
            "max_tokens": 8,
            "kv_transfer_params": params,
        }
    )
    return request.to_sampling_params(8, {})


def _responses_sampling(params: dict[str, Any]) -> SamplingParams:
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "You are a helpful assistant.",
            "max_output_tokens": 8,
            "kv_transfer_params": params,
        }
    )
    return request.to_sampling_params(8, {})


def _generate_sampling(params: dict[str, Any]) -> SamplingParams:
    request = GenerateRequest.model_validate(
        {
            "token_ids": [10, 11, 12, 13],
            "sampling_params": {"max_tokens": 8},
            "kv_transfer_params": params,
        }
    )
    # serve_tokens copies this field onto sampling extra_args before the
    # engine builds the request. Reproduce that assignment from the
    # validated generate body; the copy itself lives in serve_tokens.
    sampling = request.sampling_params
    assert request.kv_transfer_params is not None
    extra = dict(sampling.extra_args or {})
    extra["kv_transfer_params"] = request.kv_transfer_params
    sampling.extra_args = extra
    return sampling


def _prefiller(host: Any, port: Any) -> dict[str, Any]:
    return {
        "remote_prefiller": {
            "remote_host": host,
            "remote_port": port,
            "kv_request_id": "attacker-req",
        }
    }


def test_chat_remote_prefiller_is_not_dialed(p2p_manager):
    ctx, router, monitor, port = _bind_router()
    try:
        sampling = _chat_sampling(_prefiller("127.0.0.1", port))
        request = _request_from_sampling(sampling)
        req_context = _create_req_context(request, None)  # type: ignore[arg-type]
        p2p_manager.on_new_request(req_context)
        accepted, payload = _peer_contact(router, monitor, timeout_s=1.0)
        lookup = p2p_manager.lookup(make_offload_key(b"prompt-block", 0), req_context)
        _assert_not_contacted(accepted, payload, lookup)
    finally:
        _close_listener(ctx, router, monitor)


def test_completion_remote_prefiller_is_not_dialed(p2p_manager):
    ctx, router, monitor, port = _bind_router()
    try:
        sampling = _completion_sampling(_prefiller("127.0.0.1", port))
        request = _request_from_sampling(sampling)
        req_context = _create_req_context(request, None)  # type: ignore[arg-type]
        p2p_manager.on_new_request(req_context)
        accepted, payload = _peer_contact(router, monitor, timeout_s=1.0)
        lookup = p2p_manager.lookup(make_offload_key(b"prompt-block", 0), req_context)
        _assert_not_contacted(accepted, payload, lookup)
    finally:
        _close_listener(ctx, router, monitor)


def test_responses_remote_kv_source_is_not_dialed(p2p_manager):
    ctx, router, monitor, port = _bind_router()
    try:
        sampling = _responses_sampling(
            {
                "remote_kv_source": {
                    "remote_host": "127.0.0.1",
                    "remote_port": port,
                    "kv_request_id": "attacker-req",
                }
            }
        )
        request = _request_from_sampling(sampling)
        req_context = _create_req_context(request, None)  # type: ignore[arg-type]
        p2p_manager.on_new_request(req_context)
        accepted, payload = _peer_contact(router, monitor, timeout_s=1.0)
        lookup = p2p_manager.lookup(make_offload_key(b"prompt-block", 0), req_context)
        _assert_not_contacted(accepted, payload, lookup)
    finally:
        _close_listener(ctx, router, monitor)


def test_generate_remote_prefiller_is_not_dialed(p2p_manager):
    ctx, router, monitor, port = _bind_router()
    try:
        sampling = _generate_sampling(_prefiller("127.0.0.1", port))
        request = _request_from_sampling(sampling)
        req_context = _create_req_context(request, None)  # type: ignore[arg-type]
        p2p_manager.on_new_request(req_context)
        accepted, payload = _peer_contact(router, monitor, timeout_s=1.0)
        lookup = p2p_manager.lookup(make_offload_key(b"prompt-block", 0), req_context)
        _assert_not_contacted(accepted, payload, lookup)
    finally:
        _close_listener(ctx, router, monitor)


def test_remote_decoder_does_not_force_request_level_offload(p2p_manager):
    sampling = _chat_sampling({"remote_decoder": {"kv_request_id": "attacker-req"}})
    request = _request_from_sampling(sampling)
    req_context = _create_req_context(request, None)  # type: ignore[arg-type]
    offload_ctx = p2p_manager.on_new_request(req_context)
    assert offload_ctx.policy is not OffloadPolicy.REQUEST_LEVEL


def test_second_read_of_peer_address_is_not_dialed(p2p_manager):
    """A check that re-reads host or port must not dial the later value."""
    first = _bind_router()
    later = _bind_router()
    try:
        host = _SecondRead("127.0.0.1", "127.0.0.1")
        port = _SecondRead(str(first[3]), str(later[3]))
        sampling = SamplingParams(
            max_tokens=8,
            extra_args={"kv_transfer_params": _prefiller(host, port)},
        )
        request = _request_from_sampling(sampling)
        req_context = _create_req_context(request, None)  # type: ignore[arg-type]
        p2p_manager.on_new_request(req_context)
        first_hit = _peer_contact(first[1], first[2], timeout_s=1.0)
        later_hit = _peer_contact(later[1], later[2], timeout_s=0.3)
        lookup = p2p_manager.lookup(make_offload_key(b"prompt-block", 0), req_context)
        _assert_not_contacted(first_hit[0], first_hit[1], lookup)
        _assert_not_contacted(later_hit[0], later_hit[1], lookup)
    finally:
        _close_listener(*first[:3])
        _close_listener(*later[:3])
