# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Caller-supplied encoder peers must not fill the shared embedding cache.

The CPU consumer treats ``ec_transfer_params[mm_hash].peer_host`` as a
producer. These tests drive that path — API params, the public multimodal
hash, a real ZMQ dial, and a real NIXL READ — and require the secure outcome:
no dial to a caller-chosen address, and no ready cache entry for a later
request with the same image hash.
"""

import os
import time
from collections.abc import Callable
from typing import Any

import msgspec
import pytest
import torch
import zmq

from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq import (
    ZmqServerTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.protocol import (
    XferAck,
    XferReq,
    XferStatus,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    serialize_mem_descriptor,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

_MODEL_ID = "victim-model"
_IMAGE = b"victim-image-bytes"
_WIDTH = 32
_BLOCK = _WIDTH * torch.float16.itemsize
_FORGED = bytes([0xA5]) * _BLOCK
_ROLES = ("ec_consumer", "ec_both")


class _NoLora:
    lora_config = None


class _ChunkCfg:
    disable_chunked_mm_input = False


class _EncoderGate:
    """Attributes ``Scheduler._try_schedule_encoder_inputs`` reads."""

    def __init__(self, connector: ECCPUScheduler) -> None:
        self.is_encoder_decoder = False
        self.scheduler_config = _ChunkCfg()
        self.encoder_cache_manager = EncoderCacheManager(1024)
        self.ec_connector = connector


class _RebindingHost:
    """String form changes on every read. Hash and equality stay stable."""

    def __init__(self, values: list[str]) -> None:
        self._values = list(values)

    def __str__(self) -> str:
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return other is self


class _RebindingPort:
    """Integer value changes on every read."""

    def __init__(self, values: list[int]) -> None:
        self._values = list(values)

    def __int__(self) -> int:
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


def _image_hash() -> str:
    """Same digest ``ProcessorInputs.get_mm_hashes`` computes for raw bytes."""
    digest = MultiModalHasher.hash_kwargs(
        "blake3",
        model_id=_MODEL_ID,
        image=_IMAGE,
    )
    identifier = InputProcessor._get_mm_identifier(_NoLora(), digest, None)
    assert identifier == digest
    return digest


def _sampling_params(kind: str, params: dict[str, Any]) -> SamplingParams:
    if kind == "chat":
        request = ChatCompletionRequest(
            model=_MODEL_ID,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4,
            ec_transfer_params=params,
        )
        return request.to_sampling_params(4, {})
    if kind == "completion":
        request = CompletionRequest(
            model=_MODEL_ID,
            prompt="hi",
            max_tokens=4,
            ec_transfer_params=params,
        )
        return request.to_sampling_params(4, {})
    if kind == "responses":
        request = ResponsesRequest(
            model=_MODEL_ID,
            input="hi",
            max_output_tokens=4,
            ec_transfer_params=params,
        )
        return request.to_sampling_params(4, {})
    raise AssertionError(kind)


def _request(
    request_id: str,
    sampling_params: SamplingParams,
    mm_hash: str,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[10],
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(offset=0, length=1),
                mm_hash=mm_hash,
            )
        ],
    )


class _BoundPeer:
    def __init__(self, host: str) -> None:
        self.host = host
        self.server = ZmqServerTransport(host, 0)
        endpoint = self.server._router.getsockopt(zmq.LAST_ENDPOINT).decode()
        self.port = int(endpoint.rsplit(":", 1)[-1])

    def close(self) -> None:
        self.server.close()


def _bind(host: str, port: int = 0) -> _BoundPeer:
    if port == 0:
        return _BoundPeer(host)
    peer = _BoundPeer.__new__(_BoundPeer)
    peer.host = host
    peer.server = ZmqServerTransport(host, port)
    peer.port = port
    return peer


def _consumer(role: str) -> ECCPUScheduler:
    if NixlWrapper is None or nixl_agent_config is None:
        pytest.fail("ec_enable_nixl consumer requires the nixl package")
    cfg = create_ec_vllm_config(
        ec_role=role,
        dtype=torch.float16,
        encoder_output_widths={},
    )
    cfg.model_config.model = _MODEL_ID
    cfg.ec_transfer_config.ec_connector_extra_config = {
        "ec_enable_nixl": True,
        "ec_cpu_bytes": _BLOCK * 16,
    }
    # ec_both binds the producer side channel during setup.
    os.environ["VLLM_EC_SIDE_CHANNEL_HOST"] = "127.0.0.1"
    producer = _bind("127.0.0.1")
    os.environ["VLLM_EC_SIDE_CHANNEL_PORT"] = str(producer.port)
    producer.close()
    return ECCPUScheduler(cfg)


def _forged_source() -> tuple[Any, Any, bytes, bytes]:
    remote = torch.tensor(list(_FORGED), dtype=torch.uint8)
    agent = NixlWrapper(
        "attacker-encoder",
        nixl_agent_config(num_threads=1, capture_telemetry=False),
    )
    reg = agent.get_reg_descs([(remote.data_ptr(), remote.numel(), 0, "")], "DRAM")
    agent.register_memory(reg, backends=["UCX"])
    descriptor = serialize_mem_descriptor([(remote.data_ptr(), remote.numel(), 0)])
    # Keep ``remote`` alive for the READ; the agent holds no Python reference.
    agent._forged = remote  # type: ignore[attr-defined]
    return agent, reg, agent.get_agent_metadata(), descriptor


def _ack(req: XferReq, metadata: bytes, descriptor: bytes) -> bytes:
    return msgspec.msgpack.Encoder().encode(
        XferAck(
            mm_hash=req.mm_hash,
            status=XferStatus.OK,
            session_id=req.session_id,
            src_block_indices=[0],
            agent_metadata=metadata,
            mem_descriptor=descriptor,
        )
    )


def _drive(
    sched: ECCPUScheduler,
    request: Request,
    peers: list[_BoundPeer],
    ack_for: Callable[[XferReq], bytes],
) -> list[_BoundPeer]:
    """Run admission, answer any XferReq, and let a READ finish."""
    sched.ensure_cache_available(request, 0)
    contacted: list[_BoundPeer] = []
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not contacted:
        for peer in peers:
            incoming = peer.server.poll(timeout_ms=50)
            for identity, payload in incoming:
                req = msgspec.msgpack.decode(payload, type=XferReq)
                peer.server.send(identity, ack_for(req))
                contacted.append(peer)
    if not contacted:
        return contacted
    mm_hash = request.mm_features[0].identifier
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        sched.build_connector_meta(None)  # type: ignore[arg-type]
        sched.ensure_cache_available(request, 0)
        entry = sched._cache.get(mm_hash)
        if entry is not None and entry.ready:
            break
        time.sleep(0.02)
    return contacted


def _block_bytes(sched: ECCPUScheduler, mm_hash: str) -> bytes | None:
    entry = sched._cache.get(mm_hash)
    if entry is None or not entry.ready:
        return None
    block = sched._region.blocks[entry.block_ids[0]]
    return bytes(block.view(torch.uint8).tolist())


def _victim_schedule(
    sched: ECCPUScheduler, mm_hash: str
) -> tuple[list[int], list[int], dict[str, Any]]:
    victim = _request(
        "victim",
        SamplingParams(max_tokens=1),
        mm_hash,
    )
    gate = _EncoderGate(sched)
    encoder_inputs, _, _, external, _ = Scheduler._try_schedule_encoder_inputs(
        gate,  # type: ignore[arg-type]
        victim,
        num_computed_tokens=0,
        num_new_tokens=1,
        encoder_compute_budget=1024,
    )
    sched.update_state_after_alloc(victim, 0)
    meta = sched.build_connector_meta(None)  # type: ignore[arg-type]
    return encoder_inputs, external, meta.loads


@pytest.mark.parametrize("role", _ROLES)
@pytest.mark.parametrize("entrypoint", ("chat", "completion", "responses"))
def test_caller_peer_does_not_enter_shared_embedding_cache(
    role: str, entrypoint: str
) -> None:
    """An API caller's peer must not become the embedding for that image hash."""
    mm_hash = _image_hash()
    attacker = _bind("127.0.0.1")
    agent = None
    reg = None
    sched = None
    try:
        agent, reg, metadata, descriptor = _forged_source()
        params = {
            mm_hash: {
                "peer_host": "127.0.0.1",
                "peer_port": attacker.port,
                "size_bytes": _BLOCK,
                "shape": [1, _WIDTH],
            }
        }
        sampling = _sampling_params(entrypoint, params)
        request = _request("attacker", sampling, mm_hash)
        assert request.ec_transfer_params == params
        sched = _consumer(role)
        contacted = _drive(
            sched,
            request,
            [attacker],
            lambda req: _ack(req, metadata, descriptor),
        )
        encoder_inputs, external, loads = _victim_schedule(sched, mm_hash)
        assert _block_bytes(sched, mm_hash) != _FORGED
        assert not sched.has_cache_item(mm_hash)
        assert external == []
        assert encoder_inputs == [0]
        assert mm_hash not in loads
        assert contacted == []
    finally:
        if sched is not None:
            sched.shutdown()
        attacker.close()
        if agent is not None and reg is not None:
            agent.deregister_memory(reg)


def test_reread_peer_address_is_not_dialed() -> None:
    """A later read of peer_host or peer_port must not be the dialed address.

    ``make_zmq_path`` stringifies the host twice on one dial (IPv6 check, then
    the path). The next read names the other host, and the next port read
    names the other port. Listeners cover every pair. None of those addresses
    is an operator-configured producer, so none may be contacted and none may
    publish bytes under the image hash.
    """
    mm_hash = _image_hash()
    first_port = _bind("127.0.0.1")
    same_port_other_host = _bind("127.0.0.2", first_port.port)
    second_port_same_host = _bind("127.0.0.1")
    second_port_other_host = _bind("127.0.0.2", second_port_same_host.port)
    peers = [
        first_port,
        same_port_other_host,
        second_port_same_host,
        second_port_other_host,
    ]
    agent = None
    reg = None
    sched = None
    try:
        agent, reg, metadata, descriptor = _forged_source()
        params = {
            mm_hash: {
                "peer_host": "127.0.0.1",
                "peer_port": 1,
                "size_bytes": _BLOCK,
                "shape": [1, _WIDTH],
            }
        }
        sampling = _sampling_params("chat", params)
        assert sampling.extra_args is not None
        announced = sampling.extra_args["ec_transfer_params"][mm_hash]
        # Two stringifications are one dial. A third is a different host.
        announced["peer_host"] = _RebindingHost(["127.0.0.1", "127.0.0.1", "127.0.0.2"])
        announced["peer_port"] = _RebindingPort(
            [first_port.port, second_port_same_host.port]
        )
        request = _request("attacker", sampling, mm_hash)
        sched = _consumer("ec_consumer")
        contacted = _drive(
            sched,
            request,
            peers,
            lambda req: _ack(req, metadata, descriptor),
        )
        encoder_inputs, external, loads = _victim_schedule(sched, mm_hash)
        assert _block_bytes(sched, mm_hash) != _FORGED
        assert not sched.has_cache_item(mm_hash)
        assert external == []
        assert encoder_inputs == [0]
        assert mm_hash not in loads
        assert contacted == []
    finally:
        if sched is not None:
            sched.shutdown()
        for peer in peers:
            peer.close()
        if agent is not None and reg is not None:
            agent.deregister_memory(reg)
