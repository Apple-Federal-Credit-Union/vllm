# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ungranted NIXL remote block ids must not be read.

A disaggregated prefill response returns the side-channel address and the
physical blocks that were pinned for that request. A later completion,
chat, or responses request can repeat that address with ``do_remote_prefill``
and substitute other in-range block ids. The pull connector copies those
ids onto the engine request and posts a NIXL READ of them.

These tests fail while that READ is posted. They pass once a READ is
limited to the blocks the producer pinned for the named remote request
(or is not posted at all).

Not exercised here, and still open if a fix only covers the pull scheduler's
``do_remote_prefill`` branch or only one HTTP schema:

* Rust gRPC (``rust/src/server/src/grpc/convert.rs``) forwards the same
  ``kv_transfer_params`` object into sampling params.
* ``/inference/v1/generate`` copies the field in
  ``vllm/entrypoints/scale_out/token_in_token_out/serving.py``.
* Render hoists the field in
  ``vllm/entrypoints/scale_out/render/serving.py``.
* Push mode overwrites ``remote_block_ids`` before its WRITE and is not
  this READ.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlConnector,
    NixlConnectorMetadata,
    NixlConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    compute_nixl_compatibility_hash,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_cache_interface import compute_layer_kv_cache_shape_bytes
from vllm.v1.request import Request, RequestStatus

from .utils import (
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
    maybe_update_block_size,
)

# Producer engine id the decode worker's handshake accepts.
_REMOTE_ENGINE_ID = "remote_engine"
# Matches make_kv_cache_config's pool and the NIXL descriptor count.
_NUM_BLOCKS = 128
# Long enough that the transferred tail still contains the victim blocks
# after the worker trims the remote list to the local block count.
_REMOTE_LIST_LEN = 32

SamplingBuilder = Callable[[dict[str, Any]], Any]


def _chat_sampling(params: dict[str, Any]):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "facebook/opt-125m",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 8,
            "kv_transfer_params": params,
        }
    )
    return request.to_sampling_params(8, {})


def _completion_sampling(params: dict[str, Any]):
    request = CompletionRequest.model_validate(
        {
            "model": "facebook/opt-125m",
            "prompt": "hello",
            "max_tokens": 8,
            "kv_transfer_params": params,
        }
    )
    return request.to_sampling_params(8, {})


def _responses_sampling(params: dict[str, Any]):
    request = ResponsesRequest.model_validate(
        {
            "model": "facebook/opt-125m",
            "input": "hello",
            "max_output_tokens": 8,
            "kv_transfer_params": params,
        }
    )
    return request.to_sampling_params(8, {})


def _flat_desc_count(descs) -> int:
    """Descriptor count of a NIXL prep list.

    An Nx5 strided array contributes its ``count`` column. A plain list
    contributes one descriptor per element. This is the flat size
    ``nixlAgent::makeXferReq`` checks indices against (nixl 1.5.1).
    """
    arr = np.asarray(descs)
    if arr.ndim == 2 and arr.shape[1] == 5:
        return int(arr[:, 4].sum())
    return int(arr.shape[0]) if arr.ndim else len(descs)


class RecordingNixlWrapper:
    """Stand-in for ``nixl_cu12._api.nixl_agent`` 1.5.1.

    ``make_prepped_xfer`` forwards the caller's indices to ``makeXferReq``,
    which rejects a count mismatch or an index past the prepared descriptor
    list and otherwise selects those descriptors. ``transfer`` posts that
    request. There is no per-request block grant. This wrapper records the
    indices and applies those two checks.
    """

    AGENT_METADATA = b"fake_agent_metadata"
    REMOTE_AGENT_NAME = "remote_agent"

    def __init__(self, agent_name: str, *args, **kwargs):
        del agent_name, args, kwargs
        self._cycles: defaultdict[int, int] = defaultdict(int)
        self.invalid_remote_agents: set[str] = set()
        self.dlist_size: dict[int, int] = {}
        self._next_handle = 1
        self.prepared: list[dict[str, Any]] = []
        self.posted: list[int] = []

    def _handle(self) -> int:
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def get_reg_descs(self, caches_data, memory_type: str) -> list:
        del memory_type
        return [str(uuid.uuid4()) for _ in caches_data]

    def register_memory(self, descs, backends) -> None:
        del descs, backends

    def deregister_memory(self, descs) -> None:
        del descs

    def get_xfer_descs(self, blocks_data, memory_type: str) -> list:
        del memory_type
        return [str(uuid.uuid4()) for _ in blocks_data]

    def prep_xfer_dlist(
        self, agent_name: str, descs, mem_type: str | None = None
    ) -> int:
        del agent_name, mem_type
        handle = self._handle()
        self.dlist_size[handle] = _flat_desc_count(descs)
        return handle

    def get_agent_metadata(self) -> bytes:
        return self.AGENT_METADATA

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        del agent_metadata
        self.invalid_remote_agents.discard(self.REMOTE_AGENT_NAME)
        return self.REMOTE_AGENT_NAME

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return {}

    def check_xfer_state(self, handle: int) -> str:
        return "DONE" if self._cycles[handle] else "PROC"

    def release_xfer_handle(self, handle: int) -> None:
        del handle

    def release_dlist_handle(self, handle: int) -> None:
        del handle

    def remove_remote_agent(self, agent: str) -> None:
        del agent

    def check_remote_metadata(self, agent: str) -> bool:
        return agent not in self.invalid_remote_agents

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        del agent_name, notif_msg

    def make_prepped_xfer(
        self,
        operation: str,
        local_xfer_side: int,
        local_indices,
        remote_xfer_side: int,
        remote_indices,
        notif_msg: bytes | None = None,
        **kwargs,
    ) -> int:
        del kwargs
        local = [int(i) for i in np.asarray(local_indices).ravel()]
        remote = [int(i) for i in np.asarray(remote_indices).ravel()]
        # nixl 1.5.1 nixlAgent::makeXferReq: equal non-empty index lists,
        # each index inside the prepared descriptor list.
        if not local or len(local) != len(remote):
            raise RuntimeError(
                "NIXL_ERR_INVALID_PARAM: local/remote index counts "
                f"{len(local)} and {len(remote)}"
            )
        remote_size = self.dlist_size.get(remote_xfer_side)
        if remote_size is None:
            raise RuntimeError(
                "NIXL_ERR_INVALID_PARAM: remote descriptor list was not prepared"
            )
        handle = self._handle()
        self.prepared.append(
            {
                "operation": operation,
                "handle": handle,
                "local_indices": local,
                "remote_indices": remote,
                "remote_dlist_size": remote_size,
                "notif_msg": notif_msg,
                "posted": False,
            }
        )
        if any(idx < 0 or idx >= remote_size for idx in remote):
            raise RuntimeError(
                "NIXL_ERR_INVALID_PARAM: remote index out of range "
                f"size={remote_size} indices={remote}"
            )
        local_size = self.dlist_size.get(local_xfer_side)
        if local_size is not None and any(
            idx < 0 or idx >= local_size for idx in local
        ):
            raise RuntimeError(
                f"NIXL_ERR_INVALID_PARAM: local index out of range size={local_size}"
            )
        return handle

    def transfer(self, handle: int, notif_msg: bytes = b"") -> str:
        del notif_msg
        for rec in self.prepared:
            if rec["handle"] == handle:
                rec["posted"] = True
                self.posted.append(handle)
                return "PROC"
        raise RuntimeError("NIXL_ERR_INVALID_PARAM: unknown transfer handle")

    def get_xfer_telemetry(self, handle: int) -> dict:
        del handle
        return {
            "xferDuration": 0,
            "postDuration": 0,
            "totalBytes": 1,
            "descCount": 1,
        }


def _grouped(block_ids) -> list[list[int]]:
    if not block_ids:
        return []
    first = block_ids[0]
    if isinstance(first, int):
        return [[int(b) for b in block_ids]]
    return [[int(b) for b in group] for group in block_ids]


def _pin_prefill(scheduler, request_id: int, num_tokens: int, block_size: int):
    """Run a prefill with ``do_remote_decode`` and return its transfer params.

    ``Scheduler._connector_finished`` is the hook that puts the connector's
    ``request_finished`` result on the engine output.
    """
    request = create_request(
        request_id=request_id,
        num_tokens=num_tokens,
        do_remote_decode=True,
        block_size=block_size,
    )
    scheduler.add_request(request)
    scheduler.schedule()
    request.num_computed_tokens = request.num_prompt_tokens
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    _delay, params = scheduler._connector_finished(request)
    assert params is not None
    assert params["remote_block_ids"]
    assert params["remote_engine_id"] == _REMOTE_ENGINE_ID
    assert params["remote_host"]
    assert params["remote_port"]
    return params


def _producer_params(block_size: int):
    config = create_vllm_config(
        kv_role="kv_producer",
        block_size=block_size,
    )
    config.kv_transfer_config.engine_id = _REMOTE_ENGINE_ID
    scheduler = create_scheduler(config, num_blocks=_NUM_BLOCKS)
    # Two live prefills. The second request's blocks are the attacker's own
    # grant; the first request's blocks are the ones they will name instead.
    victim = _pin_prefill(scheduler, 1, block_size * 2, block_size)
    attacker = _pin_prefill(scheduler, 2, block_size * 2, block_size)
    victim_blocks = _grouped(victim["remote_block_ids"])
    attacker_blocks = _grouped(attacker["remote_block_ids"])
    assert victim_blocks != attacker_blocks
    return attacker, victim_blocks, attacker_blocks


def _client_params(attacker: dict[str, Any], remote_block_ids) -> dict[str, Any]:
    """JSON round-trip, then replace the block ids. Same shape the HTTP API uses."""
    wire = json.loads(json.dumps(attacker))
    wire["do_remote_prefill"] = True
    wire["do_remote_decode"] = False
    wire["remote_block_ids"] = remote_block_ids
    return wire


def _unauthorized_groups(
    grant: list[list[int]], victim: list[list[int]]
) -> list[list[int]]:
    granted = {block for group in grant for block in group}
    stolen = [block for group in victim for block in group]
    stolen_set = set(stolen)
    padding = [
        block
        for block in range(1, _NUM_BLOCKS)
        if block not in granted and block not in stolen_set
    ]
    assert len(padding) >= _REMOTE_LIST_LEN
    # Victim blocks sit at the tail, which is what the worker keeps when it
    # trims the remote list down to the locally allocated count.
    body = padding[: _REMOTE_LIST_LEN - len(stolen)] + stolen
    return [body]


def _engine_request(sampling, prompt_len: int, block_size: int, request_id: str):
    init_none_hash(sha256)
    engine_request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        mm_features=None,
        sampling_params=sampling,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return Request.from_engine_core_request(
        engine_request, get_request_block_hasher(block_size, sha256)
    )


class _HandshakeWorker(NixlConnectorWorker):
    """Worker whose handshake skips the ZMQ side channel.

    The block-id path under test runs after the agent is registered.
    Descriptor preparation still goes through ``prep_xfer_dlist``.
    """

    def __init__(self, *args, kv_cache_config=None, **kwargs):
        if kv_cache_config is None:
            kv_cache_config = make_kv_cache_config(
                block_size=args[0].cache_config.block_size,
                num_blocks=_NUM_BLOCKS,
            )
        super().__init__(*args, kv_cache_config=kv_cache_config, **kwargs)
        self.src_xfer_handles_by_block_size = {self.block_size: 1}
        self.src_blocks_data = np.empty((0, 5), dtype=np.uint64)
        spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        test_shape = compute_layer_kv_cache_shape_bytes(spec, 1)
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            is_mamba=False,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            tensor_shape=test_shape,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name
        )
        self.tracked_handshakes: list = []

    def _ensure_handshake(self, *args, **kwargs):
        fut = super()._ensure_handshake(*args, **kwargs)
        if fut is not None:
            self.tracked_handshakes.append(fut)
        return fut

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dcp_size: int = 1,
        remote_pp_size: int = 1,
        notif_agents_only: bool = False,
    ):
        del host, port, remote_pp_size, notif_agents_only
        assert expected_engine_id == _REMOTE_ENGINE_ID
        slot = 4096
        self.slot_size_per_layer = [slot]
        self.block_len_per_layer = [slot * self.block_size]
        self.num_blocks = self.kv_cache_config.num_blocks
        self.num_regions = 1
        self.block_stride_per_layer = list(self.block_len_per_layer)
        self.region_num_blocks = [self.num_blocks]
        self.region_group_ids = [0]
        self.region_mem_types = [self.nixl_memory_type]
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.dst_region_num_blocks[self.engine_id] = self.region_num_blocks
        self.dst_region_group_ids[self.engine_id] = self.region_group_ids
        self.dst_region_mem_types[self.engine_id] = self.region_mem_types
        remote_block_lens = list(self.block_len_per_layer)
        agent_name = self.add_remote_agent(
            NixlAgentMetadata(
                engine_id=_REMOTE_ENGINE_ID,
                agent_metadata=RecordingNixlWrapper.AGENT_METADATA,
                kv_caches_base_addr=[0],
                device_id=0,
                num_blocks=self.num_blocks,
                block_lens=remote_block_lens,
                block_strides=remote_block_lens,
                kv_cache_layout=self.kv_cache_layout,
                block_size=self.block_size,
                ssm_sizes=(0, 0),
                attn_backend_name=self.backend_name,
                physical_blocks_per_logical_kv_block=1,
                region_num_blocks=self.region_num_blocks,
                region_group_ids=self.region_group_ids,
                region_mem_types=self.region_mem_types,
            ),
            remote_tp_rank=0,
            remote_tp_size=remote_tp_size,
        )
        del agent_name, remote_dcp_size
        return {(0, 0): RecordingNixlWrapper.REMOTE_AGENT_NAME}, 0.0


def _drive_read(connector: NixlConnector, metadata) -> RecordingNixlWrapper:
    worker = connector.connector_worker
    assert isinstance(worker.nixl_wrapper, RecordingNixlWrapper)
    ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(ctx)
    for fut in worker.tracked_handshakes:
        fut.result(timeout=5)
    # The handshake callback queues the request; the next load posts the READ.
    connector.bind_connector_metadata(NixlConnectorMetadata())
    connector.start_load_kv(ctx)
    return worker.nixl_wrapper


def _assert_reads_within_grant(wrapper: RecordingNixlWrapper, grant: set[int]):
    """Posted READ indices must be the named request's pinned blocks.

    Refusing the transfer (no ``make_prepped_xfer``) satisfies this. Posting
    a READ and then ignoring it does not: the prepare call is the selection.
    """
    for rec in wrapper.prepared:
        assert rec["operation"] == "READ"
        got = set(rec["remote_indices"])
        outside = got - grant
        assert not outside, (
            "NIXL READ descriptor indices "
            f"{sorted(outside)} are not blocks pinned for the named "
            f"remote request (grant {sorted(grant)}). Posted {sorted(got)} "
            f"from a prepared list of {rec['remote_dlist_size']} blocks."
        )


def _schedule_decode(sampling, block_size: int, *, extra_config=None):
    config = create_vllm_config(
        block_size=block_size,
        kv_connector_extra_config=extra_config,
    )
    scheduler = create_scheduler(config, num_blocks=_NUM_BLOCKS)
    request = _engine_request(
        sampling,
        prompt_len=block_size * 4,
        block_size=block_size,
        request_id="decode-attacker",
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    metadata = output.kv_connector_metadata
    assert isinstance(metadata, NixlConnectorMetadata)
    return config, request, metadata


def _assert_scheduled_read(config, request, metadata, monkeypatch, grant: set[int]):
    """Drive the worker when a receive was scheduled.

    Omitting the request from ``reqs_to_recv`` is a refusal and passes.
    A receive that still names the caller's list is checked after that
    list is whatever the scheduler stored.
    """
    if request.request_id not in metadata.reqs_to_recv:
        return
    connector = _consumer_connector(config, monkeypatch)
    try:
        wrapper = _drive_read(connector, metadata)
    finally:
        connector.shutdown()
    _assert_reads_within_grant(wrapper, grant)


def _consumer_connector(config, monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
        RecordingNixlWrapper,
    )
    kv_cache_config = make_kv_cache_config(
        block_size=config.cache_config.block_size,
        num_blocks=_NUM_BLOCKS,
    )
    connector = NixlConnector(config, KVConnectorRole.WORKER, kv_cache_config)
    connector.connector_worker = _HandshakeWorker(
        config,
        connector.engine_id,
        kv_cache_config=kv_cache_config,
    )
    return connector


def _response_leaks_coordinates(params: dict[str, Any]):
    """The prefill HTTP body returns the side channel and the pinned blocks."""
    response = ChatCompletionResponse(
        model="facebook/opt-125m",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="ok"),
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        kv_transfer_params=params,
    )
    body = json.loads(response.model_dump_json())
    leaked = body["kv_transfer_params"]
    for key in (
        "remote_host",
        "remote_port",
        "remote_engine_id",
        "remote_request_id",
        "remote_block_ids",
    ):
        assert leaked[key], key


@pytest.fixture(autouse=True)
def _patch_nixl(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
        RecordingNixlWrapper,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "build_sampling",
    [_chat_sampling, _completion_sampling, _responses_sampling],
    ids=["chat", "completion", "responses"],
)
def test_client_remote_block_ids_are_not_read(
    build_sampling: SamplingBuilder,
    default_vllm_config,
    dist_init,
    monkeypatch,
):
    """A request that names another prefill's blocks must not READ them.

    The remote request id is the caller's own prefill. The block ids are
    the other prefill's live blocks, padded so they survive tail trimming.
    """
    del default_vllm_config, dist_init
    block_size = maybe_update_block_size(16)
    attacker, victim_blocks, grant_blocks = _producer_params(block_size)
    _response_leaks_coordinates(attacker)
    remote_block_ids = _unauthorized_groups(grant_blocks, victim_blocks)
    sampling = build_sampling(_client_params(attacker, remote_block_ids))
    config, request, metadata = _schedule_decode(sampling, block_size)
    grant = {block for group in grant_blocks for block in group}
    _assert_scheduled_read(config, request, metadata, monkeypatch, grant)


@pytest.mark.cpu_test
def test_block_ids_swapped_after_schedule_are_not_read(
    default_vllm_config, dist_init, monkeypatch
):
    """Swap the block-id list after the scheduler has copied it.

    ``add_new_req_to_recv`` stores the caller's list object. A check that
    accepts the list and a later read of that same object must not observe
    a different set of blocks.
    """
    del default_vllm_config, dist_init
    block_size = maybe_update_block_size(16)
    attacker, victim_blocks, grant_blocks = _producer_params(block_size)
    first = _unauthorized_groups(grant_blocks, victim_blocks)
    sampling = _chat_sampling(_client_params(attacker, first))
    config, request, metadata = _schedule_decode(sampling, block_size)
    if request.request_id not in metadata.reqs_to_recv:
        return
    stored = metadata.reqs_to_recv[request.request_id].remote
    swapped = [block + 1 if block + 1 < _NUM_BLOCKS else 1 for block in first[0]]
    grant = {block for group in grant_blocks for block in group}
    swapped = [block if block not in grant else 1 for block in swapped]
    assert swapped != first[0]
    # Mutate the caller's list and, when the scheduler kept that same object,
    # the metadata. A snapshot taken earlier is left as the scheduler stored it.
    client_blocks = request.kv_transfer_params["remote_block_ids"]
    client_blocks[0][:] = swapped
    if (
        stored is not None
        and stored.block_ids is client_blocks
        and isinstance(stored.block_ids[0], list)
    ):
        stored.block_ids[0][:] = swapped
    _assert_scheduled_read(config, request, metadata, monkeypatch, grant)


@pytest.mark.cpu_test
def test_bidirectional_decode_blocks_are_not_read(
    default_vllm_config, dist_init, monkeypatch
):
    """Bidirectional pull uses client remote blocks on ``do_remote_decode``."""
    del default_vllm_config, dist_init
    block_size = maybe_update_block_size(16)
    attacker, victim_blocks, grant_blocks = _producer_params(block_size)
    remote_block_ids = _unauthorized_groups(grant_blocks, victim_blocks)
    params = _client_params(attacker, remote_block_ids)
    # Prefill instance pulling blocks that the decode node still holds.
    params["do_remote_prefill"] = False
    params["do_remote_decode"] = True
    params["remote_num_tokens"] = block_size * 4
    sampling = _chat_sampling(params)
    config, request, metadata = _schedule_decode(
        sampling,
        block_size,
        extra_config={
            "bidirectional_kv_xfer": True,
            "kv_recompute_threshold": 0,
        },
    )
    grant = {block for group in grant_blocks for block in group}
    _assert_scheduled_read(config, request, metadata, monkeypatch, grant)


@pytest.mark.cpu_test
def test_named_request_grant_is_not_another_requests_blocks(
    default_vllm_config, dist_init, monkeypatch
):
    """Naming the victim request id still only authorizes that request's blocks.

    Supplying the victim's request id with a different in-range layout must
    not READ the substituted blocks.
    """
    del default_vllm_config, dist_init
    block_size = maybe_update_block_size(16)
    attacker, victim_blocks, grant_blocks = _producer_params(block_size)
    # Build the producer victim params by pinning order: request id is in
    # the attacker dict's remote_request_id for the attacker's own prefill.
    # Re-pin is unnecessary: victim blocks are the grant iff the client names
    # the victim request. The attacker params carry the attacker's request id;
    # point remote_request_id at the victim by rewriting it to the id the
    # producer used for request 1.
    params = _client_params(attacker, _unauthorized_groups(victim_blocks, grant_blocks))
    # create_request uses id "id-{n}". Victim was request_id=1.
    params["remote_request_id"] = "id-1"
    sampling = _chat_sampling(params)
    config, request, metadata = _schedule_decode(sampling, block_size)
    grant = {block for group in victim_blocks for block in group}
    _assert_scheduled_read(config, request, metadata, monkeypatch, grant)
