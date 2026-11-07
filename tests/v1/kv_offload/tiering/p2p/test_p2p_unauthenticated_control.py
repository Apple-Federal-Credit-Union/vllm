# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unauthenticated P2P control plane must not disclose or serve KV blocks.

These tests drive the real ZMQ control socket, ``P2PSecondaryTierManager``,
and ``NixlTransport``. They fail while any inbound peer is sent the local
NIXL registration or can have cached blocks written to it. NIXL itself is
not installed here; ``_DescriptorCopyAgent`` is the subset of
``nixl_agent`` (nixl ``v1.5.1``, ``src/api/python/_api.py``) that this
transport calls. ``make_prepped_xfer`` selects descriptors by index and
``transfer`` posts that operation; a ``WRITE`` copies each local descriptor
into the paired remote descriptor.
"""

from __future__ import annotations

import ctypes
import socket
import time
from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import pytest
import zmq

import vllm.envs as envs
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    hash_block_tokens,
    init_none_hash,
)
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.p2p.data import nixl as nixl_module
from vllm.v1.kv_offload.tiering.p2p.manager import P2PSecondaryTierManager
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    ConnectAckMsg,
    ConnectMsg,
    FetchMsg,
    LookupMsg,
)

_TOKENS = (11, 22, 33, 44)
_OTHER_TOKENS = (55, 66, 77, 88)
_HASH_SEED = "p2p-auth-test-seed"
_NUM_BLOCKS = 4
_BLOCK_LEN = 64
_SENTINEL = 0x5A
_OTHER_SENTINEL = 0xA5
# Placeholder only. The control socket must ignore it.
_API_KEY_PLACEHOLDER = "placeholder-not-a-secret"


class _DescList:
    def __init__(self, descs: list[tuple[int, int, int]]) -> None:
        self.tuples = list(descs)


class _Prepared:
    def __init__(self, descs: _DescList) -> None:
        self.descs = descs


class _Xfer:
    def __init__(
        self,
        operation: str,
        local: _Prepared,
        local_idxs: list[int],
        remote: _Prepared,
        remote_idxs: list[int],
    ) -> None:
        self.operation = operation
        self.local = local
        self.local_idxs = local_idxs
        self.remote = remote
        self.remote_idxs = remote_idxs
        self.state = "PEND"


class _DescriptorCopyAgent:
    """Stand-in for ``nixl._api.nixl_agent`` at nixl ``v1.5.1``.

    ``transfer`` copies ``WRITE`` descriptors. ``postXferReq`` is what the
    installed binding calls; that native body is not in this tree, so the
    copy is the operation ``make_prepped_xfer`` indexed.
    """

    def __init__(self, agent_name: str, _cfg: object) -> None:
        self.agent_name = agent_name
        self.metadata = b"nixl-md:" + agent_name.encode()
        self.registrations: list[list[tuple]] = []
        self.remote_agents: dict[str, bytes] = {}
        self.added_remote: list[bytes] = []
        self.writes: list[tuple[list[int], list[int]]] = []

    def create_backend(self, _name: str, _params: dict) -> None:
        return None

    def register_memory(self, reg_list, mem_type: str | None = None):
        self.registrations.append(list(reg_list))
        return reg_list

    def deregister_memory(self, _reg) -> None:
        return None

    def get_agent_metadata(self) -> bytes:
        return self.metadata

    def get_xfer_descs(self, descs, mem_type: str | None = None) -> _DescList:
        return _DescList([(int(a), int(n), int(d)) for a, n, d in descs])

    def prep_xfer_dlist(
        self,
        _agent_name: str,
        xfer_list,
        mem_type: str | None = None,
    ) -> _Prepared:
        if isinstance(xfer_list, _DescList):
            return _Prepared(xfer_list)
        return _Prepared(self.get_xfer_descs(xfer_list, mem_type))

    def add_remote_agent(self, metadata: bytes) -> str:
        name = "remote-" + metadata.hex()
        self.remote_agents[name] = metadata
        self.added_remote.append(metadata)
        return name

    def remove_remote_agent(self, name: str) -> None:
        self.remote_agents.pop(name, None)

    def make_prepped_xfer(
        self,
        operation: str,
        local_side: _Prepared,
        local_indices,
        remote_side: _Prepared,
        remote_indices,
        notif_msg: bytes = b"",
    ) -> _Xfer:
        del notif_msg
        return _Xfer(
            operation,
            local_side,
            [int(i) for i in local_indices],
            remote_side,
            [int(i) for i in remote_indices],
        )

    def transfer(self, handle: _Xfer, notif_msg: bytes = b"") -> str:
        del notif_msg
        if handle.operation != "WRITE":
            handle.state = "ERR"
            return "ERR"
        for local_i, remote_i in zip(handle.local_idxs, handle.remote_idxs):
            src, src_len, _ = handle.local.descs.tuples[local_i]
            dst, dst_len, _ = handle.remote.descs.tuples[remote_i]
            ctypes.memmove(dst, src, min(src_len, dst_len))
        self.writes.append((handle.local_idxs, handle.remote_idxs))
        handle.state = "DONE"
        return "DONE"

    def check_xfer_state(self, handle: _Xfer) -> str:
        return handle.state

    def release_xfer_handle(self, _handle: _Xfer) -> None:
        return None

    def release_dlist_handle(self, _handle: _Prepared) -> None:
        return None


class _WirePeer:
    """ZMQ peer that speaks the P2P control socket without credentials."""

    def __init__(self, victim_endpoint: str) -> None:
        self.ctx = zmq.Context()
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind("tcp://127.0.0.1:0")
        endpoint = self.router.getsockopt_string(zmq.LAST_ENDPOINT)
        _scheme, _host, port_s = endpoint.split(":")
        self.port = int(port_s)
        self.identity = f"127.0.0.1:{self.port}"
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.setsockopt(zmq.IDENTITY, self.identity.encode())
        self.dealer.connect(victim_endpoint)
        self.buf = np.zeros((_NUM_BLOCKS, _BLOCK_LEN), dtype=np.uint8)
        self.base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))

    def send(self, msg: dict) -> None:
        self.dealer.send(msgspec.msgpack.encode(msg))

    def recv(self) -> list[dict]:
        found: list[dict] = []
        while True:
            try:
                frames = self.router.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return found
            if len(frames) == 2:
                found.append(msgspec.msgpack.decode(frames[1]))

    def close(self) -> None:
        self.dealer.close(linger=0)
        self.router.close(linger=0)
        self.ctx.term()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _offloading_spec(dp_index: int = 0):
    return SimpleNamespace(
        blocks_per_chunk=1,
        config=OffloadingConfig(
            groups=(),
            worker_kv_bytes_per_block=_BLOCK_LEN,
            enable_kv_cache_events=False,
            extra_config={},
            engine_id="p2p-auth-test",
            model=OffloadingModelConfig(name="test-model", dtype="uint8"),
            cache=OffloadingCacheConfig(tokens_per_hash=4, blocks_per_chunk=1),
            parallel=OffloadingParallelConfig(
                rank=0,
                world_size=1,
                tp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=1,
                data_parallel_index=dp_index,
                data_parallel_size=1,
                data_parallel_rank_local=None,
                is_parallelism_agnostic=True,
            ),
        ),
    )


def _install_agent(monkeypatch: pytest.MonkeyPatch) -> type[_DescriptorCopyAgent]:
    monkeypatch.setattr(nixl_module, "_NixlAgent", _DescriptorCopyAgent)
    monkeypatch.setattr(nixl_module, "_NixlAgentConfig", lambda **_k: object())
    return _DescriptorCopyAgent


def _arm_hash_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", _HASH_SEED)
    monkeypatch.setenv("VLLM_API_KEY", _API_KEY_PLACEHOLDER)
    init_none_hash(sha256)


def _block_key(token_ids: tuple[int, ...]) -> OffloadKey:
    return make_offload_key(bytes(hash_block_tokens(sha256, None, token_ids)), 0)


def _key_from_seed(seed: str, token_ids: tuple[int, ...]) -> OffloadKey:
    parent = sha256(seed)
    digest = sha256((parent, tuple(token_ids), None))
    return make_offload_key(digest, 0)


def _open_manager(
    monkeypatch: pytest.MonkeyPatch,
    kv: np.ndarray,
    *,
    host: str | None,
    port: int | None,
) -> P2PSecondaryTierManager:
    _install_agent(monkeypatch)
    return P2PSecondaryTierManager(
        offloading_spec=_offloading_spec(),
        primary_kv_view=memoryview(kv),
        host=host,
        port=port,
    )


def _pump(drive, peer: _WirePeer, seconds: float) -> list[dict]:
    found: list[dict] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        drive()
        found.extend(peer.recv())
        time.sleep(0.01)
    return found


def _disclosed(messages: list[dict], mgr: P2PSecondaryTierManager) -> str | None:
    seed = mgr._get_hash_seed()
    meta = mgr._data.get_agent_metadata()
    base = mgr._data.base_addr
    blocks = mgr._data.num_blocks
    block_len = mgr._data.block_len
    fingerprint = mgr._data.config_fingerprint
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get(ConnectMsg.HASH_SEED) == seed:
            return "hash_seed"
        if msg.get(ConnectMsg.AGENT_METADATA) == meta:
            return "agent_metadata"
        if msg.get(ConnectMsg.BASE_ADDR) == base:
            return "base_addr"
        if (
            msg.get(ConnectMsg.NUM_BLOCKS) == blocks
            and msg.get(ConnectMsg.BLOCK_LEN) == block_len
        ):
            return "block_geometry"
        if fingerprint and msg.get(ConnectMsg.CONFIG_FINGERPRINT) == fingerprint:
            return "config_fingerprint"
    return None


def _connect_msg(
    peer: _WirePeer,
    seed: str,
    fingerprint: str | None,
) -> dict:
    msg: dict[str, Any] = {
        TYPE_KEY: ConnectMsg.TYPE,
        ConnectMsg.PEER_ID: peer.identity,
        ConnectMsg.AGENT_METADATA: b"attacker-nixl-metadata",
        ConnectMsg.BASE_ADDR: peer.base_addr,
        ConnectMsg.NUM_BLOCKS: _NUM_BLOCKS,
        ConnectMsg.BLOCK_LEN: _BLOCK_LEN,
        ConnectMsg.HASH_SEED: seed,
    }
    if fingerprint is not None:
        msg[ConnectMsg.CONFIG_FINGERPRINT] = fingerprint
    return msg


def _fetch_msg(kv_request_id: str, keys: list[OffloadKey]) -> dict:
    return {
        TYPE_KEY: FetchMsg.TYPE,
        FetchMsg.KV_REQUEST_ID: kv_request_id,
        FetchMsg.KEYS: [bytes(key) for key in keys],
        FetchMsg.BLOCK_INDEXES: list(range(len(keys))),
        FetchMsg.ROUND_SEQ: 0,
    }


def _assert_private(
    messages: list[dict], mgr: P2PSecondaryTierManager, peer: _WirePeer
) -> None:
    leaked = _disclosed(messages, mgr)
    agent: _DescriptorCopyAgent = mgr._data._agent
    assert leaked is None, f"unauthenticated peer received {leaked}"
    assert agent.added_remote == [], (
        "peer memory was registered before the control channel was refused"
    )
    assert agent.writes == [], f"unauthenticated peer was served blocks {agent.writes}"
    assert not np.any(peer.buf), "cached KV bytes were written to the peer"


def _store_pd(
    mgr: P2PSecondaryTierManager, kv_request_id: str, key: OffloadKey, block_id: int
) -> None:
    ctx = ReqContext(
        req_id=kv_request_id,
        kv_transfer_params={"remote_decoder": {"kv_request_id": kv_request_id}},
    )
    mgr.on_new_request(ctx)
    mgr.submit_store(
        TransferJob(
            job_id=1,
            keys=[key],
            chunk_ids=np.array([block_id]),
            is_promotion=False,
            req_context=ctx,
        )
    )


@pytest.fixture
def kv_buffer() -> np.ndarray:
    return np.zeros((_NUM_BLOCKS, _BLOCK_LEN), dtype=np.uint8)


def test_defaults_are_local_and_api_key_is_not_required(monkeypatch):
    """The control socket follows the documented local default and ignores
    ``VLLM_API_KEY``."""
    monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_HOST", raising=False)
    monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_PORT", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", _API_KEY_PLACEHOLDER)
    assert envs.environment_variables["VLLM_P2P_SIDE_CHANNEL_HOST"]() == "localhost"
    assert envs.environment_variables["VLLM_P2P_SIDE_CHANNEL_PORT"]() == 5710

    port = _free_port()
    monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_HOST", "127.0.0.1")
    monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_PORT", str(port))
    _arm_hash_seed(monkeypatch)
    kv = np.zeros((_NUM_BLOCKS, _BLOCK_LEN), dtype=np.uint8)
    mgr = _open_manager(monkeypatch, kv, host=None, port=None)
    try:
        assert mgr._local_id == f"127.0.0.1:{port}"
        endpoint = mgr._control._router.getsockopt_string(zmq.LAST_ENDPOINT)
        peer = _WirePeer(endpoint)
        try:
            peer.send({TYPE_KEY: "probe"})
            messages = _pump(mgr.get_finished_jobs, peer, 1.0)
            _assert_private(messages, mgr, peer)
        finally:
            peer.close()
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("fingerprint", [None, ""])
def test_omitted_fingerprint_does_not_register_or_serve(
    monkeypatch, kv_buffer, fingerprint
):
    """A connect that echoes the hash seed and omits the fingerprint must
    not register the peer or copy a stored block."""
    _arm_hash_seed(monkeypatch)
    key = _block_key(_TOKENS)
    assert key == _key_from_seed(_HASH_SEED, _TOKENS)
    block_id = 1
    kv_buffer[block_id] = _SENTINEL
    port = _free_port()
    mgr = _open_manager(monkeypatch, kv_buffer, host="127.0.0.1", port=port)
    try:
        assert mgr._data.config_fingerprint
        _store_pd(mgr, "req-victim", key, block_id)
        peer = _WirePeer(f"tcp://127.0.0.1:{port}")
        try:
            peer.send({TYPE_KEY: "probe"})
            seen = _pump(mgr.get_finished_jobs, peer, 1.0)
            seed = mgr._get_hash_seed()
            peer.send(_connect_msg(peer, seed, fingerprint))
            seen.extend(_pump(mgr.get_finished_jobs, peer, 1.0))
            peer.send(_fetch_msg("req-victim", [key]))
            seen.extend(_pump(mgr.get_finished_jobs, peer, 1.0))
            assert not any(
                isinstance(msg, dict) and msg.get(TYPE_KEY) == ConnectAckMsg.TYPE
                for msg in seen
            )
            _assert_private(seen, mgr, peer)
        finally:
            peer.close()
    finally:
        mgr.shutdown()


def test_matching_fingerprint_without_credentials_does_not_serve(
    monkeypatch, kv_buffer
):
    """Knowing the model fingerprint and hash seed is not authorization to
    read a block stored for another request."""
    _arm_hash_seed(monkeypatch)
    key = _block_key(_TOKENS)
    block_id = 2
    kv_buffer[block_id] = _SENTINEL
    port = _free_port()
    mgr = _open_manager(monkeypatch, kv_buffer, host="127.0.0.1", port=port)
    try:
        _store_pd(mgr, "req-victim", key, block_id)
        peer = _WirePeer(f"tcp://127.0.0.1:{port}")
        try:
            peer.send(
                _connect_msg(peer, mgr._get_hash_seed(), mgr._data.config_fingerprint)
            )
            seen = _pump(mgr.get_finished_jobs, peer, 1.0)
            peer.send(_fetch_msg("req-victim", [key]))
            seen.extend(_pump(mgr.get_finished_jobs, peer, 1.0))
            _assert_private(seen, mgr, peer)
        finally:
            peer.close()
    finally:
        mgr.shutdown()


def test_lookup_does_not_serve_another_requests_blocks(monkeypatch):
    """An inbound lookup for a primary-cache hash must not copy that block
    to a peer that does not own the request, including a second request's
    hash asked for in the same fetch."""
    _arm_hash_seed(monkeypatch)
    _install_agent(monkeypatch)
    kv = np.zeros((_NUM_BLOCKS, _BLOCK_LEN), dtype=np.uint8)
    region = type("Region", (), {})()
    region.create_kv_memoryview = lambda: memoryview(kv)
    region.cleanup = lambda: None
    primary = CPUPrimaryTierOffloadingManager(
        num_chunks=_NUM_BLOCKS, mmap_region=region
    )
    port = _free_port()
    p2p = P2PSecondaryTierManager(
        offloading_spec=_offloading_spec(),
        primary_kv_view=primary.get_kv_memoryview(),
        host="127.0.0.1",
        port=port,
    )
    tiering = TieringOffloadingManager(
        primary_tier=primary,
        secondary_tiers=[p2p],
        tier_poll_interval_s=-1,
    )
    own = _block_key(_TOKENS)
    other = _block_key(_OTHER_TOKENS)
    try:
        for key, sentinel in ((own, _SENTINEL), (other, _OTHER_SENTINEL)):
            ctx = ReqContext(req_id="store-" + key.hex()[:8])
            primary.on_new_request(ctx)
            prepared = primary.prepare_store([key], ctx)
            assert prepared is not None
            chunk_id = int(prepared.store_spec.chunk_ids[0])
            kv[chunk_id] = sentinel
            primary.complete_store([key], ctx)

        def drive() -> None:
            with tiering.lock:
                tiering.poll_tiers()

        peer = _WirePeer(f"tcp://127.0.0.1:{port}")
        try:
            peer.send({TYPE_KEY: "probe"})
            seen = _pump(drive, peer, 1.0)
            peer.send(_connect_msg(peer, p2p._get_hash_seed(), None))
            seen.extend(_pump(drive, peer, 1.0))
            peer.send(
                {
                    TYPE_KEY: LookupMsg.TYPE,
                    LookupMsg.KV_REQUEST_ID: "attacker-req",
                    LookupMsg.KEYS: [bytes(own), bytes(other)],
                    LookupMsg.ROUND_SEQ: 0,
                }
            )
            seen.extend(_pump(drive, peer, 1.0))
            peer.send(_fetch_msg("attacker-req", [own, other]))
            seen.extend(_pump(drive, peer, 1.0))
            _assert_private(seen, p2p, peer)
            assert not np.any(peer.buf == _SENTINEL)
            assert not np.any(peer.buf == _OTHER_SENTINEL)
        finally:
            peer.close()
    finally:
        tiering.shutdown()
