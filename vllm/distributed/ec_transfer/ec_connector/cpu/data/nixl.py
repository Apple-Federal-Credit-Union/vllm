# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL-backed data-plane transport for ECCPUConnector."""

from itertools import groupby
from typing import Any

from vllm.distributed.ec_transfer.ec_connector.cpu.data.base import (
    DataTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    build_block_descs,
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.logger import init_logger

logger = init_logger(__name__)
_NIXL_DRAM = "DRAM"
_NIXL_BACKENDS = ["UCX"]

# (first block index, block count) of a contiguous run of blocks.
_BlockRun = tuple[int, int]


def _contiguous_runs(block_indices: list[int]) -> list[_BlockRun]:
    runs: list[_BlockRun] = []
    ordered = enumerate(sorted(set(block_indices)))
    for _, group in groupby(ordered, lambda pair: pair[1] - pair[0]):
        blocks = [block for _, block in group]
        runs.append((blocks[0], len(blocks)))
    return runs


class NixlDataTransport(DataTransport):
    """Adapts NixlWrapper to the DataTransport interface.

    Dlist handles (prepped xfer descriptor lists for remote peers) are stored
    internally keyed by agent_name. Callers never touch raw NIXL handles —
    post_read() resolves the handle from agent_name internally.

    Peers never receive this agent's full metadata. grant_blocks() registers
    the granted blocks as their own NIXL regions and hands out partial
    metadata naming only those regions, so a peer's NIXL agent refuses to
    address any other block of the region.

    NixlWrapper is not thread-safe, and neither is this class. Every call site
    runs on the scheduler thread, which serializes the transport by
    construction; transfers are async inside NIXL and are drained from that
    same thread via get_new_notifs() / check_xfer_state().
    """

    def __init__(
        self,
        agent_name: str,
        base_ptr: int,
        num_blocks: int,
        block_size_bytes: int,
        total_size_bytes: int,
        num_threads: int = 1,
    ) -> None:
        if NixlWrapper is None or nixl_agent_config is None:
            raise RuntimeError(
                "ECCPUConnector requires NIXL; "
                "install the `nixl` package or set a different ec_connector."
            )
        # agent_name → remote dlist handle (from add_remote_peer, used by post_read)
        self._peer_handles: dict[str, Any] = {}
        # None until the local region is registered, so deregister() is safe on
        # a transport whose registration failed part-way.
        self._reg_descs: Any = None
        self._base_ptr = base_ptr
        self._num_blocks = num_blocks
        self._block_size_bytes = block_size_bytes
        # Granted run → (outstanding grants, its registration descs).
        self._granted_runs: dict[_BlockRun, tuple[int, Any]] = {}

        self._nixl = NixlWrapper(
            agent_name,
            nixl_agent_config(num_threads=num_threads, capture_telemetry=True),
        )
        self._register_local_memory(
            base_ptr, num_blocks, block_size_bytes, total_size_bytes
        )

    def _register_local_memory(
        self,
        base_ptr: int,
        num_blocks: int,
        block_size_bytes: int,
        total_size_bytes: int,
    ) -> None:
        """Register the local region with NIXL and prep its transfer dlist.

        Unwinds the registration if a later step fails: __init__ propagates the
        error, so the caller never receives an object it could deregister with.
        """
        block_descs = build_block_descs(base_ptr, num_blocks, block_size_bytes)
        self._reg_descs = self._nixl.get_reg_descs(
            [(base_ptr, total_size_bytes, 0, "")], _NIXL_DRAM
        )
        self._nixl.register_memory(self._reg_descs, backends=_NIXL_BACKENDS)
        try:
            xfer_descs = self._nixl.get_xfer_descs(block_descs, _NIXL_DRAM)
            self._local_xfer_handle = self._nixl.prep_xfer_dlist(
                "NIXL_INIT_AGENT", xfer_descs
            )
        except Exception:
            self.deregister()
            raise

    def _run_reg_descs(self, runs: list[_BlockRun]) -> Any:
        return self._nixl.get_reg_descs(
            [
                (
                    self._base_ptr + first * self._block_size_bytes,
                    count * self._block_size_bytes,
                    0,
                    "",
                )
                for first, count in runs
            ],
            _NIXL_DRAM,
        )

    def grant_blocks(self, block_indices: list[int]) -> tuple[bytes, bytes]:
        if not block_indices or not all(
            0 <= i < self._num_blocks for i in block_indices
        ):
            raise ValueError(f"EC: cannot grant blocks {block_indices}")
        runs = _contiguous_runs(block_indices)
        retained: list[_BlockRun] = []
        try:
            for run in runs:
                self._retain_run(run)
                retained.append(run)
            metadata = self._nixl.get_partial_agent_metadata(
                self._run_reg_descs(runs),
                inc_conn_info=True,
                backends=_NIXL_BACKENDS,
            )
        except Exception:
            for run in retained:
                self._release_run(run)
            raise
        mem_descriptor = serialize_mem_descriptor(
            [
                (self._base_ptr + i * self._block_size_bytes, self._block_size_bytes, 0)
                for i in block_indices
            ]
        )
        return metadata, mem_descriptor

    def revoke_blocks(self, block_indices: list[int]) -> None:
        for run in _contiguous_runs(block_indices):
            self._release_run(run)

    def _retain_run(self, run: _BlockRun) -> None:
        held = self._granted_runs.get(run)
        if held is not None:
            self._granted_runs[run] = (held[0] + 1, held[1])
            return
        reg_descs = self._run_reg_descs([run])
        self._nixl.register_memory(reg_descs, backends=_NIXL_BACKENDS)
        self._granted_runs[run] = (1, reg_descs)

    def _release_run(self, run: _BlockRun) -> None:
        held = self._granted_runs.get(run)
        if held is None:
            return
        grants, reg_descs = held
        if grants > 1:
            self._granted_runs[run] = (grants - 1, reg_descs)
            return
        del self._granted_runs[run]
        try:
            self._nixl.deregister_memory(reg_descs, backends=_NIXL_BACKENDS)
        except Exception:
            logger.warning("EC: deregister of granted blocks failed", exc_info=True)

    def add_remote_peer(self, metadata: bytes, mem_descriptor: bytes) -> str:
        """Load a peer's grant, prep its dlist, store the handle internally.

        NIXL merges metadata for a peer it already knows, so the peer stays
        registered across grants; only the dlist is replaced. A transfer built
        from the replaced dlist keeps its own copy of the descriptors.

        Returns agent_name for use in post_read() and remove_remote_peer().
        """
        agent_name = self._nixl.add_remote_agent(metadata)
        remote_blocks = deserialize_mem_descriptor(mem_descriptor)
        remote_xfer_descs = self._nixl.get_xfer_descs(remote_blocks, _NIXL_DRAM)
        handle = self._nixl.prep_xfer_dlist(agent_name, remote_xfer_descs)
        self._release_dlist(self._peer_handles.get(agent_name))
        self._peer_handles[agent_name] = handle
        return agent_name

    def _release_dlist(self, handle: Any) -> None:
        if handle is None:
            return
        try:
            self._nixl.release_dlist_handle(handle)
        except Exception:
            logger.warning("EC: release_dlist_handle failed", exc_info=True)

    def remove_remote_peer(self, agent_name: str) -> None:
        self._release_dlist(self._peer_handles.pop(agent_name, None))
        try:
            self._nixl.remove_remote_agent(agent_name)
        except Exception:
            logger.warning(
                "EC: remove_remote_peer failed for %s", agent_name, exc_info=True
            )

    def post_read(
        self,
        local_indices: list[int],
        agent_name: str,
        remote_indices: list[int],
        notif_msg: bytes,
    ) -> Any:
        if len(local_indices) != len(remote_indices):
            raise ValueError(
                f"EC: local/remote block count mismatch "
                f"({len(local_indices)} vs {len(remote_indices)})"
            )
        remote_handle = self._peer_handles[agent_name]
        handle = self._nixl.make_prepped_xfer(
            "READ",
            self._local_xfer_handle,
            local_indices,
            remote_handle,
            remote_indices,
            notif_msg=notif_msg,
        )
        self._nixl.transfer(handle)
        return handle

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return self._nixl.get_new_notifs()

    def check_xfer_state(self, handle: Any) -> str:
        return self._nixl.check_xfer_state(handle)

    def release_xfer_handle(self, handle: Any) -> None:
        try:
            self._nixl.release_xfer_handle(handle)
        except Exception:
            logger.warning("EC: release_xfer_handle failed", exc_info=True)

    def deregister(self) -> None:
        for _, reg_descs in self._granted_runs.values():
            try:
                self._nixl.deregister_memory(reg_descs, backends=_NIXL_BACKENDS)
            except Exception:
                logger.warning("EC: deregister of granted blocks failed", exc_info=True)
        self._granted_runs.clear()
        if self._reg_descs is None:
            return
        try:
            self._nixl.deregister_memory(self._reg_descs)
        except Exception:
            logger.warning("EC: deregister failed", exc_info=True)
        self._reg_descs = None
