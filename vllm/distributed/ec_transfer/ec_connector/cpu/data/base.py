# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base class for the EC data-plane transport (NIXL/UCX)."""

from abc import ABC, abstractmethod
from typing import Any


class DataTransport(ABC):
    """Interface wrapping the NIXL data-plane for EC block transfers.

    Producer side:
      grant_blocks()     — expose some blocks to peers; returns what a peer needs
      revoke_blocks()    — stop exposing blocks returned by grant_blocks()

    Peer lifecycle:
      add_remote_peer()  — register a peer's grant and store its dlist handle
      remove_remote_peer() — deregister and release the stored handle
      post_read()        — issue a READ using the stored handle for agent_name

    The dlist handle (prepped xfer descriptor list) is an implementation detail
    of the transport and is never exposed to callers.
    """

    @abstractmethod
    def grant_blocks(self, block_indices: list[int]) -> tuple[bytes, bytes]:
        """Make block_indices readable by a peer.

        Returns:
            (agent_metadata, mem_descriptor) describing only block_indices, in
            that order. Every call must be paired with revoke_blocks().

        """

    @abstractmethod
    def revoke_blocks(self, block_indices: list[int]) -> None:
        """Release one grant_blocks() call for block_indices."""

    @abstractmethod
    def add_remote_peer(self, metadata: bytes, mem_descriptor: bytes) -> str:
        """Load a peer's grant, prep its READ-source dlist, and return agent_name.

        The dlist handle is stored internally and used by post_read(). A later
        grant from the same peer replaces it.
        """

    @abstractmethod
    def remove_remote_peer(self, agent_name: str) -> None:
        """Deregister a previously registered remote peer and release its handle."""

    @abstractmethod
    def post_read(
        self,
        local_indices: list[int],
        agent_name: str,
        remote_indices: list[int],
        notif_msg: bytes,
    ) -> Any:
        """Issue a consumer-initiated READ using the stored dlist for agent_name.

        Returns a transfer handle for use with check_xfer_state / release_xfer_handle.
        """

    @abstractmethod
    def get_new_notifs(self) -> dict[str, list[bytes]]:
        """Drain NIXL completion notifications addressed to this agent."""

    @abstractmethod
    def check_xfer_state(self, handle: Any) -> str:
        """Return the NIXL state string: 'DONE', 'PROC', or an error string."""

    @abstractmethod
    def release_xfer_handle(self, handle: Any) -> None:
        """Release a transfer handle returned by post_read."""

    @abstractmethod
    def deregister(self) -> None:
        """Deregister all memory and tear down the NIXL agent."""
