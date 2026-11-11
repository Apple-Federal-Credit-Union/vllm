# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unauthenticated Mooncake encoder-cache control plane.

These tests are expected to fail until a peer that presents no credential
cannot reserve, complete, cancel, or read buffer addresses, and until cancel
ignores a missing reservation id.
"""

from __future__ import annotations

import ctypes
import socket

import pytest
import torch
import zmq

from vllm.distributed.ec_transfer.ec_connector.mooncake.control import (
    ConsumerControlServer,
    ControlClient,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    ConsumerReservation,
    ConsumerReservationManager,
    ConsumerReservationState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
    ECMooncakeWorker,
)
from vllm.utils.network_utils import make_zmq_path

pytestmark = pytest.mark.cpu_test

_SHAPE = (32,)
_DTYPE = torch.uint8
_DTYPE_NAME = "uint8"
_NBYTES = 32


class _HostMemoryTransfer(MooncakeTransfer):
    """Register the receive slab without opening a TransferEngine.

    The control plane returns ``tensor.data_ptr()`` from that slab. A Mooncake
    WRITE then copies its payload onto that address; the CPU apply of that
    write is ``ServerSession::readBody`` in mooncake-transfer-engine 0.3.14.
    """

    def register_memory(self, tensor: torch.Tensor) -> int:
        return 0

    def unregister_memory(self, tensor: torch.Tensor) -> bool:
        return True

    def local_session(self) -> str:
        return f"{self._hostname}:1"

    def ensure_ready(self) -> None:
        return None


class _ReceivingConsumer:
    """Consumer worker surface the control server and loader call."""

    def __init__(
        self,
        transfer: MooncakeTransfer,
        memory: ConsumerMemoryPool,
        reservations: ConsumerReservationManager,
    ) -> None:
        self._transfer = transfer
        self._consumer_memory = memory
        self._reservations = reservations
        self._buffer_device = "cpu"
        self._consumer_rank_resolved = True
        self._is_receiving_rank = True
        self._completed_loads: set[str] = set()
        self._failed_loads: set[str] = set()

    def _resolve_consumer_rank(self) -> None:
        ECMooncakeWorker._resolve_consumer_rank(self)

    def reserve(self, payload: dict) -> dict:
        return ECMooncakeWorker._reserve_push_destination(self, payload)

    def status(self, transfer_id: str) -> dict | None:
        return ECMooncakeWorker._push_status(self, transfer_id)

    def load(
        self,
        metadata: ECMooncakeConnectorMetadata,
        encoder_cache: dict[str, torch.Tensor],
    ) -> None:
        ECMooncakeWorker.start_load_caches(self, metadata, encoder_cache)


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _apply_write(dst_ptr: int, payload: bytes) -> None:
    """Copy ``payload`` onto ``dst_ptr`` as a completed CPU Mooncake WRITE."""
    if dst_ptr == 0 or not payload:
        return
    ctypes.memmove(dst_ptr, payload, len(payload))


def _payload(fill: int) -> bytes:
    return bytes((fill,)) * _NBYTES


class _ControlPlane:
    def __init__(self, pool_size: int) -> None:
        self.transfer = _HostMemoryTransfer("127.0.0.1", "tcp")
        self.memory = ConsumerMemoryPool(pool_size, self.transfer)
        self.memory.prepare(torch.device("cpu"))
        if self.memory.tensor is None:
            raise RuntimeError("consumer receive pool was not allocated")
        self.reservations = ConsumerReservationManager(self.memory, 300.0, 1024)
        self.consumer = _ReceivingConsumer(
            self.transfer, self.memory, self.reservations
        )
        self.port = _free_port()
        self.server = ConsumerControlServer(
            "127.0.0.1",
            self.port,
            self.consumer.reserve,
            self.consumer.status,
            self.reservations.complete,
            self.reservations.cancel,
            self.reservations.expire,
            peer_ports=[self.port],
            device=self.memory.tensor.device,
            drain_ready=self.reservations.drain_ready,
        )
        self.server.start()
        self.address = make_zmq_path("tcp", "127.0.0.1", self.port)
        self.client = ControlClient(2000)
        self.events = zmq.Context()
        self.event_pull = self.events.socket(zmq.PULL)
        self.event_pull.setsockopt(zmq.RCVTIMEO, 500)
        assert self.server.event_port is not None
        self.event_pull.connect(
            make_zmq_path("tcp", "127.0.0.1", self.server.event_port)
        )

    def close(self) -> None:
        self.event_pull.close(linger=0)
        self.events.term()
        self.client.close()
        self.server.close()
        self.memory.close()

    def request(self, payload: dict) -> dict:
        try:
            return self.client._exchange(self.address, payload)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def granted(self, response: dict) -> bool:
        return response.get("ok") is True

    def reserve_body(self, transfer_id: str, mm_hash: str) -> dict:
        return {
            "op": "reserve",
            "transfer_id": transfer_id,
            "mm_hash": mm_hash,
            "nbytes": _NBYTES,
            "shape": list(_SHAPE),
            "dtype": _DTYPE_NAME,
        }

    def plant(self, transfer_id: str, mm_hash: str) -> ConsumerReservation:
        record, created = self.reservations.reserve(
            transfer_id,
            mm_hash,
            _NBYTES,
            _SHAPE,
            _DTYPE_NAME,
            _DTYPE,
        )
        assert record is not None and created
        assert record.allocation is not None
        return record

    def drain_event(self) -> dict | None:
        try:
            event = self.event_pull.recv_json()
        except zmq.Again:
            return None
        return event if isinstance(event, dict) else {"malformed": True}


@pytest.fixture
def plane() -> _ControlPlane:
    opened = _ControlPlane(1 << 20)
    try:
        yield opened
    finally:
        opened.close()


def _pointer(response: dict) -> int:
    result = response.get("result")
    if not isinstance(result, dict):
        return 0
    try:
        return int(result.get("dst_ptr") or 0)
    except (TypeError, ValueError):
        return 0


def test_unauthenticated_reserve_cannot_serve_attacker_bytes(plane: _ControlPlane):
    """A reserve/complete with no credential must not enter the encoder cache."""
    transfer_id = "attacker-transfer"
    mm_hash = "attacker-hash"
    payload = _payload(0xA5)
    response = plane.request(plane.reserve_body(transfer_id, mm_hash))
    _apply_write(_pointer(response), payload)
    completed = plane.request(
        {
            "op": "complete",
            "transfer_id": transfer_id,
            "reservation_id": str(
                (response.get("result") or {}).get("reservation_id", "")
            ),
        }
    )
    encoder_cache: dict[str, torch.Tensor] = {}
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    plane.consumer.load(
        ECMooncakeConnectorMetadata(
            loads=[
                ECMooncakeLoadSpec(
                    mm_hash=mm_hash,
                    nbytes=_NBYTES,
                    shape=_SHAPE,
                    dtype=_DTYPE_NAME,
                    transfer_id=transfer_id,
                )
            ]
        ),
        encoder_cache,
    )
    served = encoder_cache.get(mm_hash)
    served_bytes = b""
    if served is not None:
        served_bytes = bytes(served.view(torch.uint8).tolist())
    assert served_bytes != payload
    assert mm_hash not in encoder_cache
    assert plane.reservations.status(transfer_id) is None
    assert not plane.granted(response)
    assert not (
        plane.granted(completed) and (completed.get("result") or {}).get("completed")
    )
    assert result.get("dst_session") in (None, "")
    assert plane.drain_event() is None


def test_unauthenticated_reserve_cannot_read_or_overwrite_cached_embedding(
    plane: _ControlPlane,
):
    """A known hash must not yield the resident embedding pointer."""
    mm_hash = "cached-hash"
    original = torch.arange(_NBYTES, dtype=_DTYPE)
    allocation = plane.memory.try_allocate(_NBYTES, _SHAPE, _DTYPE)
    assert allocation is not None
    allocation.tensor.copy_(original)
    plane.memory.publish(mm_hash, allocation)
    encoder_cache = {mm_hash: allocation.tensor}
    response = plane.request(plane.reserve_body("other-transfer", mm_hash))
    _apply_write(_pointer(response), _payload(0x3C))
    assert torch.equal(encoder_cache[mm_hash], original)
    assert torch.equal(allocation.tensor, original)
    assert _pointer(response) != allocation.tensor.data_ptr()
    assert not plane.granted(response)


def test_unauthenticated_reserve_cannot_evict_cached_embedding():
    """Repeated reserves must not reclaim a retired cached embedding."""
    plane = _ControlPlane(256)
    try:
        mm_hash = "resident-hash"
        original = torch.full(_SHAPE, 7, dtype=_DTYPE)
        allocation = plane.memory.try_allocate(_NBYTES, _SHAPE, _DTYPE)
        assert allocation is not None
        allocation.tensor.copy_(original)
        plane.memory.publish(mm_hash, allocation)
        plane.memory.retire_stale({}, freed=[mm_hash])
        responses = [
            plane.request(plane.reserve_body(f"fill-{index}", f"new-hash-{index}"))
            for index in range(4)
        ]
        retained = plane.memory.take_resident(mm_hash, _SHAPE, _DTYPE_NAME)
        assert retained is not None
        assert torch.equal(retained, original)
        assert mm_hash not in plane.memory.drain_reclaimed()
        assert all(not plane.granted(response) for response in responses)
        assert plane.reservations.status("fill-0") is None
    finally:
        plane.close()


def test_unauthenticated_reserve_batch_allocates_nothing(plane: _ControlPlane):
    """reserve_batch is the same unauthenticated reserve path."""
    transfer_id = "batch-transfer"
    response = plane.request(
        {
            "op": "reserve_batch",
            "items": [plane.reserve_body(transfer_id, "batch-hash")],
        }
    )
    items = (response.get("result") or {}).get("items", [])
    accepted = plane.granted(response) and any(item.get("ok") for item in items)
    assert not accepted
    assert plane.reservations.status(transfer_id) is None


def test_unauthenticated_complete_does_not_publish_readiness(plane: _ControlPlane):
    """complete must not mark a reservation ready or emit a readiness event."""
    record = plane.plant("owned-transfer", "owned-hash")
    assert record.allocation is not None
    _apply_write(record.allocation.tensor.data_ptr(), _payload(0x11))
    response = plane.request(
        {
            "op": "complete",
            "transfer_id": record.transfer_id,
            "reservation_id": record.reservation_id,
        }
    )
    status = plane.reservations.status(record.transfer_id)
    assert status is not None
    assert status.state is ConsumerReservationState.WRITING
    assert not plane.granted(response)
    assert plane.drain_event() is None


def test_unauthenticated_complete_batch_does_not_publish_readiness(
    plane: _ControlPlane,
):
    """complete_batch shares the complete handler."""
    record = plane.plant("owned-batch", "owned-batch-hash")
    response = plane.request(
        {
            "op": "complete_batch",
            "items": [
                {
                    "transfer_id": record.transfer_id,
                    "reservation_id": record.reservation_id,
                }
            ],
        }
    )
    status = plane.reservations.status(record.transfer_id)
    assert status is not None
    assert status.state is ConsumerReservationState.WRITING
    items = (response.get("result") or {}).get("items", [])
    assert not any(item.get("completed") for item in items)
    assert not plane.granted(response)


def test_unauthenticated_cancel_with_reservation_id_does_not_abort(
    plane: _ControlPlane,
):
    """cancel must not abort a reservation for a peer with no credential."""
    record = plane.plant("cancel-with-id", "cancel-hash")
    response = plane.request(
        {
            "op": "cancel",
            "transfer_id": record.transfer_id,
            "reservation_id": record.reservation_id,
            "abandon": True,
        }
    )
    status = plane.reservations.status(record.transfer_id)
    assert status is not None
    assert status.state is ConsumerReservationState.WRITING
    assert not (
        plane.granted(response) and (response.get("result") or {}).get("cancelled")
    )


def test_cancel_without_reservation_id_does_not_abort(plane: _ControlPlane):
    """An empty reservation id must not skip the reservation-id check."""
    writing = plane.plant("cancel-empty-writing", "empty-writing-hash")
    ready = plane.plant("cancel-empty-ready", "empty-ready-hash")
    assert plane.reservations.complete(ready.transfer_id, ready.reservation_id) == (
        True,
        True,
    )
    via_socket = plane.request(
        {"op": "cancel", "transfer_id": writing.transfer_id, "abandon": True}
    )
    via_manager = plane.reservations.cancel(ready.transfer_id, "")
    writing_status = plane.reservations.status(writing.transfer_id)
    ready_status = plane.reservations.status(ready.transfer_id)
    assert writing_status is not None
    assert writing_status.state is ConsumerReservationState.WRITING
    assert ready_status is not None
    assert ready_status.state is ConsumerReservationState.READY
    assert via_manager is False
    assert not (
        plane.granted(via_socket) and (via_socket.get("result") or {}).get("cancelled")
    )
    unknown = "transfer-not-yet-reserved"
    preemptive = plane.request({"op": "cancel", "transfer_id": unknown})
    assert unknown not in plane.reservations._records
    assert not (
        plane.granted(preemptive)
        and (preemptive.get("result") or {}).get("cancelled")
    )


def test_unauthenticated_status_peers_and_event_port_disclose_nothing(
    plane: _ControlPlane,
):
    """status, peers, and event_port share the unauthenticated REP socket."""
    record = plane.plant("status-transfer", "status-hash")
    status = plane.request({"op": "status", "transfer_id": record.transfer_id})
    peers = plane.request({"op": "peers"})
    event_port = plane.request({"op": "event_port"})
    assert not plane.granted(status)
    assert not plane.granted(peers)
    assert not plane.granted(event_port)
    disclosed = status.get("result")
    assert not (isinstance(disclosed, dict) and disclosed.get("reservation_id"))
    assert (peers.get("result") or {}).get("ports") in (None, [])
    assert event_port.get("result") != plane.server.event_port


def test_cancel_reservation_id_authorizes_only_that_reservation(plane: _ControlPlane):
    """A reservation id must not cancel a different transfer."""
    first = plane.plant("transfer-a", "hash-a")
    second = plane.plant("transfer-b", "hash-b")
    response = plane.request(
        {
            "op": "cancel",
            "transfer_id": second.transfer_id,
            "reservation_id": first.reservation_id,
            "abandon": True,
        }
    )
    status = plane.reservations.status(second.transfer_id)
    assert status is not None
    assert status.state is ConsumerReservationState.WRITING
    assert not (
        plane.granted(response) and (response.get("result") or {}).get("cancelled")
    )
