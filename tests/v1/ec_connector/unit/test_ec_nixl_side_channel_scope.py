# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security spec for the CPU encoder-cache NIXL side channel.

A producer answers ``XferReq`` on the ZMQ side channel. These tests are the
behavior a fix has to restore. They fail while a peer that can reach the
socket and name one cached item can obtain another item's embedding bytes.

Peer authentication, once implemented, is the ``auth_token`` field on the
request map and the ``VLLM_EC_SIDE_CHANNEL_TOKEN`` environment variable.
The value used here is an obvious placeholder, not a deployed credential.
A grant may describe and serve only the named item's blocks. Agent metadata
returned with that grant must not let the peer NIXL-READ any other block
in the region.

The Mooncake encoder-cache connector does not serve this ZMQ descriptor
grant, so it is not covered here.
"""

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import msgspec
import pytest
import torch
import zmq

from vllm.distributed.ec_transfer.ec_connector.cpu.control.zmq import (
    ZmqClientTransport,
    ZmqServerTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl import (
    NixlDataTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.protocol import (
    EC_CONNECTOR_VERSION,
    XferAck,
    XferStatus,
    compute_ec_compatibility_hash,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    CacheEntry,
    EmbeddingCache,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.session import ProducerSession
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    deserialize_mem_descriptor,
    serialize_mem_descriptor,
)

_BLOCK = 64
_NUM_BLOCKS = 4
_PATTERN_NAMED = 0x11
_PATTERN_OTHER = 0x22
_PLACEHOLDER_TOKEN = "placeholder"
_WRONG_TOKEN = "wrong-placeholder"
_TOKEN_ENV = "VLLM_EC_SIDE_CHANNEL_TOKEN"


class _XferReqOnWire(  # type: ignore[call-arg]
    msgspec.Struct,
    tag="req",
    omit_defaults=True,
):
    """Wire request. ``auth_token`` is ignored until the producer reads it."""

    mm_hash: str
    compatibility_hash: str
    session_id: str = ""
    connector_version: int = EC_CONNECTOR_VERSION
    auth_token: str = ""


class _PinCountingCache(EmbeddingCache):
    """Counts pins taken by ``pin_if_ready``, including ones later released."""

    def __init__(self, num_blocks: int) -> None:
        super().__init__(num_blocks)
        self.pins_taken = 0

    def pin_if_ready(self, key: str) -> CacheEntry | None:
        before = self.get(key)
        was_evictable = before is not None and before.evictable
        entry = super().pin_if_ready(key)
        if was_evictable and entry is not None and not entry.evictable:
            self.pins_taken += 1
        return entry


class _Producer:
    def __init__(self) -> None:
        self.engine_id = uuid.uuid4().hex
        self.region = ECSharedRegion(self.engine_id, _NUM_BLOCKS, _BLOCK)
        self.cache = _PinCountingCache(_NUM_BLOCKS)
        named = self.cache.alloc("named-item", 1)
        other = self.cache.alloc("other-item", 1)
        assert named is not None and other is not None
        self.named_block = named.block_ids[0]
        self.other_block = other.block_ids[0]
        self.cache.mark_ready("named-item")
        self.cache.mark_ready("other-item")
        self.region.blocks[self.named_block].fill_(_PATTERN_NAMED)
        self.region.blocks[self.other_block].fill_(_PATTERN_OTHER)
        self.compat_hash = compute_ec_compatibility_hash(
            vllm_version="0.test",
            model="public-model",
            dtype="float16",
            block_size_bytes=_BLOCK,
        )
        self.data = NixlDataTransport(
            agent_name=f"producer-{self.engine_id}",
            base_ptr=self.region.blocks.data_ptr(),
            num_blocks=_NUM_BLOCKS,
            block_size_bytes=_BLOCK,
            total_size_bytes=_NUM_BLOCKS * _BLOCK,
        )
        self.server = ZmqServerTransport("127.0.0.1", 0)
        endpoint = self.server._router.getsockopt(zmq.LAST_ENDPOINT)
        if isinstance(endpoint, bytes):
            endpoint = endpoint.decode()
        self.port = int(str(endpoint).rsplit(":", 1)[-1])
        self.session = ProducerSession(
            transport=self.server,
            data=self.data,
            cache=self.cache,
            compat_hash=self.compat_hash,
        )
        self.clients = ZmqClientTransport()

    def close(self) -> None:
        self.clients.close()
        self.session.close()
        self.data.deregister()
        self.region.cleanup()

    def request(self, auth_token: str | None) -> XferAck:
        conn = self.clients.connect(("127.0.0.1", self.port))
        request = _XferReqOnWire(
            mm_hash="named-item",
            compatibility_hash=self.compat_hash,
            session_id=uuid.uuid4().hex,
            auth_token=auth_token or "",
        )
        conn.send(msgspec.msgpack.Encoder().encode(request))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.session.poll(self.server.poll(timeout_ms=100))
            payloads = conn.recv()
            if payloads:
                return msgspec.msgpack.Decoder(XferAck).decode(payloads[0])
        raise AssertionError("producer sent no XferAck")


@contextmanager
def _producer(
    monkeypatch: pytest.MonkeyPatch, token: str | None
) -> Iterator[_Producer]:
    if token is None:
        monkeypatch.delenv(_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(_TOKEN_ENV, token)
    producer = _Producer()
    try:
        yield producer
    finally:
        producer.close()


def _pattern(byte: int) -> bytes:
    return bytes([byte]) * _BLOCK


def _nixl_read(metadata: bytes, remote_desc: tuple[int, int, int]) -> bytes | None:
    """READ one remote descriptor. None means the transfer was rejected."""
    if not metadata:
        return None
    local = torch.zeros(1, _BLOCK, dtype=torch.int8)
    agent_name = f"peer-{uuid.uuid4().hex}"
    attacker = NixlDataTransport(
        agent_name=agent_name,
        base_ptr=local.data_ptr(),
        num_blocks=1,
        block_size_bytes=_BLOCK,
        total_size_bytes=_BLOCK,
    )
    handle = None
    try:
        remote = attacker.add_remote_peer(
            metadata, serialize_mem_descriptor([remote_desc])
        )
        handle = attacker.post_read([0], remote, [0], notif_msg=b"probe")
        deadline = time.monotonic() + 5
        state = "PROC"
        while time.monotonic() < deadline:
            state = attacker.check_xfer_state(handle)
            if state != "PROC":
                break
        if state != "DONE":
            return None
        return local.numpy().tobytes()
    except Exception:
        return None
    finally:
        if handle is not None:
            attacker.release_xfer_handle(handle)
        attacker.deregister()


def _read_descriptor_index(ack: XferAck, index: int) -> bytes | None:
    descs = deserialize_mem_descriptor(ack.mem_descriptor)
    if index < 0 or index >= len(descs):
        return None
    addr, length, device = descs[index]
    return _nixl_read(ack.agent_metadata, (addr, length, device))


def _describes_only(ack: XferAck, addr: int) -> bool:
    """True when the ack lists exactly one block at ``addr``."""
    if not ack.mem_descriptor:
        return False
    return deserialize_mem_descriptor(ack.mem_descriptor) == [(addr, _BLOCK, 0)]


def _other_item_was_read(producer: _Producer, ack: XferAck) -> bool:
    """True if this ack lets a peer copy the item it did not name."""
    victim = _pattern(_PATTERN_OTHER)
    if ack.mem_descriptor:
        granted = set(ack.src_block_indices)
        for index in range(len(deserialize_mem_descriptor(ack.mem_descriptor))):
            if index in granted:
                continue
            copied = _read_descriptor_index(ack, index)
            if copied == victim:
                return True
    other = producer.region.blocks[producer.other_block]
    addr = other.data_ptr()
    copied = _nixl_read(ack.agent_metadata, (addr, _BLOCK, 0))
    return copied == victim


def test_named_grant_reads_that_item_and_not_any_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peer that presents the placeholder token may read only the named item."""
    with _producer(monkeypatch, _PLACEHOLDER_TOKEN) as producer:
        ack = producer.request(_PLACEHOLDER_TOKEN)
        assert ack.status == XferStatus.OK
        assert list(ack.src_block_indices) == [producer.named_block]
        copied = _read_descriptor_index(ack, ack.src_block_indices[0])
        assert copied == _pattern(_PATTERN_NAMED)
        named_addr = producer.region.blocks[producer.named_block].data_ptr()
        describes_named = _describes_only(ack, named_addr)
        leaked = _other_item_was_read(producer, ack)
        assert leaked is False
        assert describes_named


@pytest.mark.parametrize(
    ("server_token", "request_token"),
    [
        (None, None),
        (None, _PLACEHOLDER_TOKEN),
        (_PLACEHOLDER_TOKEN, None),
        (_PLACEHOLDER_TOKEN, _WRONG_TOKEN),
    ],
)
def test_xfer_req_without_matching_token_exposes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    server_token: str | None,
    request_token: str | None,
) -> None:
    """A request that does not match the configured token must not be granted.

    The producer must not pin the named item, and the reply must not carry
    agent metadata or a memory descriptor. A peer must not be able to copy
    the other item's bytes from that reply.
    """
    with _producer(monkeypatch, server_token) as producer:
        ack = producer.request(request_token)
        leaked = _other_item_was_read(producer, ack)
        assert leaked is False
        assert ack.status != XferStatus.OK
        assert ack.agent_metadata == b""
        assert ack.mem_descriptor == b""
        assert producer.cache.pins_taken == 0
