# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NIXL side channel must not give an unauthenticated peer KV metadata.

Pull and push connectors both publish ``NixlHandshakePayload`` from
``NixlBaseConnectorScheduler._nixl_handshake_listener``. That payload is
the NIXL agent metadata (the blob ``add_remote_agent`` loads before a
READ or WRITE) plus the KV base addresses and block geometry. These
tests speak the same ZMQ request the decode worker sends and require
that an unauthenticated caller not receive those bytes.

They fail while the listener answers every request. They pass once the
listener withholds the handshake until the peer is authenticated.
"""

import ctypes
import time

import msgspec
import pytest
import zmq

from tests.v1.kv_connector.unit.utils import create_vllm_config, make_kv_cache_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
    NixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlAgentMetadata,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.utils.network_utils import get_open_port, make_zmq_path

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:  # optional; the disclosure checks do not need it
    nixl_agent = None
    nixl_agent_config = None

# Distinctive bytes. They stand in for ``nixl_agent.get_agent_metadata()``,
# which is an opaque blob the listener returns unchanged.
_PULL_AGENT = b"nixl-pull-agent-metadata-sentinel"
_PUSH_AGENT = b"nixl-push-agent-metadata-sentinel"
_UPDATED_AGENT = b"nixl-updated-agent-metadata-sentinel"

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _agent(engine_id: str, agent_metadata: bytes, base_addr: int) -> NixlAgentMetadata:
    """Build the same record the worker puts in the handshake."""
    return NixlAgentMetadata(
        engine_id=engine_id,
        agent_metadata=agent_metadata,
        kv_caches_base_addr=[base_addr],
        device_id=0,
        num_blocks=8,
        block_lens=[256],
        block_strides=[256],
        kv_cache_layout="LBHNC",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        physical_blocks_per_logical_kv_block=1,
        region_num_blocks=[8],
        region_group_ids=[0],
        region_names=["layer0"],
        region_mem_types=["VRAM"],
    )


def _payload(agent: NixlAgentMetadata) -> NixlHandshakePayload:
    return NixlHandshakePayload(
        compatibility_hash="test-compat-hash",
        agent_metadata_bytes=msgspec.msgpack.encode(agent),
    )


def _secret_blobs(*agents: NixlAgentMetadata) -> list[bytes]:
    """Byte strings that must not appear in an unauthenticated reply."""
    blobs: list[bytes] = []
    for agent in agents:
        blobs.append(agent.agent_metadata)
        blobs.append(msgspec.msgpack.encode(agent))
        blobs.append(msgspec.msgpack.encode(_payload(agent)))
    return blobs


def _recv_frames(sock: zmq.Socket) -> list[bytes]:
    try:
        return list(sock.recv_multipart())
    except zmq.Again:
        return []


def _query(host: str, port: int, message: tuple) -> list[bytes]:
    """Send one handshake request the way ``_nixl_handshake`` does."""
    path = make_zmq_path("tcp", host, port)
    with zmq_ctx(zmq.REQ, path) as sock:
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.send(msgspec.msgpack.encode(message))
        return _recv_frames(sock)


def _assert_withheld(frames: list[bytes], *agents: NixlAgentMetadata) -> None:
    """Fail if any frame carries agent metadata or KV base addresses."""
    reply = b"".join(frames)
    for secret in _secret_blobs(*agents):
        assert secret not in reply
    for frame in frames:
        try:
            payload = msgspec.msgpack.decode(frame, type=NixlHandshakePayload)
        except (msgspec.DecodeError, msgspec.ValidationError):
            continue
        meta = msgspec.msgpack.decode(
            payload.agent_metadata_bytes, type=NixlAgentMetadata
        )
        pytest.fail(
            "unauthenticated peer decoded NIXL handshake for "
            f"engine {meta.engine_id} base addrs {meta.kv_caches_base_addr}"
        )


def _decoded_handshake(frames: list[bytes]) -> NixlAgentMetadata | None:
    for frame in frames:
        try:
            payload = msgspec.msgpack.decode(frame, type=NixlHandshakePayload)
            return msgspec.msgpack.decode(
                payload.agent_metadata_bytes, type=NixlAgentMetadata
            )
        except (msgspec.DecodeError, msgspec.ValidationError):
            continue
    return None


def _wait_xfer(agent, handle: object) -> str:
    state = "PROC"
    for _ in range(50):
        state = agent.check_xfer_state(handle)
        if state != "PROC":
            return state
        time.sleep(0.05)
    return state


def _read_and_write_region(
    stolen: NixlAgentMetadata, victim: ctypes.Array
) -> tuple[bytes, bytes]:
    """READ then WRITE the region described by a stolen handshake.

    Returns the bytes the READ landed locally and the victim buffer after
    the WRITE. A caller that was not given the handshake cannot do either.
    """
    assert nixl_agent is not None
    assert nixl_agent_config is not None
    length = stolen.block_lens[0]
    remote_addr = stolen.kv_caches_base_addr[0]
    mem_type = stolen.region_mem_types[0] if stolen.region_mem_types else "DRAM"
    attacker = nixl_agent(
        f"attacker-{stolen.engine_id}", nixl_agent_config(backends=["UCX"])
    )
    remote_name = attacker.add_remote_agent(stolen.agent_metadata)
    if isinstance(remote_name, bytes):
        remote_name = remote_name.decode()

    read_buf = (ctypes.c_uint8 * length)()
    read_addr = ctypes.addressof(read_buf)
    attacker.register_memory([(read_addr, length, 0, "")], mem_type)
    local = attacker.get_xfer_descs([(read_addr, length, 0)], mem_type)
    remote = attacker.get_xfer_descs([(remote_addr, length, 0)], mem_type)
    read_handle = attacker.initialize_xfer("READ", local, remote, remote_name)
    attacker.transfer(read_handle)
    assert _wait_xfer(attacker, read_handle) == "DONE"

    write_buf = (ctypes.c_uint8 * length)(*([0x5A] * length))
    write_addr = ctypes.addressof(write_buf)
    attacker.register_memory([(write_addr, length, 0, "")], mem_type)
    local = attacker.get_xfer_descs([(write_addr, length, 0)], mem_type)
    write_handle = attacker.initialize_xfer("WRITE", local, remote, remote_name)
    attacker.transfer(write_handle)
    assert _wait_xfer(attacker, write_handle) == "DONE"
    return bytes(read_buf), bytes(victim)


def _publish(connector, agents_by_rank: dict[tuple[int, int], NixlAgentMetadata]):
    connector.set_xfer_handshake_metadata_pp_aware(
        {rank: _payload(agent) for rank, agent in agents_by_rank.items()}
    )


@pytest.fixture
def side_channel(monkeypatch: pytest.MonkeyPatch):
    """Bind the listener on every interface, as cross-node P/D must."""
    port = get_open_port()
    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "0.0.0.0")
    monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_PORT", str(port))
    return "127.0.0.1", port


def _start(connector_cls, *, kv_role: str):
    config = create_vllm_config(kv_role=kv_role, kv_connector=connector_cls.__name__)
    return connector_cls(config, KVConnectorRole.SCHEDULER, make_kv_cache_config(16))


@pytest.mark.parametrize(
    ("connector_cls", "kv_role", "agent_blob"),
    [
        (NixlPullConnector, "kv_producer", _PULL_AGENT),
        (NixlPushConnector, "kv_producer", _PUSH_AGENT),
    ],
)
def test_unauthenticated_peer_cannot_read_nixl_handshake(
    side_channel, connector_cls, kv_role, agent_blob
):
    """GET_META_MSG for a published rank must not return that rank's KV.

    Covers the pull (READ) and push (WRITE) schedulers. Both serve the
    handshake from the same listener after KV registration is published.
    """
    dial_host, port = side_channel
    rank00 = _agent("engine-pp0-tp0", agent_blob + b"-0-0", 0x4E49584C0001)
    rank01 = _agent("engine-pp0-tp1", agent_blob + b"-0-1", 0x4E49584C0002)
    rank10 = _agent("engine-pp1-tp0", agent_blob + b"-1-0", 0x4E49584C0003)
    published = {(0, 0): rank00, (0, 1): rank01, (1, 0): rank10}

    connector = _start(connector_cls, kv_role=kv_role)
    try:
        scheduler = connector.connector_scheduler
        assert scheduler is not None
        assert scheduler.side_channel_host == "0.0.0.0"
        assert scheduler.side_channel_port == port
        _publish(connector, published)

        # Two peers, neither authenticated. Each asks for every rank with
        # the tag the worker uses, then with a different tag. The listener
        # currently answers both.
        for _peer in range(2):
            for pp_rank, tp_rank in published:
                reply = _query(dial_host, port, (GET_META_MSG, pp_rank, tp_rank))
                _assert_withheld(reply, rank00, rank01, rank10)
                other = _query(
                    dial_host, port, (b"not-the-handshake-tag", pp_rank, tp_rank)
                )
                _assert_withheld(other, rank00, rank01, rank10)

        # A later registration is served by the same socket. It is still
        # not a public document.
        updated = _agent("engine-pp0-tp0", _UPDATED_AGENT, 0x4E49584C0004)
        _publish(connector, {**published, (0, 0): updated})
        refreshed = _query(dial_host, port, (GET_META_MSG, 0, 0))
        _assert_withheld(refreshed, rank00, rank01, rank10, updated)
    finally:
        connector.shutdown()


def test_unauthenticated_handshake_cannot_read_or_write_registered_memory(
    side_channel,
):
    """A peer that only sends GET_META_MSG must not be able to move the region.

    The listener is the pull scheduler. Push uses the same listener; the
    other test covers that entry point. This one loads the returned blob
    in a NIXL client and issues READ and WRITE against the registered bytes.
    """
    if nixl_agent is None or nixl_agent_config is None:
        pytest.skip("nixl is not installed")

    dial_host, port = side_channel
    length = 64
    pattern = bytes([0xA5]) * length
    victim_buf = (ctypes.c_uint8 * length)(*pattern)
    victim_addr = ctypes.addressof(victim_buf)
    victim = nixl_agent(f"victim-{port}", nixl_agent_config(backends=["UCX"]))
    victim.register_memory([(victim_addr, length, 0, "")], "DRAM")
    record = _agent(f"victim-{port}", victim.get_agent_metadata(), victim_addr)
    record.block_lens = [length]
    record.block_strides = [length]
    record.region_mem_types = ["DRAM"]

    connector = _start(NixlPullConnector, kv_role="kv_producer")
    try:
        _publish(connector, {(0, 0): record})
        frames = _query(dial_host, port, (GET_META_MSG, 0, 0))
        stolen = _decoded_handshake(frames)
        if stolen is not None:
            read_back, after_write = _read_and_write_region(stolen, victim_buf)
            untouched = bytes(length)
            assert (read_back, after_write) == (untouched, pattern)
        _assert_withheld(frames, record)
    finally:
        connector.shutdown()
