# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security contract for the external elastic scale-up handshake.

The listener must authenticate peers with the secret published to the
control store before it accepts connections. The data-parallel rank is the
``dp_rank`` field of that authenticated message. The ZMQ identity is only
the router's routing key.

A peer who does not present that secret gets no reply and does not remove
an expected rank. An identity outside the new-rank set, a non-msgpack
frame, an unexpected status, or an extra frame is dropped the same way.
The listener keeps serving the ranks that do authenticate.

These tests fail until that contract holds.
"""

import time
from dataclasses import dataclass

import msgspec
import pytest
import zmq
from torch.distributed import TCPStore

from vllm.distributed.elastic_ep.external_elastic_ep import (
    ExternalElasticEPScaleCoordinator,
    ExternalElasticEPScaleUpHandshakeMetadata,
    ExternalElasticEPScaleUpHandshakeServer,
)
from vllm.distributed.utils import create_tcp_store
from vllm.utils.network_utils import get_open_port, make_zmq_socket
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.engine.utils import EngineZmqAddresses
from vllm.v1.utils import get_engine_client_zmq_addr

# Stand-in bytes for a secret the peer does not know. Not a store value.
_UNKNOWN_SECRET = b"placeholder-handshake-secret"
_SECRET_PART = "handshake_secret"
_LISTEN_GRACE_S = 2.0
_REPLY_TIMEOUT_MS = 1500

_NEW_RANKS = (2, 3)
_NUM_REDUNDANT_EXPERTS = 5
_ELASTIC_EP_MAX_DP_SIZE = 8

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@dataclass
class _ParallelConfig:
    data_parallel_master_ip: str
    data_parallel_rpc_port: int


@dataclass
class _VllmConfig:
    parallel_config: _ParallelConfig


@dataclass
class _ScaleClient:
    vllm_config: _VllmConfig
    input_address: str
    output_address: str
    coordinator_input_address: str
    coordinator_output_address: str
    stats_update_address: str


@dataclass
class _Handshake:
    coordinator: ExternalElasticEPScaleCoordinator
    control_store: TCPStore
    epoch: str
    bootstrap: ReconfigureDistributedRequest
    server: ExternalElasticEPScaleUpHandshakeServer
    addresses: EngineZmqAddresses
    ctx: zmq.Context

    def secret(self) -> bytes:
        key = self.coordinator.key(self.epoch, _SECRET_PART)
        if not self.control_store.check([key]):
            raise AssertionError(
                f"control store is missing the scale-up handshake secret at {key}"
            )
        value = self.control_store.get(key)
        if not value:
            raise AssertionError("scale-up handshake secret is empty")
        return value


def _rank_identity(rank: int, width: int = 2) -> bytes:
    return rank.to_bytes(width, "little")


def _unauthenticated(status: str) -> bytes:
    """Payload the engine client sends today, with no handshake secret."""
    return msgspec.msgpack.encode({"status": status, "local": False, "headless": True})


def _authenticated(status: str, rank: int, secret: bytes) -> bytes:
    return msgspec.msgpack.encode(
        {
            "status": status,
            "local": False,
            "headless": True,
            "dp_rank": rank,
            "handshake_secret": secret,
        }
    )


def _assert_still_serving(server: ExternalElasticEPScaleUpHandshakeServer) -> None:
    """Fail if the listener records an error or exits on its own."""
    deadline = time.monotonic() + _LISTEN_GRACE_S
    while time.monotonic() < deadline:
        if server._error is not None or not server._thread.is_alive():
            break
        time.sleep(0.05)
    server.raise_if_failed()
    assert server._thread.is_alive()


def _assert_finished(server: ExternalElasticEPScaleUpHandshakeServer) -> None:
    deadline = time.monotonic() + _LISTEN_GRACE_S
    while server._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not server._thread.is_alive()
    server.raise_if_failed()


def _assert_join_metadata(reply: bytes, handshake: _Handshake, secret: bytes) -> None:
    assert secret not in reply
    message = msgspec.msgpack.decode(
        reply, type=ExternalElasticEPScaleUpHandshakeMetadata
    )
    assert message.engine_metadata.addresses == handshake.addresses
    parallel_config = message.engine_metadata.parallel_config
    bootstrap = handshake.bootstrap
    assert parallel_config["data_parallel_master_ip"] == (
        bootstrap.new_data_parallel_master_ip
    )
    assert (
        parallel_config["data_parallel_master_port"]
        == bootstrap.new_data_parallel_master_port
    )
    assert parallel_config["_data_parallel_master_port_list"] == (
        bootstrap.new_data_parallel_master_port_list
    )
    assert parallel_config["data_parallel_size"] == bootstrap.new_data_parallel_size
    assert parallel_config["_coord_store_port"] == bootstrap.coord_store_port
    assert message.num_redundant_experts == _NUM_REDUNDANT_EXPERTS
    assert message.elastic_ep_max_dp_size == _ELASTIC_EP_MAX_DP_SIZE


@pytest.fixture
def handshake():
    rpc_port = get_open_port()
    master_ip = "127.0.0.1"
    addresses = EngineZmqAddresses(
        inputs=["tcp://10.1.0.8:18001"],
        outputs=["tcp://10.1.0.8:18002"],
        coordinator_input="tcp://10.1.0.8:18003",
        coordinator_output="tcp://10.1.0.8:18004",
        frontend_stats_publish_address="tcp://10.1.0.8:18005",
    )
    client = _ScaleClient(
        vllm_config=_VllmConfig(
            parallel_config=_ParallelConfig(
                data_parallel_master_ip=master_ip,
                data_parallel_rpc_port=rpc_port,
            )
        ),
        input_address=addresses.inputs[0],
        output_address=addresses.outputs[0],
        coordinator_input_address=addresses.coordinator_input,
        coordinator_output_address=addresses.coordinator_output,
        stats_update_address=addresses.frontend_stats_publish_address,
    )
    coordinator = ExternalElasticEPScaleCoordinator(client)
    control_store = create_tcp_store(
        master_ip, 0, is_master=True, world_size=-1, wait_for_workers=False
    )
    epoch, bootstrap = coordinator._prepare_reconfig_bootstrap(control_store, 2, 4)
    server = coordinator._start_scale_up_handshake_server(
        bootstrap,
        2,
        _NUM_REDUNDANT_EXPERTS,
        _ELASTIC_EP_MAX_DP_SIZE,
    )
    ctx = zmq.Context()
    try:
        assert server.handshake_address == get_engine_client_zmq_addr(
            False, master_ip, rpc_port
        )
        yield _Handshake(
            coordinator=coordinator,
            control_store=control_store,
            epoch=epoch,
            bootstrap=bootstrap,
            server=server,
            addresses=addresses,
            ctx=ctx,
        )
    finally:
        # stop() re-raises a listener failure. The test already observed it.
        server._error = None
        server._stop_event.set()
        server._thread.join(timeout=2)
        ctx.destroy(linger=0)


def _exchange(
    handshake: _Handshake,
    identity: bytes,
    payload: bytes | list[bytes],
    timeout_ms: int = _REPLY_TIMEOUT_MS,
) -> bytes | None:
    sock = make_zmq_socket(
        handshake.ctx,
        handshake.server.handshake_address,
        zmq.DEALER,
        identity=identity,
        bind=False,
        linger=0,
    )
    try:
        if isinstance(payload, list):
            sock.send_multipart(payload)
        else:
            sock.send(payload)
        if not sock.poll(timeout_ms):
            return None
        return sock.recv()
    finally:
        sock.close(linger=0)


def test_listener_binds_tcp_master_rpc_port(handshake: _Handshake):
    """Scale-up reopens the data-parallel RPC port as a TCP router."""
    address = handshake.server.handshake_address
    assert address.startswith("tcp://127.0.0.1:")
    assert address.endswith(
        str(
            handshake.coordinator.client.vllm_config.parallel_config.data_parallel_rpc_port
        )
    )


def test_handshake_secret_is_in_the_control_store_when_listening(
    handshake: _Handshake,
):
    """The secret is already in the control store once the port accepts."""
    secret = handshake.secret()
    assert secret != handshake.epoch.encode()
    assert secret != msgspec.msgpack.encode(handshake.bootstrap)
    assert secret != _UNKNOWN_SECRET


@pytest.mark.parametrize(
    "identity",
    [
        _rank_identity(0),
        _rank_identity(99),
        b"not-a-rank",
        b"\xff" * 16,
    ],
)
def test_unexpected_identity_does_not_abort_scale_up(handshake: _Handshake, identity):
    """An identity that is not a new rank must not tear the listener down."""
    reply = _exchange(handshake, identity, _unauthenticated("HELLO"))
    assert reply is None
    _assert_still_serving(handshake.server)


@pytest.mark.parametrize(
    "identity",
    [
        _rank_identity(2),
        _rank_identity(2, width=1),
        _rank_identity(2, width=3),
        _rank_identity(2, width=8),
    ],
)
def test_rank_shaped_identity_without_secret_gets_no_metadata(
    handshake: _Handshake, identity: bytes
):
    """Little-endian rank bytes are not a credential for the join metadata."""
    reply = _exchange(handshake, identity, _unauthenticated("HELLO"))
    assert reply is None
    _assert_still_serving(handshake.server)


def test_same_identity_handover_without_secret_gets_no_metadata(
    handshake: _Handshake,
):
    """A second DEALER that takes over an expected identity is still untrusted."""
    identity = _rank_identity(2)
    held = make_zmq_socket(
        handshake.ctx,
        handshake.server.handshake_address,
        zmq.DEALER,
        identity=identity,
        bind=False,
        linger=0,
    )
    try:
        reply = _exchange(handshake, identity, _unauthenticated("HELLO"))
        assert reply is None
        assert not held.poll(200)
    finally:
        held.close(linger=0)
    _assert_still_serving(handshake.server)


def test_unauthenticated_ready_does_not_complete_new_ranks(handshake: _Handshake):
    """READY from a rank-shaped identity must not retire that rank."""
    for rank in _NEW_RANKS:
        reply = _exchange(handshake, _rank_identity(rank), _unauthenticated("READY"))
        assert reply is None
    _assert_still_serving(handshake.server)


@pytest.mark.parametrize(
    "payload",
    [
        b"not-msgpack",
        msgspec.msgpack.encode(1),
        msgspec.msgpack.encode({"local": False}),
        msgspec.msgpack.encode({"status": "GOODBYE", "dp_rank": 2}),
        [_unauthenticated("HELLO"), b"extra"],
    ],
)
def test_malformed_peer_does_not_abort_listener(handshake: _Handshake, payload):
    """Bad frames are dropped. They must not record a listener error."""
    reply = _exchange(handshake, _rank_identity(2), payload)
    assert reply is None
    _assert_still_serving(handshake.server)


def test_wrong_secret_gets_no_metadata_and_does_not_ready(handshake: _Handshake):
    """A stand-in secret must not fetch metadata or retire a rank."""
    for rank in _NEW_RANKS:
        for status in ("HELLO", "READY"):
            reply = _exchange(
                handshake,
                _rank_identity(rank),
                _authenticated(status, rank, _UNKNOWN_SECRET),
            )
            assert reply is None
    _assert_still_serving(handshake.server)


def test_secret_does_not_authorize_a_different_rank(handshake: _Handshake):
    """Join metadata follows the authenticated dp_rank, not the ZMQ identity.

    Identity bytes of an expected rank with a body for some other rank must
    not be answered. Identity bytes of an unexpected rank with a body for an
    expected rank must be answered, because the identity is not the rank.
    """
    secret = handshake.secret()
    refused = _exchange(
        handshake,
        _rank_identity(2),
        _authenticated("HELLO", 0, secret),
    )
    assert refused is None
    _assert_still_serving(handshake.server)

    accepted = _exchange(
        handshake,
        _rank_identity(0),
        _authenticated("HELLO", 2, secret),
    )
    assert accepted is not None
    _assert_join_metadata(accepted, handshake, secret)
    _assert_still_serving(handshake.server)


def test_ready_follows_authenticated_rank_not_identity(handshake: _Handshake):
    """Retiring a rank uses dp_rank from the message that held the secret."""
    secret = handshake.secret()
    # Identity says 3, body says 2. Only rank 2 is authorized.
    _exchange(
        handshake,
        _rank_identity(3),
        _authenticated("READY", 2, secret),
    )
    _assert_still_serving(handshake.server)
    # A second READY whose identity is rank 2 still only names rank 2.
    _exchange(
        handshake,
        _rank_identity(2),
        _authenticated("READY", 2, secret),
    )
    _assert_still_serving(handshake.server)
    _exchange(
        handshake,
        b"not-a-rank",
        _authenticated("READY", 3, secret),
    )
    _assert_finished(handshake.server)


def test_attacks_then_authorized_ranks_still_complete(handshake: _Handshake):
    """Dropped peers leave the handshake able to finish for real new ranks."""
    attacks: list[tuple[bytes, bytes | list[bytes]]] = [
        (_rank_identity(0), _unauthenticated("HELLO")),
        (b"not-a-rank", _unauthenticated("READY")),
        (_rank_identity(2), b"not-msgpack"),
        (_rank_identity(2), [_unauthenticated("HELLO"), b"extra"]),
        (_rank_identity(3), _unauthenticated("HELLO")),
        (_rank_identity(2, width=1), _unauthenticated("HELLO")),
        (
            _rank_identity(2),
            _authenticated("HELLO", 2, _UNKNOWN_SECRET),
        ),
    ]
    for identity, payload in attacks:
        assert _exchange(handshake, identity, payload) is None
    _assert_still_serving(handshake.server)

    secret = handshake.secret()
    first = _exchange(
        handshake,
        b"peer-one",
        _authenticated("HELLO", 2, secret),
    )
    assert first is not None
    _assert_join_metadata(first, handshake, secret)
    assert _exchange(handshake, b"peer-one", _authenticated("READY", 2, secret)) is None
    _assert_still_serving(handshake.server)

    second = _exchange(
        handshake,
        _rank_identity(2),
        _authenticated("HELLO", 3, secret),
    )
    assert second is not None
    _assert_join_metadata(second, handshake, secret)
    assert (
        _exchange(
            handshake,
            _rank_identity(99),
            _authenticated("READY", 3, secret),
        )
        is None
    )
    _assert_finished(handshake.server)
