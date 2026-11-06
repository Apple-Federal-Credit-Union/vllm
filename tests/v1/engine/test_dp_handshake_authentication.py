# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Authentication contract for the data-parallel startup handshakes.

These tests exercise the TCP ROUTER used when a rank is remote. They fail
while an unauthenticated peer can abort startup or read frontend addresses
and rendezvous fields.

Contract the servers must implement:

* The cluster secret is the process environment variable
  ``VLLM_DATA_PARALLEL_HANDSHAKE_SECRET``, read when each message is
  handled. An unset or empty value authenticates nobody.
* A peer is authenticated only when the payload is a msgpack map whose
  ``handshake_secret`` value equals that variable.
* ``wait_for_engine_startup`` selects the engine by the exact two-byte
  little-endian ZMQ identity (``rank.to_bytes(2, "little")``). A shorter or
  longer identity that happens to decode to the same integer is a different
  peer. Payload fields such as ``data_parallel_rank`` do not select it.
* Messages that are not authenticated, not an expected identity, not exactly
  one payload frame, or not a handshake map are ignored. They must not raise,
  must not change engine state, and must not be answered with frontend ZMQ
  addresses or rendezvous fields.
* An authenticated ``HELLO`` for an expected identity in ``NEW`` state is
  answered with that engine's ``EngineHandshakeMetadata``. An authenticated
  ``READY`` then advances that same engine. ``READY`` before ``HELLO`` does
  not mark it ready.
* ``ExternalElasticEPScaleUpHandshakeServer`` uses the same secret and the
  same exact identity rule. Only an authenticated ``READY`` from an expected
  rank clears that rank. An authenticated ``HELLO`` is what returns
  ``ExternalElasticEPScaleUpHandshakeMetadata``.

``rust/src/engine-core-client/src/transport.rs`` ``connect_handshake`` accepts
any dealer identity and returns input and output addresses with no secret.
That path is not covered here.
"""

import socket
import threading
import time

import msgspec
import pytest
import zmq

from vllm.config import CacheConfig, ParallelConfig
from vllm.distributed.elastic_ep.external_elastic_ep import (
    ExternalElasticEPScaleUpHandshakeMetadata,
    ExternalElasticEPScaleUpHandshakeServer,
)
from vllm.utils.network_utils import make_zmq_socket, zmq_socket_ctx
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.engine.utils import (
    CoreEngine,
    CoreEngineLaunch,
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    wait_for_engine_startup,
)
from vllm.v1.utils import get_engine_client_zmq_addr

pytestmark = pytest.mark.skip_global_cleanup

HANDSHAKE_SECRET_ENV = "VLLM_DATA_PARALLEL_HANDSHAKE_SECRET"
HANDSHAKE_SECRET_FIELD = "handshake_secret"
PLACEHOLDER_HANDSHAKE_SECRET = "placeholder-dp-handshake-secret"

INPUT_ADDRESS = "tcp://198.51.100.10:39991"
OUTPUT_ADDRESS = "tcp://198.51.100.10:39995"
COORDINATOR_INPUT = "tcp://198.51.100.10:39996"
COORDINATOR_OUTPUT = "tcp://198.51.100.10:39997"
STATS_ADDRESS = "tcp://198.51.100.10:39998"
MASTER_IP = "198.51.100.20"
MASTER_PORT = 39990
PORT_LIST = [39993, 39994]
COORD_STORE_PORT = 39992
DP_SIZE = 4
REDUNDANT_EXPERTS = 17011
ELASTIC_MAX_DP_SIZE = 17013

STRING_MARKERS = (
    INPUT_ADDRESS,
    OUTPUT_ADDRESS,
    COORDINATOR_INPUT,
    COORDINATOR_OUTPUT,
    STATS_ADDRESS,
    MASTER_IP,
)
INT_MARKERS = (MASTER_PORT, COORD_STORE_PORT, *PORT_LIST, REDUNDANT_EXPERTS)

_REPLY_WAIT_S = 0.5
_READY_WAIT_S = 2.0


def _rank_identity(rank: int) -> bytes:
    return rank.to_bytes(2, "little")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _handshake_message(
    status: str,
    *,
    local: bool,
    headless: bool,
    secret: str | None = None,
    data_parallel_rank: int | None = None,
) -> bytes:
    body: dict[str, object] = {
        "status": status,
        "local": local,
        "headless": headless,
    }
    if secret is not None:
        body[HANDSHAKE_SECRET_FIELD] = secret
    if data_parallel_rank is not None:
        body["data_parallel_rank"] = data_parallel_rank
    return msgspec.msgpack.encode(body)


def _walk(value: object, found: list[object]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            found.append(key)
            _walk(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk(item, found)
    else:
        found.append(value)


def _discloses_cluster_internals(payload: bytes | None) -> bool:
    """True when a handshake reply carries endpoints or rendezvous fields."""
    if payload is None:
        return False
    if any(marker.encode() in payload for marker in STRING_MARKERS):
        return True
    try:
        decoded = msgspec.msgpack.decode(payload)
    except msgspec.DecodeError:
        return False
    found: list[object] = []
    _walk(decoded, found)
    return any(marker in found for marker in INT_MARKERS)


def _parallel_config(
    *,
    external_lb: bool,
    hybrid_lb: bool,
) -> ParallelConfig:
    config = ParallelConfig(
        data_parallel_size=DP_SIZE,
        data_parallel_size_local=0,
        data_parallel_rank=0,
        data_parallel_master_ip=MASTER_IP,
        data_parallel_external_lb=external_lb,
        data_parallel_hybrid_lb=hybrid_lb,
    )
    config.data_parallel_master_port = MASTER_PORT
    config._data_parallel_master_port_list = list(PORT_LIST)
    config._coord_store_port = COORD_STORE_PORT
    return config


def _addresses() -> EngineZmqAddresses:
    return EngineZmqAddresses(
        inputs=[INPUT_ADDRESS],
        outputs=[OUTPUT_ADDRESS],
        coordinator_input=COORDINATOR_INPUT,
        coordinator_output=COORDINATOR_OUTPUT,
        frontend_stats_publish_address=STATS_ADDRESS,
    )


class _StartupServer:
    """Background ``wait_for_engine_startup`` bound like the DP master."""

    def __init__(
        self,
        *,
        ranks: list[int],
        external_lb: bool,
        hybrid_lb: bool,
        coordinated_dp: bool,
        headless: bool,
    ) -> None:
        self.headless = headless
        self.coordinated_dp = coordinated_dp
        self.address = get_engine_client_zmq_addr(False, "127.0.0.1", _free_port())
        self.parallel_config = _parallel_config(
            external_lb=external_lb,
            hybrid_lb=hybrid_lb,
        )
        self.config_hash = self.parallel_config.compute_hash()
        self.engines = [CoreEngine(index=rank, local=False) for rank in ranks]
        self.launch = CoreEngineLaunch(
            engine_manager=None,
            coordinator=None,
            addresses=_addresses(),
            tensor_queue=None,
        )
        self.error: BaseException | None = None
        self._bound = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._ctx = zmq.Context()
        self._sockets: list[zmq.Socket] = []

    def start(self) -> None:
        self._thread.start()
        assert self._bound.wait(timeout=5), "handshake ROUTER did not bind"

    def _serve(self) -> None:
        try:
            with zmq_socket_ctx(self.address, zmq.ROUTER, bind=True) as sock:
                self._bound.set()
                wait_for_engine_startup(
                    sock,
                    self.engines,
                    self.parallel_config,
                    self.coordinated_dp,
                    CacheConfig(),
                    self.launch,
                )
        except BaseException as exc:
            self.error = exc
            self._bound.set()

    def connect(self, identity: bytes) -> zmq.Socket:
        sock = make_zmq_socket(
            self._ctx,
            self.address,
            zmq.DEALER,
            identity=identity,
            bind=False,
            linger=0,
        )
        self._sockets.append(sock)
        return sock

    def wait_for_reply(self, sock: zmq.Socket, timeout_s: float) -> bytes | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.error is not None or not self._thread.is_alive():
                if sock.poll(0):
                    return sock.recv()
                return None
            if sock.poll(50):
                return sock.recv()
        if sock.poll(0):
            return sock.recv()
        return None

    def complete(
        self,
        peer: zmq.Socket,
        *,
        claimed_rank: int | None = None,
    ) -> bytes:
        """Authenticated HELLO and READY on an already connected dealer."""
        peer.send(
            _handshake_message(
                "HELLO",
                local=False,
                headless=self.headless,
                secret=PLACEHOLDER_HANDSHAKE_SECRET,
                data_parallel_rank=claimed_rank,
            )
        )
        reply = self.wait_for_reply(peer, _READY_WAIT_S)
        assert reply is not None, "authenticated HELLO received no reply"
        peer.send(_ready_message(self.headless, self.config_hash))
        return reply

    def close(self) -> None:
        for sock in self._sockets:
            sock.close(linger=0)
        self._ctx.destroy(linger=0)
        self._thread.join(timeout=2)


def _ready_message(headless: bool, config_hash: str) -> bytes:
    return msgspec.msgpack.encode(
        {
            "status": "READY",
            "local": False,
            "headless": headless,
            HANDSHAKE_SECRET_FIELD: PLACEHOLDER_HANDSHAKE_SECRET,
            "parallel_config_hash": config_hash,
        }
    )


def _assert_ignored(server: _StartupServer, peer: zmq.Socket, stage: str) -> None:
    reply = server.wait_for_reply(peer, _REPLY_WAIT_S)
    assert server.error is None, f"{stage} aborted startup: {server.error!r}"
    assert server._thread.is_alive(), f"{stage} ended startup"
    assert not _discloses_cluster_internals(reply), (
        f"{stage} returned cluster endpoints or rendezvous fields"
    )


def _assert_metadata(
    reply: bytes,
    *,
    coordinated_dp: bool,
) -> None:
    metadata = msgspec.msgpack.decode(reply, type=EngineHandshakeMetadata)
    assert metadata.addresses.inputs == [INPUT_ADDRESS]
    assert metadata.addresses.outputs == [OUTPUT_ADDRESS]
    assert metadata.addresses.coordinator_input == COORDINATOR_INPUT
    assert metadata.addresses.coordinator_output == COORDINATOR_OUTPUT
    assert metadata.addresses.frontend_stats_publish_address == STATS_ADDRESS
    if coordinated_dp:
        assert metadata.parallel_config == {
            "data_parallel_master_ip": MASTER_IP,
            "data_parallel_master_port": MASTER_PORT,
            "_data_parallel_master_port_list": PORT_LIST,
            "_coord_store_port": COORD_STORE_PORT,
            "data_parallel_size": DP_SIZE,
        }
    else:
        assert metadata.parallel_config == {}


@pytest.mark.parametrize(
    "coordinated_dp",
    [True, False],
    ids=["coordinated", "addresses-only"],
)
@pytest.mark.parametrize(
    ("external_lb", "hybrid_lb", "headless"),
    [
        pytest.param(False, False, True, id="internal-lb"),
        pytest.param(True, False, False, id="external-lb"),
        pytest.param(False, True, False, id="hybrid-lb"),
    ],
)
def test_unauthenticated_dp_handshake_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    coordinated_dp: bool,
    external_lb: bool,
    hybrid_lb: bool,
    headless: bool,
) -> None:
    """A remote dealer cannot abort startup or read the handshake reply."""
    monkeypatch.setenv(HANDSHAKE_SECRET_ENV, PLACEHOLDER_HANDSHAKE_SECRET)
    rank = 1
    server = _StartupServer(
        ranks=[rank],
        external_lb=external_lb,
        hybrid_lb=hybrid_lb,
        coordinated_dp=coordinated_dp,
        headless=headless,
    )
    identity = _rank_identity(rank)
    try:
        server.start()
        unknown = server.connect(b"\xff\xfe")
        unknown.send(b"not-a-handshake")
        _assert_ignored(
            server,
            unknown,
            "unknown identity with a non-handshake payload",
        )
        unknown.send(
            _handshake_message(
                "HELLO",
                local=False,
                headless=headless,
                secret=PLACEHOLDER_HANDSHAKE_SECRET,
            )
        )
        _assert_ignored(
            server,
            unknown,
            "unknown identity presenting the cluster secret",
        )

        truncated = server.connect(bytes([rank]))
        truncated.send(
            _handshake_message(
                "HELLO",
                local=False,
                headless=headless,
                secret=PLACEHOLDER_HANDSHAKE_SECRET,
            )
        )
        _assert_ignored(
            server,
            truncated,
            "identity whose integer value matches the rank but whose bytes do not",
        )

        peer = server.connect(identity)
        peer.send(b"not-a-handshake")
        _assert_ignored(server, peer, "expected rank with a non-handshake payload")
        peer.send_multipart(
            [
                _handshake_message("HELLO", local=False, headless=headless),
                b"trailing",
            ]
        )
        _assert_ignored(server, peer, "expected rank with an extra ZMQ frame")
        peer.send(_handshake_message("HELLO", local=False, headless=headless))
        _assert_ignored(server, peer, "HELLO without the cluster secret")
        peer.send(
            _handshake_message(
                "HELLO",
                local=False,
                headless=headless,
                secret="placeholder-not-the-secret",
            )
        )
        _assert_ignored(server, peer, "HELLO with the wrong cluster secret")
        peer.send(_ready_message(headless, server.config_hash))
        _assert_ignored(server, peer, "READY before an authenticated HELLO")

        reply = server.complete(peer)
        _assert_metadata(reply, coordinated_dp=coordinated_dp)
        server._thread.join(timeout=_READY_WAIT_S)
        assert server.error is None, f"legitimate handshake failed: {server.error!r}"
        assert not server._thread.is_alive()
    finally:
        server.close()


def test_authenticated_hello_updates_only_that_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine that presented the secret is the engine the reply advances."""
    monkeypatch.setenv(HANDSHAKE_SECRET_ENV, PLACEHOLDER_HANDSHAKE_SECRET)
    server = _StartupServer(
        ranks=[1, 2],
        external_lb=False,
        hybrid_lb=False,
        coordinated_dp=True,
        headless=True,
    )
    try:
        server.start()
        first_peer = server.connect(_rank_identity(1))
        first = server.complete(first_peer, claimed_rank=2)
        _assert_metadata(first, coordinated_dp=True)

        second_peer = server.connect(_rank_identity(2))
        second_peer.send(_handshake_message("HELLO", local=False, headless=True))
        _assert_ignored(server, second_peer, "unauthenticated second rank")
        second = server.complete(second_peer)
        _assert_metadata(second, coordinated_dp=True)
        server._thread.join(timeout=_READY_WAIT_S)
        assert server.error is None, f"second rank failed: {server.error!r}"
        assert not server._thread.is_alive()
    finally:
        server.close()


def test_missing_cluster_secret_discloses_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no cluster secret configured, a well-formed HELLO learns nothing."""
    monkeypatch.delenv(HANDSHAKE_SECRET_ENV, raising=False)
    server = _StartupServer(
        ranks=[1],
        external_lb=False,
        hybrid_lb=False,
        coordinated_dp=True,
        headless=True,
    )
    try:
        server.start()
        peer = server.connect(_rank_identity(1))
        peer.send(
            _handshake_message(
                "HELLO",
                local=False,
                headless=True,
                secret=PLACEHOLDER_HANDSHAKE_SECRET,
            )
        )
        _assert_ignored(server, peer, "HELLO while the cluster secret is unset")
    finally:
        server.close()


def _elastic_bootstrap() -> ReconfigureDistributedRequest:
    return ReconfigureDistributedRequest(
        new_data_parallel_size=DP_SIZE,
        new_data_parallel_rank=0,
        new_data_parallel_rank_local=0,
        new_data_parallel_master_ip=MASTER_IP,
        new_data_parallel_master_port=MASTER_PORT,
        new_data_parallel_master_port_list=list(PORT_LIST),
        coord_store_port=COORD_STORE_PORT,
    )


def _stop_elastic(server: ExternalElasticEPScaleUpHandshakeServer) -> None:
    server._stop_event.set()
    server._thread.join(timeout=5)


def test_elastic_scale_up_handshake_requires_exact_authenticated_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External elastic scale-up uses the same unauthenticated ROUTER."""
    monkeypatch.setenv(HANDSHAKE_SECRET_ENV, PLACEHOLDER_HANDSHAKE_SECRET)
    rank = 2
    address = get_engine_client_zmq_addr(False, "127.0.0.1", _free_port())
    server = ExternalElasticEPScaleUpHandshakeServer(
        handshake_address=address,
        expected_new_ranks=[rank],
        addresses=_addresses(),
        bootstrap=_elastic_bootstrap(),
        num_redundant_experts=REDUNDANT_EXPERTS,
        elastic_ep_max_dp_size=ELASTIC_MAX_DP_SIZE,
    )
    ctx = zmq.Context()
    try:
        server.start()
        unknown = make_zmq_socket(
            ctx,
            address,
            zmq.DEALER,
            identity=b"\xff\xfe",
            bind=False,
            linger=0,
        )
        unknown.send(b"not-a-handshake")
        assert _elastic_reply(server, unknown) is None
        assert server._error is None
        assert server._thread.is_alive()

        truncated = make_zmq_socket(
            ctx,
            address,
            zmq.DEALER,
            identity=bytes([rank]),
            bind=False,
            linger=0,
        )
        truncated.send(
            _handshake_message(
                "HELLO",
                local=False,
                headless=True,
                secret=PLACEHOLDER_HANDSHAKE_SECRET,
            )
        )
        truncated_reply = _elastic_reply(server, truncated)
        assert server._error is None
        assert server._thread.is_alive()
        assert not _discloses_cluster_internals(truncated_reply)

        peer = make_zmq_socket(
            ctx,
            address,
            zmq.DEALER,
            identity=_rank_identity(rank),
            bind=False,
            linger=0,
        )
        peer.send(_handshake_message("HELLO", local=False, headless=True))
        assert not _discloses_cluster_internals(_elastic_reply(server, peer))
        peer.send(
            _handshake_message(
                "READY",
                local=False,
                headless=True,
                secret=PLACEHOLDER_HANDSHAKE_SECRET,
            )
        )
        assert _elastic_reply(server, peer) is None
        assert server._error is None
        assert server._thread.is_alive()

        peer.send(
            _handshake_message(
                "HELLO",
                local=False,
                headless=True,
                secret=PLACEHOLDER_HANDSHAKE_SECRET,
            )
        )
        reply = _elastic_reply(server, peer)
        assert reply is not None
        metadata = msgspec.msgpack.decode(
            reply,
            type=ExternalElasticEPScaleUpHandshakeMetadata,
        )
        assert metadata.num_redundant_experts == REDUNDANT_EXPERTS
        assert metadata.elastic_ep_max_dp_size == ELASTIC_MAX_DP_SIZE
        assert metadata.engine_metadata.addresses.inputs == [INPUT_ADDRESS]
        assert metadata.engine_metadata.addresses.outputs == [OUTPUT_ADDRESS]
        assert metadata.engine_metadata.parallel_config == {
            "data_parallel_master_ip": MASTER_IP,
            "data_parallel_master_port": MASTER_PORT,
            "_data_parallel_master_port_list": PORT_LIST,
            "data_parallel_size": DP_SIZE,
            "_coord_store_port": COORD_STORE_PORT,
        }
        peer.send(
            _handshake_message(
                "READY",
                local=False,
                headless=True,
                secret=PLACEHOLDER_HANDSHAKE_SECRET,
            )
        )
        server._thread.join(timeout=_READY_WAIT_S)
        assert server._error is None
        assert not server._thread.is_alive()
    finally:
        ctx.destroy(linger=0)
        _stop_elastic(server)


def _elastic_reply(
    server: ExternalElasticEPScaleUpHandshakeServer,
    sock: zmq.Socket,
) -> bytes | None:
    deadline = time.monotonic() + _REPLY_WAIT_S
    while time.monotonic() < deadline:
        if server._error is not None or not server._thread.is_alive():
            if sock.poll(0):
                return sock.recv()
            return None
        if sock.poll(50):
            return sock.recv()
    if sock.poll(0):
        return sock.recv()
    return None
