# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache events must not give prompt token ids to an unauthenticated peer.

Enabling KV cache events selects the ZMQ publisher. These tests exercise that
publisher, the block-pool and offload producers that put prompt token ids into
``BlockStored``, and ``GET /kv_event_sources`` behind the API-key middleware.

They fail while an unauthenticated subscriber can read those token ids on a
wildcard TCP socket, or while ``/kv_event_sources`` answers without the API key.

Producers that also copy prompt token ids into ``BlockStored`` but are not
invoked here, because they need a running transfer runtime, still go out
through the same publisher once the scheduler drains them:

- ``vllm/v1/simple_kv_offload/manager.py``
- ``vllm/distributed/kv_transfer/kv_connector/v1/umbp/scheduler.py``
- ``vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py``
- ``vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py``

``tcp://[::]:port`` is the IPv6 form of the same bind. On this host
``ZmqEventPublisher`` raises ``zmq.ZMQError: No such device`` for that
address, so the IPv6 bind is not exercised. IPv4 uses the same ``_bind``.
"""

import errno
import socket
import time
from collections.abc import Iterable

import msgspec.msgpack
import pytest
import torch
import zmq
from starlette.testclient import TestClient

from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    BlockStored,
    EventPublisherFactory,
    KVEventBatch,
    ZmqEventPublisher,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
)
from vllm.entrypoints.common.cli_args import FrontendArgs
from vllm.entrypoints.launchers.app import build_app
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpecKind,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.kv_offload.base import (
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    make_offload_key,
)
from vllm.v1.request import Request

# Distinct from any hash, port, or sequence number the publisher adds.
_BLOCK_TOKENS = [0x13570001, 0x13570002, 0x13570003, 0x13570004]
_LOOKAHEAD_TOKENS = [0x13570011, 0x13570012]
_PARTIAL_TOKENS = [0x24680001, 0x24680002, 0x24680003, 0x24680004]
_OFFLOAD_TOKENS = [0x35790001, 0x35790002, 0x35790003, 0x35790004]
_PROMPT_SECRETS = (
    _BLOCK_TOKENS,
    _LOOKAHEAD_TOKENS,
    _PARTIAL_TOKENS,
    _OFFLOAD_TOKENS,
)
_PLACEHOLDER_API_KEY = "placeholder-api-key"
_PRIVATE_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

pytestmark = pytest.mark.skip_global_cleanup


def _request(
    request_id: str,
    prompt_token_ids: list[int],
    hash_block_size: int,
    *,
    report_full: bool = False,
) -> Request:
    extra_args = {"kv_cache_report_mode": "full"} if report_full else None
    sampling_params = SamplingParams(max_tokens=1, extra_args=extra_args)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
        session_id=f"session-{request_id}",
    )


def _manager(block_size: int, num_hash_lookahead_tokens: int) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch_float32(),
                ),
            )
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=128,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
        enable_kv_cache_events=True,
        num_hash_lookahead_tokens=num_hash_lookahead_tokens,
    )


def torch_float32():
    return torch.float32


def _prompt_events() -> list:
    """Build BlockStored events from the real cache and offload producers."""
    init_none_hash(sha256)
    block_size = len(_BLOCK_TOKENS)
    manager = _manager(block_size, num_hash_lookahead_tokens=len(_LOOKAHEAD_TOKENS))
    prompt = _BLOCK_TOKENS + _LOOKAHEAD_TOKENS
    first = _request("store", prompt, block_size)
    allocated = manager.allocate_slots(first, len(prompt))
    assert allocated is not None
    events = list(manager.take_events())

    reused = _request("reuse", prompt, block_size, report_full=True)
    _computed, num_computed, _boundary = manager.get_computed_blocks(reused)
    assert num_computed >= block_size
    events.extend(manager.take_events())

    partial_block_size = 8
    partial_prompt = [0x24680011, 0x24680012, 0x24680013, 0x24680014, *_PARTIAL_TOKENS]
    partial_request = _request("partial", partial_prompt, hash_block_size=4)
    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=4,
        enable_kv_cache_events=True,
    )
    pool.cache_partial_block(
        request=partial_request,
        block=pool.get_new_blocks(1)[0],
        num_tokens=len(partial_prompt),
        kv_cache_group_id=0,
        block_size=partial_block_size,
    )
    events.extend(pool.take_events())

    offload_request = _request("offload", list(_OFFLOAD_TOKENS), block_size)
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch_float32(),
    )
    manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
    assert manager_cls is not None
    group = GroupOffloadConfig(
        group_idx=0,
        tokens_per_block=block_size,
        tokens_per_chunk=block_size,
        hashes_per_chunk=1,
        kv_event_group_spec=OffloadingEventGroupSpec(
            kv_cache_spec_kind=KVCacheSpecKind.FULL_ATTENTION.value,
            kv_cache_spec_sliding_window=None,
        ),
        sliding_window_size_in_chunks=None,
        kv_cache_spec=spec,
        manager_cls=manager_cls,
    )
    tracker = OffloadingEventsTracker(
        OffloadingKVEventsConfig(
            enable_kv_cache_events=True,
            self_describing_kv_events=True,
        )
    )
    tail_hash = offload_request.block_hashes[0]
    assert tail_hash is not None
    key = make_offload_key(tail_hash, group.group_idx)
    tracker.record_store(offload_request, group, 0, key)
    events.extend(
        tracker.take_events(
            [
                OffloadingEvent(
                    keys=[key],
                    medium=Medium.CPU,
                    removed=False,
                )
            ]
        )
    )
    stored = [event for event in events if isinstance(event, BlockStored)]
    assert stored, "producers emitted no BlockStored events"
    return events


def _batch(events: list) -> KVEventBatch:
    return KVEventBatch(ts=time.time(), events=events)


def _tcp_host_port(endpoint: str) -> tuple[str, int] | None:
    if "://" not in endpoint:
        return None
    scheme, rest = endpoint.split("://", 1)
    if scheme != "tcp":
        return None
    if rest.startswith("["):
        host, _, port = rest[1:].rpartition("]:")
        return host, int(port)
    host, _, port = rest.rpartition(":")
    return host, int(port)


def _is_private_endpoint(endpoint: str) -> bool:
    if endpoint.startswith(("ipc://", "inproc://")):
        return True
    parsed = _tcp_host_port(endpoint)
    return parsed is not None and parsed[0] in _PRIVATE_HOSTS


def _non_loopback_ipv4() -> list[str]:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))
        except OSError:
            return []
        address = probe.getsockname()[0]
    if address.startswith("127."):
        return []
    return [address]


def _attack_endpoints(endpoint: str) -> list[str]:
    """TCP URLs an off-host client can open for this publisher endpoint."""
    parsed = _tcp_host_port(endpoint)
    if parsed is None:
        return []
    host, port = parsed
    if host in {"*", "0.0.0.0"}:
        peers = ["127.0.0.1", *_non_loopback_ipv4()]
        return [f"tcp://{peer}:{port}" for peer in peers]
    if host == "::":
        return [f"tcp://[::1]:{port}"]
    if host in _PRIVATE_HOSTS:
        return []
    if ":" in host:
        return [f"tcp://[{host}]:{port}"]
    return [endpoint]


def _wildcard_listen_ports() -> set[int]:
    ports: set[int] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        with open(path, encoding="ascii") as proc:
            next(proc)
            for line in proc:
                fields = line.split()
                local, state = fields[1], fields[3]
                if state != "0A":
                    continue
                address, port_hex = local.split(":")
                if set(address) <= {"0"}:
                    ports.add(int(port_hex, 16))
    return ports


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _msgpack_uint(value: int) -> bytes:
    if value <= 0x7F:
        return bytes((value,))
    if value <= 0xFF:
        return b"\xcc" + bytes((value,))
    if value <= 0xFFFF:
        return b"\xcd" + value.to_bytes(2, "big")
    return b"\xce" + value.to_bytes(4, "big")


def _sequence_in_object(value, secret: list[int]) -> bool:
    if isinstance(value, list):
        if len(value) >= len(secret) and any(
            value[index : index + len(secret)] == secret
            for index in range(len(value) - len(secret) + 1)
        ):
            return True
        return any(_sequence_in_object(item, secret) for item in value)
    if isinstance(value, dict):
        return any(_sequence_in_object(item, secret) for item in value.values())
    if isinstance(value, tuple):
        return _sequence_in_object(list(value), secret)
    return False


def _secrets_in_payload(
    payload: bytes, secrets: Iterable[list[int]]
) -> list[list[int]]:
    found: list[list[int]] = []
    decoded = None
    try:
        decoded = msgspec.msgpack.decode(payload)
    except msgspec.DecodeError:
        decoded = None
    for secret in secrets:
        encoded = b"".join(_msgpack_uint(token) for token in secret)
        in_bytes = encoded in payload
        in_object = decoded is not None and _sequence_in_object(decoded, secret)
        if in_bytes or in_object:
            found.append(secret)
    return found


class _UnauthenticatedSubscriber:
    """Plain ZMQ subscriber. No CURVE keys and no other credentials."""

    def __init__(self, endpoints: list[str], replay_endpoints: list[str]):
        self.ctx = zmq.Context.instance()
        self.sockets: list[tuple[str, zmq.Socket]] = []
        for endpoint in endpoints:
            sock = self.ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.IPV6, int("[" in endpoint))
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.connect(endpoint)
            self.sockets.append((endpoint, sock))
        self.replays: list[tuple[str, zmq.Socket]] = []
        for endpoint in replay_endpoints:
            dealer = self.ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.setsockopt(zmq.IPV6, int("[" in endpoint))
            dealer.connect(endpoint)
            self.replays.append((endpoint, dealer))

    def collect(self, publisher, batch: KVEventBatch) -> list[tuple[str, bytes]]:
        received: list[tuple[str, bytes]] = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            publisher.publish(batch)
            for endpoint, sock in self.sockets:
                if not sock.poll(50):
                    continue
                frames = sock.recv_multipart()
                payload = frames[-1] if frames else b""
                if payload:
                    received.append((endpoint, payload))
            if received:
                return received
        return received

    def replay(self) -> list[tuple[str, bytes]]:
        received: list[tuple[str, bytes]] = []
        for endpoint, dealer in self.replays:
            dealer.send_multipart([b"", (0).to_bytes(8, "big")])
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if not dealer.poll(200):
                    break
                frames = dealer.recv_multipart()
                if frames and frames[0] == b"":
                    frames = frames[1:]
                if len(frames) != 3 or not frames[-1]:
                    break
                received.append((endpoint, frames[-1]))
        return received

    def close(self) -> None:
        for _endpoint, sock in self.sockets:
            sock.close(linger=0)
        for _endpoint, dealer in self.replays:
            dealer.close(linger=0)


def _assert_hidden(payloads: list[tuple[str, bytes]], context: str) -> None:
    leaked: list[tuple[str, list[int]]] = []
    for endpoint, payload in payloads:
        for secret in _secrets_in_payload(payload, _PROMPT_SECRETS):
            leaked.append((endpoint, secret))
    assert not leaked, (
        f"{context} delivered prompt token ids to an unauthenticated subscriber: "
        f"{leaked}"
    )


def _tcp_reachable(endpoint: str) -> bool:
    parsed = _tcp_host_port(endpoint)
    if parsed is None:
        return False
    host, port = parsed
    if host in {"*", "0.0.0.0"}:
        targets = [("127.0.0.1", port), *((ip, port) for ip in _non_loopback_ipv4())]
    elif host == "::":
        targets = [("::1", port, 0, 0)]
    else:
        targets = [(host, port)]
    for target in targets:
        family = socket.AF_INET6 if ":" in target[0] else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            try:
                probe.connect(target)
            except OSError:
                continue
            return True
    return False


def _curve_server(sock: zmq.Socket | None) -> bool:
    if sock is None:
        return False
    try:
        return bool(sock.getsockopt(zmq.CURVE_SERVER))
    except zmq.ZMQError:
        return False


def _assert_observable_when_open(publisher, live, replayed) -> None:
    """A reachable publisher without CURVE must actually deliver a batch.

    Otherwise an empty read would pass even though the socket is still open.
    """
    config = publisher.get_publisher_config()
    if (
        not _is_private_endpoint(config.endpoint)
        and _tcp_reachable(config.endpoint)
        and not _curve_server(getattr(publisher, "_pub", None))
        and not live
    ):
        pytest.fail(f"no batch observed on unauthenticated {config.endpoint}")
    replay_endpoint = config.replay_endpoint
    if (
        replay_endpoint
        and not _is_private_endpoint(replay_endpoint)
        and _tcp_reachable(replay_endpoint)
        and not _curve_server(getattr(publisher, "_replay", None))
        and not replayed
    ):
        pytest.fail(f"no replay observed on unauthenticated {replay_endpoint}")


def _publish_and_collect(publisher, events: list, subscriber):
    batch = _batch(events)
    live = subscriber.collect(publisher, batch)
    replayed = subscriber.replay() if subscriber.replays else []
    if subscriber.replays and not replayed:
        live.extend(subscriber.collect(publisher, batch))
        replayed = subscriber.replay()
    _assert_observable_when_open(publisher, live, replayed)
    return live + replayed


def _open_publisher(config, rank: int = 0):
    before = _wildcard_listen_ports()
    try:
        publisher = EventPublisherFactory.create(config, rank)
    except zmq.ZMQError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise
        after = _wildcard_listen_ports()
        assert after <= before, (
            f"publisher raised {exc!r} after binding wildcard ports "
            f"{sorted(after - before)}"
        )
        return None
    except (OSError, ValueError, RuntimeError) as exc:
        after = _wildcard_listen_ports()
        assert after <= before, (
            f"publisher raised {exc!r} after binding wildcard ports "
            f"{sorted(after - before)}"
        )
        return None
    return publisher


@pytest.fixture(scope="module", autouse=True)
def _hash_seed():
    init_none_hash(sha256)


def test_default_publisher_does_not_expose_prompt_tokens_on_every_interface():
    """Default KV-event config must not publish prompt tokens on INADDR_ANY."""
    events = _prompt_events()
    before = _wildcard_listen_ports()
    config = KVEventsConfig(enable_kv_cache_events=True)
    publisher = EventPublisherFactory.create(config)
    try:
        resolved = publisher.get_publisher_config()
        assert resolved is not None
        endpoints = _attack_endpoints(resolved.endpoint)
        replay_endpoint = resolved.replay_endpoint
        replay_endpoints = _attack_endpoints(replay_endpoint) if replay_endpoint else []
        subscriber = _UnauthenticatedSubscriber(endpoints, replay_endpoints)
        try:
            payloads = _publish_and_collect(publisher, events, subscriber)
            _assert_hidden(payloads, f"default endpoint {resolved.endpoint}")
            assert _is_private_endpoint(resolved.endpoint), resolved.endpoint
            new_wildcard = _wildcard_listen_ports() - before
            assert not new_wildcard, (
                "default publisher is listening on every interface: "
                f"ports {sorted(new_wildcard)}"
            )
        finally:
            subscriber.close()
    finally:
        publisher.shutdown()


def test_explicit_wildcard_publisher_and_replay_hide_prompt_tokens():
    """tcp://*:port publish and replay must not reveal prompt token ids."""
    events = _prompt_events()
    pub_port = _free_tcp_port()
    replay_port = _free_tcp_port()
    config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{pub_port}",
        replay_endpoint=f"tcp://*:{replay_port}",
        buffer_steps=100,
        hwm=1000,
        topic="",
    )
    before = _wildcard_listen_ports()
    publisher = _open_publisher(config)
    if publisher is None:
        assert _wildcard_listen_ports() <= before
        return
    try:
        resolved = publisher.get_publisher_config()
        assert resolved is not None
        subscriber = _UnauthenticatedSubscriber(
            _attack_endpoints(resolved.endpoint),
            _attack_endpoints(resolved.replay_endpoint or ""),
        )
        try:
            payloads = _publish_and_collect(publisher, events, subscriber)
            _assert_hidden(
                payloads,
                "wildcard publish "
                f"{resolved.endpoint} replay {resolved.replay_endpoint}",
            )
        finally:
            subscriber.close()
    finally:
        publisher.shutdown()


def test_data_parallel_rank_offset_keeps_wildcard_unpublished():
    """Data-parallel rank offsets the port and still publishes the same events."""
    events = _prompt_events()
    port = _free_tcp_port()
    config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{port}",
        replay_endpoint=f"tcp://*:{port + 20}",
        buffer_steps=10,
        hwm=100,
        topic="",
    )
    publisher = _open_publisher(config, rank=1)
    if publisher is None:
        return
    try:
        resolved = publisher.get_publisher_config()
        assert resolved is not None
        assert resolved.endpoint.endswith(f":{port + 1}") or _is_private_endpoint(
            resolved.endpoint
        )
        subscriber = _UnauthenticatedSubscriber(
            _attack_endpoints(resolved.endpoint),
            _attack_endpoints(resolved.replay_endpoint or ""),
        )
        try:
            payloads = _publish_and_collect(publisher, events, subscriber)
            _assert_hidden(payloads, f"dp-offset endpoint {resolved.endpoint}")
        finally:
            subscriber.close()
    finally:
        publisher.shutdown()


def test_zmq_publisher_constructor_default_hides_prompt_tokens():
    """ZmqEventPublisher's own default endpoint is the wildcard TCP port."""
    events = _prompt_events()
    before = _wildcard_listen_ports()
    try:
        publisher = ZmqEventPublisher(data_parallel_rank=0)
    except (OSError, zmq.ZMQError, ValueError, RuntimeError) as exc:
        assert _wildcard_listen_ports() <= before, exc
        return
    try:
        resolved = publisher.get_publisher_config()
        subscriber = _UnauthenticatedSubscriber(
            _attack_endpoints(resolved.endpoint),
            _attack_endpoints(resolved.replay_endpoint or ""),
        )
        try:
            payloads = _publish_and_collect(publisher, events, subscriber)
            _assert_hidden(payloads, f"constructor default {resolved.endpoint}")
            assert _is_private_endpoint(resolved.endpoint), resolved.endpoint
            assert not (_wildcard_listen_ports() - before), resolved.endpoint
        finally:
            subscriber.close()
    finally:
        publisher.shutdown()


class _KvEventSourceClient:
    """Same return shape as EngineCoreClient.get_kv_event_sources."""

    def __init__(self, sources):
        self._sources = sources

    def get_kv_event_sources(self):
        return self._sources


def test_kv_event_sources_requires_the_api_key():
    """GET /kv_event_sources must use the same API key as the guarded prefixes."""
    config = KVEventsConfig(enable_kv_cache_events=True)
    publisher = EventPublisherFactory.create(config)
    try:
        sources = publisher.get_publisher_config()
        assert sources is not None
        args = FrontendArgs(api_key=[_PLACEHOLDER_API_KEY])
        app = build_app(args, supported_tasks=())
        app.state.log_stats = False
        app.state.engine_client = _KvEventSourceClient({0: sources})
        with TestClient(app) as client:
            anonymous = client.get("/kv_event_sources")
            assert anonymous.status_code in {401, 403}, anonymous.text
            assert sources.endpoint not in anonymous.text
            for secret in _PROMPT_SECRETS:
                assert not _secrets_in_payload(anonymous.content, [secret])
            wrong = client.get(
                "/kv_event_sources",
                headers={"Authorization": "Bearer wrong-placeholder"},
            )
            assert wrong.status_code in {401, 403}, wrong.text
            assert sources.endpoint not in wrong.text
            guarded = client.get("/v1/models")
            assert guarded.status_code in {401, 403}, guarded.text
    finally:
        publisher.shutdown()
