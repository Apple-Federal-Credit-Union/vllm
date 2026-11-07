# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security spec for the Mooncake lookup IPC channel.

These tests exercise ``LookupKeyServer``. They fail while a local process
that can reach the socket can reset the store, stall later lookups, or
force a mask sized from an attacker-chosen token count.

Full-attention groups are not covered. Their mask helper returns no list, and
the hit scan stops at the hash sequence, so a large token count does not
allocate there.
"""

import contextlib
import os
import queue
import socket
import stat
import tempfile
import threading

import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    protocol,
    worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    MooncakeLookupResult,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    LookupKeyServer,
    MooncakeStoreWorker,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

from .utils import create_vllm_config

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skip_global_cleanup,
]

_REPLY_TIMEOUT_MS = 1000
_PORT = 45000


class _RecordingQueue(queue.Queue):
    """Send-queue stand-in that records ``join`` without waiting."""

    def __init__(self) -> None:
        super().__init__()
        self.join_calls = 0

    def join(self) -> None:
        self.join_calls += 1


class _RecordingStore:
    """In-memory stand-in for ``MooncakeDistributedStore.remove_all``.

    ``remove_all(force=True)`` drops every stored object. That mirrors
    ``MasterService::RemoveAll`` erasing completed objects
    (``mooncake-store/src/master_service.cpp``, ``RemoveTenantObjects``).
    """

    def __init__(self) -> None:
        self.objects = {"tenant/kv-block": b"cached"}
        self.remove_all_calls: list[bool] = []

    def remove_all(self, force: bool = False) -> int:
        self.remove_all_calls.append(force)
        if force:
            self.objects.clear()
        return 0

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [1 if key in self.objects else 0 for key in keys]


class _AdminWorker:
    def __init__(self) -> None:
        self.store = _RecordingStore()
        self.kv_send_thread = type("_SendThread", (), {})()
        self.kv_send_thread.request_queue = _RecordingQueue()

    def lookup(self, num_tokens: int, block_hashes: object) -> MooncakeLookupResult:
        del num_tokens, block_hashes
        return MooncakeLookupResult(0)


class _LookupWorker:
    """Worker whose ``lookup`` is ``MooncakeStoreWorker.lookup``."""

    def __init__(
        self,
        groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
    ) -> None:
        self.store = _RecordingStore()
        self.kv_send_thread = None
        self._capacity_only = False
        self.coord = MooncakeStoreCoordinator(
            groups,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        self._kv_cache_groups = groups
        self.hash_block_size = hash_block_size
        self.token_dbs = []
        self._lookup_key_prefixes = []
        for index, group in enumerate(groups):
            meta = KeyMetadata("m", 0, 0, 0, 0, group_id=index)
            self.token_dbs.append(
                ChunkedTokenDatabase(
                    meta, group.kv_cache_spec.block_size, hash_block_size
                )
            )
            self._lookup_key_prefixes.append(("pfx",))
        self._kv_connector_stats_lock = threading.Lock()
        self.kv_connector_stats = MooncakeStoreConnectorStats()
        self._record_kv_connector_operation = (
            MooncakeStoreWorker._record_kv_connector_operation.__get__(self)
        )
        self._tail_key_boundaries = MooncakeStoreWorker._tail_key_boundaries.__get__(
            self
        )

    def lookup(self, num_tokens: int, block_hashes: object) -> MooncakeLookupResult:
        return MooncakeStoreWorker.lookup(self, num_tokens, block_hashes)


def _next_port() -> int:
    global _PORT
    _PORT += 1
    return _PORT


def _config(max_model_len: int = 4096):
    return create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        max_model_len=max_model_len,
        kv_connector_extra_config={"lookup_rpc_port": _next_port()},
    )


def _predictable_path(config) -> str:
    """Path a local process can build without seeing the server object."""
    extra = config.kv_transfer_config.kv_connector_extra_config
    port = extra.get("lookup_rpc_port", 0)
    dp_rank = config.parallel_config.data_parallel_index
    return (
        f"{tempfile.gettempdir()}/lookup_rpc_port_{port}_host_"
        f"{socket.gethostname()}_dp_rank{dp_rank}"
    )


def _ipc_url(path: str) -> str:
    if path.startswith("ipc://"):
        return path
    return f"ipc://{path}"


def _request(path: str, frames: list[bytes], timeout_ms: int = _REPLY_TIMEOUT_MS):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        sock.connect(_ipc_url(path))
        sock.send_multipart(frames)
        return sock.recv()
    except zmq.Again:
        return None
    finally:
        sock.close(linger=0)
        ctx.term()


def _stop(server: LookupKeyServer) -> None:
    """Unblock the handler, then drop the socket file."""
    server.running = False
    if server.thread.is_alive() and os.path.exists(server._ipc_path):
        _request(server._ipc_path, [b"stop"], timeout_ms=500)
    server.thread.join(timeout=1)
    with contextlib.suppress(Exception):
        server.socket.close(linger=0)
    with contextlib.suppress(Exception):
        server.ctx.term()
    if os.path.exists(server._ipc_path):
        os.unlink(server._ipc_path)


def _hash() -> BlockHash:
    return BlockHash(b"\x11" * 16)


def _lookup_frames(num_tokens: int, width: int = 4) -> list[bytes]:
    return [
        protocol.LOOKUP_MSG,
        num_tokens.to_bytes(width, "big"),
        (16).to_bytes(2, "big"),
        bytes(_hash()),
    ]


def _valid_lookup(path: str):
    return _request(path, _lookup_frames(32))


@pytest.fixture
def admin_server():
    config = _config()
    worker_obj = _AdminWorker()
    server = LookupKeyServer(worker_obj, config)
    try:
        yield config, worker_obj, server
    finally:
        _stop(server)


def test_predictable_path_reset_does_not_wipe_the_store(admin_server):
    """A process that only knows the temp-dir path formula cannot reset."""
    config, worker_obj, server = admin_server
    reply = _request(_predictable_path(config), [protocol.RESET_MSG])

    assert worker_obj.store.remove_all_calls == []
    assert worker_obj.store.objects
    assert worker_obj.kv_send_thread.request_queue.join_calls == 0
    assert server.thread.is_alive()
    assert reply != protocol.RESP_OK
    assert _valid_lookup(server._ipc_path) is not None


def test_bound_socket_rejects_reset_without_scheduler_client(admin_server):
    """Connecting to the socket the server bound is not authorization to wipe."""
    _config_obj, worker_obj, server = admin_server
    reply = _request(server._ipc_path, [protocol.RESET_MSG])

    assert worker_obj.store.remove_all_calls == []
    assert worker_obj.store.objects
    assert worker_obj.kv_send_thread.request_queue.join_calls == 0
    assert server.thread.is_alive()
    assert reply != protocol.RESP_OK
    assert _valid_lookup(server._ipc_path) is not None


def test_placeholder_credential_does_not_select_this_store(admin_server):
    """An extra frame that is not this server's credential must not wipe it."""
    _config_obj, worker_obj, server = admin_server
    _request(
        server._ipc_path,
        [protocol.RESET_MSG, b"placeholder"],
    )

    assert worker_obj.store.remove_all_calls == []
    assert "tenant/kv-block" in worker_obj.store.objects
    assert server.thread.is_alive()


def test_reset_bytes_from_one_server_do_not_wipe_another():
    """Frames accepted for server A must not erase server B's store."""
    config_a = _config()
    config_b = _config()
    victim = _AdminWorker()
    impostor_path = _predictable_path(config_a)
    if os.path.exists(impostor_path):
        os.unlink(impostor_path)

    ctx = zmq.Context()
    impostor = ctx.socket(zmq.REP)
    impostor.setsockopt(zmq.LINGER, 0)
    impostor.bind(_ipc_url(impostor_path))
    captured: list[list[bytes]] = []

    def capture() -> None:
        captured.append(impostor.recv_multipart())
        impostor.send(protocol.RESP_OK)

    thread = threading.Thread(target=capture, daemon=True)
    thread.start()
    client = worker.LookupKeyClient(config_a)
    try:
        client.reset()
        thread.join(timeout=2)
        assert captured, "scheduler client did not send a reset"
        server = LookupKeyServer(victim, config_b)
        try:
            replayed = _request(server._ipc_path, list(captured[0]))
            assert victim.store.remove_all_calls == []
            assert victim.store.objects
            assert replayed != protocol.RESP_OK
        finally:
            _stop(server)
    finally:
        client.close()
        impostor.close(linger=0)
        ctx.term()
        if os.path.exists(impostor_path):
            os.unlink(impostor_path)


def test_scheduler_client_can_reset_its_own_store(admin_server):
    """The scheduler's ``LookupKeyClient`` remains able to reset its worker."""
    config, worker_obj, server = admin_server
    client = worker.LookupKeyClient(config)
    try:
        assert client.reset() is True
    finally:
        client.close()

    assert worker_obj.kv_send_thread.request_queue.join_calls == 1
    assert worker_obj.store.remove_all_calls == [True]
    assert worker_obj.store.objects == {}
    assert server.thread.is_alive()


@pytest.mark.parametrize(
    "frames",
    [
        [protocol.LOOKUP_MSG],
        [protocol.LOOKUP_MSG, (32).to_bytes(4, "big")],
        [
            protocol.LOOKUP_MSG,
            (32).to_bytes(4, "big"),
            (16).to_bytes(2, "big"),
        ],
    ],
)
def test_short_lookup_keeps_the_handler_alive(admin_server, frames):
    """A lookup with too few frames returns an error and stays in the loop."""
    _config_obj, worker_obj, server = admin_server
    _request(server._ipc_path, frames)

    assert server.thread.is_alive()
    assert worker_obj.store.remove_all_calls == []
    assert _valid_lookup(server._ipc_path) is not None


def test_short_lookup_does_not_stall_the_scheduler_client(admin_server):
    """After a truncated lookup, later scheduler lookups still complete."""
    config, _worker_obj, server = admin_server
    _request(server._ipc_path, [protocol.LOOKUP_MSG])
    client = worker.LookupKeyClient(config)
    done = threading.Event()

    def two_lookups() -> None:
        client.lookup("req-a", 32, [_hash()])
        client.lookup("req-b", 32, [_hash()])
        done.set()

    thread = threading.Thread(target=two_lookups, daemon=True)
    thread.start()
    try:
        assert done.wait(2), "scheduler lookup client blocked after a short frame"
        assert server.thread.is_alive()
    finally:
        if thread.is_alive():
            # Closing the dead server socket unblocks the scheduler recv.
            with contextlib.suppress(Exception):
                server.socket.close(linger=0)
            thread.join(timeout=1)
        with contextlib.suppress(Exception):
            client.close()
        with contextlib.suppress(Exception):
            client.ctx.term()


def _hybrid_groups() -> list[KVCacheGroupSpec]:
    full = FullAttentionSpec(block_size=64, num_kv_heads=2, head_size=8, dtype=None)
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=None,
        sliding_window=32,
    )
    return [
        KVCacheGroupSpec(["full"], full),
        KVCacheGroupSpec(["swa"], swa),
    ]


def _swa_groups() -> list[KVCacheGroupSpec]:
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=None,
        sliding_window=128,
    )
    return [KVCacheGroupSpec(["swa"], swa)]


@pytest.mark.parametrize("width", [4, 5])
def test_oversized_lookup_does_not_allocate_a_mask(width):
    """Hybrid sliding-window lookup must not size a mask from the token count."""
    config = _config(max_model_len=4096)
    lookup_worker = _LookupWorker(_hybrid_groups(), 64, 16)
    state: dict[str, list[int]] = {"masks": []}
    original = SlidingWindowManager.reachable_block_mask.__func__

    def record_mask(cls, start_block, end_block, *args, **kwargs):
        result = original(cls, start_block, end_block, *args, **kwargs)
        if result is not None:
            state["masks"].append(len(result))
        return result

    SlidingWindowManager.reachable_block_mask = classmethod(record_mask)
    server = LookupKeyServer(lookup_worker, config)
    try:
        reply = _request(
            server._ipc_path,
            _lookup_frames(200_000 * 16, width),
            timeout_ms=5000,
        )
        cap_blocks = config.model_config.max_model_len // 16
        assert all(length <= cap_blocks for length in state["masks"]), state["masks"]
        assert server.thread.is_alive()
        assert reply is not None
        assert _valid_lookup(server._ipc_path) is not None
    finally:
        SlidingWindowManager.reachable_block_mask = classmethod(original)
        _stop(server)


def test_oversized_sliding_window_lookup_does_not_scan_token_count():
    """A sliding-window hit scan must not be sized from the claimed length."""
    config = _config(max_model_len=4096)
    lookup_worker = _LookupWorker(_swa_groups(), 16, 16)
    state: dict[str, list[int]] = {"hits": []}
    original = SlidingWindowManager.find_longest_cache_hit.__func__

    def record_hit(cls, block_hashes, max_length, *args, **kwargs):
        state["hits"].append(max_length)
        return original(cls, block_hashes, max_length, *args, **kwargs)

    SlidingWindowManager.find_longest_cache_hit = classmethod(record_hit)
    server = LookupKeyServer(lookup_worker, config)
    try:
        reply = _request(
            server._ipc_path,
            _lookup_frames(200_000 * 16),
            timeout_ms=5000,
        )
        assert all(
            length <= config.model_config.max_model_len for length in state["hits"]
        ), state["hits"]
        assert server.thread.is_alive()
        assert reply is not None
        assert _valid_lookup(server._ipc_path) is not None
    finally:
        SlidingWindowManager.find_longest_cache_hit = classmethod(original)
        _stop(server)


def test_lookup_socket_is_private_and_unpredictable(admin_server):
    """The bound socket is not the public temp-dir name and is owner-only."""
    config, _worker_obj, server = admin_server
    bound = server._ipc_path
    predictable = _predictable_path(config)
    mode = stat.S_IMODE(os.stat(bound).st_mode)
    parent_mode = stat.S_IMODE(os.stat(os.path.dirname(bound)).st_mode)

    assert os.path.abspath(bound) != os.path.abspath(predictable)
    assert mode & 0o077 == 0
    assert parent_mode & 0o077 == 0
