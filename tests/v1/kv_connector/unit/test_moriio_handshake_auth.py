# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security contract for the MoRIIO KV handshake.

The handshake that ``register_kv_caches`` starts must bind the configured
internal host and must not answer an anonymous metadata request with the
agent descriptor or the KV memory registration.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgpack
import msgspec
import pytest
import torch
import zmq

from tests.v1.kv_connector.unit.moriio_recording_io import install_recording_mori
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    get_port_offset,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    KVConnectorRole,
    MoRIIOConnector,
    MoRIIOConnectorWorker,
)
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    compute_layer_kv_cache_shape_bytes,
)

# Importing the recording engine installs it as mori.io before the connector
# module binds MoRI. Keep the name so the import is not treated as unused.
_ = install_recording_mori

_CONFIGURED_HOST = "127.0.0.1"
_LAYER_NAMES = ("layer0", "layer1", "layer2")
_NUM_BLOCKS = 2

pytestmark = pytest.mark.skip_global_cleanup


class _NamedBackend:
    """Stand-in for the attention backend name copied into handshake metadata.

    Selecting a kernel backend needs an accelerator. The handshake does not.
    """

    @staticmethod
    def get_name() -> str:
        return "CPU_TEST"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _model_dir() -> tempfile.TemporaryDirectory[str]:
    directory = tempfile.TemporaryDirectory(prefix="moriio-handshake-")
    config = {
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "vocab_size": 32,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "max_position_embeddings": 128,
        "ffn_dim": 64,
        "word_embed_proj_dim": 16,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    Path(directory.name, "config.json").write_text(json.dumps(config))
    return directory


def _vllm_config(
    model_path: str,
    *,
    handshake_port: int | None,
    notify_port: int,
    http_port: int,
) -> VllmConfig:
    extra: dict[str, object] = {
        "host_ip": _CONFIGURED_HOST,
        "notify_port": notify_port,
        "http_port": http_port,
        "backend": "xgmi",
    }
    if handshake_port is not None:
        extra["handshake_port"] = handshake_port
    return VllmConfig(
        model_config=ModelConfig(
            model=model_path,
            tokenizer=model_path,
            dtype="float16",
            seed=1,
            max_model_len=128,
        ),
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=64,
            max_model_len=128,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MoRIIOConnector",
            kv_role="kv_producer",
            kv_connector_extra_config=extra,
        ),
    )


def _kv_cache_config() -> tuple[KVCacheConfig, tuple[int, ...]]:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
    )
    shape = compute_layer_kv_cache_shape_bytes(spec, _NUM_BLOCKS)
    page = spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=_NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=len(_LAYER_NAMES) * _NUM_BLOCKS * page,
                layers=list(_LAYER_NAMES),
                layer_stride=_NUM_BLOCKS * page,
                block_stride=page,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=list(_LAYER_NAMES), kv_cache_spec=spec)
        ],
    )
    return config, shape


def _distinct_caches(shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(*shape, dtype=torch.int8).view(torch.float16)
        for name in _LAYER_NAMES
    }


_SHARED_BUFFERS: list[tuple[object, memoryview, torch.Tensor]] = []


def _page_aligned_uint8(nbytes: int) -> torch.Tensor:
    """Byte tensor whose storage base is 4KiB-aligned.

    ``_build_shared_kv_mr`` registers from the storage base and rejects a
    negative page-alignment offset. GPU allocators satisfy that alignment;
    this allocation does the same for the CPU test.
    """
    libc = ctypes.CDLL(None)
    address = ctypes.c_void_p()
    error = libc.posix_memalign(ctypes.byref(address), 4096, nbytes)
    if error:
        raise MemoryError(error)
    ctypes.memset(address, 0, nbytes)
    view = memoryview((ctypes.c_char * nbytes).from_address(address.value))
    tensor = torch.frombuffer(view, dtype=torch.uint8)
    _SHARED_BUFFERS.append((address, view, tensor))
    return tensor


def _shared_caches(shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
    proto = torch.zeros(*shape, dtype=torch.int8).view(torch.float16)
    # One extra page so the shared registration can round up to 4KiB.
    raw = _page_aligned_uint8(proto.nbytes * len(_LAYER_NAMES) + 4096)
    caches: dict[str, torch.Tensor] = {}
    for index, name in enumerate(_LAYER_NAMES):
        start = index * proto.nbytes
        slab = raw[start : start + proto.nbytes]
        caches[name] = slab.view(proto.dtype).reshape(proto.shape)
    return caches


def _process_socket_inodes() -> set[str]:
    inodes: set[str] = set()
    for entry in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{entry}")
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[len("socket:[") : -1])
    return inodes


def _decode_proc_ip(ip_hex: str) -> str:
    raw = bytes.fromhex(ip_hex)
    if len(raw) == 4:
        return ".".join(str(byte) for byte in raw[::-1])
    words = [raw[index : index + 4][::-1] for index in range(0, 16, 4)]
    return socket.inet_ntop(socket.AF_INET6, b"".join(words))


def _listening_ips(port: int) -> set[str]:
    """Local addresses of this process's TCP listeners on ``port``."""
    inodes = _process_socket_inodes()
    port_hex = f"{port:04X}"
    found: set[str] = set()
    for proc_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        with open(proc_path, encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                parts = line.split()
                local, state, inode = parts[1], parts[3], parts[9]
                ip_hex, port_part = local.split(":")
                if port_part.upper() != port_hex or state != "0A":
                    continue
                if inode not in inodes:
                    continue
                found.add(_decode_proc_ip(ip_hex))
    return found


def _unpack_memory_desc(blob: bytes) -> dict[str, int] | None:
    try:
        fields = msgpack.loads(blob)
    except (
        msgpack.exceptions.ExtraData,
        msgpack.exceptions.UnpackException,
        ValueError,
    ):
        return None
    if not isinstance(fields, list) or len(fields) < 6:
        return None
    data, size = fields[4], fields[5]
    if not isinstance(data, int) or not isinstance(size, int):
        return None
    return {"data": data, "size": size}


def _anonymous_reply(port: int) -> list[list[bytes]]:
    """Send only the metadata request tag. No credential is attached."""
    path = make_zmq_path("tcp", _CONFIGURED_HOST, port)
    messages: list[list[bytes]] = []
    with zmq_ctx(zmq.DEALER, path) as sock:
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.send(MoRIIOConstants.GET_META_MSG)
        while len(messages) < 2:
            try:
                messages.append(list(sock.recv_multipart()))
            except zmq.Again:
                break
    return messages


def _disclosure(
    messages: list[list[bytes]], caches: dict[str, torch.Tensor]
) -> str | None:
    base_ptrs = [int(cache.data_ptr()) for cache in caches.values()]
    full_bytes = {
        name: cache.nelement() * cache.element_size() for name, cache in caches.items()
    }
    one_block = min(full_bytes.values()) // _NUM_BLOCKS
    lines: list[str] = []
    for message in messages:
        payload = message[-1] if message else b""
        try:
            metadata = msgspec.msgpack.Decoder(MoRIIOAgentMetadata).decode(payload)
        except msgspec.DecodeError:
            metadata = None
        if isinstance(metadata, MoRIIOAgentMetadata):
            lines.append(
                "agent metadata "
                f"engine_id={metadata.engine_id} "
                f"kv_caches_base_addr={list(metadata.kv_caches_base_addr)} "
                f"tensor_data_ptrs={base_ptrs} "
                f"num_blocks={metadata.num_blocks} block_len={metadata.block_len} "
                f"one_block_bytes={one_block}"
            )
        try:
            decoded = msgpack.loads(payload)
        except (
            msgpack.exceptions.ExtraData,
            msgpack.exceptions.UnpackException,
            ValueError,
        ):
            decoded = None
        if not isinstance(decoded, dict):
            continue
        for layer_name, blobs in decoded.items():
            if not isinstance(blobs, list):
                continue
            for blob in blobs:
                if not isinstance(blob, bytes | bytearray):
                    continue
                registration = _unpack_memory_desc(bytes(blob))
                if registration is None:
                    continue
                lines.append(
                    f"layer {layer_name} registration data={registration['data']} "
                    f"size={registration['size']} "
                    f"full_tensor_bytes={full_bytes.get(str(layer_name))} "
                    f"one_block_bytes={one_block}"
                )
    if not lines:
        return None
    return "anonymous GET_META_MSG received the KV registration:\n" + "\n".join(lines)


def _handshake_problems(port: int, caches: dict[str, torch.Tensor]) -> list[str]:
    problems: list[str] = []
    listeners = _listening_ips(port)
    wildcard = {"0.0.0.0", "::"}
    if listeners & wildcard:
        problems.append(
            f"handshake port {port} listens on {sorted(listeners & wildcard)}"
        )
    if _CONFIGURED_HOST not in listeners:
        problems.append(
            f"handshake port {port} did not bind {_CONFIGURED_HOST}; "
            f"listeners={sorted(listeners)}"
        )
    disclosed = _disclosure(_anonymous_reply(port), caches)
    if disclosed is not None:
        problems.append(disclosed)
    return problems


def _release(worker: MoRIIOConnectorWorker) -> None:
    for port in (worker.side_channel_port, worker.notify_port):
        path = make_zmq_path("tcp", _CONFIGURED_HOST, int(port))
        try:
            with zmq_ctx(zmq.DEALER, path) as sock:
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.SNDTIMEO, 500)
                sock.send(b"\x00")
        except zmq.ZMQError:
            pass
    listener = worker._moriio_handshake_listener_t
    if listener is not None:
        listener.join(timeout=2)
    notify = getattr(worker.moriio_wrapper, "notify_thread", None)
    if notify is not None:
        notify.join(timeout=2)
    worker.shutdown()


def _start_worker(
    *,
    handshake_port: int | None,
    tp_rank: int,
    dp_rank: int,
    shared_backing: bool,
) -> tuple[
    tempfile.TemporaryDirectory[str],
    MoRIIOConnector,
    dict[str, torch.Tensor],
]:
    model_dir = _model_dir()
    notify_port = _free_port()
    http_port = _free_port()
    vllm_config = _vllm_config(
        model_dir.name,
        handshake_port=handshake_port,
        notify_port=notify_port,
        http_port=http_port,
    )
    vllm_config.parallel_config.data_parallel_size = 4
    vllm_config.parallel_config.data_parallel_size_local = 4
    vllm_config.parallel_config.data_parallel_rank = dp_rank
    kv_cache_config, shape = _kv_cache_config()
    caches = _shared_caches(shape) if shared_backing else _distinct_caches(shape)
    group = SimpleNamespace(
        rank=tp_rank, local_rank=tp_rank, world_size=4, rank_in_group=tp_rank
    )
    patches = patch.multiple(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common",
        get_tensor_model_parallel_rank=lambda: tp_rank,
        get_tensor_model_parallel_world_size=lambda: 4,
    )
    connector_patches = patch.multiple(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector",
        get_tensor_model_parallel_world_size=lambda: 4,
        get_world_group=lambda: group,
        get_tp_group=lambda: group,
        get_current_attn_backends=lambda *_args, **_kwargs: [_NamedBackend],
    )
    patches.start()
    connector_patches.start()
    try:
        with set_current_vllm_config(vllm_config):
            connector = MoRIIOConnector(
                vllm_config, KVConnectorRole.WORKER, kv_cache_config
            )
            connector.register_kv_caches(caches)
    except Exception:
        patches.stop()
        connector_patches.stop()
        model_dir.cleanup()
        raise
    connector._handshake_patches = (patches, connector_patches)  # type: ignore[attr-defined]
    return model_dir, connector, caches


def _stop_worker(
    model_dir: tempfile.TemporaryDirectory[str], connector: MoRIIOConnector
) -> None:
    worker = connector.connector_worker
    assert worker is not None
    try:
        _release(worker)
    finally:
        for active in getattr(connector, "_handshake_patches", ()):
            active.stop()
        model_dir.cleanup()


@pytest.fixture
def distinct_worker():
    started = _start_worker(
        handshake_port=_free_port(),
        tp_rank=0,
        dp_rank=0,
        shared_backing=False,
    )
    try:
        yield started
    finally:
        _stop_worker(started[0], started[1])


@pytest.fixture
def shared_worker():
    started = _start_worker(
        handshake_port=_free_port(),
        tp_rank=0,
        dp_rank=0,
        shared_backing=True,
    )
    try:
        yield started
    finally:
        _stop_worker(started[0], started[1])


def test_anonymous_handshake_does_not_export_distinct_kv_registration(distinct_worker):
    """An unnamed peer on the handshake port learns nothing about the cache.

    Each layer is its own tensor. The registration that would be exported is
    that tensor's base address and full byte length, which spans every block.
    """
    _model_dir, connector, caches = distinct_worker
    worker = connector.connector_worker
    assert worker is not None
    problems = _handshake_problems(worker.side_channel_port, caches)
    assert not problems, "\n".join(problems)


def test_anonymous_handshake_does_not_export_shared_kv_registration(shared_worker):
    """Same contract when every layer is a view of one backing allocation.

    The shared registration covers that allocation, and the advertised base
    addresses are the per-layer pointers inside it.
    """
    _model_dir, connector, caches = shared_worker
    worker = connector.connector_worker
    assert worker is not None
    storages = {id(cache.untyped_storage()) for cache in caches.values()}
    assert len(storages) == 1
    problems = _handshake_problems(worker.side_channel_port, caches)
    assert not problems, "\n".join(problems)


def test_handshake_follows_rank_offset_and_stays_unauthenticated():
    """Nonzero TP and DP ranks listen on the offset port, still without auth."""
    handshake_port = _free_port()
    tp_rank, dp_rank = 2, 1
    started = _start_worker(
        handshake_port=handshake_port,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        shared_backing=False,
    )
    try:
        _model_dir, connector, caches = started
        worker = connector.connector_worker
        assert worker is not None
        expected = handshake_port + get_port_offset(dp_rank, tp_rank)
        assert worker.tp_rank == tp_rank
        assert worker.dp_rank == dp_rank
        assert worker.side_channel_port == expected
        assert expected != handshake_port
        problems = _handshake_problems(worker.side_channel_port, caches)
        assert not problems, "\n".join(problems)
    finally:
        _stop_worker(started[0], started[1])


def test_default_handshake_port_is_6301_and_stays_unauthenticated():
    """Omitting handshake_port selects 6301, plus the rank offset (here 0)."""
    assert MoRIIOConstants.DEFAULT_HANDSHAKE_PORT == "6301"
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((_CONFIGURED_HOST, 6301))
    except OSError as exc:
        pytest.fail(f"default handshake port 6301 is not available: {exc}")
    finally:
        probe.close()
    started = _start_worker(
        handshake_port=None, tp_rank=0, dp_rank=0, shared_backing=False
    )
    try:
        _model_dir, connector, caches = started
        worker = connector.connector_worker
        assert worker is not None
        assert worker.handshake_port == 6301
        assert worker.side_channel_port == 6301 + get_port_offset(0, 0)
        problems = _handshake_problems(worker.side_channel_port, caches)
        assert not problems, "\n".join(problems)
    finally:
        _stop_worker(started[0], started[1])
