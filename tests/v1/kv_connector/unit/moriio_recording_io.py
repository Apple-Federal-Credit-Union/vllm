# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process MoRI IO engine used when the ROCm package is not installed.

The methods below follow ``python/mori/io/engine.py`` and
``src/io/engine.cpp`` from MoRI ``v1.2.3.post1`` (the revision
``docker/Dockerfile.rocm_base`` checks out): ``register_torch_tensor``
registers ``data_ptr()`` for ``nelement * element_size`` bytes, and
``RegisterMemory`` records that pointer and length. ``pack`` emits those
fields in the ``MSGPACK_DEFINE`` order from ``include/mori/io/common.hpp``.
"""

from __future__ import annotations

import os
import socket
import sys
import types

import msgpack
import torch

_IPC_HANDLE_LEN = 64


class BackendType:
    XGMI = 1
    RDMA = 2


class PollCqMode:
    POLLING = 1


class MemoryLocationType:
    CPU = 1
    GPU = 2


class _Config:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs


class IOEngineConfig:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)


class EngineDesc:
    def __init__(
        self,
        key: str,
        node_id: str,
        hostname: str,
        host: str,
        port: int,
        pid: int,
    ) -> None:
        self.key = key
        self.node_id = node_id
        self.hostname = hostname
        self.host = host
        self.port = port
        self.pid = pid

    def pack(self) -> bytes:
        # EngineDesc::MSGPACK_DEFINE(key, nodeId, hostname, host, port, pid)
        return msgpack.packb(
            [self.key, self.node_id, self.hostname, self.host, self.port, self.pid]
        )

    @staticmethod
    def unpack(blob: bytes) -> EngineDesc:
        key, node_id, hostname, host, port, pid = msgpack.unpackb(blob)
        return EngineDesc(key, node_id, hostname, host, int(port), int(pid))


class MemoryDesc:
    def __init__(self) -> None:
        self.engine_key = ""
        self.id = 0
        self.device_id = -1
        self.device_bus_id = ""
        self.data = 0
        self.size = 0
        self.loc = MemoryLocationType.CPU
        self.ipc_handle = b"\x00" * _IPC_HANDLE_LEN
        self.numa_node = -1
        self.ipc_offset = 0
        self.fabric_handle = b"\x00" * _IPC_HANDLE_LEN
        self.vpod_id = -1
        self.vpod_ppod_id = ""
        self.fabric_offset = 0
        self.fabric_alloc_size = 0

    def pack(self) -> bytes:
        # MemoryDesc::MSGPACK_DEFINE field order.
        return msgpack.packb(
            [
                self.engine_key,
                self.id,
                self.device_id,
                self.device_bus_id,
                self.data,
                self.size,
                int(self.loc),
                list(self.ipc_handle),
                self.numa_node,
                self.ipc_offset,
                list(self.fabric_handle),
                self.vpod_id,
                self.vpod_ppod_id,
                self.fabric_offset,
                self.fabric_alloc_size,
            ]
        )

    @staticmethod
    def unpack(blob: bytes) -> MemoryDesc:
        fields = msgpack.unpackb(blob)
        desc = MemoryDesc()
        (
            desc.engine_key,
            desc.id,
            desc.device_id,
            desc.device_bus_id,
            desc.data,
            desc.size,
            desc.loc,
        ) = fields[:7]
        return desc


class IOEngine:
    """Python IOEngine surface used by ``MoRIIOWrapper``."""

    def __init__(self, key: str, config: IOEngineConfig) -> None:
        self.config = config
        hostname = socket.gethostname()
        self.desc = EngineDesc(
            key=key,
            node_id=hostname,
            hostname=hostname,
            host=config.host,
            port=int(config.port),
            pid=os.getpid(),
        )
        self._next_mem_id = 0

    def get_engine_desc(self) -> EngineDesc:
        return self.desc

    def create_backend(self, backend_type: object, config: object = None) -> None:
        del backend_type, config

    def register_remote_engine(self, engine_desc: EngineDesc) -> None:
        del engine_desc

    def register_memory(
        self, ptr: int, size: int, device_id: int, mem_loc: int
    ) -> MemoryDesc:
        # IOEngine::RegisterMemory copies the pointer and the byte length into
        # the descriptor and does not shrink the region to one request.
        desc = MemoryDesc()
        desc.engine_key = self.desc.key
        desc.id = self._next_mem_id
        self._next_mem_id += 1
        desc.device_id = device_id
        desc.data = int(ptr)
        desc.size = int(size)
        desc.loc = mem_loc
        return desc

    def register_torch_tensor(self, tensor: torch.Tensor) -> MemoryDesc:
        # mori.io.IOEngine.register_torch_tensor
        if not tensor.is_contiguous():
            raise RuntimeError("input tensor must be contiguous")
        total_bytes = tensor.nelement() * tensor.element_size()
        device_id = tensor.device.index
        if device_id is None:
            device_id = -1
        if tensor.device.type == "cpu":
            mem_loc = MemoryLocationType.CPU
        elif tensor.device.type == "cuda":
            mem_loc = MemoryLocationType.GPU
        else:
            raise RuntimeError(f"unsupported device type {tensor.device.type}")
        return self.register_memory(tensor.data_ptr(), total_bytes, device_id, mem_loc)


def install_recording_mori() -> None:
    """Expose the recording engine as ``mori.io`` before vLLM imports it."""
    if "mori.io" in sys.modules:
        return
    mori_mod = types.ModuleType("mori")
    io_mod = types.ModuleType("mori.io")
    io_mod.BackendType = BackendType
    io_mod.PollCqMode = PollCqMode
    io_mod.EngineDesc = EngineDesc
    io_mod.MemoryDesc = MemoryDesc
    io_mod.IOEngine = IOEngine
    io_mod.IOEngineConfig = IOEngineConfig
    io_mod.RdmaBackendConfig = _Config
    io_mod.XgmiBackendConfig = _Config
    mori_mod.io = io_mod
    sys.modules["mori"] = mori_mod
    sys.modules["mori.io"] = io_mod


install_recording_mori()
