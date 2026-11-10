# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safe deserialization helpers for client-supplied embedding tensors.

`check_sparse_tensor_invariants_threadsafe()` is a thread-safe wrapper for
sparse tensor invariant validation. `safe_to_dense()` bounds the memory a
deserialized tensor may allocate when densified. `bounded_torch_checkpoint()`
and `load_numpy_embed()` bound what deserializing a payload allocates.

PyTorch's `torch.sparse.check_sparse_tensor_invariants()` context manager
manipulates a **process-global** flag (save/enable/restore). When multiple
embedding-load operations run concurrently on a thread-pool executor, one
context can restore the flag to `False` while another thread is still inside
its guard, bypassing the invariant check.

All call sites MUST use `check_sparse_tensor_invariants_threadsafe()`
which serializes access behind a lock.
"""

import contextlib
import math
import threading
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Final

import numpy as np
import torch

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.utils.mem_constants import MiB_bytes

_SPARSE_LOAD_LOCK = threading.Lock()

# Budget for the records of a checkpoint that are not tensor storages
# (`data.pkl`, `version`, ...), which take a few hundred bytes for an embedding.
_MAX_CHECKPOINT_METADATA_BYTES: Final = MiB_bytes
_CHECKPOINT_COPY_CHUNK_BYTES: Final = MiB_bytes


@contextlib.contextmanager
def check_sparse_tensor_invariants_threadsafe():
    with _SPARSE_LOAD_LOCK, torch.sparse.check_sparse_tensor_invariants():
        yield


def safe_to_dense(tensor: object, *, parameter: str) -> torch.Tensor:
    """Densify a client-supplied embedding tensor within a memory budget.

    `torch.load(..., weights_only=True)` prevents arbitrary code execution but
    not memory amplification: a sparse COO tensor carries its own declared
    shape, so the dense allocation is bounded by that shape rather than by the
    request. A 2.6 KB payload declaring `(30000, 30000)` materializes 3.4 GiB,
    and the shape can be picked to fit whatever host is being targeted. The
    invariant checks only reject *invalid* tensors (indices outside the
    declared shape); a single non-zero element inside a `(2**20, 2**20)` shape
    is perfectly valid and passes them.

    So reject oversized payloads based on the dense size they declare, before
    `to_dense()` allocates anything.
    """
    if not isinstance(tensor, torch.Tensor):
        raise VLLMValidationError(
            f"`{parameter}` payload did not deserialize to a torch.Tensor.",
            parameter=parameter,
        )

    max_bytes = envs.VLLM_MAX_EMBED_DECODE_BYTES
    if max_bytes > 0:
        dense_bytes = tensor.numel() * tensor.element_size()
        if dense_bytes > max_bytes:
            raise VLLMValidationError(
                f"`{parameter}` declares shape {tuple(tensor.shape)}, which "
                f"would allocate {dense_bytes} bytes "
                f"({dense_bytes / MiB_bytes:.0f} MiB) once densified, "
                f"exceeding the {max_bytes} byte limit. Set "
                f"VLLM_MAX_EMBED_DECODE_BYTES to increase this limit.",
                parameter=parameter,
                value=dense_bytes,
            )

    return tensor.to_dense()


def _decoded_over_limit(
    parameter: str, decoded_bytes: int, max_bytes: int
) -> VLLMValidationError:
    return VLLMValidationError(
        f"`{parameter}` decodes to at least {decoded_bytes} bytes of tensor "
        f"data, exceeding the {max_bytes} byte limit. Set "
        f"VLLM_MAX_EMBED_DECODE_BYTES to increase this limit.",
        parameter=parameter,
        value=decoded_bytes,
    )


def _copy_record(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    out: zipfile.ZipFile,
    budget: int,
) -> int:
    """Copy `info` uncompressed into `out`, stopping once it exceeds `budget`.

    Returns the bytes read, which is `budget + 1` when the record is larger.
    """
    copied = 0
    with (
        archive.open(info) as src,
        out.open(info.filename, "w", force_zip64=True) as dst,
    ):
        while chunk := src.read(min(_CHECKPOINT_COPY_CHUNK_BYTES, budget - copied + 1)):
            copied += len(chunk)
            if copied > budget:
                break
            dst.write(chunk)
    return copied


def bounded_torch_checkpoint(source: bytes | Path, *, parameter: str) -> BytesIO | Path:
    """Return what to pass to `torch.load` for a client `torch.save` payload.

    `torch.load` allocates each zip record at the uncompressed size its
    central directory declares, so a deflated record of a few KiB can demand
    GiBs before `safe_to_dense` sees the tensor. The legacy (non-zip) format
    allocates each storage from the element count in the pickle before
    reading its bytes, so it cannot be bounded and is refused.

    Each record is decompressed here within `VLLM_MAX_EMBED_DECODE_BYTES` and
    written uncompressed to a new archive, and only that archive reaches
    `torch.load`: torch's zip reader can locate a different central directory
    in the caller's bytes than `zipfile` does.
    """
    max_bytes = envs.VLLM_MAX_EMBED_DECODE_BYTES
    if max_bytes <= 0:
        return BytesIO(source) if isinstance(source, bytes) else source

    storage_left = max_bytes
    metadata_left = _MAX_CHECKPOINT_METADATA_BYTES
    rewritten = BytesIO()
    try:
        with (
            zipfile.ZipFile(
                BytesIO(source) if isinstance(source, bytes) else source
            ) as archive,
            zipfile.ZipFile(rewritten, "w") as out,
        ):
            records = archive.infolist()
            if not records:
                raise ValueError("archive has no records")
            for info in records:
                parts = info.filename.split("/")
                is_storage = len(parts) >= 2 and parts[-2] == "data"
                budget = storage_left if is_storage else metadata_left
                # The declared size only lets an oversized record fail early;
                # the bound that holds is the one `_copy_record` reads under.
                copied = (
                    info.file_size
                    if info.file_size > budget
                    else _copy_record(archive, info, out, budget)
                )
                if copied > budget:
                    if is_storage:
                        raise _decoded_over_limit(
                            parameter, max_bytes - storage_left + copied, max_bytes
                        )
                    raise VLLMValidationError(
                        f"`{parameter}` checkpoint records other than tensor "
                        f"storages exceed {_MAX_CHECKPOINT_METADATA_BYTES} bytes.",
                        parameter=parameter,
                    )
                if is_storage:
                    storage_left -= copied
                else:
                    metadata_left -= copied
    except (VLLMValidationError, MemoryError):
        raise
    except Exception as exc:
        # zipfile raises many types for malformed archives (BadZipFile,
        # zlib.error, EOFError, NotImplementedError, ...); all are about the
        # caller's bytes.
        raise VLLMValidationError(
            f"`{parameter}` must be a zip checkpoint written by `torch.save` "
            f"(the legacy format is not accepted while "
            f"VLLM_MAX_EMBED_DECODE_BYTES is set): {type(exc).__name__}",
            parameter=parameter,
        ) from exc

    rewritten.seek(0)
    return rewritten


def load_numpy_embed(data: bytes, *, parameter: str) -> torch.Tensor:
    """Decode a client `.npy` payload, bounded by the bytes it carries.

    `np.load` allocates the array its header declares before reading the
    array data, so a payload of a few bytes can declare any size.
    """
    buffer = BytesIO(data)
    try:
        version = np.lib.format.read_magic(buffer)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(buffer)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(buffer)
        else:
            raise ValueError(f"unsupported .npy format version {version}")
    except ValueError as exc:
        raise VLLMValidationError(
            f"`{parameter}` is not a readable .npy array.",
            parameter=parameter,
        ) from exc

    if dtype.hasobject:
        raise VLLMValidationError(
            f"`{parameter}` .npy arrays with an object dtype are not accepted.",
            parameter=parameter,
        )

    count = math.prod(shape)
    nbytes = count * dtype.itemsize
    offset = buffer.tell()
    if nbytes > len(data) - offset:
        raise VLLMValidationError(
            f"`{parameter}` .npy header declares {nbytes} bytes of array data, "
            f"but the payload holds {len(data) - offset}.",
            parameter=parameter,
            value=nbytes,
        )
    max_bytes = envs.VLLM_MAX_EMBED_DECODE_BYTES
    if max_bytes > 0 and nbytes > max_bytes:
        raise _decoded_over_limit(parameter, nbytes, max_bytes)

    try:
        array = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
    except ValueError as exc:
        raise VLLMValidationError(
            f"`{parameter}` is not a readable .npy array.",
            parameter=parameter,
        ) from exc
    if fortran_order:
        array = array.reshape(shape[::-1]).transpose()
    else:
        array = array.reshape(shape)
    return torch.from_numpy(array.copy())
