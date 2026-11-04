# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safe deserialization helpers for client-supplied embedding tensors.

`check_sparse_tensor_invariants_threadsafe()` is a thread-safe wrapper for
sparse tensor invariant validation. `safe_to_dense()` bounds the memory a
deserialized tensor may allocate when densified, and `safe_load_numpy()`
applies the same bound to `.npy` payloads before NumPy allocates them.

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
from io import BytesIO

import torch
from numpy.lib import format as npy_format

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.utils.mem_constants import MiB_bytes

_SPARSE_LOAD_LOCK = threading.Lock()


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

    _check_dense_bytes(
        tuple(tensor.shape),
        tensor.numel() * tensor.element_size(),
        parameter=parameter,
    )
    return tensor.to_dense()


def safe_load_numpy(data: bytes, *, parameter: str) -> torch.Tensor:
    """Load a client-supplied `.npy` embedding within a memory budget.

    `numpy.load` allocates the shape declared in the header before it reads
    the body, so a header of a few hundred bytes can request any amount of
    memory. Parse the header first and reject it unless its dense size is
    within the limit and the body actually holds that many bytes.
    """
    with BytesIO(data) as buffer:
        version = npy_format.read_magic(buffer)
        if version == (1, 0):
            header = npy_format.read_array_header_1_0(buffer)
        elif version == (2, 0):
            header = npy_format.read_array_header_2_0(buffer)
        else:
            raise VLLMValidationError(
                f"`{parameter}` uses unsupported .npy format version {version}.",
                parameter=parameter,
            )
        shape, _fortran_order, dtype = header
        if any(dim < 0 for dim in shape):
            raise VLLMValidationError(
                f"`{parameter}` declares negative dimensions {shape}.",
                parameter=parameter,
            )

        dense_bytes = math.prod(shape) * dtype.itemsize
        _check_dense_bytes(shape, dense_bytes, parameter=parameter)
        if len(data) - buffer.tell() < dense_bytes:
            raise VLLMValidationError(
                f"`{parameter}` declares shape {shape} but its .npy body is "
                f"shorter than the {dense_bytes} bytes that shape requires.",
                parameter=parameter,
            )

        buffer.seek(0)
        array = npy_format.read_array(buffer, allow_pickle=False)

    return torch.from_numpy(array)


def _check_dense_bytes(
    shape: tuple[int, ...], dense_bytes: int, *, parameter: str
) -> None:
    max_bytes = envs.VLLM_MAX_EMBED_DECODE_BYTES
    if max_bytes > 0 and dense_bytes > max_bytes:
        raise VLLMValidationError(
            f"`{parameter}` declares shape {shape}, which "
            f"would allocate {dense_bytes} bytes "
            f"({dense_bytes / MiB_bytes:.0f} MiB) once densified, "
            f"exceeding the {max_bytes} byte limit. Set "
            f"VLLM_MAX_EMBED_DECODE_BYTES to increase this limit.",
            parameter=parameter,
            value=dense_bytes,
        )
