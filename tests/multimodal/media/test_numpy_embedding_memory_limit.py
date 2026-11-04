# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NumPy image and video embeddings must honor the embedding byte limit.

Torch embeddings are rejected by ``safe_to_dense`` when their dense size
exceeds ``VLLM_MAX_EMBED_DECODE_BYTES``. NumPy embeddings are loaded with
``numpy.load``, which sizes the array from the ``.npy`` header before it
reads the body. These tests require that same byte limit, and they fail if
the oversized array is materialized on the way to the refusal.

``MultiModalContentParser.parse_image_embeds``, ``parse_video_embeds``, and
the async parser's counterparts are not called directly. Each one forwards
the base64 string (including every string value in a dict payload) to
``MediaConnector.fetch_*_embedding``, which these tests do call, and the
parser does not measure the payload first. Audio embeddings have no NumPy
loader.
"""

import asyncio
import io
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pybase64
import pytest
import torch
from numpy.lib.format import write_array_header_1_0, write_array_header_2_0

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.media.image import ImageEmbeddingMediaIO
from vllm.multimodal.media.video import VideoEmbeddingMediaIO

# Pinned in requirements/test/*.txt. The allocation behavior below was read
# from this version's format.read_array and PyArray_FromFile.
NUMPY_VERSION = "2.2.6"

# Lowered limit so a complete over-limit array is a few kilobytes, not a
# multi-gigabyte allocation. The header bomb below uses the real default.
TEST_LIMIT_BYTES = 4096
# Refuse to satisfy an allocation request above this even though the product
# asked for it. Keeps the header-bomb tests from exhausting the machine.
SAFETY_CAP_BYTES = 8 * 1024 * 1024
# Real, modest allocation used only to show what NumPy itself does.
PROBE_COUNT = 1_000_000

EMBEDDING_IO = [
    pytest.param(ImageEmbeddingMediaIO, "image_embeds", id="image"),
    pytest.param(VideoEmbeddingMediaIO, "video_embeds", id="video"),
]


class _EmbeddingAllocationRefused(Exception):
    """The loader asked NumPy to allocate an unsafe number of bytes."""

    def __init__(self, nbytes: int) -> None:
        super().__init__(
            f"numpy was asked to materialize {nbytes} bytes for an embedding"
        )
        self.nbytes = nbytes


def _element_nbytes(shape: object, dtype: object) -> int:
    if isinstance(shape, (tuple, list)):
        count = 1
        for dim in shape:
            count *= int(dim)
    else:
        count = int(shape)  # type: ignore[arg-type]
    return count * int(np.dtype(dtype).itemsize)  # type: ignore[arg-type]


@contextmanager
def _guard_numpy_allocation() -> Iterator[list[int]]:
    """Record every array NumPy is asked to build while a loader runs.

    Requests above ``SAFETY_CAP_BYTES`` are refused before the real
    allocator runs. Smaller requests, including arrays that are over the
    embedding limit but under the cap, are allocated for real so a check
    that happens after ``numpy.load`` still observes the materialization.
    """
    numpy_mod = np.lib.format.read_array.__globals__["numpy"]
    real_ndarray = numpy_mod.ndarray
    real_fromfile = numpy_mod.fromfile
    requested: list[int] = []

    def guarded_ndarray(
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
    ):
        nbytes = _element_nbytes(shape, dtype)
        requested.append(nbytes)
        if nbytes > SAFETY_CAP_BYTES:
            raise _EmbeddingAllocationRefused(nbytes)
        return real_ndarray(
            shape,
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=strides,
            order=order,
        )

    def guarded_fromfile(
        file,
        dtype=float,
        count=-1,
        sep="",
        offset=0,
        *,
        like=None,
    ):
        if int(count) >= 0:
            nbytes = _element_nbytes(int(count), dtype)
            requested.append(nbytes)
            if nbytes > SAFETY_CAP_BYTES:
                raise _EmbeddingAllocationRefused(nbytes)
        return real_fromfile(
            file,
            dtype=dtype,
            count=count,
            sep=sep,
            offset=offset,
            like=like,
        )

    numpy_mod.ndarray = guarded_ndarray
    numpy_mod.fromfile = guarded_fromfile
    try:
        yield requested
    finally:
        numpy_mod.ndarray = real_ndarray
        numpy_mod.fromfile = real_fromfile


def _npy_header(
    shape: tuple[int, ...],
    *,
    descr: str = "<f4",
    version: tuple[int, int] = (1, 0),
) -> bytes:
    """A ``.npy`` header with no data body, written by NumPy's own writer."""
    buf = io.BytesIO()
    fields = {"descr": descr, "fortran_order": False, "shape": shape}
    if version == (1, 0):
        write_array_header_1_0(buf, fields)
    elif version == (2, 0):
        write_array_header_2_0(buf, fields)
    else:
        raise AssertionError(version)
    return buf.getvalue()


def _npy_array(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return pybase64.b64encode(data).decode("ascii")


def _over_limit_count(limit: int, dtype: np.dtype) -> int:
    return limit // int(dtype.itemsize) + 1


def _header_bomb() -> bytes:
    """Header whose float32 shape is one element past the live byte limit."""
    limit = envs.VLLM_MAX_EMBED_DECODE_BYTES
    assert limit > 0
    count = _over_limit_count(limit, np.dtype(np.float32))
    payload = _npy_header((count,))
    # The attack is amplification: the file is tiny, the declared array is not.
    assert len(payload) < 4096
    assert count * 4 > limit
    return payload


def _expect_rejected(load: Callable[[], object], parameter: str) -> None:
    limit = envs.VLLM_MAX_EMBED_DECODE_BYTES
    assert limit > 0
    with (
        _guard_numpy_allocation() as requested,
        pytest.raises(VLLMValidationError) as exc_info,
    ):
        load()
    assert exc_info.value.parameter == parameter
    assert "VLLM_MAX_EMBED_DECODE_BYTES" in str(exc_info.value)
    materialized = [n for n in requested if n > limit]
    assert materialized == [], (
        f"loader materialized {materialized} bytes past the {limit} byte limit"
    )


@pytest.fixture
def small_decode_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_MAX_EMBED_DECODE_BYTES", str(TEST_LIMIT_BYTES))
    assert envs.VLLM_MAX_EMBED_DECODE_BYTES == TEST_LIMIT_BYTES


def test_numpy_bytes_load_allocates_declared_size_before_the_body() -> None:
    """NumPy 2.2.6 builds the full array, then discovers the body is short."""
    assert np.__version__ == NUMPY_VERSION
    declared = PROBE_COUNT * int(np.dtype(np.float32).itemsize)
    payload = _npy_header((PROBE_COUNT,))
    assert len(payload) < 1024

    numpy_mod = np.lib.format.read_array.__globals__["numpy"]
    real_ndarray = numpy_mod.ndarray
    calls: list[int] = []

    def wrapped(shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
        calls.append(_element_nbytes(shape, dtype))
        return real_ndarray(
            shape,
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=strides,
            order=order,
        )

    numpy_mod.ndarray = wrapped
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="EOF"):
            np.load(io.BytesIO(payload))
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        numpy_mod.ndarray = real_ndarray

    assert calls == [declared]
    assert peak >= declared


def test_numpy_file_load_allocates_declared_size_before_the_body(
    tmp_path: Path,
) -> None:
    """``fromfile`` is handed the header's count, and the process holds it."""
    assert np.__version__ == NUMPY_VERSION
    declared = PROBE_COUNT * int(np.dtype(np.float32).itemsize)
    path = tmp_path / "probe.npy"
    path.write_bytes(_npy_header((PROBE_COUNT,)))

    numpy_mod = np.lib.format.read_array.__globals__["numpy"]
    real_fromfile = numpy_mod.fromfile
    calls: list[int] = []

    def wrapped(file, dtype=float, count=-1, sep="", offset=0, *, like=None):
        if int(count) >= 0:
            calls.append(_element_nbytes(int(count), dtype))
        return real_fromfile(
            file, dtype=dtype, count=count, sep=sep, offset=offset, like=like
        )

    numpy_mod.fromfile = wrapped
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="cannot reshape"):
            np.load(path)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        numpy_mod.fromfile = real_fromfile

    assert calls == [declared]
    assert peak >= declared


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_bytes_rejects_complete_array_over_limit(
    io_cls, parameter: str, small_decode_limit: None
) -> None:
    count = _over_limit_count(TEST_LIMIT_BYTES, np.dtype(np.float32))
    data = _npy_array(np.zeros((count,), dtype=np.float32))
    _expect_rejected(lambda: io_cls().load_bytes(data), parameter)


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_base64_rejects_complete_array_over_limit(
    io_cls, parameter: str, small_decode_limit: None
) -> None:
    count = _over_limit_count(TEST_LIMIT_BYTES, np.dtype(np.float32))
    data = _b64(_npy_array(np.zeros((count,), dtype=np.float32)))
    _expect_rejected(lambda: io_cls().load_base64("", data), parameter)


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_file_rejects_complete_array_over_limit(
    io_cls, parameter: str, tmp_path: Path, small_decode_limit: None
) -> None:
    count = _over_limit_count(TEST_LIMIT_BYTES, np.dtype(np.float32))
    path = tmp_path / "embeds.npy"
    path.write_bytes(_npy_array(np.zeros((count,), dtype=np.float32)))
    _expect_rejected(lambda: io_cls().load_file(path), parameter)


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_bytes_rejects_header_bomb_before_materializing(
    io_cls, parameter: str
) -> None:
    _expect_rejected(lambda: io_cls().load_bytes(_header_bomb()), parameter)


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_base64_rejects_header_bomb_before_materializing(
    io_cls, parameter: str
) -> None:
    _expect_rejected(lambda: io_cls().load_base64("", _b64(_header_bomb())), parameter)


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_file_rejects_header_bomb_before_materializing(
    io_cls, parameter: str, tmp_path: Path
) -> None:
    path = tmp_path / "embeds.npy"
    path.write_bytes(_header_bomb())
    _expect_rejected(lambda: io_cls().load_file(path), parameter)


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_bytes_rejects_version2_header_bomb(io_cls, parameter: str) -> None:
    limit = envs.VLLM_MAX_EMBED_DECODE_BYTES
    count = _over_limit_count(limit, np.dtype(np.float32))
    payload = _npy_header((count,), version=(2, 0))
    _expect_rejected(lambda: io_cls().load_bytes(payload), parameter)


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_bytes_counts_dtype_itemsize(
    io_cls, parameter: str, small_decode_limit: None
) -> None:
    """A float64 array over the limit is not a float32-sized shape check."""
    count = _over_limit_count(TEST_LIMIT_BYTES, np.dtype(np.float64))
    assert count * 4 <= TEST_LIMIT_BYTES
    assert count * 8 > TEST_LIMIT_BYTES
    data = _npy_array(np.zeros((count,), dtype=np.float64))
    _expect_rejected(lambda: io_cls().load_bytes(data), parameter)


def _fetch_image(connector: MediaConnector, data: str) -> torch.Tensor:
    return connector.fetch_image_embedding(data)


def _fetch_video(connector: MediaConnector, data: str) -> torch.Tensor:
    return connector.fetch_video_embedding(data)


CONNECTOR_FETCH = [
    pytest.param(_fetch_image, "image_embeds", id="image"),
    pytest.param(_fetch_video, "video_embeds", id="video"),
]


@pytest.mark.parametrize(("fetch", "parameter"), CONNECTOR_FETCH)
def test_connector_rejects_complete_base64_over_limit(
    fetch, parameter: str, small_decode_limit: None
) -> None:
    count = _over_limit_count(TEST_LIMIT_BYTES, np.dtype(np.float32))
    data = _b64(_npy_array(np.zeros((count,), dtype=np.float32)))
    connector = MediaConnector()
    _expect_rejected(lambda: fetch(connector, data), parameter)


@pytest.mark.parametrize(("fetch", "parameter"), CONNECTOR_FETCH)
def test_connector_rejects_header_bomb(fetch, parameter: str) -> None:
    connector = MediaConnector()
    _expect_rejected(lambda: fetch(connector, _b64(_header_bomb())), parameter)


def test_connector_async_rejects_header_bomb() -> None:
    connector = MediaConnector()
    image = _b64(_header_bomb())
    video = _b64(_header_bomb())

    async def run() -> None:
        with pytest.raises(VLLMValidationError) as image_exc:
            await connector.fetch_image_embedding_async(image)
        with pytest.raises(VLLMValidationError) as video_exc:
            await connector.fetch_video_embedding_async(video)
        assert image_exc.value.parameter == "image_embeds"
        assert video_exc.value.parameter == "video_embeds"
        assert "VLLM_MAX_EMBED_DECODE_BYTES" in str(image_exc.value)
        assert "VLLM_MAX_EMBED_DECODE_BYTES" in str(video_exc.value)

    limit = envs.VLLM_MAX_EMBED_DECODE_BYTES
    with _guard_numpy_allocation() as requested:
        asyncio.run(run())
    assert [n for n in requested if n > limit] == []


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_bytes_accepts_array_exactly_at_limit(
    io_cls, parameter: str, small_decode_limit: None
) -> None:
    assert parameter.endswith("_embeds")
    count = TEST_LIMIT_BYTES // int(np.dtype(np.float32).itemsize)
    data = _npy_array(np.zeros((count,), dtype=np.float32))
    with _guard_numpy_allocation() as requested:
        result = io_cls().load_bytes(data)
    assert tuple(result.shape) == (count,)
    assert result.dtype == torch.float32
    assert requested
    assert all(n <= TEST_LIMIT_BYTES for n in requested)


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_load_bytes_accepts_small_array(
    io_cls, parameter: str, small_decode_limit: None
) -> None:
    assert parameter.endswith("_embeds")
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = io_cls().load_bytes(_npy_array(array))
    assert result.shape == torch.Size([2, 3])
    assert torch.equal(result, torch.from_numpy(array))


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_disabled_limit_allows_array_over_the_small_limit(
    io_cls, parameter: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert parameter.endswith("_embeds")
    monkeypatch.setenv("VLLM_MAX_EMBED_DECODE_BYTES", "0")
    assert envs.VLLM_MAX_EMBED_DECODE_BYTES == 0
    count = _over_limit_count(TEST_LIMIT_BYTES, np.dtype(np.float32))
    result = io_cls().load_bytes(_npy_array(np.zeros((count,), dtype=np.float32)))
    assert tuple(result.shape) == (count,)


@pytest.mark.parametrize(("io_cls", "parameter"), EMBEDDING_IO)
def test_torch_payload_over_limit_still_rejected(
    io_cls, parameter: str, small_decode_limit: None
) -> None:
    count = _over_limit_count(TEST_LIMIT_BYTES, np.dtype(np.float32))
    buf = io.BytesIO()
    torch.save(torch.zeros(count, dtype=torch.float32), buf)
    with pytest.raises(VLLMValidationError) as exc_info:
        io_cls().load_bytes(buf.getvalue())
    assert exc_info.value.parameter == parameter
