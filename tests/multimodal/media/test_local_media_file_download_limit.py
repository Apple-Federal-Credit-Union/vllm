# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local file media must obey the remote download ceiling.

``ImageMediaIO.load_file`` and ``VideoMediaIO.load_file`` read a ``file://``
object in full. HTTP bodies are capped by ``VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB``
while they are streamed. These tests fail until a local read is rejected when
either its stat size or the bytes actually read exceed that ceiling, and until
a path opened after the check is the same object that was checked.

Not covered, because those paths do not share this read:

- ``ImageEmbeddingMediaIO.load_file``, ``AudioEmbeddingMediaIO.load_file``, and
  ``VideoEmbeddingMediaIO.load_file``. The connector embedding fetchers only
  decode base64, so a file URL never reaches them.
- ``AudioMediaIO.load_file`` already rejects ``st_size`` above
  ``VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`` before it opens the file.
- ``_validate_video_source`` returns the resolved path for the codec backend.
  It does not read the file into a ``bytes`` object.
"""

import asyncio
import builtins
import errno
import fcntl
import io
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image

import vllm.envs as envs
from vllm.connections import MediaDownloadSizeExceededError
from vllm.exceptions import VLLMUnprocessableEntityError
from vllm.multimodal.media import MediaConnector
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.media.video import VideoMediaIO
from vllm.utils.mem_constants import KiB_bytes, MiB_bytes

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

_LIMIT_MB = 1
_LIMIT_BYTES = _LIMIT_MB * MiB_bytes
# Larger than the ceiling, small enough that the unbounded read stays local.
_OVERSIZE_BYTES = 4 * MiB_bytes
# A capped reader may pull one extra chunk, plus a little incidental I/O.
_READ_SLACK = 512 * KiB_bytes
_PIPE_CAPACITY = 4 * KiB_bytes


def _task_rchar() -> int:
    """Bytes this process has passed to read(), from ``/proc/self/io``."""
    with open("/proc/self/io", encoding="ascii") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key == "rchar":
                return int(value)
    raise AssertionError("rchar missing from /proc/self/io")


def _sparse(path: Path, size: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, size)
    finally:
        os.close(fd)


def _png(path: Path) -> None:
    Image.new("RGB", (1, 1), (1, 2, 3)).save(path)


def _same_path(candidate: object, expected: Path) -> bool:
    try:
        raw = os.fspath(candidate)  # type: ignore[arg-type]
    except TypeError:
        return False
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    return os.path.abspath(raw) == str(expected)


def _is_size_rejection(exc: BaseException) -> bool:
    if isinstance(exc, MediaDownloadSizeExceededError):
        return True
    cause = exc.__cause__
    if isinstance(cause, MediaDownloadSizeExceededError):
        return True
    if isinstance(exc, VLLMUnprocessableEntityError):
        return "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB" in str(exc)
    return False


def _assert_read_bounded(before: int) -> None:
    delta = _task_rchar() - before
    assert delta <= _LIMIT_BYTES + _READ_SLACK, (
        f"read {delta} bytes from local media; ceiling is {_LIMIT_BYTES}"
    )


@contextmanager
def _download_ceiling(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", str(_LIMIT_MB))
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", _LIMIT_MB)
    yield


def _connector(media_dir: Path) -> MediaConnector:
    return MediaConnector(allowed_local_media_path=str(media_dir))


async def _fetch(connector: MediaConnector, modality: str, url: str, async_mode: bool):
    if modality == "image":
        if async_mode:
            return await connector.fetch_image_async(url)
        return connector.fetch_image(url)
    if modality == "video":
        if async_mode:
            return await connector.fetch_video_async(url)
        return connector.fetch_video(url)
    raise AssertionError(modality)


def _run_fetch(connector: MediaConnector, modality: str, url: str, async_mode: bool):
    if async_mode:
        return asyncio.run(_fetch(connector, modality, url, True))
    return asyncio.run(_fetch(connector, modality, url, False))


class _OpenResult:
    def __init__(self) -> None:
        self.exc: BaseException | None = None
        self.value: object = None


def _run_fetch_captured(
    connector: MediaConnector, modality: str, url: str, async_mode: bool
) -> _OpenResult:
    captured = _OpenResult()
    try:
        captured.value = _run_fetch(connector, modality, url, async_mode)
    except Exception as exc:
        captured.exc = exc
    return captured


def _assert_size_rejected(captured: _OpenResult, before: int) -> None:
    # Bound first. Raising the size error after the whole file is buffered
    # still fails.
    _assert_read_bounded(before)
    assert captured.exc is not None, "oversized local media was accepted"
    assert _is_size_rejection(captured.exc), (
        f"{type(captured.exc).__name__}: {captured.exc}"
    )


class _FifoWriter:
    """Write a payload into a FIFO and record how many bytes were accepted."""

    def __init__(self, fifo: Path, payload: bytes) -> None:
        self._fifo = fifo
        self._payload = payload
        self._done = threading.Event()
        self.written = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> None:
        self._done.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise AssertionError("FIFO writer did not finish")

    def _open_write_end(self) -> int | None:
        while not self._done.is_set():
            try:
                return os.open(self._fifo, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as exc:
                if exc.errno != errno.ENXIO:
                    raise
                self._done.wait(0.01)
        return None

    def _run(self) -> None:
        try:
            fd = self._open_write_end()
            if fd is None:
                return
            try:
                fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, _PIPE_CAPACITY)
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
                view = memoryview(self._payload)
                while view:
                    try:
                        count = os.write(fd, view)
                    except BrokenPipeError:
                        return
                    self.written += count
                    view = view[count:]
            finally:
                os.close(fd)
        except Exception as exc:
            self.error = exc


@contextmanager
def _retarget_on_open(link: Path, target: Path) -> Iterator[None]:
    """Point ``link`` at ``target`` when something opens ``link`` by name."""
    real_io_open = io.open
    real_os_open = os.open
    real_builtin_open = builtins.open

    def retarget(path: object) -> None:
        if not _same_path(path, link):
            return
        if link.is_symlink() and os.readlink(link) == str(target):
            return
        link.unlink()
        link.symlink_to(target)

    def wrapped_io_open(file, *args, **kwargs):
        retarget(file)
        return real_io_open(file, *args, **kwargs)

    def wrapped_os_open(path, flags, mode=0o777, *, dir_fd=None):
        retarget(path)
        if dir_fd is None:
            return real_os_open(path, flags, mode)
        return real_os_open(path, flags, mode, dir_fd=dir_fd)

    io.open = wrapped_io_open
    builtins.open = wrapped_io_open
    os.open = wrapped_os_open
    try:
        yield
    finally:
        io.open = real_io_open
        builtins.open = real_builtin_open
        os.open = real_os_open


def _assert_hook_sees_path_open(tmp_path: Path) -> None:
    """The retarget hook has to run before the kernel open, or the swap is fake."""
    link = tmp_path / "hook-link"
    original = tmp_path / "hook-original"
    swapped = tmp_path / "hook-swapped"
    original.write_bytes(b"original")
    swapped.write_bytes(b"swapped")
    link.symlink_to(original)
    with _retarget_on_open(link, swapped):
        assert link.read_bytes() == b"swapped"
    link.unlink()
    link.symlink_to(original)


@pytest.fixture(scope="session", autouse=True)
def _load_media_backends() -> None:
    """Import image and video loaders before tests measure bytes read.

    The first construction maps their shared libraries. Counting those reads
    as part of a later fetch would hide the media-file read.
    """
    VideoMediaIO(ImageMediaIO())


@pytest.fixture
def media_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    directory = tmp_path / "media"
    directory.mkdir()
    with _download_ceiling(monkeypatch):
        yield directory


def test_retarget_hook_runs_before_path_read(tmp_path: Path) -> None:
    _assert_hook_sees_path_open(tmp_path)


@pytest.mark.parametrize("async_mode", [False, True])
def test_image_file_under_download_ceiling_loads(
    media_dir: Path, async_mode: bool
) -> None:
    path = media_dir / "small.png"
    _png(path)
    image = _run_fetch(_connector(media_dir), "image", path.as_uri(), async_mode)
    assert image.size == (1, 1)


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("async_mode", [False, True])
def test_sparse_local_file_is_rejected_without_reading_it(
    media_dir: Path, modality: str, async_mode: bool
) -> None:
    path = media_dir / f"sparse-{modality}"
    _sparse(path, _OVERSIZE_BYTES)
    assert path.stat().st_size == _OVERSIZE_BYTES
    before = _task_rchar()
    captured = _run_fetch_captured(
        _connector(media_dir), modality, path.as_uri(), async_mode
    )
    _assert_size_rejected(captured, before)


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("async_mode", [False, True])
def test_fifo_local_file_stops_at_the_download_ceiling(
    media_dir: Path, modality: str, async_mode: bool
) -> None:
    fifo = media_dir / f"stream-{modality}-{async_mode}"
    os.mkfifo(fifo)
    writer = _FifoWriter(fifo, b"\x00" * _OVERSIZE_BYTES)
    writer.start()
    try:
        before = _task_rchar()
        captured = _run_fetch_captured(
            _connector(media_dir), modality, fifo.as_uri(), async_mode
        )
    finally:
        writer.finish()
    assert writer.error is None, writer.error
    assert writer.written <= _LIMIT_BYTES + _READ_SLACK, (
        f"reader accepted {writer.written} FIFO bytes; ceiling is {_LIMIT_BYTES}"
    )
    _assert_size_rejected(captured, before)


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("async_mode", [False, True])
def test_open_uses_the_file_that_passed_the_directory_check(
    media_dir: Path, modality: str, async_mode: bool
) -> None:
    """Swapping the symlink at open must not deliver the replacement body.

    The directory check resolves the symlink. The read has to use that object
    or stop before the replacement's bytes are buffered. Looking the name up
    again and reading whatever it points at now fails this test.
    """
    _assert_hook_sees_path_open(media_dir)
    allowed = media_dir / "allowed.png"
    oversized = media_dir / "oversized"
    link = media_dir / "alias"
    _png(allowed)
    _sparse(oversized, _OVERSIZE_BYTES)
    link.symlink_to(allowed)

    before = _task_rchar()
    with _retarget_on_open(link, oversized):
        captured = _run_fetch_captured(
            _connector(media_dir), modality, link.as_uri(), async_mode
        )
    _assert_read_bounded(before)
    if captured.exc is None:
        assert modality == "image"
        assert captured.value.size == (1, 1)
        return
    # A size rejection is the expected outcome when the open hits the
    # replacement. A decode error is only acceptable when the read stayed
    # on the small file that was checked, which the bound above enforces.
    if _is_size_rejection(captured.exc):
        return
    assert modality == "video"
