# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec for the media-download cache entry limit.

An API client supplies image, audio, and video URLs. With ``VLLM_MEDIA_CACHE``
enabled, each distinct URL string is stored under the cache directory before
the bytes are decoded. These tests require:

* a fetch that does not yield decodable media must not create a cache file,
  including a file that is removed again before the error propagates;
* distinct URLs whose payloads are smaller than one filesystem block must not
  leave more files than the configured byte budget can pay for at one block
  each.

Chat and completion parsing pass the client URL through to
``MediaConnector.fetch_*`` / ``fetch_*_async`` (and the decisions reader calls
``load_from_url_async`` on the same connector). Those methods are the cache
write path.
"""

from __future__ import annotations

import asyncio
import os
import select
import struct
import threading
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from vllm.connections import HTTPConnection
from vllm.multimodal.media.connector import MediaConnector

_IN_CREATE = 0x00000100
_IN_MOVED_TO = 0x00000080
_IN_Q_OVERFLOW = 0x00004000
_IN_NONBLOCK = 0x00000800
_CREATED = _IN_CREATE | _IN_MOVED_TO


class _FixedBodyHandler(BaseHTTPRequestHandler):
    """HTTP 200 with the same body for every path and query string."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        body: bytes = self.server.body  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _CacheDirWatch:
    """Record filenames created in a directory, including ones later removed."""

    def __init__(self, directory: Path) -> None:
        import ctypes

        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.inotify_init1.argtypes = [ctypes.c_int]
        self._libc.inotify_init1.restype = ctypes.c_int
        self._libc.inotify_add_watch.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self._libc.inotify_add_watch.restype = ctypes.c_int
        self._fd = self._libc.inotify_init1(_IN_NONBLOCK)
        if self._fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), "inotify_init1")
        watch = self._libc.inotify_add_watch(
            self._fd,
            os.fsencode(directory),
            _IN_CREATE | _IN_MOVED_TO | _IN_Q_OVERFLOW,
        )
        if watch < 0:
            err = ctypes.get_errno()
            os.close(self._fd)
            raise OSError(err, os.strerror(err), str(directory))
        self.created: list[str] = []
        self.overflow = False
        self._fd_open = True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_events, daemon=True)
        self._thread.start()

    def _read_events(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([self._fd], [], [], 0.05)
            if not readable:
                continue
            self._consume()

    def _consume(self) -> None:
        while True:
            try:
                blob = os.read(self._fd, 65536)
            except BlockingIOError:
                return
            offset = 0
            while offset + 16 <= len(blob):
                _wd, mask, _cookie, name_len = struct.unpack_from(
                    "iIII", blob, offset
                )
                offset += 16
                raw = blob[offset : offset + name_len]
                offset += name_len
                if mask & _IN_Q_OVERFLOW:
                    self.overflow = True
                if mask & _CREATED:
                    name = raw.split(b"\x00", 1)[0].decode()
                    if name:
                        self.created.append(name)

    def close(self) -> None:
        if not self._fd_open:
            return
        self._stop.set()
        self._thread.join(timeout=2)
        self._consume()
        os.close(self._fd)
        self._fd_open = False


@contextmanager
def _http_body(body: bytes):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixedBodyHandler)
    server.body = body  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _urls(origin: str, count: int, suffix: str) -> list[str]:
    return [f"{origin}/clip/{i}{suffix}?n={i}" for i in range(count)]


def _cache_files(cache_dir: Path) -> list[Path]:
    return sorted(p for p in cache_dir.iterdir() if p.is_file())


def _enable_cache(
    monkeypatch: pytest.MonkeyPatch,
    cache_dir: Path,
    *,
    max_mb: int,
    allowed_media_domains: list[str] | None,
) -> MediaConnector:
    monkeypatch.setenv("VLLM_MEDIA_CACHE", str(cache_dir))
    monkeypatch.setenv("VLLM_MEDIA_CACHE_MAX_SIZE_MB", str(max_mb))
    monkeypatch.setenv("VLLM_MEDIA_CACHE_TTL_HOURS", "24")
    connector = MediaConnector(
        connection=HTTPConnection(),
        allowed_media_domains=allowed_media_domains,
    )
    assert connector._media_cache_dir == str(cache_dir)
    return connector


def _assert_fetch_left_no_files(
    cache_dir: Path,
    watch: _CacheDirWatch,
    *,
    rejected: int,
) -> None:
    files = _cache_files(cache_dir)
    sizes = [path.stat().st_size for path in files]
    assert not watch.overflow
    assert files == [], (
        f"{len(files)} cache files remain after {rejected} rejected fetches "
        f"(sizes={sizes})"
    )
    assert watch.created == [], (
        f"rejected media fetches created {watch.created} in the cache "
        f"before the fetch failed ({rejected} URLs)"
    )


def _tiny_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _run_sync(
    fetch: Callable[[str], object],
    urls: list[str],
) -> None:
    for url in urls:
        with pytest.raises(ValueError):
            fetch(url)


async def _run_async(
    fetch: Callable[[str], Awaitable[object]],
    urls: list[str],
) -> None:
    results = await asyncio.gather(
        *(fetch(url) for url in urls),
        return_exceptions=True,
    )
    assert len(results) == len(urls)
    assert all(isinstance(result, ValueError) for result in results), results


@pytest.mark.parametrize(
    "body",
    [b"", b"\x00\x01not-an-image"],
    ids=["empty", "garbage"],
)
@pytest.mark.parametrize("allowed_media_domains", [None, ["127.0.0.1"]])
@pytest.mark.parametrize("async_fetch", [False, True])
def test_undecodable_image_urls_do_not_create_cache_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    allowed_media_domains: list[str] | None,
    async_fetch: bool,
    body: bytes,
) -> None:
    """Empty and garbage 200 responses are distinct cache keys and fail decode.

    An empty allowlist permits every host. An allowlist that contains the
    serving host still permits a new key per path and query string.
    """
    cache_dir = tmp_path / "media-cache"
    connector = _enable_cache(
        monkeypatch,
        cache_dir,
        max_mb=1,
        allowed_media_domains=allowed_media_domains,
    )
    with _http_body(body) as origin:
        urls = _urls(origin, 8, ".png")
        watch = _CacheDirWatch(cache_dir)
        try:
            if async_fetch:
                asyncio.run(_run_async(connector.fetch_image_async, urls))
            else:
                _run_sync(connector.fetch_image, urls)
        finally:
            watch.close()
        _assert_fetch_left_no_files(cache_dir, watch, rejected=len(urls))


@pytest.mark.parametrize("async_fetch", [False, True])
@pytest.mark.parametrize(
    ("fetch_name", "async_name", "suffix"),
    [
        ("fetch_audio", "fetch_audio_async", ".wav"),
        ("fetch_video", "fetch_video_async", ".mp4"),
    ],
)
def test_undecodable_audio_and_video_urls_do_not_create_cache_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    async_fetch: bool,
    fetch_name: str,
    async_name: str,
    suffix: str,
) -> None:
    """Audio and video downloads use the same cache write as images."""
    cache_dir = tmp_path / "media-cache"
    connector = _enable_cache(
        monkeypatch,
        cache_dir,
        max_mb=1,
        allowed_media_domains=None,
    )
    with _http_body(b"") as origin:
        urls = _urls(origin, 4, suffix)
        watch = _CacheDirWatch(cache_dir)
        try:
            if async_fetch:
                fetch = getattr(connector, async_name)
                asyncio.run(_run_async(fetch, urls))
            else:
                fetch = getattr(connector, fetch_name)
                _run_sync(fetch, urls)
        finally:
            watch.close()
        _assert_fetch_left_no_files(cache_dir, watch, rejected=len(urls))


def test_tiny_valid_images_cannot_exceed_filesystem_block_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sub-block objects must count against the byte budget as whole blocks.

    ``st_size`` of a 1x1 PNG is far below the directory's block size, so a
    budget check that sums logical sizes keeps every unique URL. The bound is
    ``max_bytes // filesystem block``.
    """
    cache_dir = tmp_path / "media-cache"
    connector = _enable_cache(
        monkeypatch,
        cache_dir,
        max_mb=1,
        allowed_media_domains=["127.0.0.1"],
    )
    block = os.statvfs(cache_dir).f_frsize
    assert block >= 512
    max_bytes = 1 * 1024 * 1024
    budget_entries = max_bytes // block
    png = _tiny_png()
    assert 0 < len(png) < block
    count = budget_entries + 16
    with _http_body(png) as origin:
        for url in _urls(origin, count, ".png"):
            image = connector.fetch_image(url)
            assert image.size == (1, 1)
    files = _cache_files(cache_dir)
    assert len(files) <= budget_entries, (
        f"{len(files)} cache files for {count} distinct tiny images; "
        f"byte budget {max_bytes} can pay for {budget_entries} blocks "
        f"of {block} bytes (file sizes={[path.stat().st_size for path in files[:4]]})"
    )
