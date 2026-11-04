# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security spec for multimodal URL fetches.

A chat or batch client must not cause the API process to open a TCP connection
to a loopback, private, link-local, or metadata address, including when that
address is reached by a redirect or by a later address in the same DNS answer.
These tests fail while that connection is still made.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any

import pytest
from PIL import Image

import vllm.entrypoints.launchers.run_batch as run_batch
import vllm.envs as envs
from vllm.connections import HTTPConnection
from vllm.entrypoints.launchers.run_batch import download_bytes_from_url
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.utils import fetch_audio, fetch_image, fetch_video

_MODALITIES = ("image", "audio", "video")
_MODES = ("sync", "async")

# Documentation range (TEST-NET-3). Never routed; the egress guard refuses it
# before a packet is sent. Returned as the first address of a mixed DNS answer
# so a check of only that address still leaves the loopback address usable.
_PUBLIC_DECOY = "203.0.113.50"


def _png() -> bytes:
    image = Image.new("RGB", (1, 1), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _wav() -> bytes:
    # One silent PCM frame. Enough for the audio decoder to accept the body
    # when the fetch is allowed through.
    sample_rate = 8000
    data = b"\x00\x00"
    header = b"".join(
        (
            b"RIFF",
            (36 + len(data)).to_bytes(4, "little"),
            b"WAVE",
            b"fmt ",
            (16).to_bytes(4, "little"),
            (1).to_bytes(2, "little"),
            (1).to_bytes(2, "little"),
            sample_rate.to_bytes(4, "little"),
            (sample_rate * 2).to_bytes(4, "little"),
            (2).to_bytes(2, "little"),
            (16).to_bytes(2, "little"),
            b"data",
            len(data).to_bytes(4, "little"),
        )
    )
    return header + data


def _endpoint(address: Any) -> tuple[str, int | None]:
    if isinstance(address, tuple):
        host = address[0]
        port = address[1] if len(address) > 1 else None
    else:
        host, port = address, None
    if isinstance(host, bytes):
        host = host.decode()
    host = str(host).split("%", 1)[0]
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    with suppress(ValueError):
        host = str(ipaddress.ip_address(host))
    return host, port


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        self.server.hits.append(path)  # type: ignore[attr-defined]
        status, headers, body = self.server.routes.get(  # type: ignore[attr-defined]
            path, (404, [], b"")
        )
        self.send_response(status)
        for key, value in headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    allow_reuse_address = True
    daemon_threads = True

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


class MediaServer:
    """Local HTTP server. Hits are recorded only after a real request arrives."""

    def __init__(self) -> None:
        self.hits: list[str] = []
        png = _png()
        wav = _wav()
        self.routes: dict[str, tuple[int, list[tuple[str, str]], bytes]] = {
            "/pixel.png": (200, [("Content-Type", "image/png")], png),
            "/audio.wav": (200, [("Content-Type", "audio/wav")], wav),
            "/video.bin": (200, [("Content-Type", "video/mp4")], png),
            "/missing": (404, [("Content-Type", "text/plain")], b"missing"),
        }
        self._servers: list[ThreadingHTTPServer] = []
        self._threads: list[threading.Thread] = []
        self.port = 0

    def start(self) -> None:
        v4 = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        v4.allow_reuse_address = True
        v4.daemon_threads = True  # type: ignore[attr-defined]
        v4.hits = self.hits  # type: ignore[attr-defined]
        v4.routes = self.routes  # type: ignore[attr-defined]
        self.port = v4.server_address[1]
        self._serve(v4)
        v6 = _V6Server(("::1", self.port), _Handler)
        v6.hits = self.hits  # type: ignore[attr-defined]
        v6.routes = self.routes  # type: ignore[attr-defined]
        self._serve(v6)
        self._install_redirects()

    def _serve(self, server: ThreadingHTTPServer) -> None:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._servers.append(server)
        self._threads.append(thread)

    def _install_redirects(self) -> None:
        meta = "http://169.254.169.254/latest/meta-data/"
        evil = f"http://evil.test:{self.port}/pixel.png"
        other = "http://other.example/secret.png"
        self.routes["/redirect-meta"] = (302, [("Location", meta)], b"")
        self.routes["/redirect-evil"] = (302, [("Location", evil)], b"")
        self.routes["/redirect-other"] = (302, [("Location", other)], b"")

    def stop(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(timeout=2)

    def url(self, host: str, path: str) -> str:
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}{path}"

    @property
    def allowed_endpoints(self) -> set[tuple[str, int | None]]:
        return {("127.0.0.1", self.port), ("::1", self.port)}

    @contextmanager
    def egress_guard(
        self,
        names: dict[str, Callable[[int], list[Any]]] | None = None,
    ) -> Iterator[list[tuple[str, int | None]]]:
        """Refuse every TCP connect except this server.

        ``names`` replaces DNS for specific hosts. The replacement is applied
        inside ``socket.getaddrinfo``, which is what urllib3 2.8.0 and aiohttp's
        threaded resolver call. Each returned sockaddr is still connected by
        the real client.
        """
        blocked: list[tuple[str, int | None]] = []
        real_connect = socket.socket.connect
        real_getaddrinfo = socket.getaddrinfo
        name_map = names or {}

        def connect(sock: socket.socket, address: Any) -> None:
            endpoint = _endpoint(address)
            if endpoint in self.allowed_endpoints:
                real_connect(sock, address)
                return
            blocked.append(endpoint)
            raise ConnectionRefusedError(111, "Connection refused")

        def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
            key = host.decode() if isinstance(host, bytes) else str(host)
            if key in name_map:
                return name_map[key](int(port))
            return real_getaddrinfo(host, port, *args, **kwargs)

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
        try:
            yield blocked
        finally:
            socket.socket.connect = real_connect  # type: ignore[method-assign]
            socket.getaddrinfo = real_getaddrinfo


def _inet(ip: str, port: int) -> list[Any]:
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]


def _records(addresses: list[str]) -> Callable[[int], list[Any]]:
    def resolve(port: int) -> list[Any]:
        return [item for ip in addresses for item in _inet(ip, port)]

    return resolve


@pytest.fixture
def media_server() -> Iterator[MediaServer]:
    server = MediaServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(autouse=True)
def _pin_fetch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    # One attempt. Retries sleep, and they are not what decides the target.
    monkeypatch.setenv("VLLM_MEDIA_FETCH_MAX_RETRIES", "1")
    envs.disable_envs_cache()


def _connector(
    connection: HTTPConnection,
    domains: list[str] | None | object = ...,
) -> MediaConnector:
    if domains is ...:
        return MediaConnector(connection=connection)
    return MediaConnector(
        connection=connection,
        allowed_media_domains=domains,  # type: ignore[arg-type]
    )


async def _close(connection: HTTPConnection) -> None:
    client = connection._async_client
    if client is not None and not client.closed:
        await client.close()


async def _fetch(
    connector: MediaConnector,
    modality: str,
    url: str,
    mode: str,
) -> Any:
    if modality == "image" and mode == "sync":
        return connector.fetch_image(url)
    if modality == "image" and mode == "async":
        return await connector.fetch_image_async(url)
    if modality == "audio" and mode == "sync":
        return connector.fetch_audio(url)
    if modality == "audio" and mode == "async":
        return await connector.fetch_audio_async(url)
    if modality == "video" and mode == "sync":
        return connector.fetch_video(url)
    if modality == "video" and mode == "async":
        return await connector.fetch_video_async(url)
    raise AssertionError(modality)


def _assert_refused(
    result: Any,
    *,
    hits: list[str],
    blocked: list[tuple[str, int | None]],
    forbidden_paths: set[str],
    forbidden_hosts: set[str],
    require_error: bool,
) -> None:
    leaked_paths = [path for path in hits if path in forbidden_paths]
    leaked_hosts = [item for item in blocked if item[0] in forbidden_hosts]
    assert leaked_paths == [], f"request reached the server: {leaked_paths}"
    assert leaked_hosts == [], f"TCP connect attempted: {leaked_hosts}"
    if require_error:
        assert isinstance(result, BaseException), (
            f"fetch returned media from a forbidden target: {result!r}"
        )


async def _run_fetch(
    server: MediaServer,
    url: str,
    *,
    modality: str,
    mode: str,
    domains: list[str] | None | object = ...,
    names: dict[str, Callable[[int], list[Any]]] | None = None,
    forbidden_paths: set[str],
    forbidden_hosts: set[str],
    require_error: bool = True,
) -> None:
    connection = HTTPConnection()
    connector = _connector(connection, domains)
    try:
        with server.egress_guard(names) as blocked:
            try:
                result: Any = await _fetch(connector, modality, url, mode)
            except BaseException as exc:  # noqa: BLE001
                result = exc
            _assert_refused(
                result,
                hits=server.hits,
                blocked=blocked,
                forbidden_paths=forbidden_paths,
                forbidden_hosts=forbidden_hosts,
                require_error=require_error,
            )
    finally:
        await _close(connection)


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", _MODALITIES)
@pytest.mark.parametrize("mode", _MODES)
async def test_default_allowlist_does_not_fetch_loopback(
    media_server: MediaServer, modality: str, mode: str
) -> None:
    """Empty allowed_media_domains is the ModelConfig default."""
    path = {"image": "/pixel.png", "audio": "/audio.wav", "video": "/video.bin"}[
        modality
    ]
    await _run_fetch(
        media_server,
        media_server.url("127.0.0.1", path),
        modality=modality,
        mode=mode,
        forbidden_paths={path},
        forbidden_hosts=set(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _MODES)
async def test_explicit_empty_domain_list_does_not_fetch_loopback(
    media_server: MediaServer, mode: str
) -> None:
    await _run_fetch(
        media_server,
        media_server.url("127.0.0.1", "/pixel.png"),
        modality="image",
        mode=mode,
        domains=[],
        forbidden_paths={"/pixel.png"},
        forbidden_hosts=set(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "::1",
        # glibc getaddrinfo maps each of these onto 127.0.0.1.
        "2130706433",
        "127.1",
        "0x7f.0.0.1",
        "0177.0.0.1",
    ],
)
async def test_loopback_spellings_are_not_fetched(
    media_server: MediaServer, mode: str, host: str
) -> None:
    await _run_fetch(
        media_server,
        media_server.url(host, "/pixel.png"),
        modality="image",
        mode=mode,
        forbidden_paths={"/pixel.png"},
        forbidden_hosts={"::1", "::ffff:127.0.0.1"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    "host",
    [
        "169.254.169.254",
        "169.254.1.1",
        "10.255.255.1",
        "192.168.0.1",
        "172.16.0.1",
        "172.31.255.1",
        "fd00::1",
        "fe80::1",
        "0.0.0.0",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
    ],
)
async def test_private_link_local_and_metadata_literals_are_not_contacted(
    media_server: MediaServer, mode: str, host: str
) -> None:
    normalized = str(ipaddress.ip_address(host))
    await _run_fetch(
        media_server,
        media_server.url(host, "/latest/meta-data/"),
        modality="image",
        mode=mode,
        forbidden_paths=set(),
        forbidden_hosts={normalized},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _MODES)
async def test_loopback_http_status_is_not_observable(
    media_server: MediaServer, mode: str
) -> None:
    connection = HTTPConnection()
    connector = _connector(connection)
    url = media_server.url("127.0.0.1", "/missing")
    try:
        with media_server.egress_guard() as blocked:
            try:
                result: Any = await _fetch(connector, "image", url, mode)
            except BaseException as exc:  # noqa: BLE001
                result = exc
            _assert_refused(
                result,
                hits=media_server.hits,
                blocked=blocked,
                forbidden_paths={"/missing"},
                forbidden_hosts=set(),
                require_error=True,
            )
            assert "HTTP 404" not in str(result)
    finally:
        await _close(connection)


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", _MODALITIES)
@pytest.mark.parametrize("mode", _MODES)
async def test_allowlisted_redirect_does_not_reach_metadata(
    media_server: MediaServer, modality: str, mode: str
) -> None:
    names = {"allowed.test": _records(["127.0.0.1"])}
    url = media_server.url("allowed.test", "/redirect-meta")
    with media_server.egress_guard(names):
        resolved = socket.getaddrinfo("allowed.test", media_server.port)
    assert resolved[0][4][0] == "127.0.0.1"
    await _run_fetch(
        media_server,
        url,
        modality=modality,
        mode=mode,
        domains=["allowed.test"],
        names=names,
        forbidden_paths=set(),
        forbidden_hosts={"169.254.169.254"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", _MODALITIES)
@pytest.mark.parametrize("mode", _MODES)
async def test_allowlisted_redirect_does_not_reach_unlisted_host(
    media_server: MediaServer, modality: str, mode: str
) -> None:
    """Checking the original hostname again still follows this redirect."""
    names = {
        "allowed.test": _records(["127.0.0.1"]),
        "evil.test": _records(["127.0.0.1"]),
    }
    await _run_fetch(
        media_server,
        media_server.url("allowed.test", "/redirect-evil"),
        modality=modality,
        mode=mode,
        domains=["allowed.test"],
        names=names,
        forbidden_paths={"/pixel.png"},
        forbidden_hosts=set(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _MODES)
async def test_allowlisted_redirect_does_not_contact_unlisted_public_host(
    media_server: MediaServer, mode: str
) -> None:
    """The target is a public address, so an address-class block does not cover it.

    Re-checking only the original hostname still lets the client follow it.
    """
    names = {
        "allowed.test": _records(["127.0.0.1"]),
        "other.example": _records(["203.0.113.51"]),
    }
    await _run_fetch(
        media_server,
        media_server.url("allowed.test", "/redirect-other"),
        modality="image",
        mode=mode,
        domains=["allowed.test"],
        names=names,
        forbidden_paths=set(),
        forbidden_hosts={"203.0.113.51"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", _MODES)
async def test_mixed_dns_answer_does_not_connect_to_loopback(
    media_server: MediaServer, mode: str
) -> None:
    """First address is public. A later address in the same answer is loopback.

    A check that accepts the name, or accepts only the first address, and then
    lets the HTTP client resolve again still connects to the loopback address.
    """
    names = {"rebind.allowed.test": _records([_PUBLIC_DECOY, "127.0.0.1"])}
    await _run_fetch(
        media_server,
        media_server.url("rebind.allowed.test", "/pixel.png"),
        modality="image",
        mode=mode,
        domains=["rebind.allowed.test"],
        names=names,
        forbidden_paths={"/pixel.png"},
        # 203.0.113.50 may be contacted. 127.0.0.1:port is the server, so a
        # connect there shows up as /pixel.png rather than a blocked attempt.
        forbidden_hosts=set(),
    )


@pytest.mark.asyncio
async def test_batch_download_without_allowlist_does_not_fetch_loopback(
    media_server: MediaServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = HTTPConnection()
    monkeypatch.setattr(run_batch, "global_http_connection", connection)
    url = media_server.url("127.0.0.1", "/audio.wav")
    try:
        with media_server.egress_guard() as blocked:
            try:
                result: Any = await download_bytes_from_url(
                    url, allowed_media_domains=None
                )
            except BaseException as exc:  # noqa: BLE001
                result = exc
            _assert_refused(
                result,
                hits=media_server.hits,
                blocked=blocked,
                forbidden_paths={"/audio.wav"},
                forbidden_hosts=set(),
                require_error=True,
            )
    finally:
        await _close(connection)


@pytest.mark.asyncio
async def test_batch_download_allowlist_does_not_follow_redirect_to_metadata(
    media_server: MediaServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = HTTPConnection()
    monkeypatch.setattr(run_batch, "global_http_connection", connection)
    names = {"allowed.test": _records(["127.0.0.1"])}
    url = media_server.url("allowed.test", "/redirect-meta")
    try:
        with media_server.egress_guard(names) as blocked:
            try:
                result: Any = await download_bytes_from_url(
                    url, allowed_media_domains=["allowed.test"]
                )
            except BaseException as exc:  # noqa: BLE001
                result = exc
            _assert_refused(
                result,
                hits=media_server.hits,
                blocked=blocked,
                forbidden_paths=set(),
                forbidden_hosts={"169.254.169.254"},
                # A 302 that is not followed is an empty body, not an error.
                require_error=False,
            )
    finally:
        await _close(connection)


@pytest.mark.asyncio
async def test_batch_download_allowlist_does_not_contact_unlisted_public_host(
    media_server: MediaServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = HTTPConnection()
    monkeypatch.setattr(run_batch, "global_http_connection", connection)
    names = {
        "allowed.test": _records(["127.0.0.1"]),
        "other.example": _records(["203.0.113.51"]),
    }
    url = media_server.url("allowed.test", "/redirect-other")
    try:
        with media_server.egress_guard(names) as blocked:
            try:
                result: Any = await download_bytes_from_url(
                    url, allowed_media_domains=["allowed.test"]
                )
            except BaseException as exc:  # noqa: BLE001
                result = exc
            _assert_refused(
                result,
                hits=media_server.hits,
                blocked=blocked,
                forbidden_paths=set(),
                forbidden_hosts={"203.0.113.51"},
                require_error=False,
            )
    finally:
        await _close(connection)


@pytest.mark.asyncio
async def test_batch_download_allowlist_does_not_follow_redirect_to_other_host(
    media_server: MediaServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = HTTPConnection()
    monkeypatch.setattr(run_batch, "global_http_connection", connection)
    names = {
        "allowed.test": _records(["127.0.0.1"]),
        "evil.test": _records(["127.0.0.1"]),
    }
    url = media_server.url("allowed.test", "/redirect-evil")
    try:
        with media_server.egress_guard(names) as blocked:
            try:
                result: Any = await download_bytes_from_url(
                    url, allowed_media_domains=["allowed.test"]
                )
            except BaseException as exc:  # noqa: BLE001
                result = exc
            _assert_refused(
                result,
                hits=media_server.hits,
                blocked=blocked,
                forbidden_paths={"/pixel.png"},
                forbidden_hosts=set(),
                require_error=False,
            )
    finally:
        await _close(connection)


def test_offline_helpers_do_not_fetch_loopback(media_server: MediaServer) -> None:
    calls = (
        ("image", fetch_image, "/pixel.png"),
        ("audio", fetch_audio, "/audio.wav"),
        ("video", fetch_video, "/video.bin"),
    )
    for _modality, fn, path in calls:
        media_server.hits.clear()
        url = media_server.url("127.0.0.1", path)
        with media_server.egress_guard() as blocked:
            try:
                result: Any = fn(url)
            except BaseException as exc:  # noqa: BLE001
                result = exc
            _assert_refused(
                result,
                hits=media_server.hits,
                blocked=blocked,
                forbidden_paths={path},
                forbidden_hosts=set(),
                require_error=True,
            )
