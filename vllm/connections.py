# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextvars
import errno
import functools
import ipaddress
import socket
import threading
import time
from collections.abc import Callable, Coroutine, Mapping, MutableMapping
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from urllib.parse import urljoin

import aiohttp
import requests
import urllib3.connection
from aiohappyeyeballs import AddrInfoType
from requests.adapters import HTTPAdapter
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import (
    ConnectTimeoutError,
    NameResolutionError,
    NewConnectionError,
)
from urllib3.util import parse_url
from urllib3.util.connection import allowed_gai_family

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.utils.mem_constants import KiB_bytes, MiB_bytes
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")

# Multiplier applied to timeout and sleep on each retry attempt.
# Attempt N uses: base_timeout * (_RETRY_BACKOFF_FACTOR ** N) for the
# per-attempt timeout and sleeps _RETRY_BACKOFF_FACTOR ** N seconds.
_RETRY_BACKOFF_FACTOR = 4
_RESPONSE_READ_CHUNK_SIZE = 64 * KiB_bytes


class HTTPResponseSizeExceededError(VLLMValidationError):
    """Raised when an HTTP response exceeds a caller-supplied byte ceiling."""

    def __init__(self, max_bytes: int, received_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.received_bytes = received_bytes
        super().__init__(
            "Maximum file size exceeded: HTTP response exceeded maximum size "
            f"of {max_bytes} bytes (received at least {received_bytes} bytes)",
            parameter="audio_filesize_mb",
            value=received_bytes / MiB_bytes,
        )


class MediaDownloadSizeExceededError(ValueError):
    """Raised when a remote media download exceeds the configured byte limit."""

    def __init__(self, max_bytes: int, received_bytes: int | None = None) -> None:
        self.max_bytes = max_bytes
        self.received_bytes = received_bytes
        max_size_mb = max_bytes / MiB_bytes
        message = (
            "Remote media download exceeds "
            f"VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB={max_size_mb:g}"
        )
        if received_bytes is not None:
            message += f" (received at least {received_bytes} bytes)"
        super().__init__(message)


class MediaRequestDownloadSizeExceededError(ValueError):
    """Raised when all HTTP media downloads in one request exceed the limit."""


class MediaRequestDownloadBudget:
    """Thread-safe total of HTTP media bytes received for one request."""

    def __init__(self) -> None:
        self.used = 0
        self._lock = threading.Lock()

    def charge(self, size: int) -> None:
        limit_mb = envs.VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB
        if limit_mb <= 0:
            return
        limit = limit_mb * MiB_bytes
        with self._lock:
            received = self.used + size
            if received > limit:
                raise MediaRequestDownloadSizeExceededError(
                    "Request media downloads exceed "
                    f"VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB={limit_mb:g} "
                    f"({received} bytes received, including retried attempts)."
                )
            self.used = received


class MediaURLNotAllowedError(VLLMValidationError):
    """Raised when a media fetch would connect to a refused address or URL."""


class _AddressNotAllowedError(OSError):
    """Raised per address inside aiohttp's connector, which only handles
    ``OSError`` when trying the resolved addresses in turn."""


_ADDRESS_NOT_ALLOWED_MSG = (
    "Media URL host is not allowed: it is not a publicly routable address. "
    "To fetch from it, add its network to VLLM_MEDIA_URL_ALLOWED_NETWORKS."
)
_PROXY_NOT_ALLOWED_MSG = (
    "Media URL fetches through a proxy require the proxy's address to be "
    "listed in VLLM_MEDIA_URL_ALLOWED_NETWORKS."
)
_MAX_REDIRECTS = 10
_NAT64_NETWORK = ipaddress.IPv6Network("64:ff9b::/96")
_IPV4_COMPATIBLE_NETWORK = ipaddress.IPv6Network("::/96")
_IPV6_GLOBAL_UNICAST_NETWORK = ipaddress.IPv6Network("2000::/3")


@functools.lru_cache(maxsize=8)
def _parse_networks(
    specs: tuple[str, ...],
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    return tuple(ipaddress.ip_network(spec, strict=False) for spec in specs)


def _embedded_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address in _NAT64_NETWORK or address in _IPV4_COMPATIBLE_NETWORK:
        return ipaddress.IPv4Address(int(address) & 0xFFFF_FFFF)
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    return None


def is_media_address_allowed(address: str, *, via_proxy: bool = False) -> bool:
    """Return whether a media fetch may connect to the IP ``address``.

    Globally routable unicast addresses (for IPv6, only within ``2000::/3``)
    are allowed, as is any address in `VLLM_MEDIA_URL_ALLOWED_NETWORKS`. IPv6
    addresses that embed an IPv4 address (mapped, IPv4-compatible, NAT64,
    6to4, Teredo) are judged by the embedded address.

    A proxy (``via_proxy``) resolves and connects to destinations itself, so
    it is allowed only when listed in `VLLM_MEDIA_URL_ALLOWED_NETWORKS`.
    """
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None:
            candidates.append(embedded)

    networks = _parse_networks(tuple(envs.VLLM_MEDIA_URL_ALLOWED_NETWORKS))
    if any(candidate in network for candidate in candidates for network in networks):
        return True
    if via_proxy:
        return False

    target = candidates[-1]
    if (
        isinstance(target, ipaddress.IPv6Address)
        and target not in _IPV6_GLOBAL_UNICAST_NETWORK
    ):
        return False
    return target.is_global and not target.is_multicast


def _literal_host_address(host: str) -> str | None:
    """Return the IPv4/IPv6 address a URL host spells, without DNS.

    Includes the legacy IPv4 forms (``127.1``, ``0x7f.0.0.1``, ``2130706433``)
    that the system resolver maps onto an address.
    """
    host = host.strip("[]")
    try:
        return str(ipaddress.ip_address(host.split("%", 1)[0]))
    except ValueError:
        pass
    try:
        return socket.inet_ntoa(socket.inet_aton(host))
    except OSError:
        return None


class _AddressCheckingConnectionMixin:
    """Connect only to resolved addresses that pass `is_media_address_allowed`.

    Each address is checked and then pinned as the connect target, so the
    socket is opened to exactly the address that was checked. TLS still
    verifies against the original host name.
    """

    host: str
    port: int
    proxy: Any
    _dns_host: str

    def _new_conn(self) -> socket.socket:
        dns_host = self._dns_host
        try:
            infos = socket.getaddrinfo(
                dns_host.strip("[]"),
                self.port,
                allowed_gai_family(),
                socket.SOCK_STREAM,
            )
        except socket.gaierror as e:
            raise NameResolutionError(self.host, self, e) from e  # type: ignore[arg-type]

        # With a proxy, the socket goes to the proxy, not the URL's host.
        via_proxy = self.proxy is not None
        addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
        allowed = [
            address
            for address in addresses
            if is_media_address_allowed(address, via_proxy=via_proxy)
        ]
        if not allowed:
            raise MediaURLNotAllowedError(
                _PROXY_NOT_ALLOWED_MSG if via_proxy else _ADDRESS_NOT_ALLOWED_MSG
            )

        error: Exception | None = None
        try:
            for address in allowed:
                self._dns_host = address
                try:
                    return super()._new_conn()  # type: ignore[misc]
                except (NewConnectionError, ConnectTimeoutError) as e:
                    error = e
        finally:
            self._dns_host = dns_host
        assert error is not None
        raise error


class _AddressCheckingHTTPConnection(
    _AddressCheckingConnectionMixin, urllib3.connection.HTTPConnection
):
    pass


class _AddressCheckingHTTPSConnection(
    _AddressCheckingConnectionMixin, urllib3.connection.HTTPSConnection
):
    pass


class _AddressCheckingHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _AddressCheckingHTTPConnection


class _AddressCheckingHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _AddressCheckingHTTPSConnection


_ADDRESS_CHECKING_POOL_CLASSES = {
    "http": _AddressCheckingHTTPConnectionPool,
    "https": _AddressCheckingHTTPSConnectionPool,
}


class _AddressCheckingAdapter(HTTPAdapter):
    """Check every address requests connects to, the proxy's included."""

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = _ADDRESS_CHECKING_POOL_CLASSES

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any) -> Any:
        if proxy.lower().startswith("socks"):
            raise MediaURLNotAllowedError(
                "SOCKS proxies are not supported for media URL fetches."
            )
        manager = super().proxy_manager_for(proxy, **proxy_kwargs)
        manager.pool_classes_by_scheme = _ADDRESS_CHECKING_POOL_CLASSES
        return manager


_connecting_via_proxy: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_connecting_via_proxy", default=False
)


def _address_checking_socket_factory(addr_info: AddrInfoType) -> socket.socket:
    family, type_, proto, _, sockaddr = addr_info
    via_proxy = _connecting_via_proxy.get()
    if not is_media_address_allowed(str(sockaddr[0]), via_proxy=via_proxy):
        raise _AddressNotAllowedError(
            errno.EACCES,
            _PROXY_NOT_ALLOWED_MSG if via_proxy else _ADDRESS_NOT_ALLOWED_MSG,
        )
    return socket.socket(family=family, type=type_, proto=proto)


class _AddressCheckingTCPConnector(aiohttp.TCPConnector):
    """Check every address aiohttp connects to, the proxy's included."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(socket_factory=_address_checking_socket_factory, **kwargs)

    async def _create_connection(
        self,
        req: aiohttp.ClientRequest,
        traces: list[Any],
        timeout: aiohttp.ClientTimeout,
    ) -> Any:
        token = _connecting_via_proxy.set(req.proxy is not None)
        try:
            return await super()._create_connection(req, traces, timeout)
        finally:
            _connecting_via_proxy.reset(token)


def _get_media_download_limit_bytes() -> int | None:
    max_size_mb = envs.VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB
    if max_size_mb <= 0:
        return None
    return max_size_mb * MiB_bytes


def _get_response_size_limit(max_bytes: int | None) -> tuple[int | None, bool]:
    media_max_bytes = _get_media_download_limit_bytes()
    if media_max_bytes is not None and (
        max_bytes is None or media_max_bytes <= max_bytes
    ):
        return media_max_bytes, True
    return max_bytes, False


def _raise_size_exceeded(
    max_bytes: int, received_bytes: int, media_download_limit: bool
) -> None:
    if media_download_limit:
        raise MediaDownloadSizeExceededError(max_bytes, received_bytes)
    raise HTTPResponseSizeExceededError(max_bytes, received_bytes)


def _check_content_length(
    content_length: str | int | None,
    max_bytes: int | None,
    media_download_limit: bool,
) -> None:
    if content_length is None or max_bytes is None:
        return
    try:
        parsed_length = int(content_length)
    except (TypeError, ValueError):
        return
    if parsed_length > max_bytes:
        _raise_size_exceeded(max_bytes, parsed_length, media_download_limit)


def _append_chunk(
    chunks: list[bytes],
    chunk: bytes,
    received_bytes: int,
    max_bytes: int | None,
    media_download_limit: bool,
    request_budget: MediaRequestDownloadBudget | None = None,
) -> int:
    if not chunk:
        return received_bytes
    received_bytes += len(chunk)
    if max_bytes is not None and received_bytes > max_bytes:
        _raise_size_exceeded(max_bytes, received_bytes, media_download_limit)
    if request_budget is not None:
        request_budget.charge(len(chunk))
    chunks.append(chunk)
    return received_bytes


def _read_response_bytes(
    response: requests.Response,
    max_bytes: int | None,
    request_budget: MediaRequestDownloadBudget | None = None,
) -> bytes:
    max_bytes, media_download_limit = _get_response_size_limit(max_bytes)
    if max_bytes is None and request_budget is None:
        return response.content

    _check_content_length(
        response.headers.get("Content-Length"),
        max_bytes,
        media_download_limit,
    )
    chunks: list[bytes] = []
    received_bytes = 0
    for chunk in response.iter_content(_RESPONSE_READ_CHUNK_SIZE):
        received_bytes = _append_chunk(
            chunks,
            chunk,
            received_bytes,
            max_bytes,
            media_download_limit,
            request_budget,
        )
    return b"".join(chunks)


async def _async_read_response_bytes(
    response: aiohttp.ClientResponse,
    max_bytes: int | None,
    request_budget: MediaRequestDownloadBudget | None = None,
) -> bytes:
    max_bytes, media_download_limit = _get_response_size_limit(max_bytes)
    if max_bytes is None and request_budget is None:
        return await response.read()

    _check_content_length(response.content_length, max_bytes, media_download_limit)
    chunks: list[bytes] = []
    received_bytes = 0
    async for chunk in response.content.iter_chunked(_RESPONSE_READ_CHUNK_SIZE):
        received_bytes = _append_chunk(
            chunks,
            chunk,
            received_bytes,
            max_bytes,
            media_download_limit,
            request_budget,
        )
    return b"".join(chunks)


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors that are worth retrying.

    Retryable:
      - Timeouts (aiohttp, requests, stdlib)
      - Connection-level failures (refused, reset, DNS)
      - Server errors (5xx) -- includes S3 503 SlowDown
    Not retryable:
      - Client errors (4xx) -- bad URL, auth, not-found
      - Programming errors (ValueError, TypeError, ...)
    """
    # Timeouts
    if isinstance(
        exc,
        (
            TimeoutError,
            asyncio.TimeoutError,
            requests.exceptions.Timeout,
            aiohttp.ServerTimeoutError,
        ),
    ):
        return True
    # Connection-level failures
    if isinstance(
        exc,
        (
            ConnectionError,
            aiohttp.ClientConnectionError,
            requests.exceptions.ConnectionError,
        ),
    ):
        return True
    # aiohttp server-side disconnects
    if isinstance(exc, aiohttp.ServerDisconnectedError):
        return True
    # requests 5xx -- raise_for_status() throws HTTPError
    if (
        isinstance(exc, requests.exceptions.HTTPError)
        and exc.response is not None
        and exc.response.status_code >= 500
    ):
        return True
    # aiohttp 5xx -- raise_for_status() throws ClientResponseError
    return isinstance(exc, aiohttp.ClientResponseError) and exc.status >= 500


def _log_retry(
    args: tuple,
    kwargs: dict,
    attempt: int,
    max_retries: int,
    attempt_timeout: float | None,
    exc: Exception,
    backoff: float,
    base_timeout: float | None,
) -> None:
    # args[0] is `self` (bound method), args[1] is the URL
    url = args[1] if len(args) > 1 else kwargs.get("url")
    timeout_info = (
        f"timeout={attempt_timeout:.3f}s" if base_timeout is not None else "no timeout"
    )
    next_timeout = (
        f" with timeout={base_timeout * (_RETRY_BACKOFF_FACTOR ** (attempt + 1)):.3f}s"
        if base_timeout is not None
        else ""
    )
    logger.warning(
        "HTTP fetch failed for %s (attempt %d/%d, %s): %s -- retrying in %.3fs%s",
        url,
        attempt + 1,
        max_retries,
        timeout_info,
        exc,
        backoff,
        next_timeout,
    )


def _sync_retry(
    fn: Callable[_P, _T],
) -> Callable[_P, _T]:
    """Add retry logic with exponential backoff to a sync method.

    The decorated method must accept ``timeout`` as a keyword argument.
    The decorator replaces it with a per-attempt timeout that grows by
    ``_RETRY_BACKOFF_FACTOR`` on each retry so transient slowness on busy
    hosts is absorbed.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        base_timeout: float | None = kwargs.get("timeout")
        max_retries = max(envs.VLLM_MEDIA_FETCH_MAX_RETRIES, 1)

        for attempt in range(max_retries):
            attempt_timeout = (
                base_timeout * (_RETRY_BACKOFF_FACTOR**attempt)
                if base_timeout is not None
                else None
            )
            kwargs["timeout"] = attempt_timeout
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if not _is_retryable(e) or attempt + 1 >= max_retries:
                    raise
                backoff = _RETRY_BACKOFF_FACTOR**attempt
                _log_retry(
                    args,
                    kwargs,
                    attempt,
                    max_retries,
                    attempt_timeout,
                    e,
                    backoff,
                    base_timeout,
                )
                time.sleep(backoff)

        raise AssertionError("unreachable")

    return wrapper  # type: ignore[return-value]


def _async_retry(
    fn: Callable[_P, Coroutine[Any, Any, _T]],
) -> Callable[_P, Coroutine[Any, Any, _T]]:
    """Add retry logic with exponential backoff to an async method.

    The decorated method must accept ``timeout`` as a keyword argument.
    The decorator replaces it with a per-attempt timeout that grows by
    ``_RETRY_BACKOFF_FACTOR`` on each retry so transient slowness on busy
    hosts is absorbed.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> _T:
        base_timeout: float | None = kwargs.get("timeout")
        max_retries = max(envs.VLLM_MEDIA_FETCH_MAX_RETRIES, 1)

        for attempt in range(max_retries):
            attempt_timeout = (
                base_timeout * (_RETRY_BACKOFF_FACTOR**attempt)
                if base_timeout is not None
                else None
            )
            kwargs["timeout"] = attempt_timeout
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if not _is_retryable(e) or attempt + 1 >= max_retries:
                    raise
                backoff = _RETRY_BACKOFF_FACTOR**attempt
                _log_retry(
                    args,
                    kwargs,
                    attempt,
                    max_retries,
                    attempt_timeout,
                    e,
                    backoff,
                    base_timeout,
                )
                await asyncio.sleep(backoff)

        raise AssertionError("unreachable")

    return wrapper  # type: ignore[return-value]


_REDIRECT_STATUSES = frozenset((301, 302, 303, 307, 308))


class HTTPConnection:
    """Helper class to send HTTP requests.

    Args:
        reuse_client: Reuse one client session across requests.
        restrict_addresses: Connect only to addresses accepted by
            `is_media_address_allowed`, for every resolved address, redirect
            hop, and proxy. Disable only for URLs supplied by the operator,
            never for URLs from API requests.

    """

    def __init__(
        self, *, reuse_client: bool = True, restrict_addresses: bool = True
    ) -> None:
        super().__init__()

        self.reuse_client = reuse_client
        self.restrict_addresses = restrict_addresses

        self._sync_client: requests.Session | None = None
        self._async_client: aiohttp.ClientSession | None = None

    def get_sync_client(self) -> requests.Session:
        if self._sync_client is None or not self.reuse_client:
            session = requests.Session()
            if self.restrict_addresses:
                adapter = _AddressCheckingAdapter()
                session.mount("http://", adapter)
                session.mount("https://", adapter)
            self._sync_client = session

        return self._sync_client

    # NOTE: We intentionally use an async function even though it is not
    # required, so that the client is only accessible inside async event loop
    async def get_async_client(self) -> aiohttp.ClientSession:
        if self._async_client is None or not self.reuse_client:
            connector = (
                _AddressCheckingTCPConnector() if self.restrict_addresses else None
            )
            self._async_client = aiohttp.ClientSession(
                trust_env=True, connector=connector
            )

        return self._async_client

    def _validate_http_url(self, url: str):
        parsed_url = parse_url(url)

        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(
                "Invalid HTTP URL: A valid HTTP URL must have scheme 'http' or 'https'."
            )

        # Literal addresses are refused before any connection, which also
        # covers requests sent through a proxy that would resolve them.
        if self.restrict_addresses and parsed_url.host:
            address = _literal_host_address(parsed_url.host)
            if address is not None and not is_media_address_allowed(address):
                raise MediaURLNotAllowedError(_ADDRESS_NOT_ALLOWED_MSG)

    def _redirect_target(
        self,
        url: str,
        location: str,
        validate_redirect: Callable[[str], None] | None,
    ) -> str:
        target = parse_url(urljoin(url, location)).url
        if validate_redirect is not None:
            try:
                validate_redirect(target)
            except ValueError as e:
                raise MediaURLNotAllowedError(
                    f"Media URL redirect was refused: {e}"
                ) from e
        return target

    def _headers(self, **extras: str) -> MutableMapping[str, str]:
        return {"User-Agent": f"vLLM/{VLLM_VERSION}", **extras}

    def get_response(
        self,
        url: str,
        *,
        stream: bool = False,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ):
        self._validate_http_url(url)

        client = self.get_sync_client()
        extra_headers = extra_headers or {}

        return client.get(
            url,
            headers=self._headers(**extra_headers),
            stream=stream,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )

    async def get_async_response(
        self,
        url: str,
        *,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ):
        self._validate_http_url(url)

        client = await self.get_async_client()
        extra_headers = extra_headers or {}

        return client.get(
            url,
            headers=self._headers(**extra_headers),
            timeout=timeout,
            allow_redirects=allow_redirects,
        )

    @_sync_retry
    def get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        allow_redirects: bool = True,
        max_bytes: int | None = None,
        request_budget: MediaRequestDownloadBudget | None = None,
        validate_redirect: Callable[[str], None] | None = None,
    ) -> bytes:
        """Fetch ``url``. Redirects are followed one hop at a time, and each
        target is passed to ``validate_redirect`` before it is requested."""
        for _ in range(_MAX_REDIRECTS + 1):
            with self.get_response(
                url,
                stream=True,
                timeout=timeout,
                allow_redirects=False,
            ) as r:
                location = (
                    self.get_sync_client().get_redirect_target(r)
                    if allow_redirects
                    else None
                )
                if location is None:
                    r.raise_for_status()
                    return _read_response_bytes(r, max_bytes, request_budget)
            url = self._redirect_target(url, location, validate_redirect)

        raise MediaURLNotAllowedError(
            f"Media URL exceeded the limit of {_MAX_REDIRECTS} redirects."
        )

    @_async_retry
    async def async_get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        allow_redirects: bool = True,
        max_bytes: int | None = None,
        request_budget: MediaRequestDownloadBudget | None = None,
        validate_redirect: Callable[[str], None] | None = None,
    ) -> bytes:
        """Async version of `get_bytes`."""
        for _ in range(_MAX_REDIRECTS + 1):
            try:
                async with await self.get_async_response(
                    url,
                    timeout=timeout,
                    allow_redirects=False,
                ) as r:
                    location = (
                        r.headers.get(aiohttp.hdrs.LOCATION)
                        if allow_redirects and r.status in _REDIRECT_STATUSES
                        else None
                    )
                    if location is None:
                        r.raise_for_status()
                        return await _async_read_response_bytes(
                            r, max_bytes, request_budget
                        )
            except aiohttp.ClientConnectorError as e:
                if isinstance(e.os_error, _AddressNotAllowedError):
                    raise MediaURLNotAllowedError(str(e.os_error.strerror)) from e
                raise
            url = self._redirect_target(url, location, validate_redirect)

        raise MediaURLNotAllowedError(
            f"Media URL exceeded the limit of {_MAX_REDIRECTS} redirects."
        )

    def get_text(self, url: str, *, timeout: float | None = None) -> str:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return r.text

    async def async_get_text(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> str:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return await r.text()

    def get_json(self, url: str, *, timeout: float | None = None) -> str:
        with self.get_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return r.json()

    async def async_get_json(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> str:
        async with await self.get_async_response(url, timeout=timeout) as r:
            r.raise_for_status()

            return await r.json()

    @_sync_retry
    def download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path:
        try:
            with self.get_response(url, timeout=timeout) as r:
                r.raise_for_status()

                with save_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size):
                        f.write(chunk)

            return save_path
        except Exception:
            # Clean up partial downloads before retrying or propagating
            if save_path.exists():
                save_path.unlink()
            raise

    @_async_retry
    async def async_download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path:
        try:
            async with await self.get_async_response(
                url,
                timeout=timeout,
            ) as r:
                r.raise_for_status()

                with save_path.open("wb") as f:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        f.write(chunk)

            return save_path
        except Exception:
            # Clean up partial downloads before retrying or propagating
            if save_path.exists():
                save_path.unlink()
            raise


global_http_connection = HTTPConnection()
"""
The global [`HTTPConnection`][vllm.connections.HTTPConnection] instance used
by vLLM.
"""

trusted_http_connection = HTTPConnection(restrict_addresses=False)
"""
An [`HTTPConnection`][vllm.connections.HTTPConnection] without the address
restriction, for URLs supplied by the operator (e.g. a `run-batch` input file).
"""
