# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
import hashlib
import hmac
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import regex as re
import zmq
from zmq.auth.thread import ThreadAuthenticator
from zmq.utils import z85

from vllm import envs
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())


MIN_SIDE_CHANNEL_KEY_LEN = 32


@dataclass(frozen=True)
class SideChannelKeys:
    """CURVE keypairs (Z85) derived from ``VLLM_NIXL_SIDE_CHANNEL_KEY``."""

    server_public: bytes
    server_secret: bytes
    client_public: bytes
    client_secret: bytes


def _derive_curve_keypair(shared_key: bytes, label: bytes) -> tuple[bytes, bytes]:
    secret = z85.encode(hmac.digest(shared_key, label, hashlib.sha256))
    return zmq.curve_public(secret), secret


def get_side_channel_keys() -> SideChannelKeys | None:
    """Derive the side-channel keys, or return None when no key is set.

    Raises:
        ValueError: If the configured key is too short.
        RuntimeError: If libzmq was built without CURVE support.

    """
    shared_key = envs.VLLM_NIXL_SIDE_CHANNEL_KEY
    if not shared_key:
        return None
    if len(shared_key) < MIN_SIDE_CHANNEL_KEY_LEN:
        raise ValueError(
            "VLLM_NIXL_SIDE_CHANNEL_KEY must be at least "
            f"{MIN_SIDE_CHANNEL_KEY_LEN} characters long."
        )
    if not zmq.has("curve"):
        raise RuntimeError(
            "The NIXL side channel requires libzmq with CURVE security support."
        )
    key = shared_key.encode()
    server_public, server_secret = _derive_curve_keypair(
        key, b"vllm-nixl-side-channel-server"
    )
    client_public, client_secret = _derive_curve_keypair(
        key, b"vllm-nixl-side-channel-client"
    )
    return SideChannelKeys(server_public, server_secret, client_public, client_secret)


class _ClientKeyAllowlist:
    """ZAP credentials provider admitting only the derived client key."""

    def __init__(self, client_public: bytes):
        self._client_public = client_public

    def callback(self, domain: str, key: bytes) -> bool:
        return hmac.compare_digest(key, self._client_public)


# TODO: merge with vllm.utils.network_utils.zmq_socket_ctx
@contextlib.contextmanager
def zmq_ctx(
    socket_type: Any, addr: str, keys: SideChannelKeys | None = None
) -> Iterator[zmq.Socket]:
    """Context manager for a side-channel socket.

    With ``keys`` the socket uses CURVE: the ROUTER only completes a handshake
    with peers that prove possession of the client secret key, so peers
    without it never reach the application. A ROUTER always requires ``keys``.
    """
    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")
    if socket_type == zmq.ROUTER and keys is None:
        raise ValueError("A side-channel listener requires SideChannelKeys.")

    ctx: zmq.Context | None = None
    auth: ThreadAuthenticator | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        # CURVE options must be set before make_zmq_socket binds or connects,
        # so they are set as defaults for sockets created after this point.
        if socket_type == zmq.ROUTER:
            assert keys is not None
            auth = ThreadAuthenticator(ctx)
            auth.start()
            auth.configure_curve_callback(
                domain="*",
                credentials_provider=_ClientKeyAllowlist(keys.client_public),
            )
            ctx.setsockopt(zmq.CURVE_SERVER, 1)
            ctx.setsockopt(zmq.CURVE_SECRETKEY, keys.server_secret)
        elif keys is not None:
            ctx.setsockopt(zmq.CURVE_SERVERKEY, keys.server_public)
            ctx.setsockopt(zmq.CURVE_PUBLICKEY, keys.client_public)
            ctx.setsockopt(zmq.CURVE_SECRETKEY, keys.client_secret)
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if auth is not None:
            auth.stop()
        if ctx is not None:
            ctx.destroy(linger=0)


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)


# Trailing 8-hex randomization suffix appended by
# ``input_processor.assign_request_id`` as ``-{random_uuid():.8}``.
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def get_base_request_id(request_id: str) -> str:
    """Strip the per-request ``-<8 hex>`` randomization suffix, if present."""
    return _RANDOM_SUFFIX_RE.sub("", request_id)
