# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import secrets
from argparse import Namespace
from collections.abc import Awaitable

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

import vllm.envs as envs

GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")


def resolve_api_tokens(args: Namespace) -> list[str]:
    """Return the API keys a server must require, or an empty list.

    The ``--api-key`` option from the CLI takes precedence over
    ``VLLM_API_KEY``.
    """
    api_keys = getattr(args, "api_key", None) or [envs.VLLM_API_KEY]
    return [key for key in api_keys if key]


def hash_api_tokens(tokens: list[str]) -> list[bytes]:
    return [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]


def verify_bearer_token(authorization: str | None, token_hashes: list[bytes]) -> bool:
    """Check an ``Authorization: Bearer <key>`` value against hashed keys."""
    if not authorization:
        return False

    scheme, _, param = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return False

    param_hash = hashlib.sha256(param.encode("utf-8")).digest()

    token_match = False
    for token_hash in token_hashes:
        token_match |= secrets.compare_digest(param_hash, token_hash)

    return token_match


class AuthenticationMiddleware:
    """Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with GUARDED_PREFIX (e.g. /health).

    """

    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = hash_api_tokens(tokens)

    def verify_token(self, headers: Headers) -> bool:
        return verify_bearer_token(headers.get("Authorization"), self.api_tokens)

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
        ):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        headers = Headers(scope=scope)
        # Type narrow to satisfy mypy.
        if url_path.startswith(GUARDED_PREFIX) and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)
