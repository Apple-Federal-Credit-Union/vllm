# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import secrets
from argparse import Namespace
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from fastapi import Request
from fastapi.routing import APIRoute
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm import envs

GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")


def resolve_api_tokens(args: Namespace | None) -> list[str]:
    """Return the configured API keys; `--api-key` takes precedence over
    `VLLM_API_KEY`."""
    return [
        key for key in (getattr(args, "api_key", None) or [envs.VLLM_API_KEY]) if key
    ]


def _hash_tokens(tokens: list[str]) -> list[bytes]:
    return [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]


def _bearer_token_matches(headers: Headers, token_hashes: list[bytes]) -> bool:
    authorization_header_value = headers.get("Authorization")
    if not authorization_header_value:
        return False

    scheme, _, param = authorization_header_value.partition(" ")
    if scheme.lower() != "bearer":
        return False

    param_hash = hashlib.sha256(param.encode("utf-8")).digest()

    token_match = False
    for token_hash in token_hashes:
        token_match |= secrets.compare_digest(param_hash, token_hash)

    return token_match


class APIKeyProtectedRoute(APIRoute):
    """Route class that requires the configured API key on routes outside
    `GUARDED_PREFIX`, before the request body is read."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def protected_handler(request: Request) -> Response:
            tokens = resolve_api_tokens(getattr(request.app.state, "args", None))
            if tokens and not _bearer_token_matches(
                request.headers, _hash_tokens(tokens)
            ):
                return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return await handler(request)

        return protected_handler


class AuthenticationMiddleware:
    """Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with GUARDED_PREFIX (e.g. /health).

    Routes outside GUARDED_PREFIX can opt in with `APIKeyProtectedRoute`.

    """

    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = _hash_tokens(tokens)

    def verify_token(self, headers: Headers) -> bool:
        return _bearer_token_matches(headers, self.api_tokens)

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
