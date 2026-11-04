# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import secrets
from collections.abc import Awaitable

from starlette._utils import get_route_path
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

PUBLIC_PATHS = frozenset({"/health", "/ping"})


class AuthenticationMiddleware:
    """Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Every route requires the token except:
        1. CORS preflight requests, which `CORSMiddleware` answers before
           routing.
        2. Paths in PUBLIC_PATHS (`/health` and `/ping`).
    """

    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]

    def verify_token(self, headers: Headers) -> bool:
        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()

        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match

    @staticmethod
    def is_public(scope: Scope, headers: Headers) -> bool:
        # Must match the condition under which Starlette's CORSMiddleware
        # answers a preflight itself; any other OPTIONS request reaches the
        # router (e.g. the /metrics mount answers OPTIONS).
        if (
            scope["type"] == "http"
            and scope["method"] == "OPTIONS"
            and "origin" in headers
            and "access-control-request-method" in headers
        ):
            return True
        # Same path the router matches routes against.
        return get_route_path(scope) in PUBLIC_PATHS

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket"):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        headers = Headers(scope=scope)
        if self.is_public(scope, headers) or self.verify_token(headers):
            return self.app(scope, receive, send)
        response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
        return response(scope, receive, send)
