# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm import envs


class RequestBodyTooLarge(Exception):
    pass


def _error_response(status: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": status.phrase,
                "param": None,
                "code": status.value,
            }
        },
        status_code=status.value,
        headers={"Connection": "close"},
    )


class RequestBodyLimitMiddleware:
    """Pure ASGI middleware that rejects HTTP request bodies larger than
    ``max_body_size`` bytes with 413, before the app has buffered them.

    A declared ``Content-Length`` over the limit is rejected without reading
    the body. Otherwise body bytes are counted as they arrive, and the chunk
    that crosses the limit is never handed to the app.
    """

    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        if max_body_size <= 0:
            raise ValueError(f"max_body_size must be positive, got {max_body_size}")
        self.app = app
        self.max_body_size = max_body_size

    def _too_large(self) -> JSONResponse:
        return _error_response(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"Request body exceeds the limit of {self.max_body_size} bytes",
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope["headers"]:
            if name.lower() != b"content-length":
                continue
            if not value.isdigit():
                response = _error_response(
                    HTTPStatus.BAD_REQUEST, "Invalid Content-Length header"
                )
                await response(scope, receive, send)
                return
            if int(value) > self.max_body_size:
                await self._too_large()(scope, receive, send)
                return

        received = 0
        rejected = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                raise RequestBodyTooLarge
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_size:
                    rejected = True
                    raise RequestBodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if rejected:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not rejected or response_started:
                raise
        if rejected:
            if response_started:
                raise RequestBodyTooLarge
            await self._too_large()(scope, receive, send)


def limit_request_body(app: ASGIApp) -> ASGIApp:
    """Wrap ``app`` with the ``VLLM_HTTP_MAX_REQUEST_BODY_SIZE`` limit; 0
    leaves it unlimited."""
    max_body_size = envs.VLLM_HTTP_MAX_REQUEST_BODY_SIZE
    if max_body_size < 0:
        raise ValueError(
            "VLLM_HTTP_MAX_REQUEST_BODY_SIZE must be a non-negative number of "
            f"bytes, got {max_body_size}"
        )
    if max_body_size == 0:
        return app
    return RequestBodyLimitMiddleware(app, max_body_size)
