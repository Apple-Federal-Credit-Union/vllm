# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI
from starlette.routing import Route

from vllm.logger import init_logger

from .exception_handling.error_response import create_error_response

logger = init_logger(__name__)

__all__ = [
    "create_error_response",
    "get_vllm_dev_api_paths",
    "register_vllm_serve_api_routers",
    "register_vllm_dev_api_routers",
]


def register_vllm_serve_api_routers(app: FastAPI):
    from .instrumentator import register_instrumentator_api_routers

    register_instrumentator_api_routers(app)

    from vllm.entrypoints.serve.lora.api_router import (
        attach_router as attach_lora_router,
    )

    attach_lora_router(app)

    from vllm.entrypoints.serve.profile.api_router import (
        attach_router as attach_profile_router,
    )

    attach_profile_router(app)

    from vllm.entrypoints.serve.tokenize.api_router import (
        attach_router as attach_tokenize_router,
    )

    attach_tokenize_router(app)


def _get_vllm_dev_api_routers() -> list[APIRouter]:
    from .dev.cache.api_router import router as cache_router
    from .dev.rpc.api_router import router as rpc_router
    from .dev.server_info.api_router import router as server_info_router
    from .dev.sleep.api_router import router as sleep_router

    return [cache_router, rpc_router, server_info_router, sleep_router]


def get_vllm_dev_api_paths() -> tuple[str, ...]:
    """Paths of the development routes, which require the API key if set."""
    return tuple(
        route.path
        for router in _get_vllm_dev_api_routers()
        for route in router.routes
        if isinstance(route, Route)
    )


def register_vllm_dev_api_routers(app: FastAPI):
    logger.warning(
        "SECURITY WARNING: Development endpoints are enabled! "
        "This should NOT be used in production!"
    )

    for router in _get_vllm_dev_api_routers():
        app.include_router(router)
