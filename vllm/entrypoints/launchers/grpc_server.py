#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""vLLM gRPC Server

Starts a gRPC server backed by AsyncLLM, using the VllmEngineServicer
from the smg-grpc-servicer package.

Usage:
    python -m vllm.entrypoints.launchers.grpc_server --model <model_path>

Example:
    python -m vllm.entrypoints.launchers.grpc_server \
        --model meta-llama/Llama-2-7b-hf \
        --host 0.0.0.0 \
        --port 50051

"""

import argparse
import asyncio
import signal
import sys
import time

try:
    import grpc
    from grpc_health.v1 import health_pb2_grpc
    from grpc_reflection.v1alpha import reflection
    from smg_grpc_proto import vllm_engine_pb2, vllm_engine_pb2_grpc
    from smg_grpc_servicer.vllm.health_servicer import VllmHealthServicer
    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
except ImportError as e:
    raise ImportError(
        "gRPC mode requires smg-grpc-servicer. "
        "If not installed, run: pip install vllm[grpc]. "
        "If already installed, there may be a broken import due to a "
        "version mismatch — see the chained exception above for details."
    ) from e

try:
    from smg_grpc_servicer.vllm import resolve_servicer_impl
except ImportError:  # Releases without the Rust servicer switch.
    resolve_servicer_impl = None

import uvloop

from vllm import envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.common.startup import (
    log_version_and_model,
    redact_sensitive_args,
)
from vllm.entrypoints.serve.middleware.authenticate import (
    hash_api_tokens,
    resolve_api_tokens,
    verify_bearer_token,
)
from vllm.logger import configure_logging_from_args, init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

# Like the HTTP /health route, the standard health service stays open so that
# load balancers and Kubernetes gRPC probes, which send no metadata, work.
_UNGUARDED_METHOD_PREFIX = "/grpc.health.v1.Health/"


async def _reject_unauthenticated(request_or_iterator, context):
    await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Unauthorized")


class ApiKeyInterceptor(grpc.aio.ServerInterceptor):
    """Require ``authorization: Bearer <api key>`` metadata on every RPC
    except the standard health service, matching the HTTP
    ``AuthenticationMiddleware``."""

    def __init__(self, tokens: list[str]) -> None:
        self.api_tokens = hash_api_tokens(tokens)

    def verify_token(self, metadata) -> bool:
        for key, value in metadata or ():
            if key == "authorization":
                return isinstance(value, str) and verify_bearer_token(
                    value, self.api_tokens
                )
        return False

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if (
            handler is None
            or handler_call_details.method.startswith(_UNGUARDED_METHOD_PREFIX)
            or self.verify_token(handler_call_details.invocation_metadata)
        ):
            return handler
        # Match the RPC's streaming shape, and leave the request undecoded.
        if handler.request_streaming and handler.response_streaming:
            make_handler = grpc.stream_stream_rpc_method_handler
        elif handler.request_streaming:
            make_handler = grpc.stream_unary_rpc_method_handler
        elif handler.response_streaming:
            make_handler = grpc.unary_stream_rpc_method_handler
        else:
            make_handler = grpc.unary_unary_rpc_method_handler
        return make_handler(_reject_unauthenticated)


def check_api_key_support(args: argparse.Namespace) -> None:
    """Refuse an API key that the selected servicer would not enforce.

    smg-grpc-servicer replaces ``serve_grpc`` with its Rust servicer when
    ``SMG_VLLM_SERVICER_IMPL`` or ``--servicer-impl`` selects ``rust``, which
    skips ``ApiKeyInterceptor``. Must run before ``serve_grpc``.
    """
    if not resolve_api_tokens(args) or resolve_servicer_impl is None:
        return
    impl = resolve_servicer_impl(args)
    if impl != "python":
        raise ValueError(
            "An API key is set (--api-key or VLLM_API_KEY), but the "
            f"smg-grpc-servicer '{impl}' servicer does not enforce API keys. "
            "Use the Python servicer or remove the API key."
        )


async def serve_grpc(args: argparse.Namespace):
    """Main gRPC serving function.

    Args:
        args: Parsed command line arguments

    """
    log_version_and_model(logger, VLLM_VERSION, args.model)
    logger.info("vLLM gRPC server args: %s", redact_sensitive_args(vars(args)))

    start_time = time.time()

    api_tokens = resolve_api_tokens(args)
    interceptors = [ApiKeyInterceptor(api_tokens)] if api_tokens else []

    # Create engine args
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Build vLLM config
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    # Create AsyncLLM
    async_llm = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_log_requests=args.enable_log_requests,
        disable_log_stats=args.disable_log_stats,
    )

    # Create servicer
    servicer = VllmEngineServicer(async_llm, start_time)

    # Create gRPC server
    server = grpc.aio.server(
        interceptors=interceptors,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            # A non-streaming Generate writes nothing while the engine
            # decodes, so client keepalive and BDP-probe PINGs arriving in
            # that window are strikes and eventually earn a
            # GOAWAY(ENHANCE_YOUR_CALM, "too_many_pings") that fails every
            # in-flight RPC. Disable the strike count to tolerate them.
            ("grpc.http2.max_ping_strikes", 0),
            # GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS expands to
            # a string with no "recv", and gRPC Core silently ignores unknown
            # keys, so the misspelt form left the 300s default in force.
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
            # Without this, gRPC Core widens the floor to a fixed two hours
            # once no streams are open, striking keepalives between requests.
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    # Add servicer to server
    vllm_engine_pb2_grpc.add_VllmEngineServicer_to_server(servicer, server)

    # Add standard gRPC health service for Kubernetes probes
    health_servicer = VllmHealthServicer(async_llm)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Enable reflection for grpcurl and other tools
    service_names = (
        vllm_engine_pb2.DESCRIPTOR.services_by_name["VllmEngine"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    # Bind to address
    host = args.host or "0.0.0.0"
    address = f"{host}:{args.port}"
    server.add_insecure_port(address)

    try:
        # Start server
        await server.start()
        logger.info("vLLM gRPC server started on %s", address)
        logger.info("Server is ready to accept requests")

        # Start periodic stats logging (mirrors the HTTP server's lifespan task)
        if not args.disable_log_stats:

            async def _force_log():
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await async_llm.do_log_stats()

            stats_task = asyncio.create_task(_force_log())
        else:
            stats_task = None

        # Handle shutdown signals
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def signal_handler():
            logger.info("Received shutdown signal")
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
    finally:
        logger.info("Shutting down vLLM gRPC server...")
        if stats_task is not None:
            stats_task.cancel()
        try:
            health_servicer.set_not_serving()
        except Exception:  # broad: must not prevent server.stop() / shutdown()
            logger.warning("Failed to set health status to NOT_SERVING", exc_info=True)
        await server.stop(grace=5.0)
        logger.info("gRPC server stopped")
        async_llm.shutdown()
        logger.info("AsyncLLM engine stopped")
        logger.info("Shutdown complete")


def main():
    """Main entry point for python -m vllm.entrypoints.launchers.grpc_server."""
    parser = FlexibleArgumentParser(
        description="vLLM gRPC Server",
    )

    # Server args
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind gRPC server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to bind gRPC server to",
    )
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    configure_logging_from_args(args)
    check_api_key_support(args)

    # Run server
    try:
        uvloop.run(serve_grpc(args))
    except Exception as e:
        logger.exception("Server failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
