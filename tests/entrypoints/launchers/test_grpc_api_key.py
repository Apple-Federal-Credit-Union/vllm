# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API keys must gate the Python gRPC engine, not only the HTTP frontend.

``vllm serve --grpc`` dispatches to ``serve_grpc`` and never installs
``AuthenticationMiddleware``. These tests start that server and call every
``vllm.grpc.engine.VllmEngine`` method. With a key configured, an
unauthenticated client must not reach the engine. Either startup refuses
before the engine is built and before the port accepts a connection, or the
server stays up and rejects the call with ``UNAUTHENTICATED`` without
touching the engine. A configured key, presented as HTTP does
(``authorization: Bearer <key>``), must be allowed through to the engine.

The same rule applies to ``VLLM_API_KEY`` and to ``python -m
vllm.entrypoints.launchers.grpc_server``, which share ``serve_grpc``. When
both the CLI list and the environment variable are set, the CLI list is the
one that authorizes, matching ``init_entrypoints_middleware``.

Not asserted here: ``grpc.health.v1.Health`` (Check, List, Watch), server
reflection, the Rust frontend's ``--grpc-port`` Inference and Control
services, and TLS or message-size limits. The deprecated
``vllm.entrypoints.grpc_server`` module calls the same ``main``.
"""

import argparse
import asyncio
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

grpc = pytest.importorskip("grpc")
pytest.importorskip("smg_grpc_servicer")
smg_grpc_proto = pytest.importorskip("smg_grpc_proto")
protobuf_symbol_database = pytest.importorskip("google.protobuf.symbol_database")

# Imported after the skips so a missing gRPC extra skips this module.
import vllm.entrypoints.launchers.grpc_server as grpc_server  # noqa: E402
import vllm.envs as envs  # noqa: E402
from vllm.entrypoints.cli.serve import ServeSubcommand  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

vllm_engine_pb2 = smg_grpc_proto.vllm_engine_pb2
vllm_engine_pb2_grpc = smg_grpc_proto.vllm_engine_pb2_grpc
protobuf_default = protobuf_symbol_database.Default

# Obvious placeholders. Not a credential from any deployment.
CLI_KEY = "placeholder-api-key"
OTHER_CLI_KEY = "placeholder-other-api-key"
ENV_KEY = "placeholder-env-api-key"
REJECTED_KEY = "placeholder-rejected-api-key"

_LISTEN_TIMEOUT_S = 20.0
_RPC_TIMEOUT_S = 5.0


class RecordingEngine:
    """Stand-in for AsyncLLM. Records attribute reads the servicer performs."""

    def __init__(self) -> None:
        object.__setattr__(self, "events", [])

    def __getattribute__(self, name: str):
        if name in {"events", "event_count"}:
            return object.__getattribute__(self, name)
        events = object.__getattribute__(self, "events")
        events.append(name)
        if name == "vllm_config":
            return None
        if name == "model_config":
            return SimpleNamespace(is_multimodal_model=False)
        if name == "errored":
            return False
        if name == "shutdown":
            return lambda: None
        if name == "check_health":

            async def _check_health():
                events.append("check_health")
                return None

            return _check_health
        raise AttributeError(name)

    def event_count(self) -> int:
        return len(object.__getattribute__(self, "events"))


class _Running:
    def __init__(self, task: asyncio.Task, engine: RecordingEngine, port: int) -> None:
        self.task = task
        self.engine = engine
        self.port = port


class _Refused:
    def __init__(
        self, exc: BaseException, engine_built: bool, port_opened: bool
    ) -> None:
        self.exc = exc
        self.engine_built = engine_built
        self.port_opened = port_opened


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _parse_serve(argv: list[str]) -> argparse.Namespace:
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    ServeSubcommand().subparser_init(subparsers)
    return parser.parse_args(["serve", *argv])


def _engine_methods():
    service = vllm_engine_pb2.DESCRIPTOR.services_by_name["VllmEngine"]
    return list(service.methods)


def _identity(payload: bytes) -> bytes:
    return payload


def _bearer(token: str) -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {token}"),)


def _refusal_names_the_key(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "api" in text and "key" in text


async def _invoke(channel, method, metadata):
    database = protobuf_default()
    request_cls = database.GetSymbol(method.input_type.full_name)
    request = request_cls()
    path = f"/{method.containing_service.full_name}/{method.name}"
    serializer = request_cls.SerializeToString
    deserializer = _identity
    try:
        if method.server_streaming:
            caller = channel.unary_stream(
                path,
                request_serializer=serializer,
                response_deserializer=deserializer,
            )
            async for _chunk in caller(
                request, metadata=metadata, timeout=_RPC_TIMEOUT_S
            ):
                return grpc.StatusCode.OK
            return grpc.StatusCode.OK
        caller = channel.unary_unary(
            path,
            request_serializer=serializer,
            response_deserializer=deserializer,
        )
        await caller(request, metadata=metadata, timeout=_RPC_TIMEOUT_S)
        return grpc.StatusCode.OK
    except grpc.aio.AioRpcError as error:
        return error.code()


async def _stop(task: asyncio.Task) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        return


async def _wait_until_serving(port: int, task: asyncio.Task) -> bool:
    deadline = asyncio.get_running_loop().time() + _LISTEN_TIMEOUT_S
    while asyncio.get_running_loop().time() < deadline:
        if _port_open(port):
            return True
        if task.done():
            return False
        await asyncio.sleep(0.05)
    return _port_open(port)


async def _start(
    args: argparse.Namespace | None, entry: str, port: int
) -> _Running | _Refused:
    """Run the real launcher. Only engine construction is replaced."""
    created = {"engine": False}
    engines: list[RecordingEngine] = []
    tasks: list[asyncio.Task] = []

    def create_engine_config(self, *args, **kwargs):
        created["engine"] = True
        return SimpleNamespace()

    def from_vllm_config(*args, **kwargs):
        created["engine"] = True
        engine = RecordingEngine()
        engines.append(engine)
        return engine

    def schedule(coro):
        tasks.append(asyncio.get_running_loop().create_task(coro))
        return None

    patches = [
        patch.object(
            grpc_server.AsyncEngineArgs,
            "create_engine_config",
            create_engine_config,
        ),
        patch.object(grpc_server.AsyncLLM, "from_vllm_config", from_vllm_config),
    ]
    if entry == "serve":
        patches.append(patch("vllm.entrypoints.cli.serve.uvloop.run", schedule))
    else:
        patches.append(patch.object(grpc_server.uvloop, "run", schedule))

    for item in patches:
        item.start()
    try:
        try:
            if entry == "serve":
                assert args is not None
                ServeSubcommand().validate(args)
                ServeSubcommand.cmd(args)
            else:
                grpc_server.main()
        except BaseException as exc:
            return _Refused(exc, created["engine"], False)

        if not tasks:
            raise AssertionError(
                "launcher returned without starting serve_grpc or refusing"
            )
        task = tasks[0]
        port_opened = await _wait_until_serving(port, task)
        if not port_opened:
            if task.done():
                caught = task.exception()
                exc = caught if caught is not None else RuntimeError("server exited")
            else:
                exc = TimeoutError("gRPC port did not accept a connection")
                await _stop(task)
            return _Refused(exc, created["engine"], port_opened)
        if not engines:
            await _stop(task)
            raise AssertionError("server listened without building an engine")
        return _Running(task, engines[0], port)
    except BaseException:
        if tasks:
            await _stop(tasks[0])
        raise
    finally:
        for item in patches:
            item.stop()


def _assert_clean_refusal(result: _Refused) -> None:
    assert not result.port_opened
    assert not result.engine_built
    assert _refusal_names_the_key(result.exc)


async def _assert_engine_closed_to(
    running: _Running, metadata: tuple[tuple[str, str], ...] | None
) -> None:
    before = running.engine.event_count()
    async with grpc.aio.insecure_channel(f"127.0.0.1:{running.port}") as channel:
        for method in _engine_methods():
            code = await _invoke(channel, method, metadata)
            assert code == grpc.StatusCode.UNAUTHENTICATED, method.name
            assert running.engine.event_count() == before, method.name


async def _assert_health_reaches_engine(running: _Running, metadata) -> None:
    before = running.engine.event_count()
    async with grpc.aio.insecure_channel(f"127.0.0.1:{running.port}") as channel:
        stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)
        response = await stub.HealthCheck(
            vllm_engine_pb2.HealthCheckRequest(),
            metadata=metadata,
            timeout=_RPC_TIMEOUT_S,
        )
    assert response.healthy
    assert running.engine.event_count() > before


async def _run_serve(argv: list[str]) -> _Running | _Refused:
    args = _parse_serve(argv)
    return await _start(args, "serve", args.port)


async def _run_module(argv: list[str]) -> _Running | _Refused:
    port = int(argv[argv.index("--port") + 1])
    with patch.object(sys, "argv", ["grpc_server", *argv]):
        return await _start(None, "module", port)


async def _expect_protection(
    result: _Running | _Refused, accepted: list[str], rejected: list[str]
):
    if isinstance(result, _Refused):
        _assert_clean_refusal(result)
        return
    try:
        await _assert_engine_closed_to(result, None)
        for token in rejected:
            await _assert_engine_closed_to(result, _bearer(token))
        for token in accepted:
            await _assert_health_reaches_engine(result, _bearer(token))
            async with grpc.aio.insecure_channel(f"127.0.0.1:{result.port}") as channel:
                for method in _engine_methods():
                    code = await _invoke(channel, method, _bearer(token))
                    assert code != grpc.StatusCode.UNAUTHENTICATED, method.name
    finally:
        await _stop(result.task)


@pytest.fixture(autouse=True)
def _isolate_api_key_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("SMG_VLLM_SERVICER_IMPL", raising=False)
    envs.disable_envs_cache()


@pytest.mark.asyncio
async def test_serve_without_api_key_reaches_engine_health():
    """The harness reaches the real servicer when no key is configured."""
    port = _free_port()
    result = await _run_serve(
        [
            "placeholder-model",
            "--grpc",
            "--port",
            str(port),
            "--disable-log-stats",
        ]
    )
    assert isinstance(result, _Running)
    try:
        await _assert_health_reaches_engine(result, None)
    finally:
        await _stop(result.task)


@pytest.mark.asyncio
async def test_serve_cli_api_key_rejects_unauthenticated_engine_rpcs():
    """``vllm serve --grpc --api-key`` must not serve engine RPCs freely."""
    port = _free_port()
    result = await _run_serve(
        [
            "placeholder-model",
            "--grpc",
            "--port",
            str(port),
            "--disable-log-stats",
            "--api-key",
            CLI_KEY,
            OTHER_CLI_KEY,
        ]
    )
    await _expect_protection(
        result,
        accepted=[CLI_KEY, OTHER_CLI_KEY],
        rejected=[REJECTED_KEY, ENV_KEY],
    )


@pytest.mark.asyncio
async def test_serve_env_api_key_rejects_unauthenticated_engine_rpcs(monkeypatch):
    """``VLLM_API_KEY`` is the HTTP server's key and must gate ``--grpc`` too."""
    monkeypatch.setenv("VLLM_API_KEY", ENV_KEY)
    envs.disable_envs_cache()
    port = _free_port()
    result = await _run_serve(
        [
            "placeholder-model",
            "--grpc",
            "--port",
            str(port),
            "--disable-log-stats",
        ]
    )
    await _expect_protection(result, accepted=[ENV_KEY], rejected=[CLI_KEY])


@pytest.mark.asyncio
async def test_cli_api_key_overrides_env_for_grpc(monkeypatch):
    """The key that authorizes a call is the CLI list, not a different env key."""
    monkeypatch.setenv("VLLM_API_KEY", ENV_KEY)
    envs.disable_envs_cache()
    port = _free_port()
    result = await _run_serve(
        [
            "placeholder-model",
            "--grpc",
            "--port",
            str(port),
            "--disable-log-stats",
            "--api-key",
            CLI_KEY,
        ]
    )
    await _expect_protection(result, accepted=[CLI_KEY], rejected=[ENV_KEY])


@pytest.mark.asyncio
async def test_module_main_env_api_key_rejects_unauthenticated_engine_rpcs(monkeypatch):
    """``python -m`` the launcher shares ``serve_grpc`` and the env key."""
    monkeypatch.setenv("VLLM_API_KEY", ENV_KEY)
    envs.disable_envs_cache()
    port = _free_port()
    result = await _run_module(
        [
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--model",
            "placeholder-model",
            "--disable-log-stats",
        ]
    )
    await _expect_protection(result, accepted=[ENV_KEY], rejected=[CLI_KEY])
