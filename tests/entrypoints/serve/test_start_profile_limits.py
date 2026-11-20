# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security spec for POST /start_profile.

An API client must not widen the operator's torch profiler settings. These
tests drive the HTTP route, the engine profile RPC, and the worker profiler.
They fail while unsanitized overrides are still applied.
"""

import asyncio
from argparse import Namespace
from pathlib import Path

import httpx
import pytest
import torch
from fastapi import FastAPI, Response

from vllm.config import ProfilerConfig
from vllm.entrypoints.serve.middleware.register import init_entrypoints_middleware
from vllm.entrypoints.serve.profile.api_router import attach_router
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.executor.abstract import Executor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, run_method
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import WorkerWrapperBase

# Obvious placeholder. Not a deployed credential.
PLACEHOLDER_API_KEY = "placeholder-api-key"
_OPERATOR_MAX_ITERATIONS = 4
_WORKER_CLASS = {
    "cpu": (CPUWorker, "vllm.v1.worker.cpu_worker.CPUWorker"),
    "cuda": (Worker, "vllm.v1.worker.gpu_worker.Worker"),
}


class _ParallelConfig:
    """Only the field Executor.profile reads when checking worker kwargs."""

    def __init__(self, worker_cls: str) -> None:
        self.worker_cls = worker_cls


class _LocalExecutor:
    """In-process stand-in for collective_rpc transport only."""

    def __init__(self, worker: WorkerWrapperBase, worker_cls: str) -> None:
        self._worker = worker
        self.vllm_config = Namespace(parallel_config=_ParallelConfig(worker_cls))

    def profile(self, *args, **kwargs) -> None:
        Executor.profile(self, *args, **kwargs)

    def collective_rpc(self, method, args=(), kwargs=None, **_unused):
        return [run_method(self._worker, method, args, kwargs or {})]


class ServingProfileEngine:
    """HTTP engine client that runs the production profile dispatch.

    AsyncLLM validates the request, the multiprocess client msgpack-encodes the
    utility call, and EngineCore invokes Executor.profile. The socket is the
    only skipped step; encode and decode still use the production codec.
    """

    def __init__(self, worker: Worker, worker_cls: str) -> None:
        self._profile_lock = asyncio.Lock()
        self._profile_session_guard_enabled = True
        self._profile_session_active = False
        self.profiler = None
        self.worker = worker
        self.engine_core = self
        self.model_executor = _LocalExecutor(
            _worker_wrapper(worker),
            worker_cls,
        )

    async def start_profile(self, *args, **kwargs) -> None:
        await AsyncLLM.start_profile(self, *args, **kwargs)

    async def stop_profile(self) -> None:
        await AsyncLLM.stop_profile(self)

    def profile(self, *args, **kwargs) -> None:
        EngineCore.profile(self, *args, **kwargs)

    async def profile_async(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
        delay_iterations: int | None = None,
        max_iterations: int | None = None,
        profiler_kwargs: dict | None = None,
    ) -> None:
        utility_args = (
            is_start,
            profile_prefix,
            delay_iterations,
            max_iterations,
            profiler_kwargs,
        )
        frames = MsgpackEncoder().encode((0, 1, "profile", utility_args))
        _client_index, _call_id, method_name, method_args = MsgpackDecoder().decode(
            frames
        )
        assert method_name == "profile"
        # Bound EngineCore.profile is what EngineCoreProc dispatches. This
        # object's own profile() is a wrapper, so convert against the real
        # signature and then call that implementation.
        method = EngineCore.profile.__get__(self, type(self))
        converted = EngineCoreProc._convert_msgspec_args(method, method_args)
        method(*converted)


def _worker_wrapper(worker: Worker) -> WorkerWrapperBase:
    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.worker = worker
    return wrapper


def _new_worker(profiler_config: ProfilerConfig) -> tuple[Worker, str]:
    worker_cls, worker_qualname = _WORKER_CLASS[current_platform.device_type]
    # CPU and CUDA workers share Worker.profile. Constructing the full worker
    # loads a model, which this limit check does not need.
    worker = object.__new__(worker_cls)
    worker.rank = 0
    worker.local_rank = 1
    worker.profiler = None
    worker.profiler_config = profiler_config
    # annotate_profile reads this for the trace annotation.
    worker.vllm_config = Namespace(profiler_config=profiler_config)
    return worker, worker_qualname


def _profiler_config(trace_dir: Path, **overrides) -> ProfilerConfig:
    values = {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir),
        "torch_profiler_with_stack": False,
        "torch_profiler_with_memory": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_record_shapes": False,
        "torch_profiler_dump_cuda_time_total": False,
        "ignore_frontend": True,
        "delay_iterations": 0,
        "max_iterations": _OPERATOR_MAX_ITERATIONS,
        "warmup_iterations": 0,
        "wait_iterations": 0,
    }
    values.update(overrides)
    return ProfilerConfig(**values)


def _trace_files(trace_dir: Path) -> list[Path]:
    if not trace_dir.exists():
        return []
    return sorted(path for path in trace_dir.rglob("*") if path.is_file())


def _session_is_active(engine: ServingProfileEngine) -> bool:
    if engine._profile_session_active:
        return True
    profiler = engine.worker.profiler
    if profiler is None:
        return False
    return bool(profiler._active or profiler.is_running)


def _run_worker_steps(worker: Worker, count: int) -> None:
    """One inference iteration: annotate_profile steps the profiler."""
    scheduler_output = SchedulerOutput.make_empty()
    for _ in range(count):
        with worker.annotate_profile(scheduler_output):
            hidden = torch.randn(16, 16)
            hidden @ hidden


def _auth_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"}


class _SameThreadClient:
    """Send ASGI requests on the calling thread.

    torch.profiler must be stepped on the thread that started it. Starlette's
    TestClient runs the app on a portal thread, which aborts Kineto.
    """

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def post(self, path: str, headers: dict[str, str] | None = None, json=None):
        transport = httpx.ASGITransport(app=self.app)

        async def _go():
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                return await client.post(path, headers=headers, json=json)

        return asyncio.run(_go())


def _build_service(
    trace_dir: Path,
    *,
    api_key: list[str] | None,
    profiler_overrides: dict | None = None,
) -> tuple[ServingProfileEngine, _SameThreadClient]:
    profiler_config = _profiler_config(trace_dir, **(profiler_overrides or {}))
    worker, worker_qualname = _new_worker(profiler_config)
    engine = ServingProfileEngine(worker, worker_qualname)
    args = Namespace(
        profiler_config=profiler_config,
        api_key=api_key,
        allowed_origins=[],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        enable_request_id_headers=False,
        middleware=[],
    )
    app = FastAPI()
    app.state.args = args
    app.state.engine_client = engine

    @app.post("/v1/completions")
    async def _completions() -> Response:
        return Response(status_code=204)

    attach_router(app)
    init_entrypoints_middleware(args, app, ())
    return engine, _SameThreadClient(app)


@pytest.fixture
def profile_service(tmp_path: Path):
    engine, client = _build_service(tmp_path, api_key=[PLACEHOLDER_API_KEY])
    try:
        yield engine, client, tmp_path
    finally:
        profiler = engine.worker.profiler
        if profiler is not None and profiler.is_running:
            profiler.stop()


def test_cpu_and_cuda_workers_share_profile_implementation() -> None:
    assert CPUWorker.profile is Worker.profile


def test_api_key_still_guards_inference_routes(profile_service) -> None:
    _engine, client, _trace_dir = profile_service
    assert client.post("/v1/completions").status_code == 401
    authed = client.post("/v1/completions", headers=_auth_header())
    assert authed.status_code == 204


def test_operator_cap_bounds_a_profile_session(profile_service) -> None:
    """A positive per-session cap still profiles and then stops."""
    engine, client, trace_dir = profile_service
    response = client.post(
        "/start_profile",
        headers=_auth_header(),
        json={"max_iterations": 2},
    )

    assert response.status_code == 200
    profiler = engine.worker.profiler
    assert isinstance(profiler, TorchProfilerWrapper)
    assert profiler.is_running
    assert profiler._max_iters == 2
    torch_profiler = profiler.profiler
    assert torch_profiler.profile_memory is False
    assert torch_profiler.with_stack is False
    assert torch_profiler.with_flops is False
    assert torch_profiler.record_shapes is False

    _run_worker_steps(engine.worker, 2)
    assert profiler.is_running
    _run_worker_steps(engine.worker, 1)
    assert not profiler.is_running
    written = _trace_files(trace_dir)
    assert written
    assert all(path.stat().st_size > 0 for path in written)


def test_unauthenticated_start_profile_does_not_run(profile_service) -> None:
    engine, client, trace_dir = profile_service
    response = client.post(
        "/start_profile",
        json={
            "max_iterations": 0,
            "profiler_kwargs": _expensive_profiler_kwargs(),
        },
    )

    assert response.status_code == 401
    assert engine.worker.profiler is None
    assert not engine._profile_session_active
    assert _trace_files(trace_dir) == []


def test_unauthenticated_stop_profile_does_not_stop_a_session(
    profile_service,
) -> None:
    engine, client, _trace_dir = profile_service
    started = client.post(
        "/start_profile",
        headers=_auth_header(),
        json={"max_iterations": 2},
    )
    assert started.status_code == 200
    assert engine.worker.profiler is not None
    assert engine.worker.profiler.is_running

    stopped = client.post("/stop_profile")

    assert stopped.status_code == 401
    assert engine.worker.profiler.is_running


def test_request_cannot_enable_disabled_torch_profiler_options(
    profile_service,
) -> None:
    engine, client, trace_dir = profile_service
    response = client.post(
        "/start_profile",
        headers=_auth_header(),
        json={
            "max_iterations": 2,
            "profiler_kwargs": _expensive_profiler_kwargs(),
        },
    )

    _assert_operator_profiler_preserved(
        response,
        engine,
        trace_dir,
        max_iterations=2,
    )


def test_zero_iteration_cap_cannot_outrun_the_operator_limit(
    profile_service,
) -> None:
    engine, client, trace_dir = profile_service
    response = client.post(
        "/start_profile",
        headers=_auth_header(),
        json={"max_iterations": 0},
    )

    if response.status_code >= 400:
        assert not _session_is_active(engine)
        assert _trace_files(trace_dir) == []
        return

    profiler = engine.worker.profiler
    assert profiler is not None
    assert profiler._max_iters == _OPERATOR_MAX_ITERATIONS
    _run_worker_steps(engine.worker, _OPERATOR_MAX_ITERATIONS + 1)
    assert not profiler.is_running


def test_profiler_kwargs_cannot_replace_handler_activities_or_schedule(
    tmp_path: Path,
) -> None:
    engine, client = _build_service(
        tmp_path,
        api_key=[PLACEHOLDER_API_KEY],
        profiler_overrides={"warmup_iterations": 1, "active_iterations": 1},
    )
    try:
        response = client.post(
            "/start_profile",
            headers=_auth_header(),
            json={
                "max_iterations": 2,
                "profiler_kwargs": {
                    "on_trace_ready": "attacker-handler",
                    "activities": ["CPU"],
                    "schedule": None,
                },
            },
        )
        if response.status_code >= 400:
            assert not _session_is_active(engine)
            assert _trace_files(tmp_path) == []
            follow_up = client.post(
                "/start_profile",
                headers=_auth_header(),
                json={"max_iterations": 2},
            )
            assert follow_up.status_code == 200
            assert engine.worker.profiler is not None
            assert engine.worker.profiler.is_running
            return

        profiler = engine.worker.profiler
        assert isinstance(profiler, TorchProfilerWrapper)
        assert profiler.is_running
        operator_schedule = profiler._base_profiler_kwargs["schedule"]
        torch_profiler = profiler.profiler
        assert torch_profiler.schedule is operator_schedule
        assert callable(torch_profiler.on_trace_ready)
        assert torch_profiler.on_trace_ready != "attacker-handler"
        assert all(
            isinstance(activity, torch.profiler.ProfilerActivity)
            for activity in torch_profiler.activities
        )
    finally:
        profiler = engine.worker.profiler
        if profiler is not None and profiler.is_running:
            profiler.stop()


def test_limits_hold_when_no_api_key_is_configured(tmp_path: Path) -> None:
    engine, client = _build_service(tmp_path, api_key=None)
    try:
        response = client.post(
            "/start_profile",
            json={
                "max_iterations": 0,
                "profiler_kwargs": _expensive_profiler_kwargs(),
            },
        )
        _assert_operator_profiler_preserved(
            response,
            engine,
            tmp_path,
            max_iterations=_OPERATOR_MAX_ITERATIONS,
        )
    finally:
        profiler = engine.worker.profiler
        if profiler is not None and profiler.is_running:
            profiler.stop()


def _expensive_profiler_kwargs() -> dict[str, bool]:
    return {
        "profile_memory": True,
        "with_stack": True,
        "with_flops": True,
        "record_shapes": True,
    }


def _assert_operator_profiler_preserved(
    response,
    engine: ServingProfileEngine,
    trace_dir: Path,
    *,
    max_iterations: int,
) -> None:
    if response.status_code >= 400:
        assert not _session_is_active(engine)
        assert _trace_files(trace_dir) == []
        return

    profiler = engine.worker.profiler
    assert isinstance(profiler, TorchProfilerWrapper)
    assert profiler.is_running
    assert 0 < profiler._max_iters <= max_iterations
    torch_profiler = profiler.profiler
    assert torch_profiler.profile_memory is False
    assert torch_profiler.with_stack is False
    assert torch_profiler.with_flops is False
    assert torch_profiler.record_shapes is False
    _run_worker_steps(engine.worker, profiler._max_iters + 1)
    assert not profiler.is_running
