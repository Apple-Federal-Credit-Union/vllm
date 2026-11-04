# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dev-mode admin routes must reject unauthenticated callers before any
worker or scheduler side effect.

These tests fail while ``VLLM_SERVER_DEV_MODE`` mounts ``/collective_rpc``
and the online RL routes outside ``AuthenticationMiddleware``'s guarded
prefixes. They pass once those routes are absent or reject the caller
before the worker method, weight write, or scheduler mutation runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.entrypoints.launchers.api_server.entry import build_app
from vllm.model_executor.model_loader.reload.inplace import (
    get_parameter_for_reload,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.skip_global_cleanup

# Obvious placeholder. This is the server's configured API key, not a secret
# from the environment.
PLACEHOLDER_API_KEY = "placeholder-api-key"
WORKER_CLS = "vllm.v1.worker.gpu_worker.Worker"


@dataclass
class DirectUpdateInfo(WeightTransferUpdateInfo):
    name: str
    values: list[float]


@dataclass
class DirectInitInfo(WeightTransferInitInfo):
    pass


class DirectWeightTransferEngine(
    WeightTransferEngine[DirectInitInfo, DirectUpdateInfo]
):
    """In-process weight sink used by the real worker update methods.

    ``WeightTransferEngine.update_weights`` parses the HTTP dict and calls
    ``receive_weights``. This backend copies that payload into the live
    parameter with ``get_parameter_for_reload``.
    """

    init_info_cls = DirectInitInfo
    update_info_cls = DirectUpdateInfo

    def __init__(self, vllm_config: VllmConfig, model: torch.nn.Module) -> None:
        super().__init__(
            WeightTransferConfig(backend="direct"),
            vllm_config,
            torch.device("cpu"),
            model,
        )
        self.inited = False
        self.started = False
        self.finished = False

    def init_transfer_engine(self, init_info: DirectInitInfo) -> None:
        self.inited = True

    def start_weight_update(self) -> None:
        self.started = True

    def finish_weight_update(self) -> None:
        self.finished = True

    def receive_weights(self, update_info: DirectUpdateInfo) -> None:
        param = get_parameter_for_reload(self.model, update_info.name)
        values = torch.tensor(update_info.values, dtype=param.dtype).view_as(param)
        param.copy_(values)

    def shutdown(self) -> None:
        return None


class PathModelConfig:
    """Attributes ``reload_weights`` and the MM-cache factory read.

    A full ``ModelConfig`` resolves a Hugging Face repo. These fields are the
    ones the weight-reload path assigns and the worker init path checks.
    """

    def __init__(self) -> None:
        self.model = "local-weights"
        self.revision = None
        self.model_weights = ""
        self.quantization = None
        self.supports_multimodal_inputs = False


class LiveModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.ones_like(self.linear.weight))
        self.linear.weight.requires_grad_(False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensor)
            loaded.add(name)
        return loaded


class ModelRunner:
    def __init__(self, model: LiveModel, vllm_config: VllmConfig) -> None:
        self.model = model
        self.vllm_config = vllm_config
        self.lora_config = vllm_config.lora_config
        self.drafter = None
        self.shut_down = False
        self.encoder_cleared = False
        self.mm_cleared = False

    get_model = GPUModelRunner.get_model
    get_draft_model = GPUModelRunner.get_draft_model
    reload_weights = GPUModelRunner.reload_weights

    def reset_lora_state(self) -> None:
        LoRAModelRunnerMixin.reset_lora_state(self)

    def reset_encoder_cache(self) -> None:
        self.encoder_cleared = True

    def reset_mm_cache(self) -> None:
        self.mm_cleared = True

    def shutdown(self) -> None:
        self.shut_down = True


class PrefixPool:
    """Fields read by ``BlockPool.reset_prefix_cache``."""

    def __init__(self) -> None:
        self.num_gpu_blocks = 2
        self.free_block_queue = type("FreeQueue", (), {"num_free_blocks": 1})()
        self.blocks = [KVCacheBlock(block_id=0), KVCacheBlock(block_id=1)]
        cached = BlockHashWithGroupId(b"cached-prefix")
        self.blocks[1].set_block_hash(cached, 1)
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hash_to_block.insert(cached, self.blocks[1])
        self.cached_block_hashes_by_block = {1: cached}
        self._reuse_watchers: dict[int, object] = {}
        self.metrics_collector = None
        self.enable_kv_cache_events = False

    reset_prefix_cache = BlockPool.reset_prefix_cache
    get_num_free_blocks = BlockPool.get_num_free_blocks


class SingleTypeManager:
    def __init__(self, pool: PrefixPool) -> None:
        self.block_pool = pool


class Coordinator:
    def __init__(self, pool: PrefixPool) -> None:
        self.single_type_managers = [SingleTypeManager(pool)]

    reset_prefix_cache = KVCacheCoordinator.reset_prefix_cache


class CacheManager:
    def __init__(self, coordinator: Coordinator) -> None:
        self.coordinator = coordinator
        self.log_stats = False

    reset_prefix_cache = KVCacheManager.reset_prefix_cache


class IdleScheduler:
    def __init__(self, pool: PrefixPool) -> None:
        self._pause_state = PauseState.UNPAUSED
        self.requests: dict = {}
        self.running: list = []
        self.prev_step_scheduled_req_ids: set = set()
        self.kv_cache_manager = CacheManager(Coordinator(pool))
        self.connector = None
        self.aux_output_connector = None
        self.pool = pool
        self.encoder_cache_reset = False

    set_pause_state = Scheduler.set_pause_state
    pause_state = Scheduler.pause_state
    finish_requests = Scheduler.finish_requests
    reset_prefix_cache = Scheduler.reset_prefix_cache
    reset_connector_cache = Scheduler.reset_connector_cache
    get_kv_connector = Scheduler.get_kv_connector

    def has_unfinished_requests(self) -> bool:
        return False

    def has_requests(self) -> bool:
        return False

    def reset_encoder_cache(self) -> None:
        self.encoder_cache_reset = True


class Renderer:
    def __init__(self) -> None:
        self.mm_cache_cleared = False

    def _join_mm_warmup(self) -> None:
        return None

    async def clear_mm_cache_async(self) -> None:
        self.mm_cache_cleared = True


class OutputTracker:
    def __init__(self) -> None:
        self.request_states: dict = {}
        self.external_req_ids = {"live-request": ["internal-live"]}
        self.parent_requests: dict = {}
        self.admission_stats = None

    abort_requests = OutputProcessor.abort_requests
    _update_admission_stats = OutputProcessor._update_admission_stats


class SyncExecutor:
    def __init__(self, wrapper: WorkerWrapperBase) -> None:
        self.driver_worker = wrapper
        self.sleeping_tags: set[str] = set()

    collective_rpc = UniProcExecutor.collective_rpc
    wake_up = UniProcExecutor.wake_up

    @property
    def is_sleeping(self) -> bool:
        return bool(self.sleeping_tags)

    def reset_mm_cache(self) -> None:
        self.collective_rpc("reset_mm_cache")

    def reset_encoder_cache(self) -> None:
        self.collective_rpc("reset_encoder_cache")


class EngineCoreState:
    def __init__(self, scheduler: IdleScheduler, executor: SyncExecutor) -> None:
        self.scheduler = scheduler
        self.model_executor = executor
        self.mm_receiver_cache = None
        self.batch_queue = None
        self.reject_new_requests = False
        self._weight_version = "original"

    pause_scheduler = EngineCore.pause_scheduler
    _finish_pause = EngineCore._finish_pause
    _reset_caches = EngineCore._reset_caches
    reset_prefix_cache = EngineCore.reset_prefix_cache
    reset_mm_cache = EngineCore.reset_mm_cache
    reset_encoder_cache = EngineCore.reset_encoder_cache
    is_scheduler_paused = EngineCore.is_scheduler_paused
    sleep = EngineCore.sleep
    wake_up = EngineCore.wake_up
    is_sleeping = EngineCore.is_sleeping
    release_kv_cache_memory = EngineCore.release_kv_cache_memory
    set_weight_version = EngineCore.set_weight_version
    get_weight_version = EngineCore.get_weight_version
    _set_kv_connector_handshake_metadata = (
        EngineCore._set_kv_connector_handshake_metadata
    )

    def resume_scheduler(self) -> None:
        self.reject_new_requests = False
        EngineCore.resume_scheduler(self)

    async def pause_generation_async(self, mode="abort", clear_cache=True) -> None:
        EngineCoreProc.pause_generation(self, mode, clear_cache)

    async def sleep_async(self, level: int = 1, mode: str = "abort") -> None:
        EngineCore.sleep(self, level, mode)

    async def wake_up_async(self, tags: list[str] | None = None) -> bool:
        return EngineCore.wake_up(self, tags)

    async def is_sleeping_async(self) -> bool:
        return EngineCore.is_sleeping(self)

    async def is_scheduler_paused_async(self) -> bool:
        return EngineCore.is_scheduler_paused(self)

    async def resume_scheduler_async(self) -> None:
        self.resume_scheduler()

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return EngineCore.reset_prefix_cache(
            self, reset_running_requests, reset_connector
        )

    async def reset_mm_cache_async(self) -> None:
        EngineCore.reset_mm_cache(self)

    async def reset_encoder_cache_async(self) -> None:
        EngineCore.reset_encoder_cache(self)

    async def release_kv_cache_memory_async(self) -> None:
        EngineCore.release_kv_cache_memory(self)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.aborted_in_core = list(request_ids)

    async def set_weight_version_async(self, weight_version: str) -> None:
        EngineCore.set_weight_version(self, weight_version)

    async def get_weight_version_async(self) -> str:
        return EngineCore.get_weight_version(self)

    async def compute_weight_checksums_async(self) -> list[dict[str, str]]:
        return self.model_executor.collective_rpc("compute_weight_checksums")


class ServingEngine:
    def __init__(
        self,
        wrapper: WorkerWrapperBase,
        vllm_config: VllmConfig,
        dispatch: str,
        scheduler: IdleScheduler,
    ) -> None:
        self.driver_worker = wrapper
        self.dispatch = dispatch
        self.vllm_config = vllm_config
        self.renderer = Renderer()
        self.logger_manager = None
        self.log_requests = False
        self.output_processor = OutputTracker()
        self.model_executor = SyncExecutor(wrapper)
        self.engine_core = EngineCoreState(scheduler, self.model_executor)
        proc = WorkerProc.__new__(WorkerProc)
        proc.worker = wrapper
        proc.rank = 0
        proc.use_async_scheduling = False
        proc.worker_response_mq = None
        self.worker_proc = proc

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        kwargs = kwargs or {}
        if self.dispatch == "uniproc":
            return UniProcExecutor.collective_rpc(self, method, timeout, args, kwargs)
        # MultiprocExecutor enqueues the string method unchanged.
        # WorkerProc._execute_worker_rpc resolves it with getattr.
        captured: dict = {}

        def handle_output(output) -> None:
            captured["output"] = output

        self.worker_proc.handle_output = handle_output
        WorkerProc._execute_worker_rpc(
            self.worker_proc, (method, tuple(args), dict(kwargs), None)
        )
        output = captured.get("output")
        if isinstance(output, Exception):
            raise output
        return [output]

    pause_generation = AsyncLLM.pause_generation
    resume_generation = AsyncLLM.resume_generation
    is_paused = AsyncLLM.is_paused
    sleep = AsyncLLM.sleep
    wake_up = AsyncLLM.wake_up
    is_sleeping = AsyncLLM.is_sleeping
    release_kv_cache_memory = AsyncLLM.release_kv_cache_memory
    reset_prefix_cache = AsyncLLM.reset_prefix_cache
    reset_mm_cache = AsyncLLM.reset_mm_cache
    reset_encoder_cache = AsyncLLM.reset_encoder_cache
    abort = AsyncLLM.abort
    init_weight_transfer_engine = AsyncLLM.init_weight_transfer_engine
    shutdown_weight_transfer_engine = AsyncLLM.shutdown_weight_transfer_engine
    start_weight_update = AsyncLLM.start_weight_update
    start_draft_weight_update = AsyncLLM.start_draft_weight_update
    update_weights = AsyncLLM.update_weights
    finish_weight_update = AsyncLLM.finish_weight_update
    update_weight_version = AsyncLLM.update_weight_version
    get_weight_version = AsyncLLM.get_weight_version
    compute_weight_checksums = AsyncLLM.compute_weight_checksums


def _build_args():
    from argparse import Namespace

    return Namespace(
        disable_fastapi_docs=True,
        enable_offline_docs=False,
        root_path=None,
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=[PLACEHOLDER_API_KEY],
        enable_request_id_headers=False,
        enable_fault_tolerance=False,
        middleware=[],
        log_error_stack=False,
    )


def _worker(vllm_config: VllmConfig, model: LiveModel) -> tuple:
    vllm_config.model_config = PathModelConfig()
    vllm_config.parallel_config.worker_cls = WORKER_CLS
    wrapper = WorkerWrapperBase(rpc_rank=0)
    with set_current_vllm_config(vllm_config):
        wrapper.init_worker(
            [
                {
                    "vllm_config": vllm_config,
                    "local_rank": 0,
                    "rank": 0,
                    "distributed_init_method": "tcp://127.0.0.1:0",
                    "is_driver_worker": True,
                }
            ]
        )
    record_metadata_for_reloading(model)
    runner = ModelRunner(model, vllm_config)
    worker = wrapper.worker
    worker.model_runner = runner
    worker.weight_transfer_engine = DirectWeightTransferEngine(vllm_config, model)
    return wrapper, worker, runner


@pytest.fixture(scope="module")
def distributed():
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1)
    yield vllm_config
    cleanup_dist_env_and_memory()


@pytest.fixture
def dev_server(distributed, tmp_path, request):
    dispatch = getattr(request, "param", "uniproc")
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    model = LiveModel()
    original = model.linear.weight.detach().clone()
    pool = PrefixPool()
    scheduler = IdleScheduler(pool)
    wrapper, worker, runner = _worker(distributed, model)
    engine = ServingEngine(wrapper, distributed, dispatch, scheduler)
    app = build_app(_build_args(), supported_tasks=())
    app.state.log_stats = False
    app.state.engine_client = engine
    app.state.vllm_config = distributed
    with TestClient(app, raise_server_exceptions=False) as client:
        yield (
            client,
            {
                "model": model,
                "original": original,
                "runner": runner,
                "worker": worker,
                "scheduler": scheduler,
                "pool": pool,
                "engine": engine,
                "tmp_path": tmp_path,
            },
        )


def _rejected(response) -> None:
    assert response.status_code in {401, 403, 404}, response.text


def _weight_unchanged(state) -> None:
    assert torch.equal(state["model"].linear.weight, state["original"])


@pytest.mark.parametrize("dev_server", ["uniproc", "multiproc"], indirect=True)
def test_unauthenticated_collective_rpc_cannot_save_weights(dev_server):
    client, state = dev_server
    dump_dir = state["tmp_path"] / "stolen"
    dump_dir.mkdir()
    response = client.post(
        "/collective_rpc",
        json={"method": "save_sharded_state", "args": [str(dump_dir)]},
    )
    _rejected(response)
    assert list(dump_dir.iterdir()) == []


@pytest.mark.parametrize("dev_server", ["uniproc", "multiproc"], indirect=True)
def test_unauthenticated_collective_rpc_cannot_reload_weights(dev_server):
    client, state = dev_server
    checkpoint = state["tmp_path"] / "attacker-checkpoint"
    checkpoint.mkdir()
    save_file(
        {"linear.weight": torch.full((2, 2), 7.0)},
        str(checkpoint / "model.safetensors"),
    )
    response = client.post(
        "/collective_rpc",
        json={
            "method": "reload_weights",
            "kwargs": {"weights_path": str(checkpoint)},
        },
    )
    _rejected(response)
    _weight_unchanged(state)


@pytest.mark.parametrize("dev_server", ["uniproc", "multiproc"], indirect=True)
def test_unauthenticated_collective_rpc_cannot_reload_kernel_weights(dev_server):
    client, state = dev_server
    checkpoint = state["tmp_path"] / "kernel-checkpoint"
    checkpoint.mkdir()
    save_file(
        {"linear.weight": torch.full((2, 2), 3.0)},
        str(checkpoint / "model.safetensors"),
    )
    response = client.post(
        "/collective_rpc",
        json={
            "method": "reload_weights",
            "kwargs": {
                "weights_path": str(checkpoint),
                "is_checkpoint_format": False,
            },
        },
    )
    _rejected(response)
    _weight_unchanged(state)


@pytest.mark.parametrize("dev_server", ["uniproc", "multiproc"], indirect=True)
def test_unauthenticated_collective_rpc_cannot_shutdown_worker(dev_server):
    client, state = dev_server
    response = client.post("/collective_rpc", json={"method": "shutdown"})
    _rejected(response)
    assert state["runner"].shut_down is False
    pool = state["worker"].elastic_ep_executor._async_executor
    assert pool.submit(lambda: 1).result(timeout=2) == 1


@pytest.mark.parametrize("dev_server", ["uniproc", "multiproc"], indirect=True)
def test_unauthenticated_collective_rpc_cannot_zero_weights(dev_server):
    client, state = dev_server
    response = client.post(
        "/weight_checker",
        json={"action": "reset"},
    )
    _rejected(response)
    _weight_unchanged(state)


def test_unauthenticated_pause_does_not_pause(dev_server):
    client, state = dev_server
    response = client.post("/pause")
    _rejected(response)
    assert state["scheduler"].pause_state == PauseState.UNPAUSED


def test_unauthenticated_sleep_does_not_pause_or_sleep(dev_server):
    client, state = dev_server
    response = client.post("/sleep", params={"level": 0, "mode": "keep"})
    _rejected(response)
    assert state["scheduler"].pause_state == PauseState.UNPAUSED
    assert state["engine"].model_executor.is_sleeping is False


def test_unauthenticated_reset_prefix_cache_keeps_cached_block(dev_server):
    client, state = dev_server
    cached = state["pool"].blocks[1]
    assert cached.block_hash is not None
    response = client.post("/reset_prefix_cache")
    _rejected(response)
    assert cached.block_hash is not None
    assert state["pool"].cached_block_hash_to_block.get_one_block(cached.block_hash)


def test_unauthenticated_update_weights_does_not_replace_parameters(dev_server):
    client, state = dev_server
    started = client.post("/start_weight_update")
    updated = client.post(
        "/update_weights",
        json={
            "update_info": {
                "name": "linear.weight",
                "values": [9.0, 9.0, 9.0, 9.0],
            }
        },
    )
    _rejected(started)
    _rejected(updated)
    _weight_unchanged(state)
    assert state["worker"].weight_transfer_engine.started is False


@pytest.mark.parametrize(
    "method,url,kwargs",
    [
        ("post", "/resume", {}),
        ("get", "/is_paused", {}),
        ("post", "/wake_up", {}),
        ("get", "/is_sleeping", {}),
        ("post", "/release_kv_cache_memory", {}),
        ("post", "/reset_mm_cache", {}),
        ("post", "/reset_encoder_cache", {}),
        ("post", "/abort_requests", {"json": {"request_ids": ["live-request"]}}),
        ("post", "/init_weight_transfer_engine", {"json": {"init_info": {}}}),
        ("post", "/shutdown_weight_transfer_engine", {}),
        ("post", "/start_draft_weight_update", {}),
        ("post", "/finish_weight_update", {}),
        ("post", "/update_weight_version", {"json": {"new_version": "attacker"}}),
        ("get", "/weight_info", {}),
        ("post", "/weight_checker", {"json": {"action": "checksum"}}),
        ("get", "/get_world_size", {}),
        ("get", "/server_info", {}),
        (
            "post",
            "/collective_rpc",
            {"json": {"method": "update_config", "kwargs": {"overrides": {}}}},
        ),
    ],
)
def test_dev_and_rl_routes_reject_missing_api_key(dev_server, method, url, kwargs):
    client, state = dev_server
    response = getattr(client, method)(url, **kwargs)
    _rejected(response)
    assert state["engine"].output_processor.external_req_ids == {
        "live-request": ["internal-live"]
    }
    assert state["engine"].engine_core._weight_version == "original"
    assert state["runner"].encoder_cleared is False
    assert state["runner"].mm_cleared is False
    assert state["scheduler"].pause_state == PauseState.UNPAUSED


def test_wrong_api_key_does_not_authorize_collective_rpc(dev_server, tmp_path):
    client, state = dev_server
    dump_dir = tmp_path / "wrong-key"
    dump_dir.mkdir()
    response = client.post(
        "/collective_rpc",
        headers={"Authorization": "Bearer wrong-placeholder"},
        json={"method": "save_sharded_state", "args": [str(dump_dir)]},
    )
    _rejected(response)
    assert list(dump_dir.iterdir()) == []
    _weight_unchanged(state)


def test_configured_api_key_still_guards_inference_prefix(dev_server):
    client, _state = dev_server
    missing = client.get("/v1/models")
    assert missing.status_code == 401
    wrong = client.get(
        "/v1/models", headers={"Authorization": "Bearer wrong-placeholder"}
    )
    assert wrong.status_code == 401


def test_dev_routes_are_absent_when_dev_mode_is_off(monkeypatch):
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "0")
    app = build_app(_build_args(), supported_tasks=())
    app.state.log_stats = False
    app.state.vllm_config = VllmConfig()
    with TestClient(app, raise_server_exceptions=False) as client:
        for path in (
            "/collective_rpc",
            "/pause",
            "/sleep",
            "/update_weights",
            "/reset_prefix_cache",
        ):
            response = client.post(path, json={})
            assert response.status_code == 404, response.text
