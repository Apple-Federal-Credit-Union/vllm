# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache identity after a same-path LoRA reload.

These tests fail while a reload of an adapter name keeps the previous
adapter's prefix-cache blocks. They pass once a same-path reload stops
later requests from being scheduled on those blocks.
"""

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.config import CacheConfig, LoRAConfig
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.lora.api_router import attach_router
from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_scheduler

pytestmark = pytest.mark.cpu_test

PROMPT_LEN = 33
BLOCK_SIZE = 16
# Full blocks only. The scheduler recomputes the final prompt token.
CACHED_PREFIX_TOKENS = (PROMPT_LEN // BLOCK_SIZE) * BLOCK_SIZE
ADAPTER_NAME = "policy"
BASE_MODEL = "facebook/opt-125m"


class _AdapterExecutor:
    """Records adapters EngineCore forwards to the worker.

    ``Executor.add_lora`` RPCs the worker and returns its bool.
    ``LRUCacheWorkerLoRAManager.add_adapter`` reloads tensors when
    ``load_inplace`` is set or the id is new, then returns whether that
    adapter is active. This executor keeps that success contract so the
    scheduler can be observed after a successful reload.
    """

    def __init__(self) -> None:
        self.loaded: dict[int, LoRARequest] = {}
        # Captured at the call. The serving layer clears load_inplace on the
        # same object after add_lora returns.
        self.inplace_flags: list[bool] = []

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0
        self.inplace_flags.append(lora_request.load_inplace)
        self.loaded[lora_request.lora_int_id] = lora_request
        return True

    def remove_lora(self, lora_id: int) -> bool:
        return self.loaded.pop(lora_id, None) is not None

    def list_loras(self) -> set[int]:
        return set(self.loaded)

    def pin_lora(self, lora_id: int) -> bool:
        return lora_id in self.loaded


class _PrefixCacheEngine(EngineCore):
    """EngineCore methods without loading a model.

    ``add_lora`` and ``reset_prefix_cache`` are the real EngineCore methods.
    The async wrappers are the utility entry points the API client awaits.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.model_executor = _AdapterExecutor()

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return EngineCore.add_lora(self, lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return EngineCore.remove_lora(self, lora_id)

    async def reset_prefix_cache_async(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        return EngineCore.reset_prefix_cache(
            self, reset_running_requests, reset_connector
        )


class _ApiEngineClient:
    """API engine client. LoRA calls go through ``AsyncLLM``."""

    def __init__(self, engine: _PrefixCacheEngine) -> None:
        self.engine_core = engine
        self.model_config = engine.scheduler.vllm_config.model_config
        self.renderer = None
        self.input_processor = None

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        return await AsyncLLM.add_lora(self, lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        return await AsyncLLM.remove_lora(self, lora_id)

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        return await AsyncLLM.reset_prefix_cache(
            self, reset_running_requests, reset_connector
        )


class _OfflineEngine:
    """Offline engine. ``LLMEngine.add_lora`` delegates to EngineCore."""

    def __init__(self, engine: _PrefixCacheEngine) -> None:
        self.engine_core = engine

    def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        return LLMEngine.reset_prefix_cache(
            self, reset_running_requests, reset_connector
        )


def _write_adapter(path: Path, weights: bytes) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        '{"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}\n',
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(weights)


def _scheduler() -> Scheduler:
    assert CacheConfig().enable_prefix_caching
    init_none_hash(sha256)
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=8192,
        lora_config=LoRAConfig(),
    )
    assert isinstance(scheduler, Scheduler)
    return scheduler


def _request(scheduler: Scheduler, request_id: str, lora: LoRARequest) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(PROMPT_LEN)),
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=lora,
        block_hasher=get_request_block_hasher(scheduler.hash_block_size, sha256),
    )


def _apply_one_token(scheduler: Scheduler, output) -> None:
    req_ids = list(output.num_scheduled_tokens.keys())
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[[100] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _finish_prompt(scheduler: Scheduler, request: Request) -> None:
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[request.request_id] == PROMPT_LEN
    _apply_one_token(scheduler, output)
    assert request.request_id in scheduler.finished_req_ids


def _admit(scheduler: Scheduler, request: Request):
    scheduler.add_request(request)
    output = scheduler.schedule()
    admitted = output.scheduled_new_reqs[0]
    assert admitted.req_id == request.request_id
    assert admitted.lora_request is not None
    assert request.lora_request is not None
    assert admitted.lora_request.lora_name == request.lora_request.lora_name
    assert admitted.lora_request.lora_path == request.lora_request.lora_path
    assert admitted.lora_request.lora_int_id == request.lora_request.lora_int_id
    # num_computed_tokens on the scheduled request is the prefix-cache hit
    # the worker will not recompute. schedule() then advances the request.
    cached = admitted.num_computed_tokens
    assert output.num_scheduled_tokens[request.request_id] == PROMPT_LEN - cached
    return output, cached


def _cache_prefix_then_show_hit(scheduler: Scheduler, lora: LoRARequest) -> None:
    """Fill the prefix cache, then show a repeated prompt reuses it."""
    _finish_prompt(scheduler, _request(scheduler, f"{lora.lora_name}-warm", lora))
    hit = _request(scheduler, f"{lora.lora_name}-hit", lora)
    output, cached = _admit(scheduler, hit)
    assert cached == CACHED_PREFIX_TOKENS
    # Finish the hit so a prefix-cache reset can drop its blocks.
    _apply_one_token(scheduler, output)
    assert hit.request_id in scheduler.finished_req_ids


def _assert_prefix_recomputed(scheduler: Scheduler, lora: LoRARequest) -> None:
    request = _request(scheduler, f"{lora.lora_name}-after-reload", lora)
    _output, cached = _admit(scheduler, request)
    assert cached == 0, (
        f"request reused {cached} prefix-cache tokens computed under the "
        "previous adapter weights"
    )


def _serving(engine: _PrefixCacheEngine) -> OpenAIServingModels:
    return OpenAIServingModels(
        _ApiEngineClient(engine),
        [BaseModelPath(name=BASE_MODEL, model_path=BASE_MODEL)],
    )


def test_same_path_inplace_reload_drops_prefix_cache(tmp_path: Path) -> None:
    """POST /v1/load_lora_adapter of the same name and path must drop old blocks."""
    scheduler = _scheduler()
    engine = _PrefixCacheEngine(scheduler)
    serving = _serving(engine)
    adapter_path = str(tmp_path / "policy-adapter")
    _write_adapter(Path(adapter_path), b"adapter-weights-v1")

    first = asyncio.run(
        serving.load_lora_adapter(
            LoadLoRAAdapterRequest(lora_name=ADAPTER_NAME, lora_path=adapter_path)
        )
    )
    assert isinstance(first, str)
    loaded = serving.lora_requests[ADAPTER_NAME]
    _cache_prefix_then_show_hit(scheduler, loaded)

    _write_adapter(Path(adapter_path), b"adapter-weights-v2")
    second = asyncio.run(
        serving.load_lora_adapter(
            LoadLoRAAdapterRequest(
                lora_name=ADAPTER_NAME,
                lora_path=adapter_path,
                load_inplace=True,
            )
        )
    )
    assert isinstance(second, str)
    reloaded = serving.lora_requests[ADAPTER_NAME]
    assert reloaded.lora_int_id == loaded.lora_int_id
    assert reloaded.lora_path == adapter_path
    assert engine.model_executor.inplace_flags[-1] is True
    _assert_prefix_recomputed(scheduler, reloaded)


def test_post_load_lora_adapter_same_path_drops_prefix_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load route must not keep prefix blocks from the previous weights."""
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")
    scheduler = _scheduler()
    engine = _PrefixCacheEngine(scheduler)
    serving = _serving(engine)
    app = FastAPI()
    app.state.openai_serving_models = serving
    attach_router(app)
    adapter_path = str(tmp_path / "policy-adapter")
    _write_adapter(Path(adapter_path), b"adapter-weights-v1")

    with TestClient(app) as client:
        first = client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": ADAPTER_NAME, "lora_path": adapter_path},
        )
        assert first.status_code == 200
        loaded = serving.lora_requests[ADAPTER_NAME]
        _cache_prefix_then_show_hit(scheduler, loaded)

        _write_adapter(Path(adapter_path), b"adapter-weights-v2")
        second = client.post(
            "/v1/load_lora_adapter",
            json={
                "lora_name": ADAPTER_NAME,
                "lora_path": adapter_path,
                "load_inplace": True,
            },
        )
        assert second.status_code == 200

    reloaded = serving.lora_requests[ADAPTER_NAME]
    assert reloaded.lora_path == adapter_path
    assert engine.model_executor.inplace_flags[-1] is True
    _assert_prefix_recomputed(scheduler, reloaded)


def test_engine_add_lora_same_path_reload_drops_prefix_cache(tmp_path: Path) -> None:
    """LLMEngine.add_lora of the same name and path must drop old blocks."""
    scheduler = _scheduler()
    engine = _PrefixCacheEngine(scheduler)
    offline = _OfflineEngine(engine)
    adapter_path = str(tmp_path / "policy-adapter")
    _write_adapter(Path(adapter_path), b"adapter-weights-v1")
    assert (
        LLMEngine.add_lora(offline, LoRARequest(ADAPTER_NAME, 1, adapter_path)) is True
    )
    _cache_prefix_then_show_hit(scheduler, LoRARequest(ADAPTER_NAME, 1, adapter_path))

    _write_adapter(Path(adapter_path), b"adapter-weights-v2")
    assert (
        LLMEngine.add_lora(
            offline,
            LoRARequest(ADAPTER_NAME, 1, adapter_path, load_inplace=True),
        )
        is True
    )
    _assert_prefix_recomputed(scheduler, LoRARequest(ADAPTER_NAME, 1, adapter_path))


def test_unload_then_same_path_load_drops_prefix_cache(tmp_path: Path) -> None:
    """Unloading and loading the same path must drop the old prefix blocks."""
    scheduler = _scheduler()
    engine = _PrefixCacheEngine(scheduler)
    serving = _serving(engine)
    adapter_path = str(tmp_path / "policy-adapter")
    _write_adapter(Path(adapter_path), b"adapter-weights-v1")
    first = asyncio.run(
        serving.load_lora_adapter(
            LoadLoRAAdapterRequest(lora_name=ADAPTER_NAME, lora_path=adapter_path)
        )
    )
    assert isinstance(first, str)
    loaded = serving.lora_requests[ADAPTER_NAME]
    _cache_prefix_then_show_hit(scheduler, loaded)

    unloaded = asyncio.run(
        serving.unload_lora_adapter(UnloadLoRAAdapterRequest(lora_name=ADAPTER_NAME))
    )
    assert isinstance(unloaded, str)
    _write_adapter(Path(adapter_path), b"adapter-weights-v2")
    second = asyncio.run(
        serving.load_lora_adapter(
            LoadLoRAAdapterRequest(lora_name=ADAPTER_NAME, lora_path=adapter_path)
        )
    )
    assert isinstance(second, str)
    reloaded = serving.lora_requests[ADAPTER_NAME]
    assert reloaded.lora_path == adapter_path
    _assert_prefix_recomputed(scheduler, reloaded)
