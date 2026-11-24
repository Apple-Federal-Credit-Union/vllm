# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rejected or aborted LMCache requests must not kill the engine.

A client can put ``ret_first_tok`` in ``kv_transfer_params``. With the default
``LMCacheConnectorV1`` (``use_native`` unset), ``request_finished`` delegates
to LMCache's adapter, which reads the first output token whenever that key is
present. Requests rejected before admission, and requests aborted before they
sample, have an empty output-token list. These tests drive that input through
the serving rejection hook and the scheduler finish path.

``use_native=True`` is not covered. Selecting it imports
``lmcache_integration``, which fails on the LMCache release this tree installs
(``lmcache==0.5.6``) before the server accepts a request.
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.exceptions import VLLMValidationError
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.fault_tolerance.engine_core_sentinel import fault_tolerant_wrapper
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

MODEL_NAME = "openai-community/gpt2"
_CLIENT_KV = {"ret_first_tok": True, "do_remote_prefill": True}


@dataclass
class _ParallelConfig:
    _api_process_rank: int = 0


@dataclass
class _RendererConfig:
    model_config: "_ModelConfig"
    parallel_config: _ParallelConfig


@dataclass
class _ModelConfig:
    task: str = "generate"
    runner_type: str = "generate"
    model: str = MODEL_NAME
    tokenizer: str = MODEL_NAME
    trust_remote_code: bool = False
    tokenizer_mode: str = "auto"
    max_model_len: int = 128
    revision: str | None = None
    code_revision: str | None = None
    tokenizer_revision: str | None = None
    multimodal_config: MultiModalConfig = field(default_factory=MultiModalConfig)
    hf_config: Any = field(default_factory=lambda: MagicMock(model_type="gpt2"))
    hf_text_config: Any = field(default_factory=lambda: MagicMock(model_type="gpt2"))
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config: Any = None
    generation_config: str = "auto"
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    enable_prompt_embeds: bool = False
    uses_mrope: bool = False
    is_diffusion: bool = False
    enable_return_routed_experts: bool = False
    use_mla: bool = False
    dtype: torch.dtype = torch.float32

    def get_diff_sampling_param(self) -> dict:
        return self.diff_sampling_param or {}

    def get_vocab_size(self) -> int:
        return 50257

    def get_num_layers(self, parallel_config: Any) -> int:
        del parallel_config
        return 1

    def get_num_attention_heads(self, parallel_config: Any) -> int:
        del parallel_config
        return 1

    def get_num_kv_heads(self, parallel_config: Any) -> int:
        del parallel_config
        return 1

    def get_head_size(self) -> int:
        return 1


class _CapturedEngine:
    """Records the request ``notify_kv_transfer_request_rejected`` submits."""

    def __init__(self) -> None:
        self.request: EngineCoreRequest | None = None

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self.request = request

    def shutdown(self, timeout: float | None = None) -> None:
        del timeout


def _transfer_config() -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )


def _scheduler_model(config: _ModelConfig) -> MagicMock:
    """Fields the scheduler and LMCache metadata read off the model config.

    LMCache's scheduler-side constructor asks the model config for layer
    counts. Those counts are not the request path under test.
    """
    model = MagicMock()
    model.skip_tokenizer_init = True
    model.is_multimodal_model = False
    model.supports_multimodal_inputs = False
    model.is_encoder_decoder = config.is_encoder_decoder
    model.is_diffusion = False
    model.max_model_len = config.max_model_len
    model.uses_mrope = False
    model.enable_return_routed_experts = False
    model.runner_type = "generate"
    model.get_num_layers.return_value = 1
    model.get_num_attention_heads.return_value = 1
    model.get_num_kv_heads.return_value = 1
    model.get_head_size.return_value = 1
    model.dtype = torch.float32
    model.model = MODEL_NAME
    model.use_mla = False
    return model


@pytest.fixture
def lmcache_scheduler() -> Scheduler:
    model = _ModelConfig()
    vllm_config = VllmConfig(
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=_transfer_config(),
    )
    # ``VllmConfig`` turns hybrid KV off for this connector. Pin the resolved
    # flag in case construction ran before the connector config was visible.
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    vllm_config.model_config = _scheduler_model(model)
    vllm_config.cache_config.num_gpu_blocks = 32
    vllm_config.cache_config.block_size = 16
    vllm_config.scheduler_config.max_model_len = model.max_model_len
    assert vllm_config.parallel_config.enable_fault_tolerance is False
    kv_cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=16,
    )
    try:
        yield scheduler
    finally:
        scheduler.connector._lmcache_engine.shutdown()


def _engine_core(scheduler: Scheduler) -> EngineCore:
    core = EngineCore.__new__(EngineCore)
    core.scheduler = scheduler
    core.mm_receiver_cache = None
    core.request_block_hasher = None
    core.structured_output_manager = scheduler.structured_output_manager
    core.enable_fault_tolerance = (
        scheduler.vllm_config.parallel_config.enable_fault_tolerance
    )
    return core


def _finish_without_killing_engine(core: EngineCore, request: Request) -> None:
    """Run one engine-core turn that finishes ``request``.

    Fault tolerance is off. The busy-loop wrapper must let a clean
    ``SystemExit`` out and must not see a connector ``IndexError``.
    """

    def busy(engine: EngineCore) -> None:
        if request.abort_immediately:
            engine.add_request(request)
        else:
            engine.scheduler.add_request(request)
            engine.abort_requests([request.request_id])
        raise SystemExit

    with pytest.raises(SystemExit):
        fault_tolerant_wrapper(busy)(core)
    assert request.request_id not in core.scheduler.requests


def _admitted_request(
    request_id: str,
    kv_transfer_params: dict[str, Any] | None,
    *,
    abort_immediately: bool = False,
) -> Request:
    extra_args = None
    if kv_transfer_params is not None:
        extra_args = {"kv_transfer_params": dict(kv_transfer_params)}
    sampling = SamplingParams(
        max_tokens=1,
        extra_args=extra_args,
        watermarking=False,
    )
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling,
        pooling_params=None,
        abort_immediately=abort_immediately,
    )


def test_abort_immediately_with_ret_first_tok_finishes(
    lmcache_scheduler: Scheduler,
) -> None:
    """A pre-aborted remote-prefill request has no output token."""
    core = _engine_core(lmcache_scheduler)
    request = _admitted_request(
        "rejected",
        _CLIENT_KV,
        abort_immediately=True,
    )
    assert request._output_token_ids == []
    _finish_without_killing_engine(core, request)


def test_client_abort_before_first_token_finishes(
    lmcache_scheduler: Scheduler,
) -> None:
    """Disconnecting before the first sample uses the same hook."""
    core = _engine_core(lmcache_scheduler)
    request = _admitted_request("disconnect", {"ret_first_tok": True})
    assert request._output_token_ids == []
    _finish_without_killing_engine(core, request)


def test_finished_request_still_returns_first_token(
    lmcache_scheduler: Scheduler,
) -> None:
    """A request that did sample still reports that token."""
    request = _admitted_request("sampled", {"ret_first_tok": True})
    request.append_output_token_ids(41)
    lmcache_scheduler.add_request(request)
    delay_free, params = lmcache_scheduler._connector_finished(request)
    assert delay_free is False
    assert params == {"first_tok": 41}
    lmcache_scheduler.finish_requests(
        request.request_id, RequestStatus.FINISHED_STOPPED
    )
    assert request.request_id not in lmcache_scheduler.requests


def _serving_engine(vllm_config: VllmConfig) -> tuple[MagicMock, _CapturedEngine]:
    model_config = _ModelConfig()
    engine = MagicMock()
    engine.errored = False
    engine.model_config = model_config
    engine.input_processor = MagicMock()
    engine.vllm_config = vllm_config
    engine.renderer = HfRenderer(
        _RendererConfig(model_config, _ParallelConfig()),
        cached_tokenizer_from_config(model_config),
    )
    captured = _CapturedEngine()
    frontend = AsyncLLM.__new__(AsyncLLM)
    frontend.engine_core = captured
    frontend.log_requests = False

    async def notify(*args: Any, **kwargs: Any) -> None:
        await AsyncLLM.notify_kv_transfer_request_rejected(frontend, *args, **kwargs)

    engine.notify_kv_transfer_request_rejected = notify
    return engine, captured


def _models(engine: MagicMock) -> OpenAIServingModels:
    return OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[
            BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
            BaseModelPath(name="gpt2", model_path="gpt2"),
        ],
    )


def _online_renderer(engine: MagicMock) -> OnlineRenderer:
    return OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template="Dummy chat template for testing {}",
        chat_template_content_format="auto",
    )


def _drive_rejected_engine_request(
    scheduler: Scheduler,
    engine_request: EngineCoreRequest,
) -> None:
    core = _engine_core(scheduler)
    request, wave = core.preprocess_add_request(engine_request)
    assert wave == engine_request.current_wave
    assert request.abort_immediately is True
    assert request.kv_transfer_params == _CLIENT_KV
    assert request._output_token_ids == []
    _finish_without_killing_engine(core, request)


@pytest.mark.asyncio
async def test_chat_temperature_rejection_does_not_kill_engine(
    lmcache_scheduler: Scheduler,
) -> None:
    engine, captured = _serving_engine(lmcache_scheduler.vllm_config)
    serving = OpenAIServingChat(
        engine,
        _models(engine),
        "assistant",
        online_renderer=_online_renderer(engine),
        request_logger=None,
        chat_template="Dummy chat template for testing {}",
        chat_template_content_format="auto",
    )
    assert serving.has_kv_connector is True
    request = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hi"}],
        temperature=5,
        kv_transfer_params=dict(_CLIENT_KV),
    )
    with pytest.raises(VLLMValidationError, match="temperature"):
        await serving.create_chat_completion(request)

    assert captured.request is not None
    _drive_rejected_engine_request(lmcache_scheduler, captured.request)


@pytest.mark.asyncio
async def test_completion_temperature_rejection_does_not_kill_engine(
    lmcache_scheduler: Scheduler,
) -> None:
    engine, captured = _serving_engine(lmcache_scheduler.vllm_config)
    serving = OpenAIServingCompletion(
        engine,
        _models(engine),
        online_renderer=_online_renderer(engine),
        request_logger=None,
    )
    assert serving.has_kv_connector is True
    request = CompletionRequest(
        model=MODEL_NAME,
        prompt="hi",
        temperature=5,
        kv_transfer_params=dict(_CLIENT_KV),
    )
    with pytest.raises(VLLMValidationError, match="temperature"):
        await serving.create_completion(request)

    assert captured.request is not None
    _drive_rejected_engine_request(lmcache_scheduler, captured.request)


@pytest.mark.asyncio
async def test_responses_temperature_rejection_does_not_kill_engine(
    lmcache_scheduler: Scheduler,
) -> None:
    engine, captured = _serving_engine(lmcache_scheduler.vllm_config)
    serving = OpenAIServingResponses(
        engine,
        _models(engine),
        _online_renderer(engine),
        request_logger=None,
        chat_template="Dummy chat template for testing {}",
        chat_template_content_format="auto",
    )
    assert serving.has_kv_connector is True
    request = ResponsesRequest(
        model=MODEL_NAME,
        input="hi",
        temperature=5,
        kv_transfer_params=dict(_CLIENT_KV),
    )
    with pytest.raises(VLLMValidationError, match="temperature"):
        await serving.create_responses(request)

    assert captured.request is not None
    _drive_rejected_engine_request(lmcache_scheduler, captured.request)
