# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client mm_hashes on /inference/v1/generate must not name encoder-cache entries.

A generate request may carry preprocessed tensors plus hashes. Those hashes
are the same identifiers a normal multimodal request computes from the media
itself. The encoder cache is keyed by that identifier and outlives the
request, so a client-chosen hash lets one caller plant embeddings (or an
embedding count) that a later request for different media reuses.

These tests fail while that aliasing is possible. They pass when the generate
path rejects the request or stores the supplied tensors under an identifier
that a content-hashed request for different media will not look up. Image is
the modality under test; video and audio use the same hash map in
``serve_tokens`` and ``InputProcessor.process_inputs``. The Rust gRPC
preprocessed path is covered by
``client_identifier_cannot_alias_another_items_encoder_cache_key``.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace

import pytest
import torch
from fastapi import FastAPI
from PIL import Image

from vllm.entrypoints.scale_out.factories import register_scale_out_api_routers
from vllm.entrypoints.scale_out.token_in_token_out.api_router import generate
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    encode_mm_kwargs_item,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.exceptions import VLLMValidationError
from vllm.inputs.engine import mm_input
from vllm.multimodal.cache.base import BaseMultiModalProcessorCache
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.sampling_params import SamplingParams
from vllm.utils.cache import CacheInfo
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

_MODEL_ID = "org/vision-model"
_PLACEHOLDER = 4


class _CapturedGenerate(Exception):
    def __init__(self, engine_input, sampling_params):
        self.engine_input = engine_input
        self.sampling_params = sampling_params


class _ModelConfig:
    max_model_len = 4096
    max_logprobs = 20
    is_diffusion = False
    return_sampling_mask = False
    enable_trace_replay = False
    runner_type = "generate"
    is_encoder_decoder = False
    supports_multimodal_inputs = True

    def get_vocab_size(self) -> int:
        return 32000


class _VllmConfig:
    def __init__(self) -> None:
        self.model_config = _ModelConfig()
        self.scheduler_config = Namespace(
            max_num_seqs=16,
            disable_chunked_mm_input=False,
        )
        self.cache_config = Namespace(kv_sharing_fast_prefill=False)
        self.parallel_config = Namespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        )
        self.use_v2_model_runner = False
        self.reasoning_config = None
        self.watermark_config = None
        self.speculative_config = None
        self.structured_outputs_config = None
        self.diffusion_config = None

    def _check_supports_watermarking(self, params) -> bool:
        del params
        return False


class _Renderer:
    tokenizer = None
    mm_processor_cache = None

    def validate_token_ids(self, token_ids, *, parameter=None) -> None:
        del token_ids, parameter

    def get_eos_token_id(self) -> None:
        return None


class _Models:
    lora_requests: dict = {}

    def model_name(self, lora_request=None) -> str:
        del lora_request
        return _MODEL_ID

    def is_base_model(self, model_name: str) -> bool:
        del model_name
        return True


class _Engine:
    errored = False
    dead_error = RuntimeError("engine dead")

    def __init__(self, config: _VllmConfig) -> None:
        self.vllm_config = config

    def check_admission(self, n: int) -> None:
        del n

    def generate(self, prompt, sampling_params, request_id, **kwargs):
        del request_id, kwargs
        raise _CapturedGenerate(prompt, sampling_params)


class _OpenProcessorCache(BaseMultiModalProcessorCache):
    """Processor cache whose inherited validation does not bind hashes."""

    def get_and_update_item(self, mm_item, mm_hash):
        del mm_item, mm_hash
        raise AssertionError("processor cache update is not part of this path")

    def is_cached_item(self, mm_hash: str) -> bool:
        del mm_hash
        return False

    def touch_sender_cache_item(self, mm_hash: str) -> None:
        del mm_hash

    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        del delta
        return CacheInfo(hits=0, total=0)

    def clear_cache(self) -> None:
        return None


class _ScheduledEncoderInputs:
    def __init__(self, scheduled: dict[str, list[int]]) -> None:
        self.scheduled_encoder_inputs = scheduled


def _content_hash(image: Image.Image) -> str:
    inputs = ProcessorInputs(
        prompt=[0],
        mm_data_items=MultiModalDataItems({"image": ImageProcessorItems([image])}),
    )
    return inputs.get_mm_hashes(_MODEL_ID, "blake3")["image"][0]


def _pixels(fill: int) -> torch.Tensor:
    return torch.full((2, 2), fill, dtype=torch.float32)


def _kwargs_item(pixels: torch.Tensor) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(
                data=pixels,
                field=MultiModalBatchedField(),
            )
        }
    )


def _features(mm_hash: str, pixels: torch.Tensor, length: int) -> MultiModalFeatures:
    return MultiModalFeatures(
        mm_hashes={"image": [mm_hash]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=length)]},
        kwargs_data={"image": [encode_mm_kwargs_item(_kwargs_item(pixels))]},
    )


def _generate_request(
    mm_hash: str,
    pixels: torch.Tensor,
    length: int,
    *,
    cache_salt: str | None = None,
) -> GenerateRequest:
    return GenerateRequest(
        token_ids=list(range(1, length + 2)),
        features=_features(mm_hash, pixels, length),
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        cache_salt=cache_salt,
    )


def _processor(config: _VllmConfig) -> InputProcessor:
    processor = InputProcessor.__new__(InputProcessor)
    processor.vllm_config = config
    processor.model_config = config.model_config
    processor.cache_config = config.cache_config
    processor.lora_config = None
    processor.scheduler_config = config.scheduler_config
    processor.speculative_config = None
    processor.structured_outputs_config = None
    processor.diffusion_config = None
    processor.generation_config_fields = {}
    processor.renderer = _Renderer()
    processor.supports_mm_inputs = True
    processor.mm_encoder_cache_size = config.model_config.max_model_len
    processor.skip_prompt_length_check = False
    processor.validate_logits_processors_params = lambda params: None
    return processor


def _serving(cache: BaseMultiModalProcessorCache | None) -> ServingTokens:
    config = _VllmConfig()
    serving = ServingTokens.__new__(ServingTokens)
    serving.models = _Models()
    serving.request_logger = None
    serving.engine_client = _Engine(config)
    serving.force_no_detokenize = False
    serving.has_tokenizer = False
    serving.model_config = config.model_config
    renderer = _Renderer()
    renderer.mm_processor_cache = cache
    serving.online_renderer = Namespace(renderer=renderer)
    return serving


def _admit(serving: ServingTokens, request: GenerateRequest):
    """Run ``serve_tokens`` through engine handoff, then ``process_inputs``."""
    processor = _processor(serving.engine_client.vllm_config)

    async def _call():
        return await serving.serve_tokens(request)

    try:
        result = asyncio.run(_call())
    except _CapturedGenerate as captured:
        engine_input = captured.engine_input
        sampling_params = captured.sampling_params
    else:
        if isinstance(result, ErrorResponse):
            return None
        raise AssertionError(f"unexpected generate result {type(result)!r}")

    try:
        return processor.process_inputs(
            "attacker",
            engine_input,
            sampling_params,
            ("generate",),
        )
    except (VLLMValidationError, ValueError):
        return None


def _victim_request(
    content_hash: str,
    pixels: torch.Tensor,
    length: int,
    *,
    cache_salt: str | None,
) -> Request:
    """Chat request for media whose content hash is ``content_hash``."""
    processor = _processor(_VllmConfig())
    engine_input = mm_input(
        prompt_token_ids=list(range(1, length + 2)),
        mm_kwargs={"image": [_kwargs_item(pixels)]},
        mm_hashes={"image": [content_hash]},
        mm_placeholders={"image": [PlaceholderRange(offset=0, length=length)]},
        cache_salt=cache_salt,
    )
    core = processor.process_inputs(
        "victim",
        engine_input,
        SamplingParams(max_tokens=1, temperature=0.0),
        ("generate",),
    )
    return Request(
        request_id="victim",
        prompt_token_ids=list(core.prompt_token_ids),
        sampling_params=core.sampling_params,
        pooling_params=None,
        mm_features=core.mm_features,
        cache_salt=core.cache_salt,
    )


def _request_from_core(core, request_id: str) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(core.prompt_token_ids),
        sampling_params=core.sampling_params,
        pooling_params=None,
        mm_features=core.mm_features,
        cache_salt=core.cache_salt,
    )


def _scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.is_encoder_decoder = False
    scheduler.scheduler_config = Namespace(disable_chunked_mm_input=False)
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=64)
    scheduler.ec_connector = None
    scheduler.encoder_cache_mismatch_reqs = set()
    return scheduler


def _runner_for(request: Request) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.encoder_cache = {}
    runner.requests = {
        request.request_id: CachedRequestState(
            req_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids or []),
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
    }
    return runner


def _schedule(scheduler: Scheduler, request: Request, num_new_tokens: int):
    return scheduler._try_schedule_encoder_inputs(
        request,
        0,
        num_new_tokens,
        10_000,
    )


def _store_scheduled_output(
    runner: GPUModelRunner,
    request: Request,
    encoder_inputs: list[int],
    output: torch.Tensor,
) -> None:
    if not encoder_inputs:
        return
    view = _ScheduledEncoderInputs({request.request_id: encoder_inputs})
    hashes, kwargs, _refs = runner._batch_mm_inputs_from_scheduler(view)
    assert kwargs, "scheduled encoder input produced no tensors to encode"
    for mm_hash in hashes:
        runner._cache_encoder_output(mm_hash, output, None, [])


def _release_finished_request(
    scheduler: Scheduler,
    runner: GPUModelRunner,
    request: Request,
) -> None:
    scheduler.encoder_cache_manager.free(request)
    for mm_hash in scheduler.encoder_cache_manager.get_freed_mm_hashes():
        runner.encoder_cache.pop(mm_hash, None)


def _iter_routes(routes):
    for route in routes:
        yield route
        nested = getattr(route, "routes", None)
        original = getattr(route, "original_router", None)
        if nested:
            yield from _iter_routes(nested)
        elif original is not None:
            yield from _iter_routes(original.routes)


def test_scale_out_registers_the_generate_route():
    """``--enable-scale-out`` mounts the real generate handler, not the 403 stub."""
    app = FastAPI()
    app.state.args = Namespace(tokens_only=False, enable_scale_out=True)
    register_scale_out_api_routers(app, ("generate",))
    mounted = [
        route.endpoint
        for route in _iter_routes(app.routes)
        if getattr(route, "path", None) == "/inference/v1/generate"
    ]
    assert generate in mounted

    disabled = FastAPI()
    register_scale_out_api_routers(disabled, ("generate",))
    disabled_endpoints = [
        route.endpoint
        for route in _iter_routes(disabled.routes)
        if getattr(route, "path", None) == "/inference/v1/generate"
    ]
    assert disabled_endpoints
    assert generate not in disabled_endpoints


def _plant_attacker(processor_cache, victim_salt: str | None, *, release: bool):
    victim_image = Image.new("RGB", (8, 8), color=(10, 20, 30))
    other_image = Image.new("RGB", (8, 8), color=(200, 10, 10))
    victim_hash = _content_hash(victim_image)
    assert victim_hash != _content_hash(other_image)

    admitted = _admit(
        _serving(processor_cache),
        _generate_request(victim_hash, _pixels(9), _PLACEHOLDER),
    )
    victim = _victim_request(
        victim_hash,
        _pixels(3),
        _PLACEHOLDER,
        cache_salt=victim_salt,
    )
    scheduler = _scheduler()
    if admitted is None:
        return scheduler, None, victim, None

    attacker = _request_from_core(admitted, "attacker")
    runner = _runner_for(attacker)
    attacker_output = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    encoder_inputs, num_new_tokens, *_rest = _schedule(
        scheduler, attacker, _PLACEHOLDER + 1
    )
    assert attacker.request_id not in scheduler.encoder_cache_mismatch_reqs
    assert num_new_tokens > 0
    scheduler.encoder_cache_manager.allocate(attacker, encoder_inputs[0])
    _store_scheduled_output(runner, attacker, encoder_inputs, attacker_output)
    if release:
        _release_finished_request(scheduler, runner, attacker)
    return scheduler, runner, victim, attacker_output


@pytest.mark.parametrize("processor_cache", [None, _OpenProcessorCache()])
@pytest.mark.parametrize("victim_salt", [None, "tenant-salt"])
def test_client_mm_hash_does_not_supply_embeddings_while_request_is_in_flight(
    processor_cache, victim_salt
):
    """In-flight encoder output under a client hash must not be gathered for
    another request whose media hashes to that value.
    """
    scheduler, runner, victim, attacker_output = _plant_attacker(
        processor_cache, victim_salt, release=False
    )
    if runner is None:
        encoder_inputs, num_new_tokens, *_rest = _schedule(
            scheduler, victim, _PLACEHOLDER + 1
        )
        assert victim.request_id not in scheduler.encoder_cache_mismatch_reqs
        assert num_new_tokens > 0
        assert encoder_inputs == [0]
        return
    _assert_victim_does_not_consume(scheduler, runner, victim, attacker_output)


@pytest.mark.parametrize("processor_cache", [None, _OpenProcessorCache()])
@pytest.mark.parametrize("victim_salt", [None, "tenant-salt"])
def test_client_mm_hash_does_not_supply_embeddings_after_request_finishes(
    processor_cache, victim_salt
):
    """Finishing the generate request must not leave its embeddings under a
    content hash a later request for different media will look up.
    """
    scheduler, runner, victim, attacker_output = _plant_attacker(
        processor_cache, victim_salt, release=True
    )
    if runner is None:
        encoder_inputs, num_new_tokens, *_rest = _schedule(
            scheduler, victim, _PLACEHOLDER + 1
        )
        assert victim.request_id not in scheduler.encoder_cache_mismatch_reqs
        assert num_new_tokens > 0
        assert encoder_inputs == [0]
        return
    _assert_victim_does_not_consume(scheduler, runner, victim, attacker_output)


@pytest.mark.parametrize("processor_cache", [None, _OpenProcessorCache()])
def test_client_mm_hash_embed_count_does_not_fail_another_image(processor_cache):
    """A planted encoder-cache count must not finish a different image's request."""
    victim_hash = _content_hash(Image.new("RGB", (8, 8), color=(1, 2, 3)))
    short = 2
    long = 6
    serving = _serving(processor_cache)
    admitted = _admit(
        serving,
        _generate_request(victim_hash, _pixels(8), short),
    )
    victim = _victim_request(victim_hash, _pixels(1), long, cache_salt=None)
    scheduler = _scheduler()
    if admitted is not None:
        attacker = _request_from_core(admitted, "attacker")
        encoder_inputs, *_rest = _schedule(scheduler, attacker, short + 1)
        scheduler.encoder_cache_manager.allocate(attacker, encoder_inputs[0])
        scheduler.encoder_cache_manager.free(attacker)

    encoder_inputs, num_new_tokens, *_rest = _schedule(scheduler, victim, long + 1)
    assert victim.request_id not in scheduler.encoder_cache_mismatch_reqs
    assert num_new_tokens > 0
    assert encoder_inputs == [0]


def _assert_victim_does_not_consume(
    scheduler: Scheduler,
    runner: GPUModelRunner,
    victim: Request,
    attacker_output: torch.Tensor,
) -> None:
    prompt_len = len(victim.prompt_token_ids or [])
    encoder_inputs, num_new_tokens, *_rest = _schedule(scheduler, victim, prompt_len)
    assert victim.request_id not in scheduler.encoder_cache_mismatch_reqs
    assert num_new_tokens > 0
    identifier = victim.mm_features[0].identifier
    cached = runner._get_encoder_output_from_cache(identifier)
    if not encoder_inputs:
        assert cached is not None
        assert not torch.equal(cached, attacker_output)
    else:
        assert cached is None or not torch.equal(cached, attacker_output)
        assert encoder_inputs == [0]
