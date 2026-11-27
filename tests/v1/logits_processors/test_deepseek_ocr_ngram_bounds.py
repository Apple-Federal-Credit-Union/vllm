# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client n-gram arguments must not stall a V1 decode step.

DeepSeek-OCR, DeepSeek-OCR-2, and Unlimited-OCR enable
``NGramPerReqLogitsProcessor``. Chat, completion, responses, Anthropic,
transcription, and translation requests copy ``vllm_xargs`` into
``SamplingParams.extra_args``. With the V1 runner, admission is
``validate_logits_processors_parameters``, and the sampler applies the
processor on every step.

These tests fail while admission accepts an ``ngram_size`` and
``window_size`` that make one batched step rescan the output in Python.
They pass when those values are rejected, or when an admitted step stays
within the budget below. Recipe values (30/90 and 35/128) must still be
accepted and must still ban a repeated n-gram.

The Rust frontend is not covered. ``VLLM_USE_RUST_FRONTEND`` defaults to
off, and that frontend copies ``vllm_xargs`` through without this
validator.
"""

import io
import time
from collections.abc import Callable

import pytest
import torch
from fastapi import UploadFile

from vllm.config import VllmConfig
from vllm.config.device import DeviceConfig
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.translation.protocol import (
    TranslationRequest,
)
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.deepseek_ocr import (
    NGramPerReqLogitsProcessor,
)
from vllm.model_executor.models.unlimited_ocr import (
    NGramPerReqLogitsProcessor as UnlimitedNGramPerReqLogitsProcessor,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdateBuilder,
    LogitsProcessors,
    validate_logits_processors_parameters,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# DeepSeek-OCR's published context length. Half of a full output is the
# costly n-gram width: each step builds one n-token tuple per position.
OUTPUT_LEN = 8000
BATCH_SIZE = 4
# A 4-request step with the sizes below takes well over 100ms today.
# Recipe sizes take about 0.1ms. 40ms sits between those.
BATCH_STEP_BUDGET_S = 0.04

DEEPSEEK_FQCN = "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"
UNLIMITED_FQCN = "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor"
PROCESSOR_FQCNS = (DEEPSEEK_FQCN, UNLIMITED_FQCN)

# Finding's parameters, plus a large n-gram whose window is only one
# context length (a window cap that still allows this stays expensive).
ATTACK_SIZES = (
    (4000, 1_000_000_000),
    (2048, 8192),
)
RECIPE_SIZES = (
    (30, 90),
    (35, 128),
)

RequestBuilder = Callable[[int, int], SamplingParams]


def _xargs(ngram_size: int, window_size: int) -> dict[str, int]:
    return {"ngram_size": ngram_size, "window_size": window_size}


def _chat(ngram_size: int, window_size: int) -> SamplingParams:
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-OCR",
        messages=[{"role": "user", "content": "Free OCR."}],
        max_tokens=8192,
        ignore_eos=True,
        vllm_xargs=_xargs(ngram_size, window_size),
    )
    params = request.to_sampling_params(8192, {})
    assert params.ignore_eos is True
    assert params.max_tokens == 8192
    return params


def _completion(ngram_size: int, window_size: int) -> SamplingParams:
    request = CompletionRequest(
        model="deepseek-ai/DeepSeek-OCR",
        prompt="Free OCR.",
        max_tokens=8192,
        ignore_eos=True,
        vllm_xargs=_xargs(ngram_size, window_size),
    )
    params = request.to_sampling_params(8192, {})
    assert params.ignore_eos is True
    assert params.max_tokens == 8192
    return params


def _responses(ngram_size: int, window_size: int) -> SamplingParams:
    request = ResponsesRequest(
        model="deepseek-ai/DeepSeek-OCR",
        input="Free OCR.",
        max_output_tokens=8192,
        ignore_eos=True,
        vllm_xargs=_xargs(ngram_size, window_size),
    )
    params = request.to_sampling_params(8192, {})
    assert params.ignore_eos is True
    assert params.max_tokens == 8192
    return params


def _anthropic(ngram_size: int, window_size: int) -> SamplingParams:
    request = AnthropicMessagesRequest(
        model="deepseek-ai/DeepSeek-OCR",
        messages=[{"role": "user", "content": "Free OCR."}],
        max_tokens=8192,
        vllm_xargs=_xargs(ngram_size, window_size),
    )
    chat = AnthropicServingMessages._build_base_request(
        request,
        [{"role": "user", "content": "Free OCR."}],
    )
    params = chat.to_sampling_params(8192, {})
    assert params.max_tokens == 8192
    return params


def _audio() -> UploadFile:
    return UploadFile(io.BytesIO(b"clip"), filename="clip.wav")


def _transcription(ngram_size: int, window_size: int) -> SamplingParams:
    request = TranscriptionRequest(
        file=_audio(),
        model="deepseek-ai/DeepSeek-OCR",
        vllm_xargs=_xargs(ngram_size, window_size),
    )
    return request.to_sampling_params(8192, {})


def _translation(ngram_size: int, window_size: int) -> SamplingParams:
    request = TranslationRequest(
        file=_audio(),
        model="deepseek-ai/DeepSeek-OCR",
        vllm_xargs=_xargs(ngram_size, window_size),
    )
    return request.to_sampling_params(8192, {})


REQUEST_BUILDERS = (
    _chat,
    _completion,
    _responses,
    _anthropic,
    _transcription,
    _translation,
)


def _vllm_config() -> VllmConfig:
    return VllmConfig(device_config=DeviceConfig(device="cpu"))


def _assert_copied(params: SamplingParams, ngram_size: int, window_size: int) -> None:
    assert params.extra_args is not None
    assert params.extra_args["ngram_size"] == ngram_size
    assert params.extra_args["window_size"] == window_size


def _decode_step_seconds(params: SamplingParams) -> float:
    """One V1 sampler step for a batch that already holds ``params``.

    Mirrors ``InputBatch.refresh_metadata`` (``update_state``) and the
    non-argmax loop in ``Sampler.apply_logits_processors``. Greedy
    sampling, which the OCR recipes use, still runs that loop.
    """
    processor = NGramPerReqLogitsProcessor(
        _vllm_config(), torch.device("cpu"), False
    )
    builder = BatchUpdateBuilder()
    for index in range(BATCH_SIZE):
        output_ids = [(token * 17 + index) % 63 for token in range(OUTPUT_LEN)]
        builder.added.append((index, params, None, output_ids))
    processor.update_state(builder.get_and_reset(BATCH_SIZE))
    logits = torch.zeros(BATCH_SIZE, 64)
    group = LogitsProcessors([processor])
    assert group.non_argmax_invariant == [processor]
    started = time.perf_counter()
    for installed in group.non_argmax_invariant:
        logits = installed.apply(logits)
    return time.perf_counter() - started


def _require_harmless(params: SamplingParams) -> None:
    """Reject the request, or keep the shared decode step cheap.

    Admission and the worker read the same ``SamplingParams``. A check
    that does not reject and does not change the values the worker
    applies still fails the budget.
    """
    admitted = False
    for fqcn in PROCESSOR_FQCNS:
        try:
            validate_logits_processors_parameters((fqcn,), params)
        except VLLMValidationError as exc:
            message = str(exc).lower()
            assert "ngram" in message or "window" in message, message
        else:
            admitted = True
    if admitted:
        elapsed = _decode_step_seconds(params)
        assert elapsed < BATCH_STEP_BUDGET_S, (
            f"admitted ngram_size={params.extra_args['ngram_size']} "
            f"window_size={params.extra_args['window_size']}; "
            f"{BATCH_SIZE}-request step took {elapsed:.3f}s"
        )


@pytest.mark.parametrize("build", REQUEST_BUILDERS, ids=lambda fn: fn.__name__)
@pytest.mark.parametrize("ngram_size,window_size", ATTACK_SIZES)
def test_client_ngram_sizes_cannot_stall_decode_step(
    build: RequestBuilder,
    ngram_size: int,
    window_size: int,
):
    params = build(ngram_size, window_size)
    _assert_copied(params, ngram_size, window_size)
    _require_harmless(params)


@pytest.mark.parametrize("ngram_size,window_size", RECIPE_SIZES)
def test_recipe_ngram_sizes_stay_admitted(ngram_size: int, window_size: int):
    params = _chat(ngram_size, window_size)
    _assert_copied(params, ngram_size, window_size)
    for fqcn in PROCESSOR_FQCNS:
        validate_logits_processors_parameters((fqcn,), params)
    assert _decode_step_seconds(params) < BATCH_STEP_BUDGET_S


def test_recipe_ngram_still_bans_repeated_token():
    params = _chat(30, 90)
    validate_logits_processors_parameters((DEEPSEEK_FQCN,), params)
    processor = NGramPerReqLogitsProcessor(
        _vllm_config(), torch.device("cpu"), False
    )
    ngram = list(range(30))
    output_ids = ngram + ngram[:-1]
    builder = BatchUpdateBuilder()
    builder.added.append((0, params, None, output_ids))
    processor.update_state(builder.get_and_reset(1))
    logits = torch.zeros(1, 30)
    group = LogitsProcessors([processor])
    logits = group.non_argmax_invariant[0].apply(logits)
    assert logits[0, ngram[-1]].item() == -float("inf")
    assert logits[0, 0].item() == 0.0


def test_request_without_ngram_skips_processor():
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-OCR",
        messages=[{"role": "user", "content": "Free OCR."}],
        max_tokens=16,
    )
    params = request.to_sampling_params(16, {})
    validate_logits_processors_parameters((DEEPSEEK_FQCN,), params)
    processor = NGramPerReqLogitsProcessor(
        _vllm_config(), torch.device("cpu"), False
    )
    assert processor.new_req_logits_processor(params) is None


def test_unlimited_ocr_exports_the_same_processor():
    assert UnlimitedNGramPerReqLogitsProcessor is NGramPerReqLogitsProcessor
