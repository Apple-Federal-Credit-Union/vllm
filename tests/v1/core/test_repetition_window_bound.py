# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client repetition windows must not make the scheduler scan unbounded.

Chat completions, completions, and batch chat completions copy
``repetition_detection`` into sampling params with no upper bound. The engine
process receives that object (msgpack), and ``Scheduler._update_request_with_output``
calls ``check_stop`` on the engine thread before the next schedule. These tests
follow that path. They fail while a huge ``max_pattern_size`` still reaches the
stop check, and pass when the request is rejected or the scan is no larger than
a window of 64.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.utils import ConstantList

pytestmark = pytest.mark.cpu_test

# Suggested clamp from the finding. A delivered window may not scan more
# pattern lengths than this on one new token.
_MAX_PATTERN_SIZE = 64
_MAX_MODEL_LEN = 131072
_OUTPUT_LEN = 4096
_CLIENT_MAX_TOKENS = 10**9

_ADMISSION_ERRORS = (VLLMValidationError, ValidationError)


class _CountingList(list):
    """Counts index reads made by the real stop check."""

    def __init__(self, items: list[int]) -> None:
        super().__init__(items)
        self.reads = 0

    def __getitem__(self, index):
        self.reads += 1
        return list.__getitem__(self, index)


def _requested_max_tokens(request) -> int | None:
    if isinstance(request, CompletionRequest):
        return request.max_tokens
    if request.max_completion_tokens is not None:
        return request.max_completion_tokens
    return request.max_tokens


def _parse(kind: str, max_pattern_size: int):
    repetition = {"max_pattern_size": max_pattern_size, "min_count": 2}
    if kind == "chat":
        return ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": _CLIENT_MAX_TOKENS,
                "ignore_eos": True,
                "repetition_detection": repetition,
            }
        )
    if kind == "completion":
        return CompletionRequest.model_validate(
            {
                "prompt": "hi",
                "max_tokens": _CLIENT_MAX_TOKENS,
                "ignore_eos": True,
                "repetition_detection": repetition,
            }
        )
    if kind == "batch":
        batch = BatchChatCompletionRequest.model_validate(
            {
                "messages": [[{"role": "user", "content": "hi"}]],
                "max_completion_tokens": _CLIENT_MAX_TOKENS,
                "ignore_eos": True,
                "repetition_detection": repetition,
            }
        )
        return batch.to_chat_completion_request(batch.messages[0])
    raise AssertionError(kind)


def _engine_sampling_params(kind: str, max_pattern_size: int) -> SamplingParams | None:
    """Return the params the scheduler would see, or None if admission rejects."""
    try:
        request = _parse(kind, max_pattern_size)
        # Client max_tokens wins over generation_config whenever it is set
        # (get_max_tokens). A short prompt leaves almost the whole context.
        max_tokens = get_max_tokens(
            _MAX_MODEL_LEN,
            _requested_max_tokens(request),
            input_length=1,
            default_sampling_params={},
            override_max_tokens=None,
        )
        params = request.to_sampling_params(max_tokens, {})
        params = params.clone()
        engine_request = EngineCoreRequest(
            request_id="client-req",
            prompt_token_ids=[11],
            mm_features=None,
            sampling_params=params,
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        decoded = MsgpackDecoder(EngineCoreRequest).decode(
            MsgpackEncoder().encode(engine_request)
        )
    except _ADMISSION_ERRORS:
        return None
    sampling_params = decoded.sampling_params
    assert sampling_params is not None
    return sampling_params


def _scheduler_stop_reads(params: SamplingParams, num_output_tokens: int) -> int:
    """One scheduler stop check after ``num_output_tokens`` non-repeating tokens."""
    request = Request(
        request_id="client-req",
        prompt_token_ids=[11],
        sampling_params=params,
        pooling_params=None,
    )
    prefix = list(range(1, num_output_tokens))
    request.append_output_token_ids(prefix)
    counted = _CountingList(prefix)
    request._output_token_ids = counted
    request.output_token_ids = ConstantList(counted)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_model_len = _MAX_MODEL_LEN
    scheduler._update_request_with_output(request, [num_output_tokens])
    return counted.reads


def _reference_reads(max_tokens: int) -> int:
    params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        repetition_detection=RepetitionDetectionParams(
            max_pattern_size=_MAX_PATTERN_SIZE,
            min_count=2,
        ),
    )
    return _scheduler_stop_reads(params, _OUTPUT_LEN)


@pytest.mark.parametrize("kind", ["chat", "completion", "batch"])
def test_client_repetition_window_cannot_scan_unbounded_patterns(kind: str):
    """A huge client max_pattern_size must not enlarge the per-token scan."""
    params = _engine_sampling_params(kind, max_pattern_size=_CLIENT_MAX_TOKENS)
    if params is None:
        return

    assert params.max_tokens is not None
    assert params.max_tokens >= _OUTPUT_LEN
    reads = _scheduler_stop_reads(params, _OUTPUT_LEN)
    allowed = _reference_reads(params.max_tokens)
    assert reads <= allowed, (
        f"{kind} stop check indexed the output {reads} times after "
        f"{_OUTPUT_LEN} tokens; a window of {_MAX_PATTERN_SIZE} indexes it "
        f"{allowed} times"
    )


@pytest.mark.parametrize("kind", ["chat", "completion", "batch"])
def test_bounded_repetition_window_still_stops(kind: str):
    """A small client window must still stop a repeated pattern."""
    params = _engine_sampling_params(kind, max_pattern_size=4)
    assert params is not None
    assert params.repetition_detection is not None
    assert params.repetition_detection.max_pattern_size <= _MAX_PATTERN_SIZE

    request = Request.from_engine_core_request(
        EngineCoreRequest(
            request_id="client-req",
            prompt_token_ids=[11],
            mm_features=None,
            sampling_params=params,
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        None,
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_model_len = _MAX_MODEL_LEN
    _new_tokens, stopped = scheduler._update_request_with_output(
        request, [7, 8, 7, 8]
    )
    assert stopped
    assert request.status == RequestStatus.FINISHED_REPETITION
    assert request.stop_reason == "repetition_detected"
