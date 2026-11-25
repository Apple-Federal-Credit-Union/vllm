# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission must reject sampling params that the V2 sampler cannot store.

The V2 GPU sampler allocates fixed per-request buffers and raises ValueError
from LogitBiasState.add_request / BadWordsState.add_request when a request
exceeds them. Those raises happen inside GPUModelRunner.execute_model, and
EngineCore.step does not turn them into a per-request error. run_engine_core
treats that exception as fatal and calls _send_engine_dead.

These tests fail while that gap is present: InputProcessor.process_inputs
accepts the request, then the same sampling params make EngineCore.step raise.
They pass once admission raises VLLMValidationError for the same caps.

The Responses API forwards stop_token_ids but not min_tokens, and the worker
only applies the stop-token cap when min_tokens is above zero, so that field
alone is not one of these cases. Beam search builds its own SamplingParams
and does not copy the caller's allowed_token_ids into the V2 sampler.
"""

import queue
import threading
from concurrent.futures import Future

import numpy as np
import pytest

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import (
    MAX_NUM_ALLOWED_TOKEN_IDS,
    MAX_NUM_LOGIT_BIAS_TOKENS,
    MAX_NUM_STOP_TOKEN_IDS,
    LogitBiasState,
)

pytestmark = [pytest.mark.cpu_test]

_TOKEN_PROMPT = {"type": "token", "prompt_token_ids": [1, 2, 3]}
_VOCAB = 4096


class _CountingTokenizer:
    """Tokenizer stand-in whose encode length is chosen by the test.

    update_from_tokenizer calls encode twice per bad word (plain, then with a
    leading space) and keeps the second sequence only when it is a different
    tokenization of the same length. Returning the same ids both times stores
    one sequence, so the flattened token count is exactly ``num_tokens``.
    """

    def __init__(self, num_tokens: int):
        self.num_tokens = num_tokens
        self.max_token_id = _VOCAB - 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(self.num_tokens))


class _HostBuffer:
    def __init__(self):
        self.np = np.zeros(4, dtype=np.int32)


class _ModelConfig:
    max_model_len = 2048
    max_logprobs = 20
    logits_processors = None
    is_diffusion = False
    return_sampling_mask = False
    supports_multimodal_inputs = False
    runner_type = "generate"

    def __init__(self, generation_config: dict):
        self._generation_config = generation_config

    def get_vocab_size(self) -> int:
        return _VOCAB

    def try_get_generation_config(self) -> dict:
        return self._generation_config


class _ParallelConfig:
    data_parallel_size = 1
    data_parallel_size_local = 1
    local_engines_only = False


class _Observability:
    enable_logging_iteration_details = False


class _VllmConfig:
    """Config surface InputProcessor and EngineCore.step actually read."""

    use_v2_model_runner = True
    speculative_config = None
    structured_outputs_config = None
    lora_config = None
    cache_config = None
    diffusion_config = None
    watermark_config = None
    scheduler_config = None
    observability_config = _Observability()
    parallel_config = _ParallelConfig()

    def __init__(self, generation_config: dict):
        self.model_config = _ModelConfig(generation_config)

    def _check_supports_watermarking(self, params) -> bool:
        return False


class _Renderer:
    # InputProcessor wraps process_inputs for async callers. The sync path
    # under test does not use the executor.
    _executor = None

    def __init__(self, tokenizer, eos_token_id):
        self.tokenizer = tokenizer
        self._eos_token_id = eos_token_id

    def get_eos_token_id(self):
        return self._eos_token_id

    def validate_token_ids(self, token_ids) -> None:
        return None


def _processor(
    *,
    generation_config: dict | None = None,
    eos_token_id: int | None = None,
    tokenizer=None,
) -> InputProcessor:
    config = _VllmConfig(generation_config or {})
    renderer = _Renderer(tokenizer, eos_token_id)
    return InputProcessor(config, renderer)


def _chat(**kwargs) -> SamplingParams:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8,
        **kwargs,
    )
    return request.to_sampling_params(8, {})


def _completion(**kwargs) -> SamplingParams:
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        max_tokens=8,
        **kwargs,
    )
    return request.to_sampling_params(8, {})


def _responses(**kwargs) -> SamplingParams:
    request = ResponsesRequest(
        model="test-model",
        input="hello",
        max_output_tokens=8,
        **kwargs,
    )
    return request.to_sampling_params(8, {})


def _batch_chat_logit_bias(n: int) -> SamplingParams:
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=[[{"role": "user", "content": "hello"}]],
        max_tokens=8,
        logit_bias={str(i): 0.0 for i in range(n)},
    )
    single = request.to_chat_completion_request(request.messages[0])
    return single.to_sampling_params(8, {})


def _logit_bias(n: int) -> dict[str, float]:
    return {str(i): 0.0 for i in range(n)}


def _run_v2_sampler_add(sampling_params: SamplingParams) -> None:
    """Call the V2 processors in Sampler.add_request order.

    Sampler.add_request runs sampling state and thinking-budget updates first,
    then logits processors. Those earlier updates do not read the capped
    fields. PenaltiesState sits between logit bias and bad words and does not
    read them either. The raises live in the two processors called here.
    """
    bias = LogitBiasState.__new__(LogitBiasState)
    bias.prompt_len_np = np.zeros(4, dtype=np.int32)
    bias.num_allowed_token_ids = _HostBuffer()
    bias.num_logit_bias = _HostBuffer()
    bias.min_lens = _HostBuffer()
    bias.num_stop_token_ids = _HostBuffer()
    bias.restore_when_all_masked = _HostBuffer()
    bias.use_logit_bias = np.zeros(4, dtype=bool)
    bias.add_request(0, sampling_params)

    bad = BadWordsState.__new__(BadWordsState)
    # The empty-bad-words path writes this counter before returning.
    bad.num_bad_words = _HostBuffer()
    bad.add_request(0, sampling_params)


class _Executor:
    """Stand-in for the worker RPC that EngineCore.step calls.

    GPUModelRunner.execute_model calls add_requests, which calls
    Sampler.add_request with the scheduled request's sampling params and does
    not catch ValueError. This executor performs that call and returns it on
    the Future that step waits on inside log_error_detail.
    """

    def __init__(self, sampling_params: SamplingParams):
        self.sampling_params = sampling_params

    def execute_model(self, scheduler_output: SchedulerOutput, non_block: bool = True):
        future: Future = Future()
        try:
            for new_req in scheduler_output.scheduled_new_reqs:
                assert new_req.sampling_params is self.sampling_params
                _run_v2_sampler_add(new_req.sampling_params)
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(None)
        return future

    def sample_tokens(self, grammar_output):
        raise AssertionError("sample_tokens ran after sampler add_request failed")


class _Scheduler:
    def __init__(self, output: SchedulerOutput):
        self._output = output

    def has_requests(self) -> bool:
        return True

    def schedule(self, throttle_prefills: bool) -> SchedulerOutput:
        return self._output

    def get_grammar_bitmask(self, scheduler_output: SchedulerOutput):
        return None

    def make_stats(self):
        return None


def _engine_step_outcome(sampling_params: SamplingParams) -> str:
    """Run EngineCore.step on params that already passed admission.

    On a worker ValueError, reproduce the fatal handler in run_engine_core:
    log the failure and queue ENGINE_CORE_DEAD via _send_engine_dead.
    """
    new_req = NewRequestData(
        req_id="req",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=sampling_params,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        prefill_token_ids=[1, 2, 3],
    )
    output = SchedulerOutput(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"req": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    core = EngineCore.__new__(EngineCore)
    core.vllm_config = _VllmConfig({})
    core.log_stats = False
    core.scheduler = _Scheduler(output)
    core.model_executor = _Executor(sampling_params)
    core._should_throttle_prefills = lambda: False

    try:
        EngineCore.step(core)
    except Exception as exc:
        core.output_queue = queue.Queue()
        thread = threading.Thread(target=lambda: None, daemon=True)
        thread.start()
        thread.join()
        core.output_thread = thread
        EngineCoreProc._send_engine_dead(core)
        dead = core.output_queue.get_nowait()
        return (
            f"EngineCore.step raised {type(exc).__name__}: {exc}. "
            f"_send_engine_dead queued {dead!r}."
        )
    return "EngineCore.step returned without error."


def _assert_rejected(processor: InputProcessor, params: SamplingParams, parameter: str):
    try:
        request = processor.process_inputs("req", _TOKEN_PROMPT, params, ("generate",))
    except VLLMValidationError as exc:
        assert exc.parameter == parameter
        return

    outcome = _engine_step_outcome(request.sampling_params)
    pytest.fail(
        f"{parameter} above the V2 sampler cap was admitted by "
        f"InputProcessor.process_inputs. {outcome}"
    )


def test_allowed_token_ids_at_cap_are_admitted():
    params = _chat(allowed_token_ids=[0] * MAX_NUM_ALLOWED_TOKEN_IDS)
    request = _processor().process_inputs("req", _TOKEN_PROMPT, params, ("generate",))
    assert (
        len(request.sampling_params.allowed_token_ids) == MAX_NUM_ALLOWED_TOKEN_IDS
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda n: _chat(allowed_token_ids=[0] * n),
        lambda n: _completion(allowed_token_ids=[0] * n),
        lambda n: SamplingParams(allowed_token_ids=[0] * n, max_tokens=8),
    ],
    ids=["chat", "completion", "sampling-params"],
)
def test_repeated_allowed_token_ids_over_cap_rejected(build):
    params = build(MAX_NUM_ALLOWED_TOKEN_IDS + 1)
    _assert_rejected(_processor(), params, "allowed_token_ids")


def test_logit_bias_at_cap_is_admitted():
    params = _chat(logit_bias=_logit_bias(MAX_NUM_LOGIT_BIAS_TOKENS))
    request = _processor().process_inputs("req", _TOKEN_PROMPT, params, ("generate",))
    assert len(request.sampling_params.logit_bias) == MAX_NUM_LOGIT_BIAS_TOKENS


@pytest.mark.parametrize(
    "build",
    [
        lambda n: _chat(logit_bias=_logit_bias(n)),
        lambda n: _completion(logit_bias=_logit_bias(n)),
        lambda n: _responses(logit_bias=_logit_bias(n)),
        lambda n: _batch_chat_logit_bias(n),
        lambda n: SamplingParams(
            logit_bias={i: 0.0 for i in range(n)},
            max_tokens=8,
        ),
    ],
    ids=["chat", "completion", "responses", "batch-chat", "sampling-params"],
)
def test_logit_bias_over_cap_rejected(build):
    params = build(MAX_NUM_LOGIT_BIAS_TOKENS + 1)
    _assert_rejected(_processor(), params, "logit_bias")


def test_stop_token_ids_at_cap_are_admitted():
    params = _chat(
        stop_token_ids=list(range(MAX_NUM_STOP_TOKEN_IDS)),
        min_tokens=1,
    )
    request = _processor().process_inputs("req", _TOKEN_PROMPT, params, ("generate",))
    assert len(request.sampling_params.all_stop_token_ids) == MAX_NUM_STOP_TOKEN_IDS


def test_duplicate_stop_token_ids_collapse_below_cap():
    params = _chat(stop_token_ids=[7] * (MAX_NUM_STOP_TOKEN_IDS + 10), min_tokens=1)
    request = _processor().process_inputs("req", _TOKEN_PROMPT, params, ("generate",))
    assert request.sampling_params.all_stop_token_ids == {7}


@pytest.mark.parametrize(
    "build",
    [
        lambda n: _chat(stop_token_ids=list(range(n)), min_tokens=1),
        lambda n: _completion(stop_token_ids=list(range(n)), min_tokens=1),
        lambda n: SamplingParams(
            stop_token_ids=list(range(n)),
            min_tokens=1,
            max_tokens=8,
        ),
    ],
    ids=["chat", "completion", "sampling-params"],
)
def test_stop_token_ids_over_cap_rejected(build):
    params = build(MAX_NUM_STOP_TOKEN_IDS + 1)
    _assert_rejected(_processor(), params, "stop_token_ids")


def test_eos_added_after_validation_still_counts_toward_stop_cap():
    """Generation-config EOS is merged after SamplingParams.verify.

    A list that is exactly at the cap before that merge must still be rejected
    when the merged all_stop_token_ids set no longer fits the worker buffer.
    """
    eos = _VOCAB - 1
    params = _chat(
        stop_token_ids=list(range(MAX_NUM_STOP_TOKEN_IDS)),
        min_tokens=1,
    )
    processor = _processor(
        generation_config={"eos_token_id": eos},
        eos_token_id=eos,
    )
    _assert_rejected(processor, params, "stop_token_ids")


def test_bad_word_tokens_at_cap_are_admitted():
    cap = envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS
    processor = _processor(tokenizer=_CountingTokenizer(cap))
    params = _chat(bad_words=["forbidden"])
    request = processor.process_inputs("req", _TOKEN_PROMPT, params, ("generate",))
    token_ids = request.sampling_params.bad_words_token_ids
    assert token_ids is not None
    assert sum(len(word) for word in token_ids) == cap


@pytest.mark.parametrize(
    "build",
    [
        lambda: _chat(bad_words=["forbidden"]),
        lambda: _completion(bad_words=["forbidden"]),
        lambda: SamplingParams(bad_words=["forbidden"], max_tokens=8),
    ],
    ids=["chat", "completion", "sampling-params"],
)
def test_bad_word_tokens_over_cap_rejected(build):
    cap = envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS
    processor = _processor(tokenizer=_CountingTokenizer(cap + 1))
    _assert_rejected(processor, build(), "bad_words")
