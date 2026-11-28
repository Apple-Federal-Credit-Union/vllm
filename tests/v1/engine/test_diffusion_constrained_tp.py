# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constrained DiffusionGemma requests must not kill a tensor-parallel engine.

A completion or chat client can set ``diffusion_constrained`` through
``vllm_xargs`` together with ``logprob_token_ids``. Admission
(``InputProcessor._validate_params`` -> ``validate_diffusion_sampling_params``)
accepts that pair and does not look at tensor-parallel size. The V2 GPU model
runner then calls ``DiffusionSampler.add_request``, which raises when the
worker was started with tensor parallel size greater than 1. That exception
is not caught per request: the multiprocess executor reports the worker
failure, ``EngineCore.step`` lets it out of the busy loop, and
``EngineCoreProc`` publishes ``ENGINE_CORE_DEAD``.

These tests fail while that gap is present. They pass only when a constrained
request on tensor parallel size greater than 1 is rejected at admission with
``VLLMValidationError`` (HTTP 400) before the engine is stepped. Containing
the sampler exception inside the request still fails the test: the request
must not be admitted.

Paths that cannot carry both ``diffusion_constrained`` and
``logprob_token_ids`` are not covered here: the Responses API, the Anthropic
messages adapter, and the speech transcription/translation protocols copy
``vllm_xargs`` but do not set ``logprob_token_ids``, so admission already
rejects them for a missing allowed set. Chat batch completions use
``ChatCompletionRequest.to_sampling_params``, which the chat case covers.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from http import HTTPStatus

import pytest
import torch

from vllm.config.diffusion import DiffusionConfig
from vllm.config.observability import ObservabilityConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.vllm import VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.exceptions import VLLMValidationError
from vllm.inputs.engine import tokens_input
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaRequestStates,
    DiffusionSampler,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.request import Request
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

_PROMPT_IDS = [11, 22, 33, 44]
_ALLOWED_IDS = [3, 5]


class _CanvasModel:
    """Diffusion model config: ``canvas_length`` makes ``is_diffusion`` true.

    Matches ``ModelConfig.is_diffusion``.
    """

    def __init__(self, canvas_length: int = 8) -> None:
        self.canvas_length = canvas_length
        self.max_logprobs = 20
        self.logits_processors = None
        self.return_sampling_mask = False
        self.enable_trace_replay = False
        self.max_model_len = 256
        self.runner_type = "generate"

    @property
    def is_diffusion(self) -> bool:
        return self.canvas_length is not None

    def get_vocab_size(self) -> int:
        return 1024


class _Renderer:
    def get_eos_token_id(self) -> None:
        return None

    def validate_token_ids(self, prompt_ids: list[int]) -> None:
        return None


class _Cache:
    kv_sharing_fast_prefill = False


class _ReqSlots:
    def __init__(self) -> None:
        self.req_id_to_index: dict[str, int] = {}

    def remove_request(self, req_id: str) -> int | None:
        return self.req_id_to_index.pop(req_id, None)

    def add_request(self, *, req_id: str, **kwargs: object) -> None:
        self.req_id_to_index[req_id] = 0


class _IdOnly:
    def remove_request(self, req_id: str) -> None:
        return None

    def add_request(self, *args: object, **kwargs: object) -> None:
        return None


class _BlockTables:
    def append_block_ids(
        self, req_index: int, block_ids: tuple[list[int], ...], *, overwrite: bool
    ) -> None:
        return None


class _ResponseQueue:
    """In-process stand-in for the worker response ``MessageQueue``."""

    def __init__(self) -> None:
        self._items: queue.Queue[tuple[object, object]] = queue.Queue()

    def enqueue(self, item: tuple[object, object]) -> None:
        self._items.put(item)

    def dequeue(self, timeout: float | None = None, indefinite: bool = False):
        return self._items.get(timeout=timeout)


class DiffusionFrontend:
    """Frontend attributes ``InputProcessor.process_inputs`` reads.

    The methods are the real ``InputProcessor`` methods. Tensor-parallel size
    lives on ``parallel_config``, which is what the API process has.
    ``get_tp_group()`` is not initialized here: that group exists only in the
    workers, where ``DiffusionSampler.tp_size`` was captured at startup.
    """

    process_inputs = InputProcessor.process_inputs
    _validate_params = InputProcessor._validate_params
    _validate_lora = InputProcessor._validate_lora
    _validate_model_inputs = InputProcessor._validate_model_inputs
    _validate_model_input = InputProcessor._validate_model_input
    _validate_prompt_len = InputProcessor._validate_prompt_len
    apply_watermarking = InputProcessor.apply_watermarking
    resolve_watermarking = InputProcessor.resolve_watermarking

    def __init__(self, tensor_parallel_size: int) -> None:
        self.model_config = _CanvasModel()
        self.cache_config = _Cache()
        self.lora_config = None
        self.speculative_config = None
        self.structured_outputs_config = None
        self.diffusion_config = DiffusionConfig(canvas_length=8)
        self.tokenizer = None
        self.renderer = _Renderer()
        self.generation_config_fields: dict = {}
        self.supports_mm_inputs = False
        self.skip_prompt_length_check = False
        self.mm_encoder_cache_size = 0
        self.validate_logits_processors_params = lambda params: None
        parallel = ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="mp",
        )
        self.vllm_config = type(
            "_Cfg",
            (),
            {
                "parallel_config": parallel,
                "cache_config": self.cache_config,
                "reasoning_config": None,
                "use_v2_model_runner": True,
                "watermark_config": None,
                "observability_config": ObservabilityConfig(),
                "speculative_config": None,
                "_check_supports_watermarking": (
                    VllmConfig._check_supports_watermarking
                ),
            },
        )()


def _completion_params(flag: int) -> SamplingParams:
    request = CompletionRequest.model_validate(
        {
            "model": "diffusion-gemma",
            "prompt": "hello",
            "max_tokens": 8,
            "logprobs": 1,
            "logprob_token_ids": list(_ALLOWED_IDS),
            "vllm_xargs": {"diffusion_constrained": flag},
        }
    )
    return request.to_sampling_params(8)


def _chat_params(flag: int) -> SamplingParams:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "diffusion-gemma",
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 8,
            "logprobs": True,
            "logprob_token_ids": list(_ALLOWED_IDS),
            "vllm_xargs": {"diffusion_constrained": flag},
        }
    )
    return request.to_sampling_params(8, {})


def _build_sampler(tp_size: int) -> DiffusionSampler:
    sampler = DiffusionSampler.__new__(DiffusionSampler)
    sampler._pending_logprobs = {}
    sampler.sampling_states = SamplingStates(max_num_reqs=2, vocab_size=1024)
    sampler.logprob_token_ids_state = LogprobTokenIdsState(2, torch.device("cpu"))
    sampler.tp_size = tp_size
    sampler.diffusion_states = DiffusionGemmaRequestStates(
        max_num_reqs=2,
        canvas_length=8,
        vocab_size=1024,
        max_denoising_steps=4,
        device=torch.device("cpu"),
        hidden_size=4,
        stability_threshold=2,
    )
    return sampler


def _sampler(tp_size: int) -> DiffusionSampler:
    """Build the sampler the worker holds after ``custom_sampler``.

    ``custom_sampler`` stores ``tp_group.world_size`` on ``tp_size`` once, at
    worker startup. Admission later runs in the API process, where that group
    is not initialized, so the stored size is the one the request will hit.

    CPU torch in this environment has no pinned-memory allocator, so the
    sampler state uses the non-UVA buffer the same module selects when UVA
    is unavailable. The request data still goes through the real
    ``SamplingStates.add_request`` and ``LogprobTokenIdsState.add_request``.
    """
    original_uva = buffer_utils.is_uva_available
    buffer_utils.is_uva_available = lambda: False
    try:
        return _build_sampler(tp_size)
    finally:
        buffer_utils.is_uva_available = original_uva


def _runner(sampler: DiffusionSampler) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.req_states = _ReqSlots()
    runner.model_state = _IdOnly()
    runner.lora_state = _IdOnly()
    runner.block_tables = _BlockTables()
    runner.pooling_runner = None
    runner.encoder_cache = None
    runner.pp_handler = None
    runner.prompt_logprobs_worker = None
    runner.adaptive_verification = None
    runner.is_last_pp_rank = True
    runner.sampler = sampler
    return runner


def _scheduler_output(request: Request) -> SchedulerOutput:
    new_req = NewRequestData.from_request(
        request,
        block_ids=([],),
        prefill_token_ids=list(request.prompt_token_ids or []),
    )
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [new_req]
    # No tokens this step, so GPUWorker.execute_model does not touch the PP
    # group. add_requests still installs the new request before that check.
    output.num_scheduled_tokens = {}
    output.total_num_scheduled_tokens = 0
    return output


def _drive_engine(request: Request, tensor_parallel_size: int) -> bool:
    """Return True if the step publishes ENGINE_CORE_DEAD.

    Mirrors ``EngineCoreProc.run_engine_core``: an exception leaving the busy
    loop calls ``_send_engine_dead`` and is re-raised.
    """
    sampler = _sampler(tensor_parallel_size)
    runner = _runner(sampler)
    worker = GPUWorker.__new__(GPUWorker)
    worker._pp_send_work = []
    worker.profiler = None
    worker.use_v2_model_runner = True
    worker.model_runner = runner

    response_q = _ResponseQueue()

    def broadcast(payload: tuple) -> None:
        worker_proc._execute_worker_rpc(payload)

    worker_proc = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = worker
    worker_proc.use_async_scheduling = False
    worker_proc.worker_response_mq = response_q

    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.output_rank = 0
    executor.futures_queue = deque()
    executor.kv_output_aggregator = None
    executor.ec_output_aggregator = None
    broadcast_mq = type("_Broadcast", (), {})()
    broadcast_mq.enqueue = broadcast
    executor.rpc_broadcast_mq = broadcast_mq
    executor.response_mqs = [response_q]
    executor.vllm_config = type(
        "_ExecCfg",
        (),
        {
            "parallel_config": ParallelConfig(
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="mp",
            )
        },
    )()

    scheduler_output = _scheduler_output(request)

    class _Scheduler:
        def has_requests(self) -> bool:
            return True

        def schedule(self, throttle: bool) -> SchedulerOutput:
            del throttle
            return scheduler_output

        def get_grammar_bitmask(self, scheduled: SchedulerOutput) -> None:
            del scheduled
            return None

        def make_stats(self) -> None:
            return None

    output_q: queue.Queue = queue.Queue()
    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join()

    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.log_stats = False
    engine.vllm_config = type(
        "_EngineCfg",
        (),
        {"observability_config": ObservabilityConfig()},
    )()
    engine.scheduler = _Scheduler()
    engine.model_executor = executor
    engine.step_fn = engine.step
    engine.output_queue = output_q
    engine.output_thread = finished

    try:
        engine._process_engine_step()
    except Exception as exc:
        # run_engine_core's handler: any exception here is fatal.
        engine._send_engine_dead()
        assert output_q.get_nowait() == EngineCoreProc.ENGINE_CORE_DEAD
        message = str(exc).lower()
        assert "tensor parallel" in message or "diffusion_constrained" in message
        raise
    return False


def _admit(frontend: DiffusionFrontend, params: SamplingParams):
    return frontend.process_inputs(
        "cmpl-constrained",
        tokens_input(_PROMPT_IDS),
        params,
        ("generate",),
    )


@pytest.mark.parametrize("build", [_completion_params, _chat_params])
def test_constrained_request_is_rejected_before_the_engine_dies(
    build: Callable[[int], SamplingParams],
) -> None:
    frontend = DiffusionFrontend(tensor_parallel_size=2)
    params = build(1)
    assert params.extra_args is not None
    assert params.extra_args["diffusion_constrained"] == 1
    assert params.logprob_token_ids == _ALLOWED_IDS
    # The worker captured this size at startup. Admission must use it, not a
    # later lookup of an uninitialized TP group in the API process.
    worker_tp = frontend.vllm_config.parallel_config.tensor_parallel_size
    assert worker_tp == 2

    try:
        core_request = _admit(frontend, params)
    except VLLMValidationError as exc:
        assert "tensor parallel" in str(exc).lower()
        err = create_error_response(exc)
        assert err.error.code == HTTPStatus.BAD_REQUEST
        return

    admitted = core_request.sampling_params
    assert admitted is not None
    assert admitted.extra_args is not None
    assert admitted.extra_args["diffusion_constrained"] == 1
    request = Request.from_engine_core_request(core_request, None)
    try:
        finished = _drive_engine(request, worker_tp)
    except Exception as exc:
        pytest.fail(
            "constrained request was admitted on tensor parallel size "
            f"{worker_tp} and killed the engine "
            f"({type(exc).__name__}: {exc})"
        )
    pytest.fail(
        "constrained request was admitted on tensor parallel size "
        f"{worker_tp}; engine step returned {finished}"
    )


def test_constrained_request_is_admitted_at_tensor_parallel_1() -> None:
    frontend = DiffusionFrontend(tensor_parallel_size=1)
    params = _completion_params(1)
    core_request = _admit(frontend, params)
    assert core_request.sampling_params is not None
    assert core_request.sampling_params.extra_args is not None
    assert core_request.sampling_params.extra_args["diffusion_constrained"] == 1


def test_unconstrained_request_is_admitted_at_tensor_parallel_2() -> None:
    frontend = DiffusionFrontend(tensor_parallel_size=2)
    request = CompletionRequest.model_validate(
        {
            "model": "diffusion-gemma",
            "prompt": "hello",
            "max_tokens": 8,
        }
    )
    params = request.to_sampling_params(8)
    core_request = _admit(frontend, params)
    extra = (
        core_request.sampling_params.extra_args
        if core_request.sampling_params
        else None
    )
    assert not extra or not extra.get("diffusion_constrained")
