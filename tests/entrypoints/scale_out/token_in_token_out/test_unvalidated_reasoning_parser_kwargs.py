# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scale-out generate must not crash the engine on bad reasoning-parser kwargs.

An API client can POST ``/inference/v1/generate`` with
``reasoning_parser_kwargs.chat_template_kwargs`` while structured output is
requested and ``reasoning_ended`` is unset or false. ``ServingTokens`` forwards
that dict to the engine. ``StructuredOutputManager._get_reasoner`` constructs
the configured parser only when the scheduler builds a grammar bitmask, and
``K2HorizonReasoningParser`` raises ``ValueError`` for an unknown
``reasoning_effort``. That exception leaves ``EngineCore.step`` and, with
fault tolerance off (the default), the engine-core process treats it as fatal.

These tests describe the fixed behavior. They fail while the engine still
dies. Chat and responses construct the parser on the API side before
``generate``; this file does not cover those routes. The Rust generate
lowerer (``prepare_generate_request``) also forwards the kwargs without
constructing a parser; the engine-step test below is the shared crash, and
this file does not drive the Rust HTTP stack.
"""

import queue
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.v1.core.utils import create_scheduler
from vllm.config.parallel import ParallelConfig
from vllm.config.structured_outputs import StructuredOutputsConfig
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.scale_out.token_in_token_out.api_router import attach_router
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.renderers import renderer_from_config
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.fault_tolerance.engine_core_sentinel import fault_tolerant_wrapper
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager

MODEL_NAME = "openai-community/gpt2"
_BAD_EFFORT = "not-a-real-effort"
_JSON_SCHEMA = {"type": "object"}


class _HFConfig:
    model_type = "gpt2"


@dataclass
class _ModelConfig:
    task: str = "generate"
    runner_type: str = "generate"
    model: str = MODEL_NAME
    tokenizer: str = MODEL_NAME
    trust_remote_code: bool = False
    tokenizer_mode: str = "auto"
    max_model_len: int = 100
    tokenizer_revision: str | None = None
    hf_config: _HFConfig = field(default_factory=_HFConfig)
    hf_text_config: _HFConfig = field(default_factory=_HFConfig)
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config: Any = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self) -> dict:
        return self.diff_sampling_param or {}


@dataclass
class _ParallelConfig:
    _api_process_rank: int = 0


@dataclass
class _SchedulerConfig:
    max_num_seqs: int = 128


@dataclass
class _VllmConfig:
    model_config: _ModelConfig
    parallel_config: _ParallelConfig
    scheduler_config: _SchedulerConfig = field(default_factory=_SchedulerConfig)
    structured_outputs_config: StructuredOutputsConfig = field(
        default_factory=lambda: StructuredOutputsConfig(
            reasoning_parser="k2_horizon",
            backend="xgrammar",
        )
    )
    kv_transfer_config: Any = None


class _RecordingEngine:
    """Engine client that records generate() and refuses to run it."""

    errored = False

    def __init__(self) -> None:
        self.model_config = _ModelConfig()
        self.vllm_config = _VllmConfig(
            self.model_config, parallel_config=_ParallelConfig()
        )
        self.input_processor = None
        self.renderer = renderer_from_config(self.vllm_config)
        self.calls: list[dict[str, Any] | None] = []

    def check_admission(self, n: int = 1) -> None:
        del n

    @property
    def dead_error(self) -> Exception:
        return RuntimeError("engine dead")

    async def generate(self, *args: Any, **kwargs: Any):
        del args
        self.calls.append(kwargs.get("reasoning_parser_kwargs"))
        raise RuntimeError("request was submitted to the engine")
        yield  # pragma: no cover


def _attack_payload(*, stream: bool, reasoning_ended: bool | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "token_ids": [1, 2, 3],
        "sampling_params": {
            "max_tokens": 8,
            "structured_outputs": {"json": _JSON_SCHEMA},
        },
        "reasoning_parser_kwargs": {
            "chat_template_kwargs": {"reasoning_effort": _BAD_EFFORT},
        },
        "stream": stream,
    }
    if reasoning_ended is not None:
        payload["reasoning_ended"] = reasoning_ended
    return payload


def _build_app(engine: _RecordingEngine) -> FastAPI:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=[BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)],
    )
    serving = ServingTokens(
        engine,
        models,
        online_renderer=_OnlineRenderer(engine),
        online_derenderer=None,  # type: ignore[arg-type]
        request_logger=None,
    )

    async def _preprocess(*args: Any, **kwargs: Any):
        del args, kwargs
        return [{"prompt_token_ids": [1, 2, 3]}]

    serving.online_renderer.preprocess_completion = (  # type: ignore[method-assign]
        _preprocess
    )
    app = FastAPI()
    app.state.serving_tokens = serving
    app.state.args = type("Args", (), {"tokens_only": False})()
    attach_router(app)
    return app


class _OnlineRenderer:
    """Stand-in for the renderer fields ServingTokens reads before generate().

    ``preprocess_completion`` is replaced per test. It does not receive
    ``reasoning_parser_kwargs``; ``ServingTokens.serve_tokens`` reads those
    only when it calls ``engine_client.generate``.
    """

    def __init__(self, engine: _RecordingEngine) -> None:
        self.renderer = engine.renderer
        self.parser = None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reasoning_ended", [None, False])
def test_generate_rejects_unsupported_reasoning_effort_before_engine(
    stream: bool, reasoning_ended: bool | None
) -> None:
    """POST /inference/v1/generate returns 400 and does not call generate()."""
    engine = _RecordingEngine()
    app = _build_app(engine)
    payload = _attack_payload(stream=stream, reasoning_ended=reasoning_ended)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/inference/v1/generate", json=payload)

    assert response.status_code == 400, (
        response.status_code,
        response.text,
        engine.calls,
    )
    assert _BAD_EFFORT in response.text
    assert engine.calls == []


def _parsed_attack_request(reasoning_ended: bool | None) -> GenerateRequest:
    return GenerateRequest.model_validate(
        _attack_payload(stream=False, reasoning_ended=reasoning_ended)
    )


def _wait_for_grammar(request: Request) -> None:
    structured = request.structured_output_request
    assert structured is not None
    for _ in range(500):
        if structured._check_grammar_completion():
            grammar = structured.grammar
            assert grammar is not None and not isinstance(grammar, Exception), grammar
            return
        time.sleep(0.01)
    pytest.fail("grammar compilation timed out")


class _CpuExecutor:
    def execute_model(self, scheduler_output: Any, non_block: bool = True) -> Future:
        del non_block
        req_ids = list(scheduler_output.num_scheduled_tokens)
        future: Future = Future()
        future.set_result(
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
                sampled_token_ids=[[] for _ in req_ids],
                prompt_logprobs_dict={req_id: None for req_id in req_ids},
            )
        )
        return future


def _engine_core(scheduler: Any) -> EngineCore:
    core = EngineCore.__new__(EngineCore)
    core.scheduler = scheduler
    core.model_executor = _CpuExecutor()
    core.vllm_config = scheduler.vllm_config
    core.log_stats = False
    core._weight_version = "default"
    core.aborts_queue = queue.Queue()
    core.check_for_draft_tokens = False
    core.async_scheduling = False
    core.output_queue = queue.Queue()
    # Product default. The busy-loop wrapper re-raises when this is false,
    # and the process entry then sends ENGINE_CORE_DEAD.
    assert ParallelConfig().enable_fault_tolerance is False
    core.enable_fault_tolerance = False
    core.step_fn = core.step
    return core


def _prepare_scheduler(*, parallel: bool):
    max_num_seqs = 129 if parallel else 16
    scheduler = create_scheduler(
        model="gpt2",
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=256,
        max_model_len=64,
        enable_chunked_prefill=False,
    )
    config = scheduler.vllm_config.structured_outputs_config
    config.reasoning_parser = "k2_horizon"
    config.backend = "xgrammar"
    old = scheduler.structured_output_manager
    scheduler.structured_output_manager = StructuredOutputManager(scheduler.vllm_config)
    old.clear_backend()
    if parallel:
        # Force the batched bitmask branch. It calls the same constructor.
        scheduler.structured_output_manager.fill_bitmask_parallel_threshold = 0
    return scheduler


def _request_from_wire(
    scheduler: Any, reasoning_ended: bool | None, request_id: str
) -> Request:
    parsed = _parsed_attack_request(reasoning_ended)
    assert parsed.reasoning_parser_kwargs is not None
    sampling_params = parsed.sampling_params
    manager = scheduler.structured_output_manager
    sampling_params.verify(
        scheduler.vllm_config.model_config,
        scheduler.vllm_config.speculative_config,
        scheduler.vllm_config.structured_outputs_config,
        manager.tokenizer,
    )
    structured = sampling_params.structured_outputs
    assert structured is not None and structured._backend
    wire = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(parsed.token_ids),
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        reasoning_ended=parsed.reasoning_ended,
        reasoning_parser_kwargs=parsed.reasoning_parser_kwargs.model_dump(),
    )
    decoded = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(wire))
    kwargs = decoded.reasoning_parser_kwargs
    assert kwargs is not None
    assert kwargs["chat_template_kwargs"]["reasoning_effort"] == _BAD_EFFORT
    assert decoded.sampling_params.structured_outputs._backend
    request = Request.from_engine_core_request(decoded, None)
    manager.grammar_init(request)
    _wait_for_grammar(request)
    return request


def _bystander(scheduler: Any) -> Request:
    parsed = GenerateRequest.model_validate(
        {
            "token_ids": [4, 5, 6],
            "sampling_params": {
                "max_tokens": 8,
                "structured_outputs": {"json": _JSON_SCHEMA},
            },
            "reasoning_ended": True,
        }
    )
    sampling_params = parsed.sampling_params
    manager = scheduler.structured_output_manager
    sampling_params.verify(
        scheduler.vllm_config.model_config,
        scheduler.vllm_config.speculative_config,
        scheduler.vllm_config.structured_outputs_config,
        manager.tokenizer,
    )
    wire = EngineCoreRequest(
        request_id="survivor",
        prompt_token_ids=list(parsed.token_ids),
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        reasoning_ended=True,
        reasoning_parser_kwargs=None,
    )
    decoded = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(wire))
    request = Request.from_engine_core_request(decoded, None)
    manager.grammar_init(request)
    _wait_for_grammar(request)
    return request


def _error_outputs(core: EngineCore, request_id: str) -> list[Any]:
    found = []
    while not core.output_queue.empty():
        _client_index, outputs = core.output_queue.get_nowait()
        for output in outputs.outputs:
            if output.request_id == request_id:
                found.append(output)
    return found


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("reasoning_ended", [None, False])
def test_engine_step_errors_only_the_bad_request(
    parallel: bool, reasoning_ended: bool | None
) -> None:
    """A bad effort finishes that request; the busy loop keeps stepping."""
    scheduler = _prepare_scheduler(parallel=parallel)
    try:
        victim = _request_from_wire(scheduler, reasoning_ended, "victim")
        survivor = _bystander(scheduler)
        scheduler.add_request(victim)
        scheduler.add_request(survivor)
        core = _engine_core(scheduler)

        @fault_tolerant_wrapper
        def _drive(engine: EngineCore) -> None:
            for _ in range(4):
                if victim.status == RequestStatus.FINISHED_ERROR:
                    break
                EngineCoreProc._process_engine_step(engine)
            else:
                raise AssertionError(
                    "unsupported reasoning_effort did not finish the request"
                )
            EngineCoreProc._process_engine_step(engine)
            raise SystemExit(0)

        with pytest.raises(SystemExit):
            _drive(core)

        assert victim.status == RequestStatus.FINISHED_ERROR
        errors = _error_outputs(core, victim.request_id)
        assert errors
        assert any(output.finish_reason == FinishReason.ERROR for output in errors)
        assert any(
            output.error is not None and _BAD_EFFORT in output.error.message
            for output in errors
        )
        assert survivor.status != RequestStatus.FINISHED_ERROR
        assert survivor.request_id in scheduler.requests
    finally:
        scheduler.structured_output_manager.clear_backend()
