# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request chat templates on POST /v1/systemone.

The chat completion path refuses a request chat template unless the server
was started with --trust-request-chat-template. Structured decisions must
do the same before any read strategy renders. The template that is rendered
is the one the check observed: a later read of chat_template_kwargs must
not supply a different template.

Qwen3, Qwen3.5, Gemma 4, and Liquid strategies render through
NextTokenStrategy._render. DiffusionGemma also renders there, including
read_joint. D3Strategy and JointSchemaStrategy accept the kwarg and do not
render a chat template, so they are not covered here.
"""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.structured_decisions.api_router import (
    register_structured_decisions_api_router,
)
from vllm.entrypoints.generate.structured_decisions.protocol import (
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)
from vllm.entrypoints.generate.structured_decisions.serving import (
    ServingStructuredDecisions,
)
from vllm.entrypoints.openai.decisions.strategies import (
    DiffusionGemmaCanvasStrategy,
    LiquidStrategy,
    NextTokenStrategy,
    ReadContext,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers import get_tokenizer

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

MARKER = "REQUEST_TEMPLATE_MARKER"
QWEN = "Qwen/Qwen3-0.6B"
DGEMMA = "google/diffusiongemma-26B-A4B-it"
# Bounded so a render that ignores the deadline still ends. Each range stays
# inside the Jinja sandbox limit; together they outlast the render deadline.
LOOPING_TEMPLATE = (
    "{% for i in range(2000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}"
)
REFUSAL = "untrusted chat template"


def marker_template(tokenizer) -> str:
    return "{{ '" + MARKER + "' }}" + tokenizer.get_chat_template()


def choice_request(**kwargs) -> StructuredDecisionRequest:
    body = {
        "state": "billing",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "pay", "shipping": "ship"},
            }
        },
    }
    body.update(kwargs)
    return StructuredDecisionRequest(**body)


def joint_request(**kwargs) -> StructuredDecisionRequest:
    body = {
        "state": "Charged twice, please refund.",
        "questions": {
            "refund": {"type": "noul", "criteria": {"true": "a refund is due"}},
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "pay", "ship": None},
            },
        },
        "joint": True,
    }
    body.update(kwargs)
    return StructuredDecisionRequest(**body)


class RenderView(dict):
    """The mapping the renderer merges, and a different value for .get().

    merge_kwargs reads items(). validate_chat_template reads .get(). A check
    that only calls .get() and then renders this mapping uses a template the
    check did not see.
    """

    def __init__(self, template: str):
        super().__init__()
        self._template = template

    def get(self, key, default=None):
        if key == "chat_template":
            return default
        return super().get(key, default)

    def items(self):
        return iter((("chat_template", self._template),))

    def keys(self):
        return iter(("chat_template",))

    def __iter__(self):
        return iter(("chat_template",))

    def __getitem__(self, key):
        if key == "chat_template":
            return self._template
        return super().__getitem__(key)

    def __contains__(self, key):
        return key == "chat_template" or super().__contains__(key)

    def __len__(self):
        return 1


def jinja_render_threads() -> list[str]:
    frames = sys._current_frames()
    names = []
    for thread in threading.enumerate():
        frame = frames.get(thread.ident)
        while frame is not None:
            if frame.f_code.co_name == "render_jinja_template":
                names.append(thread.name)
                break
            frame = frame.f_back
    return names


async def _wait_for_render_threads():
    deadline = time.monotonic() + 20
    while jinja_render_threads() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)


def _renderer(model_config: ModelConfig, tokenizer) -> HfRenderer:
    return HfRenderer(
        SimpleNamespace(
            model_config=model_config,
            parallel_config=SimpleNamespace(_api_process_rank=0),
        ),
        tokenizer,
    )


def _serving(engine, strategy) -> ServingStructuredDecisions:
    name = engine.model_config.model
    models = OpenAIServingModels(engine, [BaseModelPath(name=name, model_path=name)])
    return ServingStructuredDecisions(models, strategy)


def _engine(model_config, renderer, tokenizer, vllm_config=None):
    recorded: list[str] = []

    async def generate(prompt, params, request_id, **kwargs):
        recorded.append(tokenizer.decode(prompt["prompt_token_ids"]))
        ids = list(params.logprob_token_ids or [0])
        width = params.max_tokens or 1
        logprobs = {token_id: Logprob(-1.0) for token_id in ids}
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=prompt["prompt_token_ids"],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="A",
                    token_ids=[ids[0]] * width,
                    cumulative_logprob=None,
                    logprobs=[logprobs] * width,
                    finish_reason="length",
                )
            ],
            finished=True,
        )

    engine = MagicMock(spec=EngineClient)
    engine.errored = False
    engine.model_config = model_config
    engine.renderer = renderer
    engine.input_processor = MagicMock()
    engine.vllm_config = vllm_config
    engine.generate.side_effect = generate
    engine.recorded = recorded
    return engine


@pytest.fixture(scope="module")
def qwen_stack():
    tokenizer = get_tokenizer(QWEN)
    model_config = ModelConfig(model=QWEN, dtype="float32", enforce_eager=True)
    renderer = _renderer(model_config, tokenizer)
    engine = _engine(model_config, renderer, tokenizer)
    online = OnlineRenderer(
        model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    assert online.trust_request_chat_template is False
    context = ReadContext(engine, online, None, "auto", {})
    strategy = NextTokenStrategy(context)
    yield _serving(engine, strategy), engine, tokenizer, strategy
    renderer.shutdown()


@pytest.fixture(scope="module")
def diffusion_stack():
    tokenizer = get_tokenizer(DGEMMA)
    model_config = ModelConfig(model=DGEMMA, dtype="float32", enforce_eager=True)
    # Device detection needs the compiled extension. Name CPU so the real
    # config, including the diffusion canvas, can be built here.
    vllm_config = VllmConfig(
        model_config=model_config, device_config=DeviceConfig(device="cpu")
    )
    renderer = HfRenderer(vllm_config, tokenizer)
    engine = _engine(model_config, renderer, tokenizer, vllm_config)
    online = OnlineRenderer(
        model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    assert online.trust_request_chat_template is False
    context = ReadContext(engine, online, None, "auto", {})
    strategy = DiffusionGemmaCanvasStrategy(context)
    assert strategy.joint_reads
    yield _serving(engine, strategy), engine, tokenizer, strategy
    renderer.shutdown()


def _assert_refused(result, engine) -> None:
    assert isinstance(result, ErrorResponse)
    assert result.error.code == 400
    assert REFUSAL in result.error.message.lower()
    assert engine.generate.call_count == 0
    assert not any(MARKER in text for text in engine.recorded)
    assert jinja_render_threads() == []


def _clear(engine) -> None:
    engine.generate.reset_mock()
    engine.recorded.clear()


@pytest.mark.asyncio
async def test_systemone_renders_without_a_request_template(qwen_stack):
    serving, engine, _, _ = qwen_stack
    _clear(engine)
    result = await serving.create_decision(choice_request())
    assert isinstance(result, StructuredDecisionResponse)
    assert engine.generate.call_count == 1
    assert MARKER not in engine.recorded[0]


@pytest.mark.asyncio
async def test_systemone_allows_template_kwargs_that_are_not_a_template(
    qwen_stack,
):
    serving, engine, _, _ = qwen_stack
    _clear(engine)
    result = await serving.create_decision(
        choice_request(chat_template_kwargs={"enable_thinking": False})
    )
    assert isinstance(result, StructuredDecisionResponse)
    assert engine.generate.call_count == 1


@pytest.mark.asyncio
async def test_trusted_request_template_is_rendered(qwen_stack):
    serving, engine, tokenizer, strategy = qwen_stack
    renderer = strategy.context.online_renderer
    _clear(engine)
    renderer.trust_request_chat_template = True
    try:
        result = await serving.create_decision(
            choice_request(
                chat_template_kwargs={"chat_template": marker_template(tokenizer)}
            )
        )
    finally:
        renderer.trust_request_chat_template = False
    assert isinstance(result, StructuredDecisionResponse)
    assert engine.recorded and engine.recorded[0].startswith(MARKER)


@pytest.mark.asyncio
async def test_untrusted_request_template_is_refused_before_render(qwen_stack):
    serving, engine, tokenizer, _ = qwen_stack
    _clear(engine)
    result = await serving.create_decision(
        choice_request(
            chat_template_kwargs={"chat_template": marker_template(tokenizer)}
        )
    )
    _assert_refused(result, engine)


@pytest.mark.asyncio
async def test_untrusted_template_is_refused_when_server_has_a_template(
    qwen_stack,
):
    serving, engine, tokenizer, strategy = qwen_stack
    server_template = tokenizer.get_chat_template()
    online = strategy.context.online_renderer
    previous = strategy.context.chat_template
    _clear(engine)
    strategy.context.chat_template = server_template
    online.chat_template = server_template
    try:
        result = await serving.create_decision(
            choice_request(
                chat_template_kwargs={"chat_template": marker_template(tokenizer)}
            )
        )
    finally:
        strategy.context.chat_template = previous
        online.chat_template = previous
    _assert_refused(result, engine)


@pytest.mark.asyncio
async def test_render_uses_the_template_the_check_observed(qwen_stack):
    """items() is the view merge_kwargs renders. .get() must not be a
    separate lookup that hides the template the renderer will compile."""
    serving, engine, tokenizer, _ = qwen_stack
    _clear(engine)
    request = choice_request()
    request.chat_template_kwargs = RenderView(marker_template(tokenizer))
    result = await serving.create_decision(request)
    _assert_refused(result, engine)


@pytest.mark.asyncio
async def test_liquid_strategy_refuses_an_untrusted_request_template(qwen_stack):
    _, engine, tokenizer, strategy = qwen_stack
    liquid = LiquidStrategy(strategy.context)
    serving = _serving(engine, liquid)
    _clear(engine)
    result = await serving.create_decision(
        choice_request(
            chat_template_kwargs={"chat_template": marker_template(tokenizer)}
        )
    )
    _assert_refused(result, engine)


@pytest.mark.asyncio
async def test_diffusion_read_refuses_an_untrusted_request_template(
    diffusion_stack,
):
    serving, engine, tokenizer, _ = diffusion_stack
    _clear(engine)
    result = await serving.create_decision(
        choice_request(
            chat_template_kwargs={"chat_template": marker_template(tokenizer)}
        )
    )
    _assert_refused(result, engine)


@pytest.mark.asyncio
async def test_diffusion_joint_read_refuses_an_untrusted_request_template(
    diffusion_stack,
):
    serving, engine, tokenizer, _ = diffusion_stack
    _clear(engine)
    result = await serving.create_decision(
        joint_request(
            chat_template_kwargs={"chat_template": marker_template(tokenizer)}
        )
    )
    _assert_refused(result, engine)


@pytest.mark.asyncio
async def test_diffusion_joint_render_view_is_the_checked_template(
    diffusion_stack,
):
    serving, engine, tokenizer, _ = diffusion_stack
    _clear(engine)
    request = joint_request()
    request.chat_template_kwargs = RenderView(marker_template(tokenizer))
    result = await serving.create_decision(request)
    _assert_refused(result, engine)


@pytest.mark.asyncio
async def test_route_refuses_untrusted_template_from_the_json_body(qwen_stack):
    serving, engine, tokenizer, _ = qwen_stack
    _clear(engine)
    app = FastAPI()
    app.state.serving_structured_decisions = serving
    app.state.args = SimpleNamespace(log_error_stack=False)
    register_structured_decisions_api_router(app)
    payload = {
        "state": "billing",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "pay", "shipping": "ship"},
            }
        },
        "chat_template_kwargs": {"chat_template": marker_template(tokenizer)},
    }
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 400
    assert REFUSAL in response.json()["error"]["message"].lower()
    assert engine.generate.call_count == 0
    assert not any(MARKER in text for text in engine.recorded)


@pytest.mark.asyncio
async def test_looping_request_template_does_not_keep_a_render_worker(
    qwen_stack, monkeypatch
):
    serving, engine, _, _ = qwen_stack
    _clear(engine)
    monkeypatch.setenv("VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT", "0.5")
    request = choice_request(chat_template_kwargs={"chat_template": LOOPING_TEMPLATE})
    try:
        started = time.monotonic()
        first = await serving.create_decision(request)
        assert time.monotonic() - started < 0.25
        _assert_refused(first, engine)
        _clear(engine)
        started = time.monotonic()
        second = await serving.create_decision(request)
        assert time.monotonic() - started < 0.25
        _assert_refused(second, engine)
    finally:
        await _wait_for_render_threads()
