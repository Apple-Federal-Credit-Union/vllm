# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request chat templates must stop running when rendering returns.

With ``--trust-request-chat-template``, a client template is rendered by
``tokenizer.apply_chat_template`` on the API render pool. These tests drive
that path and require the template's own work to have stopped by the time
the render call returns. A timeout that only abandons the pool thread leaves
the template running, and these tests fail until that work is actually
stopped.

Not covered here: synchronous ``HfRenderer.render_messages`` (pooling) and
the Mistral, Cohere, Deepseek, Kimi, and Inkling renderers. They do not use
this timeout, so they are a different unbounded-render path.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pybase64
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import TokenizersBackend

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer

_RENDER_TIMEOUT_S = "0.35"
_CALL_DEADLINE_S = 3.0
_SETTLE_S = 0.25
_SERVER_TEMPLATE = "SERVER {{ messages[0].content }}"
_BENIGN_TEMPLATE = "REQUEST {{ messages[0].content }}"
# Nested loops stay inside Jinja's sandboxed range limit, and ``tick`` lets
# the test observe this exact template. ``break`` ends the loop only after
# the test asks it to, so a passing run does not leave a worker behind.
_LOOP_TEMPLATE = (
    "{% for i in range(100000) %}"
    "{% for j in range(100000) %}"
    "{% if tick() %}{% break %}{% endif %}"
    "{% endfor %}"
    "{% if tick() %}{% break %}{% endif %}"
    "{% endfor %}"
)


@dataclass
class _HFConfig:
    model_type: str = "gpt2"


@dataclass
class _ModelConfig:
    runner_type: str = "generate"
    model: str = "local-test"
    tokenizer: str = "local-test"
    trust_remote_code: bool = False
    tokenizer_revision: str | None = None
    revision: str | None = None
    code_revision: str | None = None
    tokenizer_mode: str = "auto"
    hf_config: _HFConfig = field(default_factory=_HFConfig)
    hf_text_config: _HFConfig = field(default_factory=_HFConfig)
    encoder_config: dict[str, Any] | None = None
    enable_prompt_embeds: bool = False
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    hidden_size: int = 4
    dtype: torch.dtype = torch.float32
    multimodal_config: Any = None
    allowed_local_media_path: str | None = None
    allowed_media_domains: list[str] | None = None
    max_model_len: int = 2048

    def get_hidden_size(self) -> int:
        return self.hidden_size


@dataclass
class _ParallelConfig:
    _api_process_rank: int = 0


@dataclass
class _VllmConfig:
    model_config: _ModelConfig
    parallel_config: _ParallelConfig


def _make_tokenizer() -> TokenizersBackend:
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "user": 1,
                "hi": 2,
                "REQUEST": 3,
                "SERVER": 4,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        name_or_path="local-test",
    )


def _make_renderer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trust: bool,
    server_template: str | None,
    prompt_embeds: bool = False,
) -> tuple[OnlineRenderer, HfRenderer, TokenizersBackend]:
    monkeypatch.setenv("VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT", _RENDER_TIMEOUT_S)
    model_config = _ModelConfig(enable_prompt_embeds=prompt_embeds)
    tokenizer = _make_tokenizer()
    if server_template is not None:
        tokenizer.chat_template = server_template
    renderer = HfRenderer(
        _VllmConfig(model_config, _ParallelConfig()),
        tokenizer=tokenizer,
    )
    online = OnlineRenderer(
        model_config=model_config,  # type: ignore[arg-type]
        renderer=renderer,
        request_logger=None,
        chat_template=server_template,
        chat_template_content_format="string",
        trust_request_chat_template=trust,
    )
    return online, renderer, tokenizer


def _make_tick() -> tuple[Any, dict[str, Any]]:
    state: dict[str, Any] = {"n": 0, "stop": threading.Event()}

    def tick() -> bool:
        state["n"] += 1
        return state["stop"].is_set()

    return tick, state


def _release(renderer: HfRenderer, *states: dict[str, Any]) -> None:
    for state in states:
        state["stop"].set()
    renderer.shutdown()


def _run(coro: Any) -> Any:
    async def _drive() -> Any:
        try:
            return await coro
        except Exception as exc:
            return exc

    try:
        return asyncio.run(asyncio.wait_for(_drive(), _CALL_DEADLINE_S))
    except TimeoutError as exc:
        return exc


def _assert_template_stopped(state: dict[str, Any]) -> None:
    """The request template ran, and it is not still running."""
    assert state["n"] > 0
    seen = state["n"]
    time.sleep(_SETTLE_S)
    assert state["n"] == seen


def _prompt_embeds_data() -> str:
    buffer = io.BytesIO()
    torch.save(torch.ones(2, 4), buffer)
    return pybase64.b64encode(buffer.getvalue()).decode()


def test_trusted_request_template_is_what_gets_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The template that was allowed is the template that is rendered."""
    online, renderer, tokenizer = _make_renderer(
        monkeypatch,
        trust=True,
        server_template=_SERVER_TEMPLATE,
    )
    request = ChatCompletionRequest(
        model="local-test",
        messages=[{"role": "user", "content": "hi"}],
        chat_template=_BENIGN_TEMPLATE,
    )
    try:
        refusal = online.validate_chat_template(
            request.chat_template,
            request.chat_template_kwargs,
            online.trust_request_chat_template,
        )
        assert refusal is None
        params = request.build_chat_params(
            online.chat_template,
            online.chat_template_content_format,
        )
        assert params.chat_template == request.chat_template
        _, prompt = asyncio.run(
            renderer.render_messages_async(request.messages, params)
        )
        decoded = tokenizer.decode(prompt["prompt_token_ids"])
        assert "REQUEST" in decoded
        assert "hi" in decoded
        assert "SERVER" not in decoded
    finally:
        renderer.shutdown()


@pytest.mark.parametrize("field", ["chat_template", "chat_template_kwargs"])
def test_untrusted_request_template_does_not_run(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Refusing an untrusted template must not start rendering it."""
    online, renderer, _ = _make_renderer(
        monkeypatch,
        trust=False,
        server_template=_SERVER_TEMPLATE,
    )
    tick, state = _make_tick()
    kwargs: dict[str, Any] = {"tick": tick}
    template: str | None = _LOOP_TEMPLATE
    if field == "chat_template_kwargs":
        kwargs["chat_template"] = _LOOP_TEMPLATE
        template = None
    request = ChatCompletionRequest(
        model="local-test",
        messages=[{"role": "user", "content": "hi"}],
        chat_template=template,
        chat_template_kwargs=kwargs,
    )
    try:
        result = asyncio.run(online.render_chat(request))
        assert isinstance(result, ErrorResponse)
        assert "trust-request-chat-template" in result.error.message
        time.sleep(_SETTLE_S)
        assert state["n"] == 0
    finally:
        _release(renderer, state)


def test_chat_completion_template_stops_when_render_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A looping chat-completion template must not keep running after return."""
    online, renderer, _ = _make_renderer(
        monkeypatch,
        trust=True,
        server_template=_SERVER_TEMPLATE,
    )
    tick, state = _make_tick()
    request = ChatCompletionRequest(
        model="local-test",
        messages=[{"role": "user", "content": "hi"}],
        chat_template=_LOOP_TEMPLATE,
        chat_template_kwargs={"tick": tick},
    )
    try:
        outcome = _run(online.render_chat(request))
        assert isinstance(outcome, Exception)
        _assert_template_stopped(state)
    finally:
        _release(renderer, state)


def test_chat_template_kwargs_stop_when_render_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A template carried only in chat_template_kwargs must stop too.

    The server has no default template, which is when that kwarg is used.
    """
    online, renderer, _ = _make_renderer(
        monkeypatch,
        trust=True,
        server_template=None,
    )
    tick, state = _make_tick()
    request = ChatCompletionRequest(
        model="local-test",
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"chat_template": _LOOP_TEMPLATE, "tick": tick},
    )
    try:
        refusal = online.validate_chat_template(
            request.chat_template,
            request.chat_template_kwargs,
            online.trust_request_chat_template,
        )
        assert refusal is None
        params = request.build_chat_params(None, "string")
        assert params.chat_template is None
        outcome = _run(renderer.render_messages_async(request.messages, params))
        assert isinstance(outcome, Exception)
        _assert_template_stopped(state)
    finally:
        _release(renderer, state)


def test_tokenize_chat_template_stops_when_render_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/tokenize`` chat requests use the same render and must stop too."""
    online, renderer, _ = _make_renderer(
        monkeypatch,
        trust=True,
        server_template=_SERVER_TEMPLATE,
    )
    tick, state = _make_tick()
    request = TokenizeChatRequest(
        model="local-test",
        messages=[{"role": "user", "content": "hi"}],
        chat_template=_LOOP_TEMPLATE,
        chat_template_kwargs={"tick": tick},
    )
    try:
        refusal = online.validate_chat_template(
            request.chat_template,
            request.chat_template_kwargs,
            online.trust_request_chat_template,
        )
        assert refusal is None
        params = request.build_chat_params(
            online.chat_template,
            online.chat_template_content_format,
        )
        assert params.chat_template == request.chat_template
        outcome = _run(renderer.render_messages_async(request.messages, params))
        assert isinstance(outcome, Exception)
        _assert_template_stopped(state)
    finally:
        _release(renderer, state)


def test_responses_template_kwarg_stops_when_render_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Responses uses a kwargs template when the server has no template."""
    online, renderer, _ = _make_renderer(
        monkeypatch,
        trust=True,
        server_template=None,
    )
    tick, state = _make_tick()
    request = ResponsesRequest(
        input="hi",
        chat_template_kwargs={"chat_template": _LOOP_TEMPLATE, "tick": tick},
    )
    try:
        outcome = _run(online.render_responses(request))
        assert isinstance(outcome, Exception)
        _assert_template_stopped(state)
    finally:
        _release(renderer, state)


def test_prompt_embeds_template_stops_when_render_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prompt-embeds visibility render must stop the same way."""
    online, renderer, _ = _make_renderer(
        monkeypatch,
        trust=True,
        server_template=_SERVER_TEMPLATE,
        prompt_embeds=True,
    )
    tick, state = _make_tick()
    request = ChatCompletionRequest(
        model="local-test",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": _prompt_embeds_data()},
                    {"type": "text", "text": "hi"},
                ],
            }
        ],
        chat_template=_LOOP_TEMPLATE,
        chat_template_kwargs={"tick": tick},
    )
    try:
        outcome = _run(online.render_chat(request))
        assert isinstance(outcome, Exception)
        _assert_template_stopped(state)
    finally:
        _release(renderer, state)


def test_repeated_timeouts_do_not_leave_workers_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each timed-out request must leave no template still running."""
    online, renderer, _ = _make_renderer(
        monkeypatch,
        trust=True,
        server_template=_SERVER_TEMPLATE,
    )
    tick_a, state_a = _make_tick()
    tick_b, state_b = _make_tick()
    messages = [{"role": "user", "content": "hi"}]

    def _request(tick: Any) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="local-test",
            messages=messages,
            chat_template=_LOOP_TEMPLATE,
            chat_template_kwargs={"tick": tick},
        )

    try:
        first = _run(online.render_chat(_request(tick_a)))
        second = _run(online.render_chat(_request(tick_b)))
        assert isinstance(first, Exception)
        assert isinstance(second, Exception)
        _assert_template_stopped(state_a)
        _assert_template_stopped(state_b)
    finally:
        _release(renderer, state_a, state_b)
