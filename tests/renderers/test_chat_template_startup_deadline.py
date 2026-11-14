# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup renders of a model chat template must finish or abort.

The template is loaded from the model directory the server is asked to
serve (``chat_template.jinja``, or the legacy ``chat_template`` field in
``tokenizer_config.json``). ``trust_remote_code`` stays false.

These tests fail while that render has no deadline: a loop the Jinja
sandbox actually runs never returns, so API startup never reaches
``serve_http``. A single ``range`` bigger than the sandbox cap is not the
case under test; Jinja raises ``OverflowError`` for that, and warmup
already swallows exceptions. The loop here stays inside ``MAX_RANGE`` on
each ``range`` and still does not finish.

A fix should make every startup render honor
``VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT`` and raise (not hang, and not return)
when the template exceeds it. Fast templates must still render.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.decisions.strategies import (
    NextTokenStrategy,
    ReadContext,
)
from vllm.renderers.hf import HfRenderer
from vllm.renderers.kimi_k3 import KimiK3Renderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers.registry import get_tokenizer

# One second is long enough for a normal template and far shorter than the
# nested loop below (about 1e10 iterations).
_RENDER_TIMEOUT_S = "1"
# Parent wait after the child reports that rendering has started.
_DEADLINE_WAIT_S = 4.0
_READY_WAIT_S = 90.0

# Each range stays within jinja2.sandbox.MAX_RANGE (100000). Nested, the
# loop does not raise and does not finish within _DEADLINE_WAIT_S.
_NONTERMINATING = (
    "{% for i in range(100000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}"
)
_BENIGN = "ok"
# Runs only for the two-system probe in AnthropicServingMessages.
_SYSTEM_LOOP = (
    "{% set ns = namespace(systems=0) %}"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}"
    "{% set ns.systems = ns.systems + 1 %}{% endif %}{% endfor %}"
    "{% if ns.systems > 1 %}" + _NONTERMINATING + "{% endif %}ok"
)
# Runs only for the decisions strategy's generation-prompt probe.
_GENERATION_LOOP = "{% if add_generation_prompt %}" + _NONTERMINATING + "{% endif %}ok"

_SOURCES = ("chat_template.jinja", "tokenizer_config.json")


@dataclass
class _HFConfig:
    model_type: str = "gpt2"
    use_unified_vision_chunk: bool = False


@dataclass
class _ModelConfig:
    runner_type: str = "generate"
    model: str = ""
    tokenizer: str = ""
    trust_remote_code: bool = False
    tokenizer_revision: str | None = None
    revision: str | None = None
    code_revision: str | None = None
    tokenizer_mode: str = "auto"
    hf_config: _HFConfig = field(default_factory=_HFConfig)
    encoder_config: dict | None = None
    enable_prompt_embeds: bool = False
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    multimodal_config: object | None = None
    renderer_num_workers: int = 1
    max_model_len: int = 2048
    architecture: str = "GPT2LMHeadModel"
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None

    def get_multimodal_config(self):
        return self.multimodal_config


@dataclass
class _ParallelConfig:
    _api_process_rank: int = 0
    _api_process_count: int = 1


@dataclass
class _VllmConfig:
    model_config: _ModelConfig
    parallel_config: _ParallelConfig


@dataclass
class _Engine:
    model_config: _ModelConfig


def _write_model(directory: Path, template: str, source: str) -> None:
    """Write a local tokenizer whose chat template is ``template``.

    ``source`` is the file a model repo would ship. The other location is
    removed so the load cannot pick the template up from somewhere else.
    """
    raw = Tokenizer(WordLevel({"[UNK]": 0, "ok": 1}, unk_token="[UNK]"))
    fast = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    fast.chat_template = "placeholder"
    directory.mkdir(parents=True)
    fast.save_pretrained(directory)

    jinja_path = directory / "chat_template.jinja"
    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("chat_template", None)
    if source == "chat_template.jinja":
        jinja_path.write_text(template, encoding="utf-8")
    elif source == "tokenizer_config.json":
        jinja_path.unlink()
        config["chat_template"] = template
    else:
        raise AssertionError(source)
    config_path.write_text(json.dumps(config), encoding="utf-8")


def _loaded_template(directory: str) -> str:
    tokenizer = get_tokenizer(directory, trust_remote_code=False)
    template = tokenizer.chat_template
    if isinstance(template, dict):
        template = template.get("default")
    assert isinstance(template, str)
    return template


def _online(directory: str, renderer_cls: type) -> OnlineRenderer:
    model_config = _ModelConfig(model=directory, tokenizer=directory)
    tokenizer = get_tokenizer(directory, trust_remote_code=False)
    renderer = renderer_cls(
        _VllmConfig(model_config, _ParallelConfig()),
        tokenizer,
    )
    return OnlineRenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


def _exercise(directory: str, operation: str, result_q: mp.Queue) -> None:
    os.environ["VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT"] = _RENDER_TIMEOUT_S
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        online = _online(
            directory,
            KimiK3Renderer if operation == "kimi_warmup" else HfRenderer,
        )
    except Exception as exc:
        result_q.put(("setup_failed", type(exc).__name__, str(exc)))
        return

    result_q.put(("ready",))
    started = time.perf_counter()
    try:
        if operation in ("warmup", "kimi_warmup"):
            online.warmup()
        elif operation == "messages_probe":
            AnthropicServingMessages._should_merge_inline_system(online)
        elif operation == "decisions_probe":
            NextTokenStrategy(
                ReadContext(
                    engine_client=_Engine(online.model_config),
                    online_renderer=online,
                    chat_template=None,
                    chat_template_content_format="auto",
                    default_chat_template_kwargs={},
                )
            )
        else:
            raise AssertionError(operation)
    except Exception as exc:
        result_q.put(
            (
                "raised",
                type(exc).__name__,
                str(exc),
                time.perf_counter() - started,
            )
        )
        return
    result_q.put(("returned", time.perf_counter() - started))


def _run(directory: Path, operation: str) -> tuple:
    ctx = mp.get_context("spawn")
    result_q: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_exercise,
        args=(str(directory), operation, result_q),
    )
    process.start()
    try:
        try:
            ready = result_q.get(timeout=_READY_WAIT_S)
        except queue.Empty:
            return ("not_ready", process.exitcode)
        if ready[0] != "ready":
            return ready
        try:
            return result_q.get(timeout=_DEADLINE_WAIT_S)
        except queue.Empty:
            return ("hung",)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)


def _assert_aborted(outcome: tuple) -> None:
    assert outcome[0] == "raised", outcome
    detail = f"{outcome[1]} {outcome[2]}".lower()
    assert outcome[3] < _DEADLINE_WAIT_S
    assert any(
        token in detail for token in ("timeout", "timed out", "deadline", "fuel")
    ), outcome


def _assert_finished(outcome: tuple) -> None:
    assert outcome[0] == "returned", outcome
    assert outcome[1] < _DEADLINE_WAIT_S


@pytest.fixture
def model_dir(tmp_path: Path):
    def write(template: str, source: str) -> Path:
        directory = tmp_path / source.replace(".", "_")
        _write_model(directory, template, source)
        loaded = _loaded_template(str(directory))
        assert loaded == template
        return directory

    return write


@pytest.mark.parametrize("source", _SOURCES)
def test_benign_model_template_renders_at_startup(model_dir, source):
    directory = model_dir(_BENIGN, source)
    _assert_finished(_run(directory, "warmup"))
    _assert_finished(_run(directory, "messages_probe"))


@pytest.mark.parametrize("source", _SOURCES)
def test_nonterminating_template_aborts_startup_warmup(model_dir, source):
    directory = model_dir(_NONTERMINATING, source)
    _assert_aborted(_run(directory, "warmup"))


@pytest.mark.parametrize("source", _SOURCES)
def test_multi_system_loop_aborts_messages_probe(model_dir, source):
    """Warmup's single user message finishes; the messages probe does not."""
    directory = model_dir(_SYSTEM_LOOP, source)
    _assert_finished(_run(directory, "warmup"))
    _assert_aborted(_run(directory, "messages_probe"))


def test_generation_prompt_loop_aborts_decisions_strategy(model_dir):
    directory = model_dir(_GENERATION_LOOP, "chat_template.jinja")
    _assert_finished(_run(directory, "warmup"))
    _assert_aborted(_run(directory, "decisions_probe"))


def test_nonterminating_template_aborts_kimi_k3_warmup(model_dir):
    directory = model_dir(_NONTERMINATING, "tokenizer_config.json")
    _assert_aborted(_run(directory, "kimi_warmup"))
