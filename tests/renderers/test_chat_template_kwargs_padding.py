# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client chat_template_kwargs must not control tokenizer padding.

These tests exercise the OpenAI chat, tokenize, responses, pooling, Anthropic,
and structured-decision request objects through the HF and Kimi K3 renderers.
They fail while a request can make ``apply_chat_template`` materialize a
sequence padded to a client-chosen ``max_length`` before vLLM's own length
check. They pass once that padding is no longer applied.

Not covered here, because those paths do not forward ``padding`` into the
Hugging Face fast tokenizer: the Mistral, Cohere, Inkling, and DeepSeek
renderers, and the scoring chat-template call (it forces ``tokenize=False``).
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import TokenizersBackend

from vllm.entrypoints.anthropic.protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.decisions.strategies import ReadPromptRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages
from vllm.entrypoints.pooling.classify.protocol import ClassificationChatRequest
from vllm.entrypoints.pooling.embed.protocol import EmbeddingChatRequest
from vllm.entrypoints.pooling.pooling.protocol import PoolingChatRequest
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
from vllm.exceptions import VLLMValidationError
from vllm.renderers.hf import HfRenderer
from vllm.renderers.kimi_k3 import KimiK3Renderer
from vllm.renderers.params import ChatParams

_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}assistant {% endif %}"
    "{% if enable_thinking is defined and enable_thinking %}thinking {% endif %}"
    "{% if tools %}tool {% endif %}"
)
_MESSAGES = [{"role": "user", "content": "hi"}]
# Large enough that a padded result cannot be the real prompt, small enough
# that the fast tokenizer can actually build it.
_ATTACK_LENGTH = 2048
# Above this, the probe refuses to enter the Rust padder. The product must
# not ask for a pad of this size; the test must not allocate it either.
_MAX_SAFE_PAD = 4096
_HUGE_LENGTH = 2147483647
_PADDING_KEYS = (
    "padding",
    "max_length",
    "truncation",
    "return_tensors",
    "return_dict",
    "return_assistant_tokens_mask",
    "tokenizer_kwargs",
)


class _BlockedPad(Exception):
    def __init__(self, length: int) -> None:
        super().__init__(length)
        self.length = length


@dataclass
class _MockHFConfig:
    model_type: str = "any"


@dataclass
class _MockModelConfig:
    runner_type = "generate"
    model: str = "local-pad-tokenizer"
    tokenizer: str = "local-pad-tokenizer"
    trust_remote_code: bool = False
    tokenizer_revision = None
    revision = None
    code_revision = None
    tokenizer_mode = "auto"
    hf_config = _MockHFConfig()
    hf_text_config = hf_config
    encoder_config = None
    enable_prompt_embeds: bool = False
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    hidden_size: int = 32
    max_model_len: int = 32
    multimodal_config = None

    def get_hidden_size(self) -> int:
        return self.hidden_size


@dataclass
class _MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class _MockVllmConfig:
    model_config: _MockModelConfig
    parallel_config: _MockParallelConfig


class _PadProbe:
    """Record the token list ``apply_chat_template`` actually builds.

    A later length check does not undo that list. Pads above ``_MAX_SAFE_PAD``
    are refused here so the huge-length case cannot exhaust this process; the
    refusal is still a failed test because the product asked for the pad.
    """

    def __init__(self) -> None:
        self.lengths: list[int] = []
        self.blocked: list[int] = []
        self._orig = TokenizersBackend.apply_chat_template

    def __enter__(self) -> "_PadProbe":
        probe = self

        def wrapped(tokenizer, *args, **kwargs):
            target = _pad_target(kwargs)
            if target is not None and target > _MAX_SAFE_PAD:
                probe.blocked.append(target)
                raise _BlockedPad(target)
            result = probe._orig(tokenizer, *args, **kwargs)
            probe.lengths.append(_token_len(result))
            return result

        TokenizersBackend.apply_chat_template = wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        TokenizersBackend.apply_chat_template = self._orig  # type: ignore[method-assign]


def _pad_target(kwargs: dict[str, Any]) -> int | None:
    padding = kwargs.get("padding", False)
    max_length = kwargs.get("max_length")
    if padding == "max_length" and isinstance(max_length, int):
        return max_length
    return None


def _token_len(result: Any) -> int:
    if isinstance(result, list):
        if result and isinstance(result[0], list):
            return len(result[0])
        return len(result)
    ids = result["input_ids"]
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    if ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


def _build_tokenizer() -> TokenizersBackend:
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "user": 2,
        "assistant": 3,
        "hi": 4,
        "<s>": 5,
        "</s>": 6,
        "thinking": 7,
        "tool": 8,
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = TokenizersBackend(
        tokenizer_object=raw,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.chat_template = _TEMPLATE
    return tokenizer


def _renderer(tokenizer: TokenizersBackend) -> HfRenderer:
    config = _MockVllmConfig(_MockModelConfig(), _MockParallelConfig())
    return HfRenderer(config, tokenizer=tokenizer)


def _unpadded_len(
    tokenizer: TokenizersBackend,
    messages: list[dict[str, Any]],
    params: ChatParams,
) -> int:
    kwargs = params.get_apply_chat_template_kwargs()
    for key in _PADDING_KEYS:
        kwargs.pop(key, None)
    kwargs["padding"] = False
    kwargs["truncation"] = False
    kwargs["max_length"] = None
    kwargs["return_dict"] = False
    kwargs["tokenize"] = True
    result = tokenizer.apply_chat_template(messages, **kwargs)
    assert isinstance(result, list)
    return len(result)


def _assert_unpadded(probe: _PadProbe, natural: int, token_ids: list[int] | None):
    assert probe.blocked == [], probe.blocked
    assert all(length == natural for length in probe.lengths), probe.lengths
    if token_ids is not None:
        assert probe.lengths, "renderer returned tokens without calling the tokenizer"
        assert len(token_ids) == natural


def _render_messages(renderer: HfRenderer, params: ChatParams, messages: list):
    try:
        _, prompt = renderer.render_messages(messages, params)
    except _BlockedPad:
        return None
    except (ValueError, VLLMValidationError):
        return None
    return list(prompt["prompt_token_ids"])


def _render_chat_async(
    renderer: HfRenderer,
    params: ChatParams,
    messages: list,
    tok_params,
):
    async def run():
        return await renderer.render_chat_async([messages], params, tok_params)

    try:
        _, engine_inputs = asyncio.run(run())
    except _BlockedPad:
        return None
    except (ValueError, VLLMValidationError):
        return None
    return list(engine_inputs[0]["prompt_token_ids"])


def _chat_params(max_length: int) -> ChatParams:
    request = ChatCompletionRequest.model_validate(
        {
            "messages": _MESSAGES,
            "chat_template_kwargs": {
                "padding": "max_length",
                "max_length": max_length,
            },
        }
    )
    return request.build_chat_params(_TEMPLATE, "auto")


def test_chat_completion_padding_is_not_materialized_before_length_check():
    """The context-length check must not see a sequence the tokenizer already padded."""
    tokenizer = _build_tokenizer()
    renderer = _renderer(tokenizer)
    params = _chat_params(_ATTACK_LENGTH)
    natural = _unpadded_len(tokenizer, _MESSAGES, params)
    assert natural < renderer.model_config.max_model_len
    request = ChatCompletionRequest.model_validate(
        {
            "messages": _MESSAGES,
            "chat_template_kwargs": {
                "padding": "max_length",
                "max_length": _ATTACK_LENGTH,
            },
        }
    )
    tok_params = request.build_tok_params(renderer.model_config)

    probe = _PadProbe()
    with probe:
        token_ids = _render_chat_async(renderer, params, _MESSAGES, tok_params)
    _assert_unpadded(probe, natural, token_ids)


def test_huge_max_length_never_reaches_the_tokenizer():
    tokenizer = _build_tokenizer()
    renderer = _renderer(tokenizer)
    params = _chat_params(_HUGE_LENGTH)
    natural = _unpadded_len(tokenizer, _MESSAGES, params)

    probe = _PadProbe()
    with probe:
        token_ids = _render_messages(renderer, params, _MESSAGES)
    _assert_unpadded(probe, natural, token_ids)


def test_tokenize_chat_does_not_return_padded_ids():
    """``/tokenize`` chat sets no context limit, so a padded list would be returned."""
    tokenizer = _build_tokenizer()
    renderer = _renderer(tokenizer)
    request = TokenizeChatRequest.model_validate(
        {
            "messages": _MESSAGES,
            "chat_template_kwargs": {
                "padding": "max_length",
                "max_length": _ATTACK_LENGTH,
            },
        }
    )
    params = request.build_chat_params(_TEMPLATE, "auto")
    natural = _unpadded_len(tokenizer, _MESSAGES, params)
    tok_params = request.build_tok_params(renderer.model_config)
    assert tok_params.max_total_tokens is None

    probe = _PadProbe()
    with probe:
        token_ids = _render_chat_async(renderer, params, _MESSAGES, tok_params)
    _assert_unpadded(probe, natural, token_ids)


@pytest.mark.parametrize(
    "params_and_messages",
    [
        "responses",
        "embedding",
        "classification",
        "pooling",
        "anthropic_messages",
        "anthropic_count_tokens",
        "structured_decision",
    ],
)
def test_chat_template_kwargs_padding_is_ignored(params_and_messages: str):
    tokenizer = _build_tokenizer()
    renderer = _renderer(tokenizer)
    params, messages = _request_params(params_and_messages)
    natural = _unpadded_len(tokenizer, messages, params)

    probe = _PadProbe()
    with probe:
        token_ids = _render_messages(renderer, params, messages)
    _assert_unpadded(probe, natural, token_ids)


def test_kimi_k3_renderer_does_not_pad_to_client_max_length():
    tokenizer = _build_tokenizer()
    config = _MockVllmConfig(_MockModelConfig(), _MockParallelConfig())
    renderer = KimiK3Renderer(config, tokenizer=tokenizer)
    params = _chat_params(_ATTACK_LENGTH)
    natural = _unpadded_len(tokenizer, _MESSAGES, params)

    probe = _PadProbe()
    with probe:
        token_ids = _render_messages(renderer, params, _MESSAGES)
    _assert_unpadded(probe, natural, token_ids)


def test_template_variables_still_reach_the_chat_template():
    """Padding keys are the problem. Template variables such as tools still apply."""
    tokenizer = _build_tokenizer()
    renderer = _renderer(tokenizer)
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    request = ChatCompletionRequest.model_validate(
        {
            "messages": _MESSAGES,
            "tools": tools,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )
    params = request.build_chat_params(_TEMPLATE, "auto")
    # ``tools`` is attached by the serving layer, not by ``build_chat_params``.
    params = ChatParams(
        chat_template=params.chat_template,
        chat_template_content_format=params.chat_template_content_format,
        chat_template_kwargs={**params.chat_template_kwargs, "tools": tools},
    )
    _, prompt = renderer.render_messages(_MESSAGES, params)
    token_ids = list(prompt["prompt_token_ids"])
    assert tokenizer.convert_ids_to_tokens(token_ids) == [
        "user",
        "hi",
        "assistant",
        "thinking",
        "tool",
    ]


def _request_params(kind: str) -> tuple[ChatParams, list]:
    attack = {"padding": "max_length", "max_length": _ATTACK_LENGTH}
    if kind == "responses":
        request = ResponsesRequest.model_validate(
            {"input": "hi", "chat_template_kwargs": attack}
        )
        messages = construct_input_messages(
            request_input=request.input,
            request_instructions=request.instructions,
        )
        return request.build_chat_params(_TEMPLATE, "auto"), messages
    if kind == "embedding":
        request = EmbeddingChatRequest.model_validate(
            {"messages": _MESSAGES, "chat_template_kwargs": attack}
        )
        return request.build_chat_params(_TEMPLATE, "auto"), _MESSAGES
    if kind == "classification":
        request = ClassificationChatRequest.model_validate(
            {"messages": _MESSAGES, "chat_template_kwargs": attack}
        )
        return request.build_chat_params(_TEMPLATE, "auto"), _MESSAGES
    if kind == "pooling":
        request = PoolingChatRequest.model_validate(
            {"messages": _MESSAGES, "chat_template_kwargs": attack}
        )
        return request.build_chat_params(_TEMPLATE, "auto"), _MESSAGES
    if kind == "anthropic_messages":
        request = AnthropicMessagesRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "chat_template_kwargs": attack,
            }
        )
        chat = AnthropicServingMessages._build_base_request(request, _MESSAGES)
        return chat.build_chat_params(_TEMPLATE, "auto"), _MESSAGES
    if kind == "anthropic_count_tokens":
        request = AnthropicCountTokensRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": attack,
            }
        )
        chat = AnthropicServingMessages._build_base_request(request, _MESSAGES)
        return chat.build_chat_params(_TEMPLATE, "auto"), _MESSAGES
    if kind == "structured_decision":
        request = ReadPromptRequest(chat_template_kwargs=attack)
        return request.build_chat_params(_TEMPLATE, "auto"), _MESSAGES
    raise AssertionError(kind)
