# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV3 tool parsing must treat tokenizer ``token_suffix`` as a literal.

``tokenizer_config.json`` is loaded into ``tokenizer.init_kwargs`` and the
HYV3 parser interpolates ``token_suffix`` into its tool-call expressions.
A suffix that is a quantified group has to be escaped. These tests fail
while that value is compiled raw: a completion that reaches the injected
prefix and then fails the rest of the expression does not return, and a
completion that uses the suffix as literal marker text is not extracted.
"""

import json
import signal
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.parser_manager import ParserManager
from vllm.tokenizers import get_tokenizer

# Ambiguous quantified group. The shipped ``regex`` module finishes a pure
# nested ``(a+)+`` immediately, and stalls on this form once it is compiled
# into the HYV3 tool-call expressions.
_SUFFIX = "(?:a|a)+"
_RUN = "a" * 32
_BOUND_S = 2.0


class _SearchDidNotReturn(BaseException):
    """Raised when a parser search is still running after the bound."""


def _write_tokenizer(directory: Path) -> None:
    """Write a local checkpoint whose config carries ``token_suffix``."""
    markers = [
        f"<tool_calls{_SUFFIX}>",
        f"</tool_calls{_SUFFIX}>",
        f"<tool_call{_SUFFIX}>",
        f"</tool_call{_SUFFIX}>",
        f"<tool_sep{_SUFFIX}>",
        f"<arg_key{_SUFFIX}>",
        f"</arg_key{_SUFFIX}>",
        f"<arg_value{_SUFFIX}>",
        f"</arg_value{_SUFFIX}>",
    ]
    pieces = ["<unk>", "a", "x", "get_weather", "city", "Paris", *markers]
    vocab = {piece: index for index, piece in enumerate(pieces)}
    Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")).save(
        str(directory / "tokenizer.json")
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "TokenizersBackend",
                "token_suffix": _SUFFIX,
                "unk_token": "<unk>",
                "model_max_length": 2048,
            }
        ),
        encoding="utf-8",
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": len(vocab),
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "intermediate_size": 32,
            }
        ),
        encoding="utf-8",
    )


def _call_within(fn):
    """Run ``fn`` and fail if it is still inside the search after the bound."""

    def _on_alarm(signum, frame):
        raise _SearchDidNotReturn

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, _BOUND_S)
    try:
        try:
            return fn()
        except _SearchDidNotReturn:
            pytest.fail(
                "HYV3 tool-call search did not return within "
                f"{_BOUND_S:.0f}s for token_suffix={_SUFFIX!r}"
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture(scope="module")
def hy_v3_suffix_tokenizer(tmp_path_factory):
    directory = tmp_path_factory.mktemp("hy-v3-suffix")
    _write_tokenizer(directory)
    tokenizer = get_tokenizer(str(directory), trust_remote_code=False)
    assert tokenizer.init_kwargs.get("token_suffix") == _SUFFIX
    return tokenizer


@pytest.fixture
def hy_v3_parser(hy_v3_suffix_tokenizer):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="hy_v3",
        enable_auto_tools=True,
        tokenizer=hy_v3_suffix_tokenizer,
    )
    return parser_cls(hy_v3_suffix_tokenizer)


@pytest.fixture
def tool_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="hy-v3-suffix",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                )
            )
        ],
        tool_choice="auto",
    )


def _literal_call() -> str:
    """One well-formed call whose markers contain the suffix verbatim."""
    return (
        f"<tool_calls{_SUFFIX}>"
        f"<tool_call{_SUFFIX}>get_weather<tool_sep{_SUFFIX}>"
        f"<arg_key{_SUFFIX}>city</arg_key{_SUFFIX}>"
        f"<arg_value{_SUFFIX}>Paris</arg_value{_SUFFIX}>"
        f"</tool_call{_SUFFIX}>"
        f"</tool_calls{_SUFFIX}>"
    )


def test_literal_suffix_markers_are_extracted(hy_v3_parser, tool_request):
    """Marker text from the suffix is a literal, including argument tags."""
    _, _, calls = _call_within(
        lambda: hy_v3_parser.parse(
            _literal_call(),
            tool_request,
            enable_auto_tools=True,
        )
    )
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Paris"}


def test_tool_call_search_returns_on_failing_suffix_run(hy_v3_parser, tool_request):
    """A run of the quantified character after the open marker must return."""
    completion = f"<tool_calls{_SUFFIX}><tool_call{_RUN}!"
    _call_within(
        lambda: hy_v3_parser.parse(
            completion,
            tool_request,
            enable_auto_tools=True,
        )
    )


def test_partial_tool_call_search_returns(hy_v3_parser, tool_request):
    """The portion expression is bounded after a complete literal call."""
    completion = f"{_literal_call()}<tool_call{_RUN}!"
    _call_within(
        lambda: hy_v3_parser.parse(
            completion,
            tool_request,
            enable_auto_tools=True,
        )
    )


def test_argument_search_returns_while_streaming(hy_v3_parser, tool_request):
    """Streaming argument extraction is bounded for the same suffix."""
    start_id = hy_v3_parser.model_tokenizer.get_vocab()[f"<tool_calls{_SUFFIX}>"]
    arguments = f"<arg_key{_RUN}>x</arg_key{_RUN}!"
    delta = (
        f"<tool_calls{_SUFFIX}>"
        f"<tool_call{_SUFFIX}>get_weather<tool_sep{_SUFFIX}>"
        f"{arguments}"
        f"</tool_call{_SUFFIX}>"
    )
    _call_within(
        lambda: hy_v3_parser.parse_delta(
            delta_text=delta,
            delta_token_ids=[start_id],
            request=tool_request,
            prompt_token_ids=[0],
            finished=True,
        )
    )
