# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounds for response_template pattern searches on a chat prompt.

`start_anchor_pattern`, `open_pattern`, and `close_pattern` are compiled as
regexes and searched against the decoded prompt. A nested pattern must not
keep that search running: loading the template may reject it, or the search
may fail the request, but either way the call returns within `DEADLINE_S`.

`REPEAT` is past the knee where `((a|aa)+)+b` on a run of `a` takes seconds
in regex 2026.2.28 (about 1.1s at 24 characters and 3.2s at 26, through
`ResponseParser`). A literal anchor on the same text finishes immediately.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from typing import Any

import pytest

from vllm.entrypoints.generate.base.protocol import FunctionDefinition
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.chat_parsing.response_parser import ResponseParser
from vllm.parser.parser_manager import ParserManager

# Nested group whose failing search backtracks without a time limit.
NESTED_PATTERN = r"((a|aa)+)+b"
ANCHOR = "<|start|>"
REPEAT = 30
DEADLINE_S = 2.0
_IMPORT_TIMEOUT_S = 90.0


class _CharTokenizer:
    """One code point per id, so the decoded prompt is the attack text."""

    def __init__(self, response_template: dict[str, Any]) -> None:
        self.response_template = response_template
        self.init_kwargs: dict[str, Any] = {}
        self.name_or_path = "char"
        self.all_special_tokens: list[str] = []
        self.all_special_ids: list[int] = []

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        **_kwargs: Any,
    ) -> str:
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def get_vocab(self) -> dict[str, int]:
        return {chr(code): code for code in range(128)}


def _ids(text: str) -> list[int]:
    return [ord(char) for char in text]


def _attack(repeat: int) -> str:
    return "a" * repeat


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[{"role": "user", "content": "ping"}])


def _tool_request() -> ChatCompletionRequest:
    tool = ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition(
            name="ping",
            parameters={"type": "object", "properties": {}},
        ),
    )
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "ping"}],
        tools=[tool],
        tool_choice="none",
    )


def _template(
    *,
    start_anchor: str | None = None,
    start_anchor_pattern: str | None = None,
    fields: dict[str, Any],
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "version": 1,
        "defaults": {"role": "assistant"},
        "fields": fields,
    }
    if start_anchor is not None:
        spec["start_anchor"] = start_anchor
    if start_anchor_pattern is not None:
        spec["start_anchor_pattern"] = start_anchor_pattern
    return spec


def _thinking(**field: Any) -> dict[str, Any]:
    body = {"content": "text"}
    body.update(field)
    return body


def _parser(name: str, template: dict[str, Any], *, tool: bool):
    tokenizer = _CharTokenizer(template)
    if tool:
        parser_cls = ParserManager.get_parser(
            tool_parser_name=name,
            enable_auto_tools=True,
            tokenizer=tokenizer,
        )
    else:
        parser_cls = ParserManager.get_parser(
            reasoning_parser_name=name,
            tokenizer=tokenizer,
        )
    assert parser_cls is not None
    tools = _tool_request().tools if tool else None
    return parser_cls(tokenizer, tools), tokenizer


def _finish_open_or_close(parser: Any, prompt: str) -> None:
    """Run the end-of-stream search, which does not pass partial=True."""
    ResponseParser(parser.response_template, prefix=prompt).finalize()


def _case_hf_start_anchor(repeat: int) -> bool:
    prompt = _attack(repeat)
    parser, _tokenizer = _parser(
        "hf",
        _template(
            start_anchor_pattern=NESTED_PATTERN,
            fields={"thinking": _thinking(open="<think>", close="</think>")},
        ),
        tool=False,
    )
    token_ids = _ids(prompt)
    parser.set_prompt_token_ids(token_ids)
    request = _request()
    ended = request.resolve_reasoning_ended(parser, token_ids)
    parser.count_reasoning_tokens(token_ids)
    return ended is None or isinstance(ended, bool)


def _case_hf_open_pattern(repeat: int) -> bool:
    prompt = ANCHOR + _attack(repeat)
    parser, _tokenizer = _parser(
        "hf",
        _template(
            start_anchor=ANCHOR,
            fields={
                "thinking": _thinking(
                    open_pattern=NESTED_PATTERN,
                    close="</think>",
                )
            },
        ),
        tool=False,
    )
    token_ids = _ids(prompt)
    parser.set_prompt_token_ids(token_ids)
    request = _request()
    ended = request.resolve_reasoning_ended(parser, token_ids)
    parser.is_reasoning_end(token_ids)
    _finish_open_or_close(parser, prompt)
    return isinstance(ended, bool)


def _case_hf_close_pattern(repeat: int) -> bool:
    prompt = ANCHOR + _attack(repeat)
    parser, _tokenizer = _parser(
        "hf",
        _template(
            start_anchor=ANCHOR,
            fields={"thinking": _thinking(close_pattern=NESTED_PATTERN)},
        ),
        tool=False,
    )
    token_ids = _ids(prompt)
    parser.set_prompt_token_ids(token_ids)
    request = _request()
    ended = request.resolve_reasoning_ended(parser, token_ids)
    parser.is_reasoning_end_streaming(token_ids, token_ids[-1:])
    _finish_open_or_close(parser, prompt)
    return isinstance(ended, bool)


def _case_beam_start_anchor(repeat: int) -> bool:
    prompt = _attack(repeat)
    parser, _tokenizer = _parser(
        "beam",
        _template(
            start_anchor_pattern=NESTED_PATTERN,
            fields={"thinking": _thinking(open="<think>", close="</think>")},
        ),
        tool=False,
    )
    token_ids = _ids(prompt)
    parser.set_prompt_token_ids(token_ids)
    request = _request()
    parser.adjust_request(request)
    ended = request.resolve_reasoning_ended(parser, token_ids)
    return isinstance(ended, bool)


def _case_beam_tool_open_pattern(repeat: int) -> bool:
    prompt = ANCHOR + _attack(repeat)
    parser, _tokenizer = _parser(
        "beam",
        _template(
            start_anchor=ANCHOR,
            fields={
                "tool_calls": {
                    "content": "text",
                    "open_pattern": NESTED_PATTERN,
                    "close": "</tool>",
                }
            },
        ),
        tool=True,
    )
    token_ids = _ids(prompt)
    parser.set_prompt_token_ids(token_ids)
    request = _tool_request()
    parser.adjust_request(request)
    parser.parse("z", request)
    return True


def _case_hf_tool_close_pattern(repeat: int) -> bool:
    prompt = ANCHOR + _attack(repeat)
    parser, _tokenizer = _parser(
        "hf",
        _template(
            start_anchor=ANCHOR,
            fields={
                "tool_calls": {
                    "content": "text",
                    "close_pattern": NESTED_PATTERN,
                }
            },
        ),
        tool=True,
    )
    token_ids = _ids(prompt)
    parser.set_prompt_token_ids(token_ids)
    request = _tool_request()
    parser.adjust_request(request)
    parser.parse("z", request)
    return True


def _case_literal_anchor(repeat: int) -> bool:
    prompt = ANCHOR + _attack(repeat)
    parser, _tokenizer = _parser(
        "hf",
        _template(
            start_anchor=ANCHOR,
            fields={"thinking": _thinking(open="<think>", close="</think>")},
        ),
        tool=False,
    )
    token_ids = _ids(prompt)
    parser.set_prompt_token_ids(token_ids)
    request = _request()
    ended = request.resolve_reasoning_ended(parser, token_ids)
    parser.adjust_request(request)
    _finish_open_or_close(parser, prompt)
    return isinstance(ended, bool)


_CASES: dict[str, Callable[[int], bool]] = {
    "hf_start_anchor": _case_hf_start_anchor,
    "hf_open_pattern": _case_hf_open_pattern,
    "hf_close_pattern": _case_hf_close_pattern,
    "beam_start_anchor": _case_beam_start_anchor,
    "beam_tool_open_pattern": _case_beam_tool_open_pattern,
    "hf_tool_close_pattern": _case_hf_tool_close_pattern,
    "literal_anchor": _case_literal_anchor,
}


def _worker(case: str, repeat: int, conn: Any) -> None:
    conn.send("ready")
    conn.recv()
    try:
        ok = _CASES[case](repeat)
    except Exception as exc:
        conn.send(("raised", type(exc).__name__, str(exc)))
        return
    conn.send(("ok", ok))


def _run_bounded(case: str, repeat: int) -> tuple[str, Any]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=_worker, args=(case, repeat, child))
    proc.start()
    child.close()
    try:
        if not parent.poll(_IMPORT_TIMEOUT_S):
            pytest.fail(f"{case} did not finish importing within {_IMPORT_TIMEOUT_S}s")
        ready = parent.recv()
        if ready != "ready":
            pytest.fail(f"{case} worker setup returned {ready!r}")
        parent.send("go")
        if not parent.poll(DEADLINE_S):
            pytest.fail(f"{case} was still searching the prompt after {DEADLINE_S}s")
        try:
            return parent.recv()
        except EOFError:
            pytest.fail(f"{case} exited before reporting a result")
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        else:
            proc.join(5)


@pytest.mark.parametrize(
    "case",
    [
        "hf_start_anchor",
        "hf_open_pattern",
        "hf_close_pattern",
        "beam_start_anchor",
        "beam_tool_open_pattern",
        "hf_tool_close_pattern",
    ],
)
def test_nested_response_template_pattern_search_is_bounded(case: str) -> None:
    """A nested pattern searched against the prompt must not stall the worker.

    The hf and beam reasoning parsers reach the search from
    `ChatCompletionRequest.resolve_reasoning_ended`. Beam also searches the
    start anchor from `adjust_request`, and a tool parser searches while
    parsing the reply. Finishing with an error is the safe outcome.
    """
    status, payload = _run_bounded(case, REPEAT)
    assert status in {"ok", "raised"}
    if status == "ok":
        assert payload is True


def test_literal_response_template_anchor_stays_bounded() -> None:
    """The same prompt length with a literal anchor still parses."""
    status, payload = _run_bounded("literal_anchor", REPEAT)
    assert status == "ok"
    assert payload is True
