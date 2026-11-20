# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam tool-grammar preparation must stay linear in a parameter name.

These tests fail while ``_endings`` joins every suffix of a tool parameter
name while adjusting a request. They pass when that preparation either
rejects the name or builds the grammar without a quadratic allocation.
"""

from __future__ import annotations

import contextlib
import tracemalloc
from collections.abc import Callable

import pytest
from openai.types.responses import FunctionTool, ToolChoiceFunction
from pydantic import ValidationError

import vllm.parser.response_template_grammar as response_template_grammar
from tests.parser.test_beam_parser import _tool, adjust, chat_request, named
from vllm import envs
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError

# The joined suffixes of a name this long are about two megabytes before the
# regular expression is compiled. Copying the name through adjustment is not.
_NAME_LEN = 2048
_MAX_PREP_BYTES = 1_000_000

RequestFactory = Callable[[int], object]
pytestmark = pytest.mark.skip_global_cleanup


def _clear_suffix_caches() -> None:
    """Drop suffix memoization so each case pays for its own name.

    A fix that removes the memoized helpers leaves this as a no-op.
    """
    for name in ("_endings", "_spells_closer"):
        clear = getattr(
            getattr(response_template_grammar, name, None), "cache_clear", None
        )
        if clear is not None:
            clear()


def _properties(length: int, schema: dict | None = None) -> dict:
    return {"p" * length: {"type": "string"} if schema is None else schema}


def _chat(length: int, **body):
    return chat_request([_tool("lookup", _properties(length))], **body)


def _chat_omitted(length: int):
    request = _chat(length)
    assert request.tool_choice == "auto"
    return request


def _chat_required_key(length: int):
    tool = _tool("lookup", {"city": {"type": "string"}})
    tool.function.parameters["required"] = ["r" * length]
    return chat_request([tool])


def _chat_empty_schema(length: int):
    return chat_request([_tool("lookup", _properties(length, {}))])


def _chat_closed_object(length: int):
    tool = _tool("lookup", _properties(length))
    tool.function.parameters["additionalProperties"] = False
    return chat_request([tool], tool_choice="required")


def _chat_strict(length: int):
    return chat_request(
        [_tool("lookup", _properties(length), strict=True)],
        tool_choice="auto",
    )


def _responses(length: int, **body):
    return ResponsesRequest(
        input="Weather?",
        tools=[
            FunctionTool(
                type="function",
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": _properties(length),
                },
            )
        ],
        **body,
    )


def _responses_required_key(length: int):
    return ResponsesRequest(
        input="Weather?",
        tools=[
            FunctionTool(
                type="function",
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["r" * length],
                },
            )
        ],
    )


@pytest.fixture(scope="module", autouse=True)
def _warmup_beam_grammar():
    """Import and compile the template grammar before measurements."""
    assert envs.VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT
    assert envs.VLLM_ENFORCE_STRICT_TOOL_CALLING
    adjust(chat_request([_tool("lookup", {"city": {"type": "string"}})]))
    _clear_suffix_caches()


def _peak_bytes(make_request: RequestFactory) -> int:
    _clear_suffix_caches()
    tracemalloc.start()
    try:
        # Rejecting the name is a fix when the refusal itself stays small.
        with contextlib.suppress(VLLMValidationError, ValidationError):
            adjust(make_request(_NAME_LEN))
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        _clear_suffix_caches()
    return peak


@pytest.mark.parametrize(
    "make_request",
    [
        pytest.param(_chat_omitted, id="chat_omitted_tool_choice"),
        pytest.param(
            lambda length: _chat(length, tool_choice="auto"),
            id="chat_auto",
        ),
        pytest.param(
            lambda length: _chat(length, tool_choice="required"),
            id="chat_required",
        ),
        pytest.param(
            lambda length: _chat(length, tool_choice=named("lookup")),
            id="chat_named",
        ),
        pytest.param(_chat_strict, id="chat_strict_tool"),
        pytest.param(_chat_empty_schema, id="chat_empty_property_schema"),
        pytest.param(_chat_closed_object, id="chat_additional_properties_false"),
        pytest.param(_chat_required_key, id="chat_required_key"),
        pytest.param(_responses, id="responses_auto"),
        pytest.param(
            lambda length: _responses(length, tool_choice="required"),
            id="responses_required",
        ),
        pytest.param(
            lambda length: _responses(
                length,
                tool_choice=ToolChoiceFunction(type="function", name="lookup"),
            ),
            id="responses_named",
        ),
        pytest.param(_responses_required_key, id="responses_required_key"),
    ],
)
def test_parameter_name_preparation_is_not_quadratic(make_request: RequestFactory):
    """One client-supplied parameter name must not allocate every suffix."""
    peak = _peak_bytes(make_request)
    assert peak < _MAX_PREP_BYTES, (
        f"preparing a {_NAME_LEN}-character parameter name allocated {peak} bytes"
    )


def test_short_parameter_name_still_constrains_beam_output():
    request = chat_request([_tool("lookup", {"city": {"type": "string"}})])

    adjust(request)

    assert request.tool_choice == "auto"
    grammar = request.structured_outputs
    assert grammar is not None
    assert grammar.structural_tag
