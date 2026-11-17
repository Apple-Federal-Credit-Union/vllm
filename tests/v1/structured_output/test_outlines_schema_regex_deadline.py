# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Outlines schema admission must finish within the regex deadline.

A client can send a shallow JSON schema that pairs ``patternProperties``
with a wide optional ``properties`` map. The nesting limit allows it,
xgrammar and guidance both reject it, and the default ``auto`` backend
then expands it with outlines and parses the regex on the calling thread.
That call has to reject the request within
``VLLM_REGEX_COMPILATION_TIMEOUT_S``. A small schema of the same shape
must still be accepted.

These tests fail while that conversion and parse run without a deadline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm.config import StructuredOutputsConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import (
    SamplingParams,
    StructuredOutputsParams,
    check_json_nesting,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Wide enough that outlines' optional-property expansion makes the Python
# regex parse run far longer than the deadline below. The schema itself
# stays well under the 64-level nesting limit and under a normal request body.
_WIDE_PROPERTIES = 1200
_SMALL_PROPERTIES = 8
# Shorter than the untimed parse of ``_WIDE_PROPERTIES``, longer than
# process startup plus one deadline per admission below.
_ADMISSION_DEADLINE_S = 20
_REGEX_TIMEOUT_S = "1"


class _StubModelConfig:
    is_diffusion = False


def _schema(n_properties: int) -> dict:
    return {
        "type": "object",
        "properties": {f"p{i}": {"type": "string"} for i in range(n_properties)},
        "patternProperties": {"^x$": {"type": "string"}},
    }


def _response_format(schema: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "payload", "schema": schema},
    }


def _chat_params(schema: dict) -> SamplingParams:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        response_format=_response_format(schema),
    )
    return request.to_sampling_params(16, {})


def _completion_params(schema: dict) -> SamplingParams:
    request = CompletionRequest(
        model="m",
        prompt="hi",
        response_format=_response_format(schema),
    )
    return request.to_sampling_params(16, {})


def _responses_params(schema: dict) -> SamplingParams:
    request = ResponsesRequest(
        model="m",
        input="hi",
        text={
            "format": {
                "type": "json_schema",
                "name": "payload",
                "schema": schema,
            }
        },
    )
    return request.to_sampling_params(16, {})


def _admit(params: SamplingParams, backend: str) -> None:
    """Run request admission. Rejection is success; acceptance is not."""
    try:
        params._validate_structured_outputs(
            _StubModelConfig(),
            StructuredOutputsConfig(backend=backend),  # type: ignore[arg-type]
            tokenizer=object(),
        )
    except (VLLMValidationError, ValueError, TimeoutError):
        return
    raise AssertionError(
        "structured-output admission accepted a wide patternProperties schema"
    )


def _assert_schema_attached(params: SamplingParams) -> None:
    assert params.structured_outputs is not None
    payload = params.structured_outputs.json
    assert payload is not None
    decoded = json.loads(payload) if isinstance(payload, str) else payload
    assert "patternProperties" in decoded
    assert len(decoded["properties"]) == _WIDE_PROPERTIES


def _reject_wide_schemas(mode: str) -> None:
    os.environ["VLLM_REGEX_COMPILATION_TIMEOUT_S"] = _REGEX_TIMEOUT_S
    schema = _schema(_WIDE_PROPERTIES)
    # Same depth as a normal object schema. The nesting check must not be
    # what stops this request.
    check_json_nesting(schema)
    if mode == "auto-dict":
        # Dict schema, default backend: xgrammar rejects patternProperties
        # combined with properties, guidance rejects patternProperties, and
        # admission falls through to outlines.
        cases = (
            ("chat response_format auto", _chat_params(schema), "auto"),
            ("completion response_format auto", _completion_params(schema), "auto"),
            ("responses text.format auto", _responses_params(schema), "auto"),
        )
    elif mode == "outlines-string":
        # Explicit outlines backend, JSON schema supplied as a string.
        cases = (
            (
                "sampling params json string outlines",
                SamplingParams(
                    structured_outputs=StructuredOutputsParams(
                        json=json.dumps(schema)
                    )
                ),
                "outlines",
            ),
        )
    else:
        raise AssertionError(f"unknown worker mode {mode}")
    for name, params, backend in cases:
        print(f"CASE {name}", flush=True)
        _assert_schema_attached(params)
        _admit(params, backend)


def test_wide_schema_is_within_nesting_limit():
    """The depth cap allows the schema that stalls outlines."""
    check_json_nesting(_schema(_WIDE_PROPERTIES))


def _run_worker(mode: str) -> None:
    """Run admission in a child so a hang can be killed at the deadline."""
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    worker = subprocess.Popen(  # noqa: S603
        [sys.executable, str(Path(__file__).resolve()), "--worker", mode],
        cwd=repo_root,
        env=env,
    )
    try:
        return_code = worker.wait(timeout=_ADMISSION_DEADLINE_S)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()
        pytest.fail(
            f"{mode} outlines schema admission was still running after "
            f"{_ADMISSION_DEADLINE_S}s "
            f"(VLLM_REGEX_COMPILATION_TIMEOUT_S={_REGEX_TIMEOUT_S})"
        )
    assert return_code == 0, f"{mode} admission worker exited {return_code}"


def test_auto_backend_wide_schema_rejects_within_regex_deadline():
    """Chat, completion, and responses schemas must be rejected in time.

    Admission is the call ``AsyncLLM`` makes on the API thread, before a
    response is returned, for an already rendered engine input.
    """
    _run_worker("auto-dict")


def test_outlines_backend_wide_json_string_rejects_within_regex_deadline():
    """An explicit outlines backend parses a string schema with no deadline."""
    _run_worker("outlines-string")


@pytest.mark.parametrize("backend", ["auto", "outlines"])
@pytest.mark.parametrize(
    "build_params",
    [_chat_params, _completion_params, _responses_params],
    ids=["chat", "completion", "responses"],
)
def test_small_pattern_properties_schema_still_admitted(backend, build_params):
    """A small schema of the same shape still selects outlines and returns."""
    params = build_params(_schema(_SMALL_PROPERTIES))
    params._validate_structured_outputs(
        _StubModelConfig(),
        StructuredOutputsConfig(backend=backend),  # type: ignore[arg-type]
        tokenizer=object(),
    )
    assert params.structured_outputs is not None
    assert params.structured_outputs._backend == "outlines"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _reject_wide_schemas(sys.argv[2])
