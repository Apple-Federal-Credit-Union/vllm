# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.renderers import render_timeout
from vllm.renderers.hf import validate_request_chat_template


def test_nested_request_chat_template_is_validation_error():
    """A nested template raises RecursionError in Jinja's parser, which is not
    a TemplateError. It must be a client error instead of an unhandled 500.
    """
    template = "{% if true %}" * 400 + "x" + "{% endif %}" * 400
    with pytest.raises(VLLMValidationError, match="recursion depth") as exc_info:
        validate_request_chat_template(template)
    assert isinstance(exc_info.value.__cause__, RecursionError)


def test_request_chat_template_rejects_pow():
    with pytest.raises(VLLMValidationError, match="may not use the '\\*\\*' operator"):
        validate_request_chat_template("{{ 2 ** 8 }}")


def test_request_chat_template_accepts_template_without_pow():
    validate_request_chat_template("{{ messages }}")


def test_invalid_request_chat_template_is_validation_error():
    with pytest.raises(VLLMValidationError):
        validate_request_chat_template("{% if %}")


def test_large_request_chat_template_is_bounded(monkeypatch: pytest.MonkeyPatch):
    """Parsing runs under the render timeout, so a huge template cannot block
    the serving thread for the full parse.
    """
    monkeypatch.setattr(render_timeout.envs, "VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT", 0.05)
    template = "{{ x }}" * 400_000
    started = time.perf_counter()
    with pytest.raises(VLLMValidationError, match="timed out"):
        validate_request_chat_template(template)
    assert time.perf_counter() - started < 5
