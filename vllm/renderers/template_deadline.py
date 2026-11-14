# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound chat template renders that run on the calling thread.

Jinja compiles a template to Python bytecode and runs it, together with the
``jinja2`` helpers it calls (filters, ``lipsum``, the parser and optimizer),
in the thread that renders it. ``render_with_deadline`` installs a per-thread
trace function that checks ``VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT`` on every
call and line event in that code and raises from inside the template once the
deadline has passed, so the render actually stops instead of being abandoned.

Exceptions are raised only from template or ``jinja2`` frames, which hold no
locks outside ``with`` blocks, and use a ``BaseException`` so that
``except Exception`` handlers between the template and this module cannot
swallow them.

A single integer ``**`` is one native call that no trace function can
interrupt, so templates parsed under the deadline are refused if they raise
anything other than a small constant to a small constant power.
"""

import os
import sys
import time
from collections.abc import Callable
from types import FrameType
from typing import Any, ParamSpec, TypeVar

import jinja2
import jinja2.nodes

import vllm.envs as envs

_P = ParamSpec("_P")
_T = TypeVar("_T")

_JINJA_DIR = os.path.dirname(jinja2.__file__) + os.sep
_PARSE_CODE = jinja2.Environment._parse.__code__
_MAX_POWER_BASE = 2**64
_MAX_POWER_EXPONENT = 64


class ChatTemplateDeadlineError(TimeoutError):
    """A chat template did not finish within VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT,
    or uses an operation that the deadline cannot bound."""


class _DeadlineExceeded(BaseException):
    pass


def _is_template_frame(frame: FrameType) -> bool:
    if frame.f_code.co_filename.startswith(_JINJA_DIR):
        return True
    # Jinja executes compiled templates in a namespace holding their environment.
    return isinstance(frame.f_globals.get("environment"), jinja2.Environment)


def _is_small_number(node: jinja2.nodes.Expr, limit: int) -> bool:
    return (
        isinstance(node, jinja2.nodes.Const)
        and isinstance(node.value, int | float)
        and abs(node.value) <= limit
    )


def _has_unbounded_power(ast: jinja2.nodes.Template) -> bool:
    return any(
        not (
            _is_small_number(node.left, _MAX_POWER_BASE)
            and _is_small_number(node.right, _MAX_POWER_EXPONENT)
        )
        for node in ast.find_all(jinja2.nodes.Pow)
    )


class _Deadline:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.expires_at = time.monotonic() + timeout
        self.error: str | None = None

    def _fail(self, error: str) -> None:
        self.error = error
        raise _DeadlineExceeded(error)

    def _check(self) -> None:
        if time.monotonic() >= self.expires_at:
            self._fail(
                f"Chat template rendering timed out after {self.timeout}s. "
                "This may indicate a malicious or excessively complex "
                "template. Adjust VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT to override."
            )

    def on_call(self, frame: FrameType, event: str, arg: Any):
        if not _is_template_frame(frame):
            return None
        self._check()
        return self.on_event

    def on_event(self, frame: FrameType, event: str, arg: Any):
        if event == "line":
            self._check()
        elif (
            event == "return"
            and frame.f_code is _PARSE_CODE
            and isinstance(arg, jinja2.nodes.Template)
            and _has_unbounded_power(arg)
        ):
            self._fail(
                "Chat template uses '**' on operands that are not small "
                "constants, which VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT cannot "
                "bound."
            )
        return self.on_event


def render_with_deadline(
    fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    """Call ``fn`` and stop any chat template it renders at the deadline.

    Raises:
        ChatTemplateDeadlineError: A template rendered by ``fn`` ran past
            ``VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT`` or cannot be bounded by it.

    """
    timeout = envs.VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT
    previous = sys.gettrace()
    if timeout <= 0 or isinstance(getattr(previous, "__self__", None), _Deadline):
        return fn(*args, **kwargs)

    deadline = _Deadline(timeout)
    sys.settrace(deadline.on_call)
    try:
        result = fn(*args, **kwargs)
    except BaseException:
        if deadline.error is None:
            raise
    finally:
        sys.settrace(previous)

    if deadline.error is not None:
        raise ChatTemplateDeadlineError(deadline.error)
    return result
