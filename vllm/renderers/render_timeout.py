# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound chat template rendering by stopping the thread that renders it.

A Jinja template compiles to Python bytecode, so an exception scheduled with
``PyThreadState_SetAsyncExc`` interrupts a runaway template at its next
bytecode boundary. Abandoning the thread instead would leave the template
running for as long as the process lives.

The exception is only scheduled while the rendering thread is executing the
template or code the template called. Raised anywhere else (e.g. inside
``Condition.__enter__`` of a tokenizer pool queue) it could leave a lock
acquired forever.
"""

import concurrent.futures
import contextlib
import ctypes
import heapq
import itertools
import logging
import os
import queue
import sys
import threading
import time
import warnings
from collections.abc import Callable
from types import FrameType, ModuleType
from typing import TypeVar

import jinja2
import markupsafe
from typing_extensions import ParamSpec

from vllm import envs

P = ParamSpec("P")
T = TypeVar("T")

_set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
_set_async_exc.argtypes = (ctypes.c_ulong, ctypes.py_object)
_set_async_exc.restype = ctypes.c_int


def _package_dir(module: ModuleType) -> str:
    assert module.__file__ is not None
    return os.path.dirname(module.__file__) + os.sep


def _module_file(module: ModuleType) -> str:
    assert module.__file__ is not None
    return module.__file__


_TEMPLATE_FILENAME = "<template>"
_TEMPLATE_PACKAGE_DIRS = (_package_dir(jinja2), _package_dir(markupsafe))
# Python-level lock handling that an exception raised part-way through could
# leave acquired.
_LOCKING_CODE = (
    *(_module_file(m) for m in (contextlib, queue, threading, warnings)),
    *(_package_dir(m) for m in (concurrent.futures, logging)),
    "<frozen importlib.",
)
_RETRY_INTERVAL_S = 0.001


class _RenderInterrupted(BaseException):
    """Raised inside a render that ran past its deadline.

    Not an ``Exception`` so that ``except Exception`` handlers on the render
    path cannot swallow it and keep the template running.
    """


def _is_interruptible(frame: FrameType) -> bool:
    filename = frame.f_code.co_filename
    if filename == _TEMPLATE_FILENAME or filename.startswith(_TEMPLATE_PACKAGE_DIRS):
        return True
    if filename.startswith(_LOCKING_CODE):
        return False
    # Helpers a template calls (filters, globals) count as the template, or a
    # template could keep the thread outside its own frames whenever sampled.
    caller = frame.f_back
    while caller is not None:
        if caller.f_code.co_filename == _TEMPLATE_FILENAME:
            return True
        caller = caller.f_back
    return False


class _Deadline:
    __slots__ = ("thread_id", "fired")

    def __init__(self, thread_id: int) -> None:
        # None once the render has returned; the watchdog only interrupts a
        # thread while this is set, and both sides hold the watchdog lock.
        self.thread_id: int | None = thread_id
        self.fired = False


class _Watchdog:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._heap: list[tuple[float, int, _Deadline]] = []
        self._seq = itertools.count()
        self._thread: threading.Thread | None = None

    def _push(self, when: float, deadline: _Deadline) -> bool:
        entry = (when, next(self._seq), deadline)
        heapq.heappush(self._heap, entry)
        return self._heap[0] is entry

    def _drop_finished(self) -> None:
        while self._heap and self._heap[0][2].thread_id is None:
            heapq.heappop(self._heap)

    def arm(self, deadline: _Deadline, timeout: float) -> None:
        with self._cond:
            self._drop_finished()
            if self._push(time.monotonic() + timeout, deadline):
                self._cond.notify()
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="ChatTemplateRenderWatchdog",
                    daemon=True,
                )
                self._thread.start()

    def disarm(self, deadline: _Deadline) -> None:
        with self._cond:
            deadline.thread_id = None
            if deadline.fired:
                # Never let a pending interrupt outlive the render it targets.
                _set_async_exc(threading.get_ident(), ctypes.py_object())

    def _run(self) -> None:
        with self._cond:
            while True:
                self._drop_finished()
                if not self._heap:
                    self._cond.wait()
                    continue
                when, _, deadline = self._heap[0]
                now = time.monotonic()
                if when > now:
                    self._cond.wait(when - now)
                    continue
                heapq.heappop(self._heap)
                thread_id = deadline.thread_id
                assert thread_id is not None
                # The target cannot run while this thread holds the GIL, so it
                # resumes exactly where this frame shows it to be.
                frame = sys._current_frames().get(thread_id)
                interruptible = frame is not None and _is_interruptible(frame)
                del frame
                if interruptible:
                    deadline.fired = True
                    _set_async_exc(thread_id, _RenderInterrupted)
                else:
                    self._push(now + _RETRY_INTERVAL_S, deadline)


_watchdog = _Watchdog()


def _reset_watchdog_after_fork() -> None:
    # The watchdog thread does not survive fork, and its lock may be held.
    global _watchdog
    _watchdog = _Watchdog()


os.register_at_fork(after_in_child=_reset_watchdog_after_fork)


def render_with_timeout(fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """Call ``fn`` and stop it after ``VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT``.

    ``fn`` runs in the calling thread. When the deadline passes, the Jinja
    template it is rendering is interrupted in that thread, so it has stopped
    by the time this raises ``TimeoutError``.
    """
    timeout = envs.VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT
    if timeout <= 0:
        return fn(*args, **kwargs)

    deadline = _Deadline(threading.get_ident())
    try:
        try:
            _watchdog.arm(deadline, timeout)
            return fn(*args, **kwargs)
        finally:
            _watchdog.disarm(deadline)
    except _RenderInterrupted:
        if not deadline.fired:
            raise
        raise TimeoutError(
            f"Chat template rendering timed out after {timeout}s. This may "
            "indicate a malicious or excessively complex template. Adjust "
            "VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT to override."
        ) from None
