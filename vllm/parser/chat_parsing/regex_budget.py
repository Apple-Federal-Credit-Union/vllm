# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time limits for the regexes a `response_template` supplies.

Template patterns come from tokenizer metadata and are searched against chat
prompts and model output, so a nested pattern can backtrack for as long as the
input allows. Every such search runs through a `RegexBudget`, which caps each
search and all of a template's searches together, and raises
`ResponseTemplateTimeoutError` once either cap is hit.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

import regex as re

logger = logging.getLogger(__name__)

SEARCH_TIMEOUT_S = 0.25
TEMPLATE_BUDGET_S = 1.0


class ResponseTemplateTimeoutError(ValueError):
    """A `response_template` regex exceeded its time budget."""


class RegexBudget:
    """Wall-clock budget shared by every search of one template's patterns.

    Each search may run for `search_s`, and all searches together for
    `total_s`. Once a search times out or the total runs out, the budget is
    spent, and later searches raise without running.
    """

    def __init__(
        self, total_s: float = TEMPLATE_BUDGET_S, search_s: float = SEARCH_TIMEOUT_S
    ) -> None:
        self._remaining = total_s
        self._search_s = search_s
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def compile(self, pattern: str, flags: int = 0) -> BoundedPattern:
        return BoundedPattern(re.compile(pattern, flags), self)

    def _exhaust(self) -> ResponseTemplateTimeoutError:
        if not self._exhausted:
            self._exhausted = True
            logger.warning(
                "response_template regexes exceeded their time budget; the "
                "template matches no more delimiters in this parse."
            )
        return ResponseTemplateTimeoutError(
            "response_template regex exceeded its time budget"
        )

    def _timeout(self) -> float:
        if self._exhausted or self._remaining <= 0:
            raise self._exhaust()
        return min(self._search_s, self._remaining)

    def _spend(self, run: Callable[[], Any]) -> Any:
        start = time.monotonic()
        try:
            return run()
        except TimeoutError as e:
            raise self._exhaust() from e
        finally:
            self._remaining -= time.monotonic() - start

    def call(self, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a `regex` method that takes `timeout=` under this budget."""
        timeout = self._timeout()
        return self._spend(lambda: method(*args, timeout=timeout, **kwargs))

    def iterate(self, matches: Callable[[float], Iterator[Any]]) -> list[Any]:
        """Collect `matches(timeout)`, charging every step to this budget.

        `regex` restarts an iterator's timeout at each match, so the budget is
        checked between matches as well."""
        iterator = matches(self._timeout())
        results: list[Any] = []
        while True:
            self._timeout()
            match = self._spend(lambda: next(iterator, None))
            if match is None:
                return results
            results.append(match)


class BoundedPattern:
    """A compiled pattern whose searches all run under a `RegexBudget`."""

    def __init__(self, compiled: Any, budget: RegexBudget) -> None:
        self._compiled = compiled
        self.budget = budget

    @property
    def pattern(self) -> str:
        return self._compiled.pattern

    @property
    def groupindex(self) -> dict[str, int]:
        return self._compiled.groupindex

    def search(self, string: str, *args: Any, **kwargs: Any) -> Any:
        return self.budget.call(self._compiled.search, string, *args, **kwargs)

    def fullmatch(self, string: str, *args: Any, **kwargs: Any) -> Any:
        return self.budget.call(self._compiled.fullmatch, string, *args, **kwargs)

    def finditer(self, string: str, *args: Any, **kwargs: Any) -> list[Any]:
        return self.budget.iterate(
            lambda timeout: self._compiled.finditer(
                string, *args, timeout=timeout, **kwargs
            )
        )

    def sub(self, repl: Any, string: str, *args: Any, **kwargs: Any) -> str:
        return self.budget.call(self._compiled.sub, repl, string, *args, **kwargs)


__all__ = [
    "BoundedPattern",
    "RegexBudget",
    "ResponseTemplateTimeoutError",
]
