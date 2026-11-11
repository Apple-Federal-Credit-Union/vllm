# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regular expressions read from untrusted checkpoint configs.

``regex`` expands nested and counted repeats while compiling, and compiling
has no timeout. Patterns are therefore restricted to a subset whose compiled
size can be computed up front: no repeat may apply to a group that already
contains a repeat (except an optional ``?``), and recursion, backreferences,
fuzzy matching, verbose mode, nested sets and Unicode property escapes are
rejected. The expanded size of each pattern, and of all patterns compiled for
one config, is capped, and compiling and matching share one time budget.
"""

import time

import regex as re

MAX_PATTERN_LENGTH = 512
# Sizes count atoms after expanding counted repeats; ``regex`` needs roughly
# 150 bytes per atom.
MAX_PATTERN_SIZE = 4096
MAX_TOTAL_SIZE = 1 << 16
PATTERN_SIZE_OVERHEAD = 16
TIME_BUDGET_S = 10.0

_GROUP_PREFIXES = (":", "=", "!", "<=", "<!", ">")
_GROUP_NAME = re.compile(r"P?<[A-Za-z_][A-Za-z0-9_]*>")
_INLINE_FLAGS = re.compile(r"[aimsu]*(?:-[aimsu]+)?[:)]")
_COUNTED_REPEAT = re.compile(r"\{(\d*)(,?)(\d*)\}")
_ESCAPE_CLASSES = frozenset("dDwWsSbBAZ")

# State of the most recent item, which decides what a following repeat does.
_NONE = 0  # nothing that can be repeated
_ATOM = 1  # literal, class, escape, or group without a repeat inside
_NESTED = 2  # group that contains a repeat
_REPEATED = 3  # item that already has a repeat


def _skip_escape(pattern: str, i: int) -> int:
    if i + 1 >= len(pattern):
        raise ValueError("trailing backslash")
    ch = pattern[i + 1]
    if ch.isalnum() and ch not in _ESCAPE_CLASSES:
        raise ValueError(f"unsupported escape \\{ch}")
    return i + 2


def _skip_class(pattern: str, i: int) -> int:
    j = i + 1
    if pattern.startswith("^", j):
        j += 1
    if pattern.startswith("]", j):
        j += 1
    while j < len(pattern):
        ch = pattern[j]
        if ch == "\\":
            j = _skip_escape(pattern, j)
        elif ch == "[":
            raise ValueError("unescaped '[' inside a character class")
        elif ch == "]":
            return j + 1
        else:
            j += 1
    raise ValueError("unterminated character class")


def _open_group(pattern: str, i: int) -> tuple[int, bool]:
    """Returns the index after the group opener, and whether it opens a group
    (``False`` for a standalone inline flag such as ``(?i)``)."""
    if not pattern.startswith("?", i + 1):
        return i + 1, True
    j = i + 2
    for prefix in _GROUP_PREFIXES:
        if pattern.startswith(prefix, j):
            return j + len(prefix), True
    if m := _GROUP_NAME.match(pattern, j):
        return m.end(), True
    if m := _INLINE_FLAGS.match(pattern, j):
        return m.end(), pattern[m.end() - 1] == ":"
    raise ValueError(f"unsupported group syntax at position {i}")


def _skip_repeat(pattern: str, i: int) -> tuple[int, int, bool]:
    """Returns the index after the repeat, how many copies of the repeated
    item it expands to, and whether it is a plain ``?``."""
    if pattern[i] == "{":
        m = _COUNTED_REPEAT.match(pattern, i)
        if m is None or not (m.group(1) or m.group(3)):
            raise ValueError(f"unsupported '{{' at position {i}")
        copies = max(int(b) for b in (m.group(1), m.group(3)) if b)
        j, optional = m.end(), False
    else:
        copies = 1
        j, optional = i + 1, pattern[i] == "?"
    if j < len(pattern) and pattern[j] in "?+":
        j += 1
    return j, max(copies, 1), optional


def checkpoint_regex_size(pattern: str) -> int:
    """Returns the expanded size of a checkpoint regex.

    Raises:
        ValueError: If the pattern is outside the supported subset or larger
            than ``MAX_PATTERN_SIZE``.

    """
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise ValueError(f"pattern is longer than {MAX_PATTERN_LENGTH} characters")
    open_groups: list[tuple[bool, int]] = []
    has_repeat = False
    size = 0
    last = _NONE
    last_size = 0
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i = _skip_escape(pattern, i)
            last, last_size = _ATOM, 1
            size += 1
        elif ch == "[":
            i = _skip_class(pattern, i)
            last, last_size = _ATOM, 1
            size += 1
        elif ch == "(":
            i, is_group = _open_group(pattern, i)
            if is_group:
                open_groups.append((has_repeat, size))
                has_repeat, size = False, 0
            last = _NONE
        elif ch == ")":
            if not open_groups:
                raise ValueError(f"unbalanced ')' at position {i}")
            last = _NESTED if has_repeat else _ATOM
            last_size = size
            outer_has_repeat, outer_size = open_groups.pop()
            has_repeat = outer_has_repeat or has_repeat
            size += outer_size
            i += 1
        elif ch in "*+?{":
            j, copies, optional = _skip_repeat(pattern, i)
            if last in (_NONE, _REPEATED) or (last == _NESTED and not optional):
                raise ValueError(f"unsupported repeat at position {i}")
            size += last_size * (copies - 1)
            has_repeat = True
            last = _REPEATED
            i = j
        else:
            last = _NONE if ch in "|^$" else _ATOM
            last_size = 1
            size += 1
            i += 1
        if size > MAX_PATTERN_SIZE:
            raise ValueError(f"pattern expands to more than {MAX_PATTERN_SIZE} atoms")
    if open_groups:
        raise ValueError("unbalanced '('")
    return size


class CheckpointRegexMatcher:
    """Compiles checkpoint regexes and matches names against them.

    All patterns compiled by one matcher share a size budget, and all compiles
    and matches share a time budget, so a config with many or slow patterns
    fails to load instead of stalling it.
    """

    def __init__(
        self,
        time_budget_s: float = TIME_BUDGET_S,
        size_budget: int = MAX_TOTAL_SIZE,
    ) -> None:
        self._remaining_s = time_budget_s
        self._remaining_size = size_budget

    def compile(self, pattern: str) -> re.Pattern:
        """Compiles a regular expression taken from a checkpoint config.

        Raises:
            ValueError: If the pattern is outside the supported subset or
                invalid, or a budget runs out.

        """
        try:
            size = checkpoint_regex_size(pattern) + PATTERN_SIZE_OVERHEAD
            if size > self._remaining_size:
                raise ValueError("checkpoint regexes exceed their total size budget")
            self._remaining_size -= size
            return self._timed(re.compile, pattern, flags=re.V0)
        except (ValueError, re.error) as err:
            raise ValueError(
                f"Unsupported checkpoint regex {pattern!r}: {err}"
            ) from err

    def fullmatch(self, pattern: re.Pattern, name: str) -> bool:
        """Returns whether ``pattern`` fully matches ``name``.

        Raises:
            ValueError: If the time budget runs out.

        """
        try:
            match = self._timed(pattern.fullmatch, name, timeout=self._remaining_s)
        except TimeoutError as err:
            raise ValueError(
                f"Checkpoint regex {pattern.pattern!r} exceeded its time budget"
            ) from err
        return match is not None

    def _timed(self, fn, *args, **kwargs):
        if self._remaining_s <= 0:
            raise ValueError("checkpoint regexes exceeded their time budget")
        start = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            self._remaining_s -= time.monotonic() - start
