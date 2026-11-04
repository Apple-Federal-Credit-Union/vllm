# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confine LoRA adapter directories to a root directory."""

import contextlib
import os
from collections.abc import Iterator

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC


def resolve_adapter_dir(root: str, name: str) -> str | None:
    """Resolve `name` to a directory strictly below `root`.

    Args:
        root: Canonical (symlink-free) path of the directory to confine to.
        name: Untrusted path of the adapter relative to `root`.

    Returns:
        The canonical path of the adapter directory, or None if `name` is
        absolute, contains a parent-directory segment, or resolves (through
        symlinks) to `root` itself or anywhere outside it.

    """
    if not name or "\0" in name or os.path.isabs(name):
        return None
    segments = name.split(os.sep)
    if os.altsep:
        segments = [part for seg in segments for part in seg.split(os.altsep)]
    if os.pardir in segments:
        return None
    path = os.path.realpath(os.path.join(root, name))
    if _relative_parts(root, path) is None:
        return None
    return path


def open_adapter_dir(root: str, path: str) -> int:
    """Open the adapter directory `path` below `root` without following
    symlinks below `root`.

    Args:
        root: Canonical path of the directory the adapter must stay in.
        path: Canonical adapter path returned by `resolve_adapter_dir`.

    Returns:
        A file descriptor for the directory. The caller must close it.

    Raises:
        ValueError: If `path` is not strictly below `root`.
        OSError: If a component of `path` is missing, is not a directory,
            or is a symlink.

    """
    parts = _relative_parts(root, path)
    if parts is None:
        raise ValueError(f"LoRA adapter path {path!r} is not inside {root!r}")
    fd = os.open(root, _DIR_FLAGS)
    try:
        for part in parts:
            next_fd = os.open(part, _DIR_FLAGS | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
    except BaseException:
        os.close(fd)
        raise
    return fd


@contextlib.contextmanager
def bound_adapter_dir(root: str, path: str) -> Iterator[str]:
    """Yield a path that refers to the directory opened by `open_adapter_dir`
    for as long as the context is active, so later opens by path cannot be
    redirected by swapping a component of `path`."""
    fd = open_adapter_dir(root, path)
    try:
        fd_path = f"/proc/self/fd/{fd}"
        yield fd_path if os.path.isdir(fd_path) else path
    finally:
        os.close(fd)


def _relative_parts(root: str, path: str) -> list[str] | None:
    if not os.path.isabs(root) or not os.path.isabs(path):
        return None
    relative = os.path.relpath(path, root)
    parts = relative.split(os.sep)
    if relative == os.curdir or os.pardir in parts:
        return None
    return parts
