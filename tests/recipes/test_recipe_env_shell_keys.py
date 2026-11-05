# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recipe env names must be shell identifiers before env.sh is written.

`recipe_json_to_vllm_config.py` turns a recipe `env` object into `env.sh`,
and `serve_with_recipe.sh` sources that file. Values are shell-quoted. Names
are not. These tests drive those entry points and fail while a name can
break out of `export` and run as shell.

A refusal that still creates the env file, including creating it and then
deleting it, is a failure. An HTTP recipe that is fetched again must not be
able to substitute a different document for the one that was checked: each
request is a distinct hostile document.
"""

from __future__ import annotations

import ctypes
import json
import os
import select
import shlex
import struct
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = REPO_ROOT / "tools" / "recipes" / "recipe_json_to_vllm_config.py"
SERVE_SCRIPT = REPO_ROOT / "tools" / "recipes" / "serve_with_recipe.sh"

# glibc <sys/inotify.h> and <bits/inotify.h> (Debian glibc 2.39).
_IN_CLOEXEC = 0o2000000
_IN_NONBLOCK = 0o4000
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_Q_OVERFLOW = 0x00004000
_IN_HEADER = struct.Struct("=iIII")

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.inotify_init1.argtypes = (ctypes.c_int,)
_LIBC.inotify_init1.restype = ctypes.c_int
_LIBC.inotify_add_watch.argtypes = (
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_uint32,
)
_LIBC.inotify_add_watch.restype = ctypes.c_int

# Names that execute when bash sources `export <name>=<quoted value>`.
_HOSTILE_KEY_TEMPLATES = (
    "A;touch {marker};true",
    "A$(touch {marker})",
    "$(touch {marker})",
    "A`touch {marker}`",
    "A\ntouch {marker}\n#",
    "A&&touch {marker}&&true",
    "A|touch {marker}|true",
)

# Names outside [A-Za-z_][A-Za-z0-9_]* that are not themselves commands.
_INVALID_ENV_NAMES = (
    "1BAD",
    "HAS-HYPHEN",
    "HAS.DOT",
    "HAS SPACE",
    "",
    "A=B",
    "FOO[0]",
    "Ä",
)


def _recipe(env: dict[str, str]) -> dict[str, Any]:
    return {
        "deploy_type": "single_node",
        "argv": ["vllm", "serve", "org/example-model"],
        "env": env,
    }


def _base_env() -> dict[str, str]:
    path = os.environ.get("PATH")
    if not path:
        raise AssertionError("PATH is required to run the recipe converter")
    return {"PATH": path, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}


class _DirectoryWatch:
    """Record file names created in a directory while a command runs."""

    def __init__(self, directory: Path) -> None:
        self.names: list[str] = []
        self.error: BaseException | None = None
        self._buffer = b""
        self._stop = threading.Event()
        fd = _LIBC.inotify_init1(_IN_CLOEXEC | _IN_NONBLOCK)
        if fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), str(directory))
        watch = _LIBC.inotify_add_watch(
            fd,
            os.fsencode(directory),
            _IN_CREATE | _IN_MOVED_TO | _IN_CLOSE_WRITE,
        )
        if watch < 0:
            err = ctypes.get_errno()
            os.close(fd)
            raise OSError(err, os.strerror(err), str(directory))
        self._fd = fd
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                readable, _, _ = select.select([self._fd], [], [], 0.05)
                if readable:
                    self._drain()
            self._drain()
        except BaseException as exc:
            self.error = exc

    def _drain(self) -> None:
        while True:
            try:
                chunk = os.read(self._fd, 4096)
            except BlockingIOError:
                return
            if not chunk:
                return
            self._buffer += chunk
            self._parse()

    def _parse(self) -> None:
        data = self._buffer
        offset = 0
        while offset + _IN_HEADER.size <= len(data):
            _wd, mask, _cookie, length = _IN_HEADER.unpack_from(data, offset)
            total = _IN_HEADER.size + length
            if offset + total > len(data):
                break
            raw = data[offset + _IN_HEADER.size : offset + total]
            offset += total
            if mask & _IN_Q_OVERFLOW:
                raise OSError("inotify queue overflow")
            name = raw.rstrip(b"\x00").decode("utf-8", "surrogateescape")
            if name and (
                mask & (_IN_CREATE | _IN_MOVED_TO | _IN_CLOSE_WRITE)
            ):
                self.names.append(name)
        self._buffer = data[offset:]

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("inotify reader did not stop")
        os.close(self._fd)
        if self.error is not None:
            raise self.error


class _Watching:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.watch = _DirectoryWatch(directory)

    def __enter__(self) -> _DirectoryWatch:
        return self.watch

    def __exit__(self, *exc: object) -> None:
        self.watch.close()


def _watch(directory: Path) -> _Watching:
    return _Watching(directory)


def _run_converter(
    source: str,
    work: Path,
    *,
    env_name: str = "env.sh",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(CONVERTER),
            source,
            "--env-out",
            str(work / env_name),
            "--config-out",
            str(work / "config.yml"),
        ],
        cwd=work,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _source(env_path: Path) -> subprocess.CompletedProcess[str]:
    """Source env.sh the way the converter tells the operator to run it."""
    return subprocess.run(
        ["bash", "-c", 'set -euo pipefail; source "$1"', "bash", str(env_path)],
        cwd=env_path.parent,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_rejected(
    proc: subprocess.CompletedProcess[str],
    env_path: Path,
    created: list[str],
    markers: list[Path],
) -> None:
    executed = [str(path) for path in markers if path.exists()]
    script = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    assert proc.returncode != 0, _failure(proc, env_path, created, executed, script)
    assert env_path.name not in created, _failure(
        proc, env_path, created, executed, script
    )
    assert not env_path.exists(), _failure(proc, env_path, created, executed, script)
    assert not executed, _failure(proc, env_path, created, executed, script)


def _failure(
    proc: subprocess.CompletedProcess[str],
    env_path: Path,
    created: list[str],
    executed: list[str],
    script: str,
) -> str:
    return (
        f"returncode={proc.returncode} created={created} "
        f"env_exists={env_path.exists()} executed={executed}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}\nscript={script}"
    )


def _write_vllm_stub(directory: Path, capture: Path) -> None:
    stub = directory / "vllm"
    stub.write_text(
        "#!/bin/sh\n"
        "printf '%s' \"$SAFE_VALUE\" > "
        + shlex.quote(str(capture))
        + "\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)


def _run_serve(
    source: str,
    work: Path,
    stub_dir: Path,
) -> subprocess.CompletedProcess[str]:
    env = _base_env()
    env["PATH"] = str(stub_dir) + os.pathsep + env["PATH"]
    return subprocess.run(
        ["bash", str(SERVE_SCRIPT), source],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class _RecipeServer:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        bodies = [json.dumps(document).encode("utf-8") for document in documents]
        if not bodies:
            raise AssertionError("recipe server needs at least one document")
        state = {"n": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                index = state["n"]
                state["n"] = index + 1
                body = bodies[min(index, len(bodies) - 1)]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        self.requests = state
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> _RecipeServer:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/recipe.json"


def test_watch_records_file_created_then_removed(tmp_path: Path) -> None:
    """The refusal check has to observe a file that does not survive."""
    target = tmp_path / "env.sh"
    with _watch(tmp_path) as watch:
        target.write_text("export A=1\n", encoding="utf-8")
        target.unlink()
    assert "env.sh" in watch.names
    assert not target.exists()


@pytest.mark.parametrize("template", _HOSTILE_KEY_TEMPLATES)
def test_converter_rejects_hostile_env_key_before_writing_env_sh(
    tmp_path: Path,
    template: str,
) -> None:
    marker = tmp_path / "marker"
    key = template.format(marker=marker)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(_recipe({key: "1", "SAFE_VALUE": "kept"})),
        encoding="utf-8",
    )
    env_path = tmp_path / "env.sh"
    with _watch(tmp_path) as watch:
        proc = _run_converter(str(recipe), tmp_path)
        if env_path.exists():
            _source(env_path)
    _assert_rejected(proc, env_path, watch.names, [marker])


@pytest.mark.parametrize("key", _INVALID_ENV_NAMES)
def test_converter_rejects_non_identifier_env_names(
    tmp_path: Path,
    key: str,
) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(_recipe({key: "1"})), encoding="utf-8")
    env_path = tmp_path / "env.sh"
    with _watch(tmp_path) as watch:
        proc = _run_converter(str(recipe), tmp_path)
    _assert_rejected(proc, env_path, watch.names, [])


def test_converter_http_recipe_rejects_each_fetched_document(
    tmp_path: Path,
) -> None:
    first = tmp_path / "marker-first"
    second = tmp_path / "marker-second"
    documents = [
        _recipe({f"A;touch {first};true": "1"}),
        _recipe({f"A$(touch {second})": "1"}),
    ]
    env_path = tmp_path / "env.sh"
    with _RecipeServer(documents) as server, _watch(tmp_path) as watch:
        proc = _run_converter(server.url, tmp_path)
        if env_path.exists():
            _source(env_path)
        fetched = server.requests["n"]
    assert fetched >= 1
    _assert_rejected(proc, env_path, watch.names, [first, second])


def test_serve_with_recipe_file_does_not_source_hostile_env(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "marker"
    key = f"A\ntouch {marker}\n#"
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(_recipe({key: "1"})), encoding="utf-8")
    work = tmp_path / "work"
    stub_dir = tmp_path / "bin"
    work.mkdir()
    stub_dir.mkdir()
    invoked = tmp_path / "vllm-invoked"
    _write_vllm_stub(stub_dir, invoked)
    env_path = work / "env.sh"
    with _watch(work) as watch:
        proc = _run_serve(str(recipe), work, stub_dir)
    _assert_rejected(proc, env_path, watch.names, [marker])
    assert not invoked.exists(), proc.stdout + proc.stderr


def test_serve_with_recipe_http_does_not_source_hostile_env(
    tmp_path: Path,
) -> None:
    first = tmp_path / "marker-first"
    second = tmp_path / "marker-second"
    documents = [
        _recipe({f"A`touch {first}`": "1"}),
        _recipe({f"A;touch {second};true": "1"}),
    ]
    work = tmp_path / "work"
    stub_dir = tmp_path / "bin"
    work.mkdir()
    stub_dir.mkdir()
    invoked = tmp_path / "vllm-invoked"
    _write_vllm_stub(stub_dir, invoked)
    env_path = work / "env.sh"
    with _RecipeServer(documents) as server, _watch(work) as watch:
        proc = _run_serve(server.url, work, stub_dir)
        fetched = server.requests["n"]
    assert fetched >= 1
    _assert_rejected(proc, env_path, watch.names, [first, second])
    assert not invoked.exists(), proc.stdout + proc.stderr


def test_converter_quotes_env_values_and_exports_identifiers(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "marker"
    raw_value = "a'b $(touch " + str(marker) + ") `touch " + str(marker) + "`"
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(_recipe({"SAFE_VALUE": raw_value, "_OK1": "plain"})),
        encoding="utf-8",
    )
    env_path = tmp_path / "env.sh"
    proc = _run_converter(str(recipe), tmp_path)
    assert proc.returncode == 0, proc.stderr
    sourced = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; printf "%s\\n%s" "$SAFE_VALUE" "$_OK1"',
            "bash",
            str(env_path),
        ],
        cwd=tmp_path,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert sourced.returncode == 0, sourced.stderr
    assert sourced.stdout == raw_value + "\nplain"
    assert not marker.exists()


def test_converter_http_recipe_exports_valid_env_key(tmp_path: Path) -> None:
    recipe_env = {"SAFE_VALUE": "from-http"}
    env_path = tmp_path / "env.sh"
    with _RecipeServer([_recipe(recipe_env)]) as server:
        proc = _run_converter(server.url, tmp_path)
        assert server.requests["n"] == 1
    assert proc.returncode == 0, proc.stderr
    sourced = _source(env_path)
    assert sourced.returncode == 0, sourced.stderr
    shown = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; printf "%s" "$SAFE_VALUE"',
            "bash",
            str(env_path),
        ],
        cwd=tmp_path,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert shown.stdout == "from-http"


def test_serve_with_recipe_exports_valid_env_then_execs_vllm(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(_recipe({"SAFE_VALUE": "served"})),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    stub_dir = tmp_path / "bin"
    work.mkdir()
    stub_dir.mkdir()
    captured = tmp_path / "captured"
    _write_vllm_stub(stub_dir, captured)
    proc = _run_serve(str(recipe), work, stub_dir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert captured.read_text(encoding="utf-8") == "served"
