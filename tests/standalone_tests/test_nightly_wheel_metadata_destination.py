# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Refuse off-host nightly wheel URLs and filenames that escape the temp dir.

``setup.py`` loads the wheel utility class at import time and then calls
``setup()``. These tests exec that class and ``get_rocm_version`` so the
metadata, ``urljoin``, ``os.path.join``, and ``urlretrieve`` path runs
without torch. Hostile metadata must raise ``ValueError`` before any
download or any open of a destination outside the temporary directory.
"""

import ast
import contextlib
import ctypes
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from shutil import which
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_SETUP_PATH = Path(__file__).resolve().parents[2] / "setup.py"
_SETUP_NODES = [
    node
    for node in ast.parse(_SETUP_PATH.read_text()).body
    if (
        isinstance(node, ast.ClassDef) and node.name == "precompiled_wheel_utils"
    )
    or (isinstance(node, ast.FunctionDef) and node.name == "get_rocm_version")
]
_NAMESPACE: dict[str, Any] = {
    "subprocess": subprocess,
    "os": os,
    "re": re,
    "shutil": shutil,
    "which": which,
    "Path": Path,
    "ctypes": ctypes,
    "ROCM_HOME": None,
    "logger": logging.getLogger("setup"),
    "envs": SimpleNamespace(
        VLLM_DOCKER_BUILD_CONTEXT=False,
        VLLM_USE_PRECOMPILED=True,
    ),
    "PRECOMPILED_RUST_EXTENSION_MEMBER_REGEX": re.compile(
        r"vllm/_rust_[^/]*\.so$"
    ),
}
exec(
    compile(
        ast.Module(body=_SETUP_NODES, type_ignores=[]),
        str(_SETUP_PATH),
        "exec",
    ),
    _NAMESPACE,
)
wheel_utils = _NAMESPACE["precompiled_wheel_utils"]

_COMMIT = "a" * 40
_ESCAPE_MARK = "vllm-nightly-escape"
_PAYLOAD = b"vllm-nightly-metadata-body\n"
_WHEELS: dict[str, Any] = {"commit": _COMMIT, "entries": []}
_OPENED_ESCAPE_PATHS: list[str] = []
_AUDIT_INSTALLED = False
_REAL_URLOPEN = urlopen


def _empty_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w"):
        pass
    return buffer.getvalue()


_EMPTY_ZIP = _empty_zip()


def _audit_open(event: str, args: tuple[Any, ...]) -> None:
    if event != "open" or not args:
        return
    path = args[0]
    if isinstance(path, bytes):
        path = path.decode("utf-8", "surrogateescape")
    if isinstance(path, str) and _ESCAPE_MARK in path:
        _OPENED_ESCAPE_PATHS.append(path)


def _install_audit() -> None:
    global _AUDIT_INSTALLED
    if _AUDIT_INSTALLED:
        return
    sys.addaudithook(_audit_open)
    _AUDIT_INSTALLED = True


class _BytesResponse(io.BytesIO):
    """Response object ``urlretrieve`` and ``urlopen`` actually read.

    ``urllib.request.urlretrieve`` calls ``info()`` and ``read(size)``
    (Lib/urllib/request.py). Metadata loading calls ``read()``.
    """

    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = EmailMessage()
        self.headers["Content-Length"] = str(len(payload))

    def info(self) -> EmailMessage:
        return self.headers


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = _PAYLOAD
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(body)
        server = self.server
        assert isinstance(server, _Server)
        server.requests.append(self.path)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], handler: type[_Handler]):
        super().__init__(address, handler)
        self.requests: list[str] = []


def _route(url: str | Request, *args: Any, **kwargs: Any) -> Any:
    text = url.full_url if isinstance(url, Request) else str(url)
    parsed = urlsplit(text)
    if parsed.hostname == "wheels.vllm.ai":
        return _wheels_response(parsed.path)
    if parsed.hostname == "127.0.0.1":
        return _REAL_URLOPEN(url, *args, **kwargs)
    raise AssertionError(f"unexpected URL {text}")


def _wheels_response(path: str) -> _BytesResponse:
    if path.endswith("metadata.json"):
        return _BytesResponse(json.dumps(_WHEELS["entries"]).encode())
    trimmed = path.rstrip("/")
    if trimmed.endswith(str(_WHEELS["commit"])):
        return _BytesResponse(b'<a href="rocm70/">rocm70/</a>')
    if path.endswith("/"):
        return _BytesResponse(b'<a href="vllm/">vllm/</a>')
    if path.endswith("vllm-real.whl"):
        return _BytesResponse(_EMPTY_ZIP)
    if path.endswith(".whl"):
        return _BytesResponse(_PAYLOAD)
    raise AssertionError(f"unexpected wheels path {path}")


class TestNightlyWheelMetadataDestination(unittest.TestCase):
    server: _Server
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        _install_audit()
        cls.server = _Server(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="vllm-meta-dest-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.workdir = self.root / "work"
        self.tmpdir = self.root / "tmp"
        self.outside = self.root / "outside"
        self.workdir.mkdir()
        self.tmpdir.mkdir()
        self.outside.mkdir()
        self.rocm = False
        self.selected: tuple[str, str | None] | None = None

    @contextlib.contextmanager
    def _patched(self):
        env = {
            "VLLM_PRECOMPILED_WHEEL_COMMIT": _COMMIT,
            "VLLM_PRECOMPILED_WHEEL_VARIANT": "rocm70" if self.rocm else "cu129",
            "TMPDIR": str(self.tmpdir),
            "VLLM_ROCM_WHEEL_INDEX": f"http://127.0.0.1:{self.port}/not-used",
        }
        if self.rocm:
            env["ROCM_PATH"] = str(self.tmpdir)
        previous_cwd = os.getcwd()
        previous_tempdir = tempfile.tempdir
        os.chdir(self.workdir)
        tempfile.tempdir = None
        try:
            with (
                patch.dict(os.environ, env, clear=True),
                patch("urllib.request.urlopen", _route),
            ):
                yield
        finally:
            os.chdir(previous_cwd)
            tempfile.tempdir = previous_tempdir

    def _publish(self, path: str, filename: str | None) -> None:
        entry: dict[str, str] = {
            "package_name": "vllm",
            "platform_tag": f"manylinux2014_{platform.machine()}",
            "path": path,
        }
        if filename is not None:
            entry["filename"] = filename
        _WHEELS["entries"] = [entry]
        _WHEELS["commit"] = _COMMIT

    def _run(self, path: str, filename: str | None) -> tuple[str, str | None]:
        self._publish(path, filename)
        with self._patched():
            wheel_url, download_filename = wheel_utils.determine_wheel_url()
            self.selected = (wheel_url, download_filename)
            wheel_utils.extract_precompiled_and_patch_package(
                wheel_url,
                download_filename,
                extract_extensions=True,
                extract_rust_frontend=True,
            )
        assert self.selected is not None
        return self.selected

    def _created_files(self, opened: list[str]) -> list[Path]:
        created: list[Path] = []
        seen: set[Path] = set()
        for raw in opened:
            resolved = Path(os.path.normpath(raw))
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            created.append(resolved)
        for item in self.root.rglob(f"*{_ESCAPE_MARK}*"):
            if item.is_file() and item not in seen:
                created.append(item)
        return created

    def _assert_refused(self, path: str, filename: str | None) -> None:
        self.server.requests.clear()
        _OPENED_ESCAPE_PATHS.clear()
        self.selected = None
        try:
            self._run(path, filename)
            outcome = "returned"
        except ValueError:
            outcome = "refused"
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
        opened = list(_OPENED_ESCAPE_PATHS)
        created = self._created_files(opened)
        problems: list[str] = []
        if outcome != "refused":
            problems.append(f"outcome={outcome} selected={self.selected}")
        if opened:
            problems.append(f"opened={opened}")
        if created:
            rendered = [f"{item}={item.read_bytes()!r}" for item in created]
            problems.append(f"created={rendered}")
        if self.server.requests:
            problems.append(f"fetched={list(self.server.requests)}")
        for item in created:
            item.unlink()
        self.assertFalse(problems, "; ".join(problems))

    def _assert_path_rejected(self, path: str) -> None:
        self.server.requests.clear()
        self._publish(path, "wheel.whl")
        try:
            with self._patched():
                selected = wheel_utils.determine_wheel_url()
        except ValueError:
            selected = None
        self.assertIsNone(selected, f"selected {selected}")
        self.assertEqual(self.server.requests, [])

    def _payload_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/payload"

    def test_relative_metadata_path_stays_on_the_wheel_host(self) -> None:
        self.server.requests.clear()
        _OPENED_ESCAPE_PATHS.clear()
        url, filename = self._run("../vllm-real.whl", "vllm-real.whl")
        self.assertEqual(filename, "vllm-real.whl")
        self.assertEqual(
            url,
            f"https://wheels.vllm.ai/{_COMMIT}/cu129/vllm-real.whl",
        )
        self.assertEqual(self.server.requests, [])
        self.assertEqual(_OPENED_ESCAPE_PATHS, [])

    def test_absolute_url_and_absolute_filename_are_refused(self) -> None:
        destination = self.outside / f"{_ESCAPE_MARK}-abs.bin"
        self._assert_refused(self._payload_url(), str(destination))

    def test_absolute_url_and_parent_filename_are_refused(self) -> None:
        self._assert_refused(self._payload_url(), f"../{_ESCAPE_MARK}-up.bin")

    def test_absolute_url_and_deep_parent_filename_are_refused(self) -> None:
        self._assert_refused(
            self._payload_url(),
            f"../../{_ESCAPE_MARK}-up2.bin",
        )

    def test_absolute_url_and_nested_parent_filename_are_refused(self) -> None:
        self._assert_refused(
            self._payload_url(),
            f"missing/../../{_ESCAPE_MARK}-nest.bin",
        )

    def test_absolute_url_with_safe_filename_is_not_downloaded(self) -> None:
        self._assert_refused(self._payload_url(), "wheel.whl")

    def test_absolute_url_without_filename_is_not_downloaded(self) -> None:
        self._assert_refused(self._payload_url(), None)

    def test_on_host_path_with_absolute_filename_is_refused(self) -> None:
        destination = self.outside / f"{_ESCAPE_MARK}-abs.bin"
        self._assert_refused("../payload.whl", str(destination))

    def test_on_host_path_with_parent_filename_is_refused(self) -> None:
        self._assert_refused("../payload.whl", f"../{_ESCAPE_MARK}-up.bin")

    def test_protocol_relative_metadata_path_is_rejected(self) -> None:
        self._assert_path_rejected(f"//{self._payload_url().split('://', 1)[1]}")

    def test_lookalike_wheel_host_is_rejected(self) -> None:
        self._assert_path_rejected("https://wheels.vllm.ai.example/payload.whl")

    def test_userinfo_wheel_host_is_rejected(self) -> None:
        self._assert_path_rejected(
            f"http://wheels.vllm.ai@127.0.0.1:{self.port}/payload"
        )

    def test_rocm_absolute_url_and_absolute_filename_are_refused(self) -> None:
        self.rocm = True
        destination = self.outside / f"{_ESCAPE_MARK}-abs.bin"
        self._assert_refused(self._payload_url(), str(destination))


if __name__ == "__main__":
    unittest.main()
