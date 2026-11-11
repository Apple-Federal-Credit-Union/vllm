# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise setup-time wheel selection without importing torch or running setup."""

import ast
import importlib.util
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError, URLError

# setup.py executes setup() on import; load only the wheel utility class.
setup_path = Path(__file__).resolve().parents[2] / "setup.py"
setup_body = ast.parse(setup_path.read_text()).body
node = next(
    node
    for node in setup_body
    if isinstance(node, ast.ClassDef) and node.name == "precompiled_wheel_utils"
)
rust_member_regex = next(
    node
    for node in setup_body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name)
        and target.id == "PRECOMPILED_RUST_EXTENSION_MEMBER_REGEX"
        for target in node.targets
    )
)
namespace: dict[str, Any] = {
    "subprocess": subprocess,
    "os": os,
    "re": re,
    "shutil": shutil,
    "logger": logging.getLogger("setup"),
    "envs": SimpleNamespace(VLLM_DOCKER_BUILD_CONTEXT=False, VLLM_USE_PRECOMPILED=True),
}
exec(
    compile(
        ast.Module(body=[rust_member_regex, node], type_ignores=[]),
        str(setup_path),
        "exec",
    ),
    namespace,
)
wheel_utils = namespace["precompiled_wheel_utils"]
WHEEL = {
    "package_name": "vllm",
    "platform_tag": "manylinux_x86_64",
    "path": "wheel.whl",
    "filename": "wheel.whl",
}
NOT_FOUND = HTTPError("url", 404, "missing", Message(), None)


class TestPublishedWheelSearch(unittest.TestCase):
    """Walk back from the merge-base in a real Git repo; only the index is mocked."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.repo)
        self.git("init", "-q")
        self.published = {self.commit("vllm/a.py")}

    def git(self, *args: str) -> str:
        identity = ["-c", "user.name=t", "-c", "user.email=t@t"]
        return subprocess.check_output(["git", *identity, *args], text=True).strip()

    def commit(self, path: str) -> str:
        file = self.repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(file.read_text() + "x" if file.exists() else "x")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", path)
        return self.git("rev-parse", "HEAD")

    def search(self, base: str, arch: str = "x86_64") -> str:
        def fetch(commit, variant):
            if commit not in self.published:
                raise NOT_FOUND
            return [WHEEL], "url"

        with patch.object(wheel_utils, "fetch_metadata_for_variant", fetch):
            return wheel_utils.find_published_wheel_commit(base, "cu130", arch)

    def test_published_base_is_used_despite_local_native_edits(self):
        base = self.commit("csrc/ops.cpp")
        self.published.add(base)
        self.commit("csrc/ops.cpp")
        (self.repo / "pyproject.toml").write_text("dirty")
        self.assertEqual(self.search(base), base)

    def test_publication_lag_uses_nearest_published_ancestor(self):
        published = self.commit("vllm/b.py")
        self.published.add(published)
        self.commit("vllm/c.py")
        self.assertEqual(self.search(self.commit("vllm/d.py")), published)

    def test_compiled_changes_since_wheel_are_rejected(self):
        (published,) = self.published
        for path in wheel_utils.WHEEL_BLOCKING_PATHS:
            with self.subTest(path=path):
                self.git("reset", "-q", "--hard", published)
                file = path if "." in path else f"{path}/file"
                with self.assertRaisesRegex(ValueError, file):
                    self.search(self.commit(file))

    def test_rust_changes_since_wheel_only_warn(self):
        (published,) = self.published
        base = self.commit("rust/src/main.rs")
        with self.assertLogs("setup", "WARNING") as logs:
            self.assertEqual(self.search(base), published)
        self.assertIn("rust/src/main.rs", logs.output[0])

    def test_wrong_architecture_is_not_a_match(self):
        with self.assertRaisesRegex(ValueError, "No published precompiled wheel"):
            self.search("HEAD", arch="aarch64")

    def test_search_stops_at_depth_limit(self):
        with patch.object(wheel_utils, "WHEEL_SEARCH_DEPTH", 2):
            self.commit("vllm/b.py")
            with self.assertRaisesRegex(ValueError, "within 2 commits"):
                self.search(self.commit("vllm/c.py"))

    def test_network_and_server_errors_do_not_trigger_walkback(self):
        for error in (
            URLError("offline"),
            HTTPError("url", 503, "busy", Message(), None),
        ):
            with (
                self.subTest(error=error),
                patch.object(
                    wheel_utils, "fetch_metadata_for_variant", side_effect=error
                ) as fetch,
                self.assertRaises(type(error)),
            ):
                wheel_utils.find_published_wheel_commit("HEAD", "cu130", "x86_64")
            fetch.assert_called_once()


class TestDetermineWheelUrl(unittest.TestCase):
    def determine(self, base: str, **env: str):
        with (
            patch.dict(
                os.environ,
                {"VLLM_PRECOMPILED_WHEEL_VARIANT": "cu130", **env},
                clear=True,
            ),
            patch("platform.machine", return_value="x86_64"),
            patch.object(wheel_utils, "is_rocm_system", return_value=False),
            patch.object(
                wheel_utils, "get_base_commit_in_main_branch", return_value=base
            ),
            patch.object(
                wheel_utils,
                "fetch_metadata_for_variant",
                return_value=([WHEEL], "https://wheels.test/"),
            ) as fetch,
            patch.object(
                wheel_utils, "find_published_wheel_commit", return_value="older"
            ) as search,
        ):
            url = wheel_utils.determine_wheel_url()
        return url, fetch, search

    def test_automatic_install_downloads_the_searched_commit(self):
        url, fetch, search = self.determine("base")
        self.assertEqual(url, ("https://wheels.test/wheel.whl", "wheel.whl"))
        search.assert_called_once_with("base", "cu130", "x86_64")
        fetch.assert_called_once_with("older", "cu130")

    def test_search_is_skipped_when_it_cannot_apply(self):
        cases = {
            "nightly fallback": ("nightly", {}, {}),
            "explicit commit": (
                "base",
                {"VLLM_PRECOMPILED_WHEEL_COMMIT": "a" * 40},
                {},
            ),
            "rust only": ("base", {}, {"VLLM_USE_PRECOMPILED": False}),
            "docker": ("base", {}, {"VLLM_DOCKER_BUILD_CONTEXT": True}),
        }
        for name, (base, env, flags) in cases.items():
            with self.subTest(name), patch.dict(vars(namespace["envs"]), flags):
                _, fetch, search = self.determine(base, **env)
                search.assert_not_called()
                commit = env.get("VLLM_PRECOMPILED_WHEEL_COMMIT", base)
                fetch.assert_called_once_with(commit, "cu130")

    def test_explicit_wheel_location_bypasses_selection(self):
        for location in ("/tmp/local.whl", "https://example.com/custom.whl"):
            with self.subTest(location=location):
                url, fetch, search = self.determine(
                    "base", VLLM_PRECOMPILED_WHEEL_LOCATION=location
                )
                self.assertEqual(url, (location, None))
                fetch.assert_not_called()
                search.assert_not_called()


# Wheel members selected by the deep_gemm, fmha_sm100, and tml_fa4 patterns.
# Each prefix is three directories under the build root, so three ".."
# segments return to that root and four leave it.
_ESCAPING_PREFIXES = (
    "vllm/third_party/deep_gemm/",
    "vllm/third_party/fmha_sm100/",
    "vllm/third_party/tml_fa4/",
)
_CANARY_BYTES = b"build-tree-canary"
_SENTINEL_BYTES = b"outside-sentinel"
_LEGIT_BYTES = b"kept-in-prefix"
_KEPT_MODE = 0o640
_ATTACK_MODE = 0o707
_PLANTED_MODE = 0o750
_WRITE_EVENTS = frozenset(
    {
        "open",
        "os.chmod",
        "os.link",
        "os.makedirs",
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.symlink",
        "os.unlink",
    }
)
_watch: dict[str, Any] = {
    "on": False,
    "build": "",
    "prefix": "",
    "violations": [],
}


def _audit_filesystem_calls(event: str, args: tuple[Any, ...]) -> None:
    if not _watch["on"] or event not in _WRITE_EVENTS:
        return
    raw = _path_from_audit_args(args)
    if raw is None:
        return
    if event == "open" and not _open_writes(args):
        return
    normalized = _normalize_audit_path(raw, _watch["build"])
    if _call_stays_in_prefix(event, normalized, _watch["build"], _watch["prefix"]):
        return
    _watch["violations"].append((event, raw, normalized))


def _path_from_audit_args(args: tuple[Any, ...]) -> str | None:
    if not args:
        return None
    raw = args[0]
    if isinstance(raw, bytes):
        return os.fsdecode(raw)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, os.PathLike):
        return os.fspath(raw)
    return None


def _open_writes(args: tuple[Any, ...]) -> bool:
    if len(args) < 2:
        return False
    mode = args[1]
    if isinstance(mode, str):
        return mode[:1] in {"w", "a", "x"} or "+" in mode
    # os.open audits mode as None and puts the POSIX flags in args[2].
    if len(args) < 3 or not isinstance(args[2], int):
        return False
    write_bits = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    return bool(args[2] & write_bits)


def _normalize_audit_path(raw: str, build: str) -> str:
    if not os.path.isabs(raw):
        raw = os.path.join(build, raw)
    return os.path.normpath(raw)


def _within(root: str, path: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _call_stays_in_prefix(event: str, path: str, build: str, prefix: str) -> bool:
    """Allow package directories and files under the selected prefix only."""
    if event in {"os.mkdir", "os.makedirs"}:
        # Ancestors of the prefix have to exist before a member inside it
        # can be created. A directory beside or above the build root does not.
        if not _within(build, path):
            return False
        return _within(path, prefix) or _within(prefix, path)
    return _within(prefix, path) and path != prefix


def _install_filesystem_audit_hook() -> None:
    if getattr(_install_filesystem_audit_hook, "done", False):
        return
    sys.addaudithook(_audit_filesystem_calls)
    _install_filesystem_audit_hook.done = True


def _load_setup_envs():
    path = setup_path.parent / "vllm" / "envs.py"
    spec = importlib.util.spec_from_file_location("_vllm_envs_for_wheel_extract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _zip_member(name: str, payload: bytes, mode: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name)
    info.external_attr = (mode & 0xFFFF) << 16
    return info


def _file_signature(path: Path) -> tuple[int, int, int, bytes]:
    metadata = path.stat()
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IMODE(metadata.st_mode),
        path.read_bytes(),
    )


class TestPrecompiledWheelMemberEscape(unittest.TestCase):
    """Selected precompiled members must not be written outside their prefix.

    ``extract_precompiled_and_patch_package`` is the setup.py routine a
    ``VLLM_PRECOMPILED_WHEEL_LOCATION`` install uses. A member is rejected
    only when every directory it creates and every file it writes or chmods
    stays inside that member's own prefix. Creating a path and then removing
    it still fails: the audit hook records the call itself.
    """

    def test_parent_segments_do_not_escape_selected_prefixes(self):
        for prefix in _ESCAPING_PREFIXES:
            with self.subTest(prefix=prefix):
                self._assert_prefix_confines_parent_segments(prefix)

    def _assert_prefix_confines_parent_segments(self, prefix: str) -> None:
        _install_filesystem_audit_hook()
        envs = _load_setup_envs()
        root = Path(tempfile.mkdtemp(prefix="vllm-wheel-escape-"))
        self.addCleanup(shutil.rmtree, root, True)
        build = root / "build"
        outside = root / "outside"
        build.mkdir()
        outside.mkdir()
        canary = build / "canary.txt"
        sentinel = outside / "sentinel.bin"
        canary.write_bytes(_CANARY_BYTES)
        sentinel.write_bytes(_SENTINEL_BYTES)
        os.chmod(canary, _KEPT_MODE)
        os.chmod(sentinel, _KEPT_MODE)
        canary_before = _file_signature(canary)
        sentinel_before = _file_signature(sentinel)
        fresh = outside / "fresh"
        prefix_dir = os.path.normpath(os.path.join(build, prefix))
        components = prefix.strip("/").split("/")
        up_to_build = "../" * len(components)
        up_above_build = "../" * (len(components) + 1)
        legit_name = f"{prefix}legit.py"
        canary_name = f"{prefix}{up_to_build}canary.txt"
        sentinel_name = f"{prefix}{up_above_build}outside/sentinel.bin"
        planted_name = f"{prefix}{up_above_build}outside/fresh/planted.bin"
        wheel_path = root / "attacker.whl"
        with zipfile.ZipFile(wheel_path, "w") as archive:
            archive.writestr(
                _zip_member(legit_name, _LEGIT_BYTES, 0o644),
                _LEGIT_BYTES,
            )
            archive.writestr(
                _zip_member(canary_name, b"replaced-canary", _ATTACK_MODE),
                b"replaced-canary",
            )
            archive.writestr(
                _zip_member(sentinel_name, b"replaced-sentinel", _ATTACK_MODE),
                b"replaced-sentinel",
            )
            archive.writestr(
                _zip_member(planted_name, b"planted-outside", _PLANTED_MODE),
                b"planted-outside",
            )

        previous = os.getcwd()
        caught: Exception | None = None
        _watch["violations"] = []
        _watch["build"] = str(build)
        _watch["prefix"] = prefix_dir
        try:
            os.chdir(build)
            with patch.dict(
                os.environ,
                {"VLLM_PRECOMPILED_WHEEL_LOCATION": str(wheel_path)},
            ):
                os.environ.pop("VLLM_USE_PRECOMPILED", None)
                os.environ.pop("VLLM_USE_PRECOMPILED_RUST", None)
                # Same gates as setup.py: a wheel location turns extension
                # extraction on, and that path is not a metadata-only build.
                extract_extensions = envs.VLLM_USE_PRECOMPILED
                extract_rust_frontend = (
                    envs.VLLM_USE_PRECOMPILED or envs.VLLM_USE_PRECOMPILED_RUST
                )
                self.assertTrue(extract_extensions)
                self.assertTrue(extract_rust_frontend)
                self.assertFalse(
                    {"egg_info", "dist_info"}.intersection(sys.argv[1:])
                )
                wheel_url, download_filename = wheel_utils.determine_wheel_url()
                self.assertEqual(wheel_url, str(wheel_path))
                self.assertIsNone(download_filename)
                _watch["on"] = True
                try:
                    wheel_utils.extract_precompiled_and_patch_package(
                        wheel_url,
                        download_filename,
                        extract_extensions=extract_extensions,
                        extract_rust_frontend=extract_rust_frontend,
                    )
                except Exception as exc:
                    # Rejection may raise. A raise that already wrote or
                    # chmod'd outside the prefix is still a failure.
                    caught = exc
                finally:
                    _watch["on"] = False
        finally:
            os.chdir(previous)

        problems: list[str] = []
        if _watch["violations"]:
            rendered = ", ".join(
                f"{event} {raw!r} -> {normalized}"
                for event, raw, normalized in _watch["violations"]
            )
            problems.append(f"calls outside {prefix_dir}: {rendered}")
        if _file_signature(canary) != canary_before:
            problems.append(f"build-tree canary changed: {_file_signature(canary)}")
        if _file_signature(sentinel) != sentinel_before:
            problems.append(f"outside sentinel changed: {_file_signature(sentinel)}")
        if fresh.exists():
            problems.append(f"created {fresh}")
        legit = Path(prefix_dir) / "legit.py"
        if not legit.is_file() or legit.read_bytes() != _LEGIT_BYTES:
            problems.append(
                f"prefix member was not extracted: {legit}; extract raised {caught!r}"
            )
        self.assertFalse(problems, "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
