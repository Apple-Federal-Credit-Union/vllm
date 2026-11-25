# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security tests for the NIXL source installers' wheel cache.

The installers default to a fixed directory under /tmp and pip-install whatever
matching wheel is there. Another local user can create that directory and
choose the wheel. These tests are the expected behavior and fail until that
stops: an unsafe cache must not be installed from, a wheel discovered in the
cache must not be installed if it is replaced before pip reads it, and a wheel
planted next to one the script just built must not be the wheel that is
installed.

Source and install trees such as /tmp/ucx_source are not covered. On git 2.43
a checkout of a repository owned by another user fails before the script
runs that tree's scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UBUNTU_INSTALLER = REPO / "tools" / "install_nixl_from_source_ubuntu.py"
RUBIN_INSTALLER = REPO / "tools" / "build_nixl_with_torch215_for_rubin.py"
VERSION = "9.9.9"
MARKER_ENV = "NIXL_PTH_MARKER"
PTH_LINE = (
    "import os, pathlib; p = os.environ.get('NIXL_PTH_MARKER'); "
    "p and pathlib.Path(p).write_text('executed\\n')\n"
)
FIXED_SHARED_NAMES = {
    "wheels_cache",
    "nixl_temp_wheelhouse",
    "ucx_source",
    "nixl_source",
    "ucx_install",
    "nixl_build",
    "nixl_build_tools",
}
SHARED_TEMP_PREFIXES = ("/tmp/", "/var/tmp/", "/dev/shm/")


def write_wheel(
    path: Path,
    dist_name: str,
    version: str,
    tag: str,
    *,
    payload: bool,
) -> None:
    """Write a pip-installable wheel. A payload wheel's .pth runs at startup."""
    info = f"{dist_name}-{version}.dist-info"
    project = dist_name.replace("_", "-")
    metadata = (
        f"Metadata-Version: 2.1\nName: {project}\nVersion: {version}\n"
    ).encode()
    wheel_meta = (
        f"Wheel-Version: 1.0\nGenerator: nixl-cache-trust-test\n"
        f"Root-Is-Purelib: true\nTag: {tag}\n"
    ).encode()
    files = {
        f"{info}/METADATA": metadata,
        f"{info}/WHEEL": wheel_meta,
    }
    if payload:
        files["nixl_probe.pth"] = PTH_LINE.encode()
    record = "".join(
        f"{name},sha256={hashlib.sha256(data).hexdigest()},{len(data)}\n"
        for name, data in sorted(files.items())
    )
    record += f"{info}/RECORD,,\n"
    files[f"{info}/RECORD"] = record.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    os.chmod(path, 0o644)


def _option(command: list[str], prefix: str) -> str:
    for arg in command:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    raise AssertionError(f"missing {prefix} in {command}")


def _is_wheel_install(command: list[str]) -> bool:
    return (
        len(command) >= 4
        and command[1:3] == ["-m", "pip"]
        and "install" in command
        and command[-1].endswith(".whl")
    )


def _is_pip_wheel(command: list[str]) -> bool:
    return (
        len(command) >= 4
        and command[1:3] == ["-m", "pip"]
        and "wheel" in command
    )


def _load(path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "nixl_installer_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _drive() -> None:
    kind = os.environ["NIXL_TEST_DRIVER"]
    if kind == "print-defaults":
        os.environ.pop("WHEELS_CACHE_HOME", None)
        ubuntu = _load(UBUNTU_INSTALLER)
        rubin = _load(RUBIN_INSTALLER)
        print(ubuntu.WHEELS_CACHE_HOME)
        print(rubin.WHEELS_CACHE_HOME)
        print(rubin.TEMP_WHEEL_DIR)
        return

    evil = (
        Path(os.environ["NIXL_DRIVER_EVIL"])
        if os.environ.get("NIXL_DRIVER_EVIL")
        else None
    )

    if kind in {"ubuntu-cache", "ubuntu-cache-retarget"}:
        module = _load(UBUNTU_INSTALLER)
        if kind == "ubuntu-cache-retarget":
            link = Path(os.environ["WHEELS_CACHE_HOME"])
            real_run = module.run_command

            def run_command(command, cwd=".", env=None):
                if _is_wheel_install(command) and link.is_symlink():
                    link.unlink()
                    link.symlink_to(evil)
                return real_run(command, cwd=cwd, env=env)

            module.run_command = run_command
        module.build_and_install_prerequisites(
            argparse.Namespace(force_reinstall=False)
        )
        return

    if kind in {"rubin-cache", "rubin-cache-retarget"}:
        module = _load(RUBIN_INSTALLER)
        if kind == "rubin-cache-retarget":
            link = Path(os.environ["WHEELS_CACHE_HOME"])
            real_run = module.run_command

            def run_command(command, cwd=".", env=None):
                if _is_wheel_install(command) and link.is_symlink():
                    link.unlink()
                    link.symlink_to(evil)
                return real_run(command, cwd=cwd, env=env)

            module.run_command = run_command
        module.build_and_install_prerequisites(
            argparse.Namespace(
                force_reinstall=False,
                build_only=False,
                build_nixl_ep=False,
                skip_system_deps=True,
                ucx_path=None,
                cuda_arch_list=None,
                nixl_plugins="UCX",
            )
        )
        return

    if kind == "ubuntu-build":
        module = _load(UBUNTU_INSTALLER)
        module.ROOT_DIR = os.environ["NIXL_DRIVER_ROOT"]
        link = Path(os.environ["WHEELS_CACHE_HOME"])
        safe = Path(os.environ["NIXL_DRIVER_SAFE"])
        state = {"planted": False}
        real_check_call = subprocess.check_call

        def run_command(command, cwd=".", env=None):
            if not state["planted"]:
                state["planted"] = True
                write_wheel(
                    safe / f"nixl-{VERSION}-py3-none-any.whl",
                    "nixl",
                    VERSION,
                    "py3-none-any",
                    payload=True,
                )
                write_wheel(
                    evil / f"nixl-{VERSION}-py3-none-any.whl",
                    "nixl",
                    VERSION,
                    "py3-none-any",
                    payload=True,
                )
                os.chmod(safe, 0o777)
                os.chmod(evil, 0o777)
                if link.is_symlink():
                    link.unlink()
                    link.symlink_to(evil)
            if command[0] in {"rm", "cp"}:
                real_check_call(command, cwd=cwd or ".", env=env)
                return
            if _is_pip_wheel(command):
                wheel_dir = Path(_option(command, "--wheel-dir="))
                write_wheel(
                    wheel_dir / f"nixl-{VERSION}-cp312-cp312-linux_x86_64.whl",
                    "nixl",
                    VERSION,
                    "cp312-cp312-linux_x86_64",
                    payload=False,
                )
                return
            if command[0] == "auditwheel":
                wheel_dir = Path(_option(command, "--wheel-dir="))
                write_wheel(
                    wheel_dir
                    / f"nixl-{VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl",
                    "nixl",
                    VERSION,
                    "cp312-cp312-manylinux_2_28_x86_64",
                    payload=False,
                )
                return
            if _is_wheel_install(command):
                real_check_call(command, cwd=cwd or ".", env=env)
                return

        module.run_command = run_command
        module.build_and_install_prerequisites(
            argparse.Namespace(force_reinstall=False)
        )
        return

    if kind == "rubin-build":
        module = _load(RUBIN_INSTALLER)
        module.NIXL_DIR = os.environ["NIXL_DRIVER_NIXL_SRC"]
        module.TEMP_WHEEL_DIR = os.environ["NIXL_DRIVER_TEMP_WHEEL"]
        module.UCX_DIR = os.environ["NIXL_DRIVER_UCX"]
        module.UCX_INSTALL_DIR = os.environ["NIXL_DRIVER_UCX_INSTALL"]
        module.NIXL_BUILD_DIR = os.environ["NIXL_DRIVER_BUILD"]
        module.BUILD_TOOLS_DIR = os.environ["NIXL_DRIVER_TOOLS"]
        real_check_call = subprocess.check_call

        def run_command(command, cwd=".", env=None):
            if command[0] in {"rm", "cp"}:
                real_check_call(command, cwd=cwd or ".", env=env)
                return
            if _is_pip_wheel(command):
                wheel_dir = Path(_option(command, "--wheel-dir="))
                write_wheel(
                    wheel_dir
                    / f"nixl_cu12-{VERSION}-cp312-cp312-linux_x86_64.whl",
                    "nixl_cu12",
                    VERSION,
                    "cp312-cp312-linux_x86_64",
                    payload=False,
                )
                return
            if command[0] == "auditwheel":
                wheel_dir = Path(_option(command, "--wheel-dir="))
                write_wheel(
                    wheel_dir
                    / (
                        f"nixl_cu12-{VERSION}-cp312-cp312-"
                        "manylinux_2_28_x86_64.whl"
                    ),
                    "nixl_cu12",
                    VERSION,
                    "cp312-cp312-manylinux_2_28_x86_64",
                    payload=False,
                )
                return
            if _is_wheel_install(command):
                real_check_call(command, cwd=cwd or ".", env=env)
                return

        module.run_command = run_command
        module.build_and_install_prerequisites(
            argparse.Namespace(
                force_reinstall=False,
                build_only=False,
                build_nixl_ep=False,
                skip_system_deps=True,
                ucx_path=None,
                cuda_arch_list=None,
                nixl_plugins="UCX",
            )
        )
        return

    raise AssertionError(f"unknown driver {kind}")


if os.environ.get("NIXL_TEST_DRIVER"):
    _drive()
    raise SystemExit(0)


def _give_to_nobody(path: Path, *, symlink: bool = False) -> None:
    command = ["sudo", "-n", "chown"]
    if symlink:
        command.append("-h")
    command.extend(["-R", "nobody:nogroup", str(path)])
    subprocess.check_call(command)


def _give_to_self(path: Path) -> None:
    if not os.path.lexists(path):
        return
    subprocess.check_call(
        [
            "sudo",
            "-n",
            "chown",
            "-R",
            f"{os.getuid()}:{os.getgid()}",
            str(path),
        ]
    )


def _fingerprint(root: Path) -> list[tuple[str, int, str]]:
    target = Path(os.path.realpath(root))
    rows = []
    for dirpath, dirnames, names in os.walk(target):
        dirnames.sort()
        current = Path(dirpath)
        rows.append(
            (
                str(current.relative_to(target)) or ".",
                stat.S_IMODE(current.stat().st_mode),
                "",
            )
        )
        for name in sorted(names):
            path = current / name
            if path.is_symlink():
                rows.append((str(path.relative_to(target)), 0, "symlink"))
                continue
            data = path.read_bytes()
            rows.append(
                (
                    str(path.relative_to(target)),
                    stat.S_IMODE(path.stat().st_mode),
                    hashlib.sha256(data).hexdigest(),
                )
            )
    return rows


def _assert_not_shared_temp(path: str) -> None:
    raw = os.path.expanduser(path)
    real = os.path.realpath(raw)
    under_shared = raw.startswith(SHARED_TEMP_PREFIXES) or real.startswith(
        SHARED_TEMP_PREFIXES
    )
    if not under_shared:
        home = os.path.realpath(os.path.expanduser("~"))
        if real != home and not real.startswith(home + os.sep):
            raise AssertionError(
                f"default cache {path!r} is neither a private directory "
                f"under the user home nor a private temp directory"
            )
        return
    if not os.path.lexists(real):
        raise AssertionError(
            f"{path!r} is a fixed path under a world-writable directory; "
            "another local user can create it before the installer runs"
        )
    info = os.lstat(real)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise AssertionError(f"{path!r} is not a private directory")
    mode = stat.S_IMODE(info.st_mode)
    if info.st_uid != os.getuid() or mode & 0o077:
        raise AssertionError(
            f"{path!r} is under a shared temp and is writable by others"
        )
    if os.path.basename(real) in FIXED_SHARED_NAMES:
        raise AssertionError(
            f"{path!r} uses a predictable name another user can create first"
        )


def _make_venv(path: Path) -> Path:
    subprocess.check_call([sys.executable, "-m", "venv", str(path)])
    return path / "bin" / "python"


def _child_env(cache: Path | None, extra: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    env["NIXL_VERSION"] = VERSION
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env.pop(MARKER_ENV, None)
    env.pop("NIXL_TEST_DRIVER", None)
    if cache is None:
        env.pop("WHEELS_CACHE_HOME", None)
    else:
        env["WHEELS_CACHE_HOME"] = str(cache)
    if extra:
        env.update(extra)
    return env


def _run_driver(
    python: Path,
    kind: str,
    cache: Path | None,
    extra: dict[str, str] | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    env = _child_env(cache, extra)
    env["NIXL_TEST_DRIVER"] = kind
    return subprocess.run(
        [str(python), str(Path(__file__).resolve())],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _payload_installed(python: Path, marker: Path) -> bool:
    env = os.environ.copy()
    env[MARKER_ENV] = str(marker)
    subprocess.run(
        [str(python), "-c", "import sys"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if marker.exists():
        return True
    listed = subprocess.check_output(
        [
            str(python),
            "-c",
            "import site\nprint('\\n'.join(site.getsitepackages()))",
        ],
        text=True,
    )
    for directory in listed.splitlines():
        for pth in Path(directory).glob("*.pth"):
            if "NIXL_PTH_MARKER" in pth.read_text():
                return True
    return False


def _assert_clean(proc: subprocess.CompletedProcess[str], python: Path, marker: Path):
    if _payload_installed(python, marker):
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        tail = "\n".join(output.splitlines()[-80:])
        raise AssertionError(
            "the installer installed a planted wheel and its .pth ran on "
            f"interpreter startup\n{tail}"
        )


def _sorted_last_is_pure(names: list[str]) -> None:
    if sorted(names)[-1] != names[-1]:
        raise AssertionError(f"expected the pure wheel to sort last: {names}")


class NixlWheelCacheTrustTest(unittest.TestCase):
    def test_default_cache_paths_are_not_shared_temp_locations(self):
        proc = _run_driver(Path(sys.executable), "print-defaults", None)
        self.assertEqual(
            proc.returncode,
            0,
            proc.stderr,
        )
        ubuntu_cache, rubin_cache, temp_wheel = proc.stdout.splitlines()
        for label, path in (
            ("ubuntu", ubuntu_cache),
            ("rubin", rubin_cache),
            ("temp wheel", temp_wheel),
        ):
            with self.subTest(cache=label):
                _assert_not_shared_temp(path)

    def test_ubuntu_does_not_install_from_foreign_default_cache(self):
        cache = Path("/tmp/wheels_cache")
        parked = None
        if os.path.lexists(cache):
            parked = Path(tempfile.mkdtemp(prefix="nixl-cache-parked-")) / "old"
            try:
                cache.rename(parked)
            except OSError:
                subprocess.check_call(
                    ["sudo", "-n", "mv", str(cache), str(parked)]
                )
        venv = Path(tempfile.mkdtemp(prefix="nixl-venv-"))
        marker = venv / "marker"
        try:
            cache.mkdir(mode=0o755)
            write_wheel(
                cache / f"nixl-{VERSION}-py3-none-any.whl",
                "nixl",
                VERSION,
                "py3-none-any",
                payload=True,
            )
            os.chmod(cache, 0o755)
            before = _fingerprint(cache)
            _give_to_nobody(cache)
            python = _make_venv(venv / "venv")
            proc = _run_driver(python, "ubuntu-cache", None)
            self.assertEqual(before, _fingerprint(cache))
            _assert_clean(proc, python, marker)
        finally:
            _give_to_self(cache)
            if os.path.lexists(cache):
                subprocess.check_call(["sudo", "-n", "rm", "-rf", str(cache)])
            if parked is not None and parked.exists():
                try:
                    parked.rename(cache)
                except OSError:
                    subprocess.check_call(
                        ["sudo", "-n", "mv", str(parked), str(cache)]
                    )

    def test_ubuntu_does_not_install_from_world_or_group_writable_cache(self):
        for mode in (0o777, 0o775):
            with self.subTest(mode=oct(mode)):
                self._reject_ubuntu_mode(mode)

    def test_ubuntu_does_not_install_symlink_to_foreign_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            foreign = root / "foreign"
            foreign.mkdir(mode=0o755)
            write_wheel(
                foreign / f"nixl-{VERSION}-py3-none-any.whl",
                "nixl",
                VERSION,
                "py3-none-any",
                payload=True,
            )
            os.chmod(foreign, 0o755)
            _give_to_nobody(foreign)
            link = root / "cache"
            link.symlink_to(foreign)
            try:
                self._install_and_expect_clean(
                    "ubuntu-cache",
                    link,
                    {},
                )
            finally:
                _give_to_self(foreign)

    def test_ubuntu_does_not_install_wheel_replaced_after_discovery(self):
        pure = f"nixl-{VERSION}-py3-none-any.whl"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = root / "safe"
            evil = root / "evil"
            safe.mkdir(mode=0o700)
            evil.mkdir(mode=0o700)
            write_wheel(
                safe / pure,
                "nixl",
                VERSION,
                "py3-none-any",
                payload=False,
            )
            write_wheel(
                evil / pure,
                "nixl",
                VERSION,
                "py3-none-any",
                payload=True,
            )
            link = root / "cache"
            link.symlink_to(safe)
            os.chmod(safe, 0o700)
            self._install_and_expect_clean(
                "ubuntu-cache-retarget",
                link,
                {"NIXL_DRIVER_EVIL": str(evil)},
            )

    def test_ubuntu_build_does_not_install_wheel_planted_during_build(self):
        names = [
            f"nixl-{VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl",
            f"nixl-{VERSION}-py3-none-any.whl",
        ]
        _sorted_last_is_pure(names)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = root / "safe"
            evil = root / "evil"
            safe.mkdir(mode=0o700)
            evil.mkdir(mode=0o700)
            link = root / "cache"
            link.symlink_to(safe)
            self._install_and_expect_clean(
                "ubuntu-build",
                link,
                {
                    "NIXL_DRIVER_SAFE": str(safe),
                    "NIXL_DRIVER_EVIL": str(evil),
                    "NIXL_DRIVER_ROOT": str(root / "tools"),
                },
                timeout=90,
            )

    def test_rubin_does_not_install_from_foreign_cache(self):
        cu_pure = f"nixl_cu12-{VERSION}-py3-none-any.whl"
        cu_built = f"nixl_cu12-{VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl"
        meta = f"nixl-{VERSION}-py3-none-any.whl"
        _sorted_last_is_pure([cu_built, cu_pure])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir(mode=0o755)
            for name, dist, tag in (
                (cu_built, "nixl_cu12", "cp312-cp312-manylinux_2_28_x86_64"),
                (cu_pure, "nixl_cu12", "py3-none-any"),
                (meta, "nixl", "py3-none-any"),
            ):
                write_wheel(
                    cache / name,
                    dist,
                    VERSION,
                    tag,
                    payload=True,
                )
            os.chmod(cache, 0o755)
            before = _fingerprint(cache)
            _give_to_nobody(cache)
            try:
                venv = root / "venv"
                python = _make_venv(venv)
                marker = root / "marker"
                proc = _run_driver(python, "rubin-cache", cache)
                self.assertEqual(before, _fingerprint(cache))
                _assert_clean(proc, python, marker)
            finally:
                _give_to_self(cache)

    def test_rubin_does_not_install_wheel_replaced_after_discovery(self):
        cu = f"nixl_cu12-{VERSION}-py3-none-any.whl"
        meta = f"nixl-{VERSION}-py3-none-any.whl"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = root / "safe"
            evil = root / "evil"
            safe.mkdir(mode=0o700)
            evil.mkdir(mode=0o700)
            for directory, payload in ((safe, False), (evil, True)):
                write_wheel(
                    directory / cu,
                    "nixl_cu12",
                    VERSION,
                    "py3-none-any",
                    payload=payload,
                )
                write_wheel(
                    directory / meta,
                    "nixl",
                    VERSION,
                    "py3-none-any",
                    payload=payload,
                )
            link = root / "cache"
            link.symlink_to(safe)
            self._install_and_expect_clean(
                "rubin-cache-retarget",
                link,
                {"NIXL_DRIVER_EVIL": str(evil)},
            )

    def test_rubin_repair_does_not_install_planted_wheel(self):
        planted = f"nixl_cu12-{VERSION}-py3-none-any.whl"
        built = f"nixl_cu12-{VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl"
        _sorted_last_is_pure([built, planted])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "nixl_source"
            self._git_repo_with_tags(source)
            temp_wheel = root / "temp_wheel"
            repaired = temp_wheel / "repaired"
            repaired.mkdir(parents=True)
            os.chmod(temp_wheel, 0o777)
            os.chmod(repaired, 0o777)
            write_wheel(
                repaired / planted,
                "nixl_cu12",
                VERSION,
                "py3-none-any",
                payload=True,
            )
            cache = root / "cache"
            cache.mkdir(mode=0o700)
            for path in (
                root / "ucx",
                root / "ucx_install",
                root / "build",
                root / "tools",
            ):
                path.mkdir()
            self._install_and_expect_clean(
                "rubin-build",
                cache,
                {
                    "NIXL_DRIVER_NIXL_SRC": str(source),
                    "NIXL_DRIVER_TEMP_WHEEL": str(temp_wheel),
                    "NIXL_DRIVER_UCX": str(root / "ucx"),
                    "NIXL_DRIVER_UCX_INSTALL": str(root / "ucx_install"),
                    "NIXL_DRIVER_BUILD": str(root / "build"),
                    "NIXL_DRIVER_TOOLS": str(root / "tools"),
                },
                timeout=90,
            )

    def _reject_ubuntu_mode(self, mode: int) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            write_wheel(
                cache / f"nixl-{VERSION}-py3-none-any.whl",
                "nixl",
                VERSION,
                "py3-none-any",
                payload=True,
            )
            os.chmod(cache, mode)
            before = _fingerprint(cache)
            venv = root / "venv"
            python = _make_venv(venv)
            marker = root / "marker"
            proc = _run_driver(python, "ubuntu-cache", cache)
            self.assertEqual(before, _fingerprint(cache))
            _assert_clean(proc, python, marker)

    def _install_and_expect_clean(
        self,
        kind: str,
        cache: Path,
        extra: dict[str, str],
        timeout: int = 60,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = _make_venv(root / "venv")
            marker = root / "marker"
            proc = _run_driver(python, kind, cache, extra, timeout=timeout)
            _assert_clean(proc, python, marker)

    def _git_repo_with_tags(self, path: Path) -> None:
        path.mkdir()
        git = ["git", "-C", str(path)]
        subprocess.check_call(git + ["init", "-q"])
        subprocess.check_call(
            git
            + [
                "-c",
                "user.email=nixl-test@example.com",
                "-c",
                "user.name=nixl-test",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "init",
            ]
        )
        subprocess.check_call(git + ["tag", f"v{VERSION}"])
        subprocess.check_call(git + ["tag", VERSION])


if __name__ == "__main__":
    unittest.main()
