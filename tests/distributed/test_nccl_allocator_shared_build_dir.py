# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-temp NCCL allocator build must not load another user's library.

``compile_nccl_allocator`` passes ``tempfile.gettempdir()`` to
``torch.utils.cpp_extension.load_inline`` and then
``CUDAPluggableAllocator`` maps ``nccl_allocator.so`` from that directory.
``get_nccl_mem_pool`` is the product entry. ``nccl_symm_mem_context`` and
``CudaCommunicator`` (all-reduce and symmetric scratch) call it and are not
invoked directly here.

The child processes below are the test, not a stand-in for the loader.
They call ``get_nccl_mem_pool``. Stubs exist only so the first link can
finish on a machine without the CUDA wheel's ``libc10_cuda``,
``libtorch_cuda``, and ``libcudart`` (added in torch 2.14.0
``cpp_extension._prepare_ldflags``). ``CUDA_HOME`` is set because a CPU
torch build leaves that global unset; a CUDA build of the same torch
release sets it from the toolkit. ``is_cuda`` is forced on because this
host has no GPU, and the vulnerable branch is the CUDA branch.
"""

from __future__ import annotations

import json
import os
import pwd
import stat
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
from pathlib import Path

import pytest
import torch.utils.cpp_extension as cpp_extension

import vllm.distributed.device_communicators.pynccl_allocator as allocator

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(os.environ.get("VLLM_TEST_PYTHON", sys.executable))
ATTACKER = "ncclatk"
VICTIM = "ncclvic"

# Constructor only writes a marker file. The path comes from the environment
# so the test never embeds a credential.
_PAYLOAD_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
__attribute__((constructor)) static void nccl_alloc_hijack_init(void) {
    const char *path = getenv("NCCL_ALLOC_MARKER");
    if (path == NULL) {
        return;
    }
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        return;
    }
    fputs("loaded\n", file);
    fclose(file);
}
"""

_STUB_NCCL_H = r"""
#include <stddef.h>
typedef int ncclResult_t;
#ifdef __cplusplus
extern "C" {
#endif
ncclResult_t ncclMemAlloc(void **ptr, size_t size);
ncclResult_t ncclMemFree(void *ptr);
#ifdef __cplusplus
}
#endif
"""

_STUB_NCCL_C = r"""
#include <stdlib.h>
int ncclMemAlloc(void **ptr, size_t size) {
    *ptr = malloc(size ? size : 1);
    return 0;
}
int ncclMemFree(void *ptr) {
    free(ptr);
    return 0;
}
"""

_EMPTY_C = "static int nccl_alloc_stub_anchor;\n"


def _run(
    cmd: list[str], *, check: bool = True, **kwargs
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        **kwargs,
    )


def _ensure_user(name: str) -> int:
    probe = _run(["sudo", "-n", "id", "-u", name], check=False, capture_output=True)
    if probe.returncode != 0:
        _run(
            ["sudo", "-n", "useradd", "-M", "-s", "/usr/sbin/nologin", name],
            check=True,
        )
        probe = _run(["sudo", "-n", "id", "-u", name], check=True, capture_output=True)
    return int(probe.stdout.strip())


def _gcc(*args: str) -> None:
    _run(["gcc", *args], check=True, capture_output=True)


def _prepare_toolchain(root: Path) -> dict[str, str]:
    cuda = root / "cuda"
    include = root / "include"
    lib = root / "lib"
    cuda.joinpath("bin").mkdir(parents=True)
    cuda.joinpath("include").mkdir()
    cuda.joinpath("lib64").mkdir()
    include.mkdir()
    lib.mkdir()
    nvcc = cuda / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvcc.chmod(0o755)
    (include / "nccl.h").write_text(_STUB_NCCL_H, encoding="utf-8")
    nccl_c = root / "nccl_stub.c"
    nccl_c.write_text(_STUB_NCCL_C, encoding="utf-8")
    _gcc("-shared", "-fPIC", "-o", str(lib / "libnccl.so.2"), str(nccl_c))
    empty = root / "empty.c"
    empty.write_text(_EMPTY_C, encoding="utf-8")
    for soname in ("libc10_cuda.so", "libtorch_cuda.so", "libcudart.so"):
        _gcc("-shared", "-fPIC", "-o", str(lib / soname), str(empty))
    payload_c = root / "payload.c"
    payload_c.write_text(_PAYLOAD_SRC, encoding="utf-8")
    payload = root / "payload.so"
    _gcc("-shared", "-fPIC", "-o", str(payload), str(payload_c))
    for path in (cuda, include, lib):
        path.chmod(0o755)
    payload.chmod(0o755)
    return {
        "cuda": str(cuda),
        "include": str(include),
        "lib": str(lib),
        "payload": str(payload),
    }


def _child_env(
    role: str,
    paths: dict[str, str],
    shared: Path,
    marker: Path,
    result: Path,
    request: Path,
    ack: Path,
) -> dict[str, str]:
    return {
        "PATH": f"{PYTHON.resolve().parent}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": f"/tmp/{role}-home-nccl-alloc",
        "TMPDIR": str(shared),
        "CXX": "g++",
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONNOUSERSITE": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "VLLM_NCCL_INCLUDE_PATH": paths["include"],
        "VLLM_USE_NCCL_SYMM_MEM": "1",
        "LIBRARY_PATH": paths["lib"],
        "NCCL_ALLOC_HIJACK_ROLE": role,
        "NCCL_ALLOC_SHARED": str(shared),
        "NCCL_ALLOC_CUDA_HOME": paths["cuda"],
        "NCCL_ALLOC_PAYLOAD": paths["payload"],
        "NCCL_ALLOC_MARKER": str(marker),
        "NCCL_ALLOC_RESULT": str(result),
        "NCCL_ALLOC_ATTACKER": ATTACKER,
        "NCCL_ALLOC_REQUEST": str(request),
        "NCCL_ALLOC_ACK": str(ack),
    }


def _user_argv(user: str, env: dict[str, str], *, umask: str | None) -> list[str]:
    home = env["HOME"]
    Path(home).mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o777)
    script = 'exec "$PYTHON" "$TESTFILE"'
    if umask is not None:
        script = f"umask {umask}; {script}"
    return [
        "sudo",
        "-n",
        "-u",
        user,
        "env",
        "-i",
        *[f"{key}={value}" for key, value in env.items()],
        f"PYTHON={PYTHON}",
        f"TESTFILE={Path(__file__).resolve()}",
        "bash",
        "-c",
        script,
    ]


def _as_user(
    user: str, env: dict[str, str], *, umask: str | None, timeout: float
) -> subprocess.CompletedProcess[str]:
    return _run(
        _user_argv(user, env, umask=umask),
        check=False,
        capture_output=True,
        timeout=timeout,
    )


def _swap_library(library: str) -> None:
    """Ask the attacker process to replace ``library`` before it is mapped.

    The victim cannot sudo. A process running as the attacker waits on a fifo
    and copies the payload over a file it can still write. That is the window
    between an ownership check in ``compile_nccl_allocator`` and ``dlopen``
    inside ``load_inline``.
    """
    request = os.environ["NCCL_ALLOC_REQUEST"]
    ack = os.environ["NCCL_ALLOC_ACK"]
    with open(request, "w", encoding="utf-8") as fifo:
        fifo.write(library + "\n")
    with open(ack, encoding="utf-8") as fifo:
        fifo.read()


def _run_swapper() -> None:
    request = os.environ["NCCL_ALLOC_REQUEST"]
    ack = os.environ["NCCL_ALLOC_ACK"]
    payload = os.environ["NCCL_ALLOC_PAYLOAD"]
    while True:
        with open(request, encoding="utf-8") as fifo:
            library = fifo.read().strip()
        if library in {"", "STOP"}:
            break
        try:
            with open(payload, "rb") as source, open(library, "wb") as dest:
                dest.write(source.read())
            os.utime(library, None)
        except OSError:
            pass
        with open(ack, "w", encoding="utf-8") as fifo:
            fifo.write("ok\n")


def _run_child() -> None:
    """Executed as the attacker or the victim, never under pytest collection."""
    if os.environ["NCCL_ALLOC_HIJACK_ROLE"] == "swapper":
        _run_swapper()
        return

    os.environ["TMPDIR"] = os.environ["NCCL_ALLOC_SHARED"]
    tempfile.tempdir = None

    # CUDA torch 2.14 sets this at import. The CPU wheel used in this
    # environment does not, so point it at the fixture toolkit before the
    # real loader asks for nvcc.
    cpp_extension.CUDA_HOME = os.environ["NCCL_ALLOC_CUDA_HOME"]
    allocator.current_platform.is_cuda = lambda: True
    allocator._allocator = None
    allocator._allocator_wrapper = None
    allocator._mem_pool = None
    allocator._nccl_allocator_failed_to_compile = False

    recorded: dict[str, str | None] = {"build_directory": None}
    real_load_inline = allocator.load_inline

    def load_inline_recording_then_swap(*args, **kwargs):
        build_directory = kwargs.get("build_directory")
        recorded["build_directory"] = build_directory
        if build_directory:
            library = os.path.join(build_directory, "nccl_allocator.so")
            _swap_library(library)
        return real_load_inline(*args, **kwargs)

    role = os.environ["NCCL_ALLOC_HIJACK_ROLE"]
    marker = os.environ.get("NCCL_ALLOC_MARKER", "")
    if role == VICTIM:
        allocator.load_inline = load_inline_recording_then_swap
    else:
        # The real allocator has no constructor. Drop the marker variable so a
        # later payload copy cannot be confused with this process's load.
        os.environ.pop("NCCL_ALLOC_MARKER", None)

    allocator.get_nccl_mem_pool()

    if role == ATTACKER:
        library = os.path.join(os.environ["NCCL_ALLOC_SHARED"], "nccl_allocator.so")
        if os.path.exists(library):
            subprocess.run(
                ["cp", "-f", os.environ["NCCL_ALLOC_PAYLOAD"], library],
                check=True,
            )
            os.utime(library, None)

    mapped = False
    try:
        maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
        library = os.path.join(os.environ["NCCL_ALLOC_SHARED"], "nccl_allocator.so")
        mapped = library in maps
    except OSError:
        mapped = False
    result = {
        "role": role,
        "build_directory": recorded["build_directory"],
        "marker": os.path.exists(marker),
        "mapped_shared_library": mapped,
        "shared": os.environ["NCCL_ALLOC_SHARED"],
    }
    result_path = Path(os.environ["NCCL_ALLOC_RESULT"])
    result_path.write_text(json.dumps(result), encoding="utf-8")
    result_path.chmod(0o644)


if os.environ.get("NCCL_ALLOC_HIJACK_ROLE"):
    _run_child()


def test_other_user_shared_temp_allocator_is_not_executed():
    """A library planted in the shared temp build dir must not run as the victim.

    The attacker creates the build with mode bits that let the victim take
    torch's file lock (umask 000). Default umask 022 leaves ``lock`` as 0644,
    and filelock then refuses the other user before the load. The attacker
    owns that file and can choose the mode. Immediately before the real
    ``load_inline`` dlopen, the attacker tries again, which still succeeds
    if the library or its directory is writable by someone else.
    """
    if _run(["sudo", "-n", "true"], check=False).returncode != 0:
        pytest.fail("sudo is required to run the two-account allocator test")
    python_header = Path(sysconfig.get_path("include")) / "Python.h"
    if not python_header.exists():
        pytest.fail(f"{python_header} is required to compile the inline extension")

    attacker_uid = _ensure_user(ATTACKER)
    victim_uid = _ensure_user(VICTIM)
    assert attacker_uid != victim_uid
    assert attacker_uid != os.getuid()

    root = Path(f"/tmp/vllm-nccl-alloc-{os.getpid()}")
    shared = root / "shared"
    marker = root / "marker"
    attacker_result = root / "attacker.json"
    victim_result = root / "victim.json"
    root.mkdir(parents=True)
    shared.mkdir()
    os.chmod(shared, 0o1777)
    paths = _prepare_toolchain(root)
    # Both accounts create the marker and result files here. Toolchain
    # directories stay 0755; this directory stays sticky and shared.
    os.chmod(root, 0o1777)
    request = root / "swap.request"
    ack = root / "swap.ack"
    os.mkfifo(request)
    os.mkfifo(ack)
    os.chmod(request, 0o666)
    os.chmod(ack, 0o666)

    attacker = _as_user(
        ATTACKER,
        _child_env(ATTACKER, paths, shared, marker, attacker_result, request, ack),
        umask="000",
        timeout=180,
    )
    built = shared / "nccl_allocator.so"
    ninja_file = shared / "build.ninja"
    if ninja_file.exists() and not built.exists():
        pytest.fail(
            "shared-temp build did not produce nccl_allocator.so\n"
            f"stdout:\n{attacker.stdout}\nstderr:\n{attacker.stderr}"
        )
    if attacker.returncode != 0:
        pytest.fail(
            "attacker process failed\n"
            f"stdout:\n{attacker.stdout}\nstderr:\n{attacker.stderr}"
        )

    if marker.exists():
        marker.unlink()
    swapper_proc = subprocess.Popen(
        _user_argv(
            ATTACKER,
            _child_env("swapper", paths, shared, marker, attacker_result, request, ack),
            umask=None,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        victim = _as_user(
            VICTIM,
            _child_env(VICTIM, paths, shared, marker, victim_result, request, ack),
            umask=None,
            timeout=180,
        )
    finally:
        if swapper_proc.poll() is None:
            swapper_proc.kill()
        swapper_proc.wait(timeout=30)
    if victim.returncode != 0:
        pytest.fail(
            f"victim process failed\nstdout:\n{victim.stdout}\nstderr:\n{victim.stderr}"
        )
    outcome = json.loads(victim_result.read_text(encoding="utf-8"))
    build_directory = outcome["build_directory"]

    assert outcome["marker"] is False, textwrap.dedent(
        f"""
        the victim process executed a constructor from a library prepared by
        uid {attacker_uid} ({ATTACKER})
        victim uid {victim_uid} ({pwd.getpwuid(victim_uid).pw_name})
        build_directory={build_directory}
        mapped_shared_library={outcome["mapped_shared_library"]}
        """
    )
    assert outcome["mapped_shared_library"] is False
    assert build_directory, "get_nccl_mem_pool did not reach load_inline"
    build_path = Path(build_directory)
    directory_stat = build_path.stat()
    assert directory_stat.st_uid == victim_uid
    assert stat.S_IMODE(directory_stat.st_mode) & 0o077 == 0
    assert os.path.realpath(build_path) != os.path.realpath(shared)
    library = build_path / "nccl_allocator.so"
    if library.exists():
        library_stat = library.stat()
        assert library_stat.st_uid == victim_uid
        assert stat.S_IMODE(library_stat.st_mode) & 0o022 == 0
