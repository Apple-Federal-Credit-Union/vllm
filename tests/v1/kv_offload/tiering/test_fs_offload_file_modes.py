# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem KV offload must not publish cached blocks to other users.

The secondary filesystem tier writes KV blocks, its run config, and the
O_DIRECT probe into ``root_dir``. With a normal umask (022) those creates
must still be owner-only: files 0600 and directories 0700. These tests
fail while a block, config, probe file, or any directory the tier creates
is readable or traversable by another uid.
"""

import importlib.util
import json
import mmap
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    OffloadingSpec,
    ReqContext,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OTHER_USER_PYTHON = "/usr/bin/python3"
_BLOCK_MARK = (b"KVBLOCK0", b"KVBLOCK1")


class _OffloadingSpec(OffloadingSpec):
    def get_manager(self):
        raise NotImplementedError

    def get_worker(self, kv_caches):
        raise NotImplementedError


def _spec() -> OffloadingSpec:
    return _OffloadingSpec(
        OffloadingConfig(
            groups=(),
            worker_kv_bytes_per_block=0,
            enable_kv_cache_events=False,
            extra_config={},
            engine_id="perm-test",
            model=OffloadingModelConfig(name="perm-model", dtype="uint8"),
            cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
            parallel=OffloadingParallelConfig(
                rank=0,
                world_size=1,
                tp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=1,
                data_parallel_index=0,
                data_parallel_size=1,
                data_parallel_rank_local=None,
                is_parallelism_agnostic=False,
            ),
        )
    )


def _load_compiled_fs_io_c():
    """Compile ``csrc/fs_io.cpp`` and load that module.

    The store path that ships in the package is this extension. Building it
    from the tree under test keeps the assertion on the current source.
    """
    src = _REPO_ROOT / "csrc" / "fs_io.cpp"
    cache = Path(tempfile.gettempdir()) / "vllm-fs-io-c-mode-test"
    cache.mkdir(exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so_path = cache / f"fs_io_C{suffix}"
    if not so_path.exists() or so_path.stat().st_mtime < src.stat().st_mtime:
        include = sysconfig.get_path("include")
        proc = subprocess.run(
            [
                "g++",
                "-shared",
                "-fPIC",
                "-O2",
                "-std=c++17",
                f"-I{include}",
                str(src),
                "-o",
                str(so_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr or proc.stdout)
    spec = importlib.util.spec_from_file_location("vllm.fs_io_C", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def io_mod():
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    previous_flag = io_mod._HAS_FSIO_C
    previous_store = getattr(io_mod, "batch_store_block_C", None)
    previous_load = getattr(io_mod, "batch_load_block_C", None)
    extension = _load_compiled_fs_io_c()
    sys.modules["vllm.fs_io_C"] = extension
    io_mod._HAS_FSIO_C = True
    io_mod.batch_store_block_C = extension.batch_store_block
    io_mod.batch_load_block_C = extension.batch_load_block
    try:
        yield io_mod
    finally:
        io_mod._HAS_FSIO_C = previous_flag
        if previous_store is None:
            if hasattr(io_mod, "batch_store_block_C"):
                delattr(io_mod, "batch_store_block_C")
        else:
            io_mod.batch_store_block_C = previous_store
        if previous_load is None:
            if hasattr(io_mod, "batch_load_block_C"):
                delattr(io_mod, "batch_load_block_C")
        else:
            io_mod.batch_load_block_C = previous_load


@pytest.fixture
def normal_umask():
    """Pin umask 022 so a mode of 0644 is not hidden by a tighter umask."""
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture
def shared_root(normal_umask):
    """Operator root other users can traverse, as on a shared volume."""
    root = tempfile.mkdtemp(prefix="vllm-kv-offload-perm-")
    os.chmod(root, 0o755)
    current = root
    while current != os.path.sep:
        info = os.lstat(current)
        assert not stat.S_ISLNK(info.st_mode), current
        mode = stat.S_IMODE(info.st_mode)
        assert mode & stat.S_IXOTH, (
            f"{current} is mode {mode:04o}; other users cannot reach the root"
        )
        current = os.path.dirname(current)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _aligned_blocks(num_blocks: int, block_size: int) -> np.ndarray:
    page = mmap.PAGESIZE
    backing = np.zeros(num_blocks * block_size + page, dtype=np.uint8)
    start = (-int(backing.ctypes.data)) % page
    view = backing[start : start + num_blocks * block_size]
    return view.reshape(num_blocks, block_size)


def _other_user_access(paths: list[str]) -> dict[str, str]:
    """Ask uid nobody to list or read each path.

    Returns ``denied`` when the kernel refuses, ``list`` or ``read`` when
    it allows the operation, or raises if sudo itself cannot run.
    """
    if not paths:
        return {}
    script = (
        "import json, os, sys\n"
        "out = {}\n"
        "for path in sys.argv[1:]:\n"
        "    try:\n"
        "        if os.path.isdir(path):\n"
        "            os.listdir(path)\n"
        "            out[path] = 'list'\n"
        "        else:\n"
        "            open(path, 'rb').read(16)\n"
        "            out[path] = 'read'\n"
        "    except PermissionError:\n"
        "        out[path] = 'denied'\n"
        "json.dump(out, sys.stdout)\n"
    )
    result = subprocess.run(
        ["sudo", "-n", "-u", "nobody", "--", _OTHER_USER_PYTHON, "-c", script, *paths],
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise AssertionError(
            "could not run the other-user access check: " + (detail or "sudo failed")
        )
    report = json.loads(result.stdout.decode("utf-8"))
    missing = [path for path in paths if path not in report]
    if missing:
        raise AssertionError(f"other-user check omitted {missing}")
    return report


def _privacy_violations(root: str) -> list[str]:
    """Mode and other-uid access violations under ``root`` (not ``root``)."""
    files: list[str] = []
    directories: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            directories.append(os.path.join(dirpath, name))
        for name in filenames:
            files.append(os.path.join(dirpath, name))
    access = _other_user_access(directories + files)
    violations = []
    for path in directories:
        mode = stat.S_IMODE(os.lstat(path).st_mode)
        if mode != 0o700 or access[path] != "denied":
            violations.append(
                f"directory {path} mode {mode:04o} other-user {access[path]}"
            )
    for path in files:
        mode = stat.S_IMODE(os.lstat(path).st_mode)
        if mode != 0o600 or access[path] != "denied":
            violations.append(f"file {path} mode {mode:04o} other-user {access[path]}")
    return violations


def _fill_blocks() -> np.ndarray:
    blocks = _aligned_blocks(len(_BLOCK_MARK), mmap.PAGESIZE)
    for index, mark in enumerate(_BLOCK_MARK):
        blocks[index, : len(mark)] = np.frombuffer(mark, dtype=np.uint8)
    return blocks


@pytest.mark.parametrize("use_c_extension", [False, True])
@pytest.mark.parametrize("use_o_direct", [False, True])
def test_batch_store_keeps_kv_blocks_private(
    shared_root, io_mod, use_c_extension, use_o_direct
):
    """Stored blocks and every directory above them are owner-only."""
    io_mod._HAS_FSIO_C = use_c_extension
    blocks = _fill_blocks()
    block_size = blocks.shape[1]
    paths = [
        os.path.join(shared_root, "aaa", "bb", "one.bin"),
        os.path.join(shared_root, "ccc", "dd", "two.bin"),
    ]
    io_mod.batch_store_block(
        paths,
        memoryview(blocks),
        [0, block_size],
        block_size,
        use_o_direct,
    )
    for index, path in enumerate(paths):
        assert Path(path).read_bytes() == blocks[index].tobytes()
    violations = _privacy_violations(shared_root)
    assert not violations, "KV blocks are exposed:\n" + "\n".join(violations)


def test_o_direct_probe_file_is_private(shared_root, io_mod, monkeypatch):
    """The direct-I/O probe file is owner-only for the moment it exists."""
    observed: dict[str, object] = {}
    real_remove = os.remove

    def spy_remove(path, *args, **kwargs):
        if os.path.basename(path).startswith(".o_direct_probe"):
            observed["path"] = path
            observed["mode"] = stat.S_IMODE(os.lstat(path).st_mode)
            observed["access"] = _other_user_access([path])[path]
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", spy_remove)
    io_mod.probe_o_direct(shared_root)

    assert os.O_DIRECT, "this host has no O_DIRECT, so the probe path did not run"
    path = observed.get("path")
    assert isinstance(path, str), "probe_o_direct did not create a file"
    assert not os.path.exists(path)
    violations = []
    if observed["mode"] != 0o600 or observed["access"] != "denied":
        violations.append(
            f"probe file {path} mode {observed['mode']:04o} "
            f"other-user {observed['access']}"
        )
    assert not violations, "probe file is exposed:\n" + "\n".join(violations)


def test_filesystem_tier_offload_is_unreadable_by_other_users(
    shared_root, io_mod, monkeypatch
):
    """A tier store publishes hash-named KV files no other uid can read."""
    assert io_mod._HAS_FSIO_C, "tier store did not run the C extension"
    violations: list[str] = []
    real_remove = os.remove

    def spy_remove(path, *args, **kwargs):
        if os.path.basename(path).startswith(".o_direct_probe"):
            mode = stat.S_IMODE(os.lstat(path).st_mode)
            access = _other_user_access([path])[path]
            if mode != 0o600 or access != "denied":
                violations.append(
                    f"probe file {path} mode {mode:04o} other-user {access}"
                )
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", spy_remove)
    blocks = _fill_blocks()
    tier = FileSystemTierManager(
        offloading_spec=_spec(),
        primary_kv_view=memoryview(blocks),
        tier_type="fs",
        root_dir=shared_root,
        n_read_threads=1,
        n_write_threads=1,
    )
    try:
        keys = [
            make_offload_key(bytes([0x11]) * 32, 0),
            make_offload_key(bytes([0x22]) * 32, 1),
        ]
        tier.submit_store(
            TransferJob(
                job_id=1,
                keys=keys,
                chunk_ids=np.array([0, 1], dtype=np.int64),
                is_promotion=False,
                req_context=ReqContext(req_id="perm-test"),
            )
        )
        tier.drain_jobs()
        results = list(tier.get_finished_jobs())
        assert len(results) == 1 and results[0].success
        for index, key in enumerate(keys):
            path = tier.file_mapper.get_file_name(key)
            assert get_offload_block_hash(key).hex() in path
            assert Path(path).read_bytes() == blocks[index].tobytes()
        violations.extend(_privacy_violations(shared_root))
    finally:
        tier.shutdown()
    assert not violations, "filesystem tier is exposed:\n" + "\n".join(violations)
