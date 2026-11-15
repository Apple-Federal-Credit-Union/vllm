# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Joiner opens of the CPU encoder-cache shm file must not trust the path.

These tests fail while ``ECSharedRegion`` maps a pre-created file or symlink
without checking the inode it actually opened. They pass once that joiner
path refuses anything other than a regular file owned by the current uid,
opened without following a symlink, and checked on the file descriptor that
``mmap`` uses.

``ECCPUWorker`` is constructed here as well. Its later accelerator setup may
raise on this host; a refusal has to happen in the shared-region open, before
that setup.
"""

import os
import subprocess
import uuid
from collections.abc import Callable

import pytest
import torch

from vllm.config.ec_transfer import ECTransferConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker

pytestmark = pytest.mark.skip_global_cleanup

_OTHER_UID = 65534
_PREFIX_LEN = 8
_FILE_SIZE = 4096


class _EmbedModel:
    """Enough of a model config for ``create_ec_shared_region``."""

    dtype = torch.float16

    def get_inputs_embeds_size(self) -> int:
        return 32


def _instance_config(instance_id: str):
    parallel = ParallelConfig(distributed_executor_backend="mp")
    ec = ECTransferConfig(
        ec_connector="ECCPUConnector",
        ec_role="ec_both",
        engine_id=f"eng-{instance_id}",
        ec_connector_extra_config={"ec_cpu_bytes": 256},
    )

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.instance_id = instance_id
    cfg.parallel_config = parallel
    cfg.ec_transfer_config = ec
    cfg.model_config = _EmbedModel()
    return cfg


def _region_path(instance_id: str, dp_index: int) -> str:
    # Same formula as create_ec_shared_region + ECSharedRegion.
    return f"/dev/shm/vllm_ec_{instance_id}_dp{dp_index}.mmap"


def _payload(marker: bytes) -> bytes:
    if len(marker) < _PREFIX_LEN:
        marker = marker.ljust(_PREFIX_LEN, b"\x00")
    return marker[:_PREFIX_LEN] + b"\x5a" * (_FILE_SIZE - _PREFIX_LEN)


def _sudo(*args: str) -> None:
    subprocess.check_call(["sudo", "-n", *args])


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except PermissionError:
        _sudo("rm", "-f", path)


def _write_other_user_file(path: str, payload: bytes) -> None:
    """Create ``path`` owned by another uid and writable by this process."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o666)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    if os.geteuid() == 0:
        os.chown(path, _OTHER_UID, _OTHER_UID)
        os.chmod(path, 0o666)
        return
    _sudo("chown", f"{_OTHER_UID}:{_OTHER_UID}", path)
    _sudo("chmod", "666", path)


def _write_owned_file(path: str, payload: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _snapshot(path: str) -> os.stat_result:
    return os.lstat(path)


def _prefix(path: str) -> bytes:
    target = os.path.realpath(path) if os.path.islink(path) else path
    with open(target, "rb") as fh:
        return fh.read(_PREFIX_LEN)


def _assert_unchanged(path: str, before: os.stat_result, prefix: bytes) -> None:
    after = os.lstat(path)
    assert after.st_ino == before.st_ino
    assert after.st_uid == before.st_uid
    assert after.st_size == before.st_size
    assert (before.st_mode & 0o170000) == (after.st_mode & 0o170000)
    assert _prefix(path) == prefix


def _shutdown(obj: object) -> None:
    shutdown = getattr(obj, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return
    cleanup = getattr(obj, "cleanup", None)
    if callable(cleanup):
        cleanup()


def _mapped_prefix(obj: object) -> bytes:
    region = getattr(obj, "_region", obj)
    blocks = getattr(region, "blocks", None)
    if blocks is None:
        return b""
    raw = blocks[0, :_PREFIX_LEN].tolist()
    return bytes(int(value) & 0xFF for value in raw)


def _expect_refusal(construct: Callable[[], object], path: str, prefix: bytes) -> None:
    before = _snapshot(path)
    obj = None
    mapped = b""
    try:
        obj = construct()
        mapped = _mapped_prefix(obj)
    except OSError:
        _assert_unchanged(path, before, prefix)
        return
    except Exception as exc:
        pytest.fail(
            "untrusted shm path was opened, then "
            f"{type(exc).__name__} was raised: {exc}"
        )
    finally:
        if obj is not None:
            _shutdown(obj)
    pytest.fail(
        "untrusted shm path was mapped; prefix "
        f"{mapped!r} matches planted bytes {prefix!r}"
    )


def _expect_refusal_after_reopen_swap(
    path: str,
    swap: Callable[[], None],
    construct: Callable[[], object],
    prefix: bytes,
) -> None:
    """Refuse the inode present when the joiner ``os.open`` returns.

    ``swap`` runs inside that open, after any earlier look-up of ``path`` and
    before the descriptor is returned. A check of the path from before this
    open must not authorize the mapping.
    """
    real_open = os.open
    did_swap = False

    def racing_open(file, flags, mode=0o777, *, dir_fd=None):
        nonlocal did_swap
        if not did_swap and os.fspath(file) == path and not (flags & os.O_EXCL):
            did_swap = True
            swap()
        if dir_fd is None:
            return real_open(file, flags, mode)
        return real_open(file, flags, mode, dir_fd=dir_fd)

    os.open = racing_open
    obj = None
    mapped = b""
    try:
        try:
            obj = construct()
            mapped = _mapped_prefix(obj)
        except OSError:
            assert did_swap, "joiner path never reopened the shm file"
            after = os.lstat(path)
            assert after.st_uid == _OTHER_UID or os.path.islink(path)
            assert _prefix(path) == prefix
            if not os.path.islink(path):
                assert after.st_size == _FILE_SIZE
            return
        except Exception as exc:
            pytest.fail(
                "swapped shm path was opened, then "
                f"{type(exc).__name__} was raised: {exc}"
            )
        pytest.fail(
            "swapped shm path was mapped; prefix "
            f"{mapped!r} matches planted bytes {prefix!r}"
        )
    finally:
        os.open = real_open
        if obj is not None:
            _shutdown(obj)


def test_region_refuses_other_user_regular_file():
    """A regular file another uid planted at the shm path is not mapped."""
    engine_id = f"other-{uuid.uuid4()}"
    path = f"/dev/shm/vllm_ec_{engine_id}.mmap"
    payload = _payload(b"OTHUSER!")
    _write_other_user_file(path, payload)
    try:
        _expect_refusal(
            lambda: ECSharedRegion(
                engine_id=engine_id,
                num_blocks=4,
                block_size_bytes=64,
            ),
            path,
            payload[:_PREFIX_LEN],
        )
    finally:
        _remove(path)


def test_region_refuses_symlink_to_other_user_file():
    """A symlink planted at the shm path is not followed into a shared map."""
    engine_id = f"link-{uuid.uuid4()}"
    path = f"/dev/shm/vllm_ec_{engine_id}.mmap"
    target = f"{path}.target"
    payload = _payload(b"SYMLINK!")
    _write_other_user_file(target, payload)
    os.symlink(target, path)
    try:
        _expect_refusal(
            lambda: ECSharedRegion(
                engine_id=engine_id,
                num_blocks=4,
                block_size_bytes=64,
            ),
            path,
            payload[:_PREFIX_LEN],
        )
    finally:
        _remove(path)
        _remove(target)


def test_region_refuses_other_user_file_swapped_in_at_reopen():
    """A same-uid file replaced by another uid's file before reopen is refused.

    The replacement happens inside the joiner ``os.open``, after any earlier
    look-up of the path and before that open returns the descriptor to map.
    """
    engine_id = f"swap-{uuid.uuid4()}"
    path = f"/dev/shm/vllm_ec_{engine_id}.mmap"
    attacker = f"{path}.attacker"
    owned = _payload(b"OWNEDOK!")
    planted = _payload(b"SWAPPED!")
    _write_owned_file(path, owned)
    _write_other_user_file(attacker, planted)

    def swap() -> None:
        os.unlink(path)
        os.link(attacker, path)

    try:
        _expect_refusal_after_reopen_swap(
            path,
            swap,
            lambda: ECSharedRegion(
                engine_id=engine_id,
                num_blocks=4,
                block_size_bytes=64,
            ),
            planted[:_PREFIX_LEN],
        )
    finally:
        _remove(path)
        _remove(attacker)


def test_region_refuses_symlink_swapped_in_at_reopen():
    """A same-uid file replaced by a symlink before reopen is not followed."""
    engine_id = f"swlink-{uuid.uuid4()}"
    path = f"/dev/shm/vllm_ec_{engine_id}.mmap"
    target = f"{path}.target"
    owned = _payload(b"OWNEDOK!")
    planted = _payload(b"LINKSWAP")
    _write_owned_file(path, owned)
    _write_other_user_file(target, planted)

    def swap() -> None:
        os.unlink(path)
        os.symlink(target, path)

    try:
        _expect_refusal_after_reopen_swap(
            path,
            swap,
            lambda: ECSharedRegion(
                engine_id=engine_id,
                num_blocks=4,
                block_size_bytes=64,
            ),
            planted[:_PREFIX_LEN],
        )
    finally:
        _remove(path)
        _remove(target)


def test_create_ec_shared_region_refuses_file_named_from_instance_id():
    """The path derived from ``instance_id`` is not mapped when another uid owns it."""
    instance_id = f"inst-{uuid.uuid4()}"
    cfg = _instance_config(instance_id)
    path = _region_path(instance_id, cfg.parallel_config.data_parallel_index)
    payload = _payload(b"INSTPATH")
    _write_other_user_file(path, payload)
    try:
        _expect_refusal(
            lambda: create_ec_shared_region(cfg),
            path,
            payload[:_PREFIX_LEN],
        )
    finally:
        _remove(path)


def test_scheduler_refuses_file_named_from_instance_id():
    """``ECCPUScheduler`` construction uses that same untrusted path."""
    instance_id = f"sched-{uuid.uuid4()}"
    cfg = _instance_config(instance_id)
    path = _region_path(instance_id, cfg.parallel_config.data_parallel_index)
    payload = _payload(b"SCHEDMAP")
    _write_other_user_file(path, payload)
    try:
        _expect_refusal(
            lambda: ECCPUScheduler(cfg),
            path,
            payload[:_PREFIX_LEN],
        )
    finally:
        _remove(path)


def test_worker_refuses_file_named_from_instance_id():
    """``ECCPUWorker`` construction uses that same untrusted path."""
    instance_id = f"worker-{uuid.uuid4()}"
    cfg = _instance_config(instance_id)
    path = _region_path(instance_id, cfg.parallel_config.data_parallel_index)
    payload = _payload(b"WORKRMAP")
    _write_other_user_file(path, payload)
    try:
        _expect_refusal(
            lambda: ECCPUWorker(cfg),
            path,
            payload[:_PREFIX_LEN],
        )
    finally:
        _remove(path)
