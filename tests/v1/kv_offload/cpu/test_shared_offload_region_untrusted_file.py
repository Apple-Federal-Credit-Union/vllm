# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refuse a pre-created CPU offload shm file that this process did not create.

These tests fail while the joiner path maps whatever inode is at
``/dev/shm/vllm_offload_<engine_id>.mmap`` (and the EC copy of that path).
They pass once that inode is not mapped, modified, or removed unless it is
the same regular file the check authorized.

The segment-size constant is lowered so the second shm file is opened
without allocating a multi-gigabyte region. The open itself is the
production ``_map_segments`` path.
"""

import mmap
import os
import stat
import subprocess
import time
import uuid
from collections.abc import Callable

import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.cpu import shared_offload_region as region_module
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

_PAGE = mmap.PAGESIZE
_NOBODY = "nobody:nogroup"
# Wall-clock bound for refusing an undersized planted file. The joiner
# currently spin-waits for 30s, so this fails while that wait is reachable.
_REFUSE_WITHIN_S = 2.0


def _engine_id() -> str:
    return uuid.uuid4().hex


def _offload_path(engine_id: str) -> str:
    return f"/dev/shm/vllm_offload_{engine_id}.mmap"


def _ec_path(engine_id: str) -> str:
    return f"/dev/shm/vllm_ec_{engine_id}.mmap"


def _remove(path: str) -> None:
    subprocess.call(["sudo", "-n", "rm", "-f", path], stdout=subprocess.DEVNULL)


def _plant(path: str, size: int, mode: int, payload: bytes, *, nobody: bool) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, mode)
    try:
        os.ftruncate(fd, size)
        if payload:
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
    finally:
        os.close(fd)
    os.chmod(path, mode)
    if nobody:
        subprocess.check_call(["sudo", "-n", "chown", _NOBODY, path])


def _snapshot(path: str) -> tuple:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        body: bytes | str = os.readlink(path)
    elif stat.S_ISREG(st.st_mode):
        with open(path, "rb") as fh:
            body = fh.read()
    else:
        body = None
    return (st.st_ino, st.st_uid, st.st_mode, st.st_size, body)


def _mapped_prefix(region: SharedOffloadRegion, offset: int, length: int) -> bytes:
    view = region.base_tensor
    return bytes(int(view[offset + i].item()) & 0xFF for i in range(length))


def _ec_prefix(region: ECSharedRegion, length: int) -> bytes:
    view = region.blocks.view(-1)
    return bytes(int(view[i].item()) & 0xFF for i in range(length))


class _LookupRace:
    """Replace a directory entry between a path check and a later open.

    The swap is a real rename. ``os.fstat`` on the fd returned by ``open``
    still describes the inode that was opened. A check that only looked up
    the path, then opens it again, maps the replacement.
    """

    def __init__(self, swaps: dict[str, str]) -> None:
        self._swaps = swaps
        self._checked: set[str] = set()
        self._real_open = os.open
        self._real_stat = os.stat
        self._real_lstat = os.lstat

    def _swap(self, path: str) -> None:
        evil = self._swaps.get(path)
        if evil is None or not os.path.lexists(evil):
            return
        held = path + ".race-held"
        if os.path.lexists(held):
            return
        os.rename(path, held)
        os.rename(evil, path)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        swaps = self._swaps

        def open_hook(file, flags, mode=0o777, *, dir_fd=None):
            if file in swaps and not (flags & os.O_EXCL):
                nofollow = bool(flags & getattr(os, "O_NOFOLLOW", 0))
                if (not nofollow) or file in self._checked:
                    self._swap(file)
            kwargs = {}
            if dir_fd is not None:
                kwargs["dir_fd"] = dir_fd
            return self._real_open(file, flags, mode, **kwargs)

        def stat_hook(path, *args, **kwargs):
            result = self._real_stat(path, *args, **kwargs)
            if path in swaps:
                self._checked.add(path)
            return result

        def lstat_hook(path, *args, **kwargs):
            result = self._real_lstat(path, *args, **kwargs)
            if path in swaps:
                self._checked.add(path)
            return result

        monkeypatch.setattr(os, "open", open_hook)
        monkeypatch.setattr(os, "stat", stat_hook)
        monkeypatch.setattr(os, "lstat", lstat_hook)


def _assert_inode(before: tuple, candidates: list[str]) -> None:
    """The planted inode is still present and unmodified, wherever it was renamed."""
    ino, uid, mode, size, body = before
    for path in candidates:
        if not os.path.lexists(path):
            continue
        st = os.lstat(path)
        if st.st_ino != ino:
            continue
        assert (st.st_uid, st.st_mode, st.st_size) == (uid, mode, size)
        if stat.S_ISLNK(st.st_mode):
            assert os.readlink(path) == body
        elif stat.S_ISREG(st.st_mode):
            with open(path, "rb") as fh:
                assert fh.read() == body
        return
    pytest.fail(f"planted inode {ino} is no longer in {candidates}")


def _assert_not_adopted(
    planted: list[tuple[tuple, list[str]]],
    construct: Callable[[], object],
    read_prefix: Callable[[object], bytes],
    marker: bytes,
    poke_offset: int,
) -> None:
    """Fail if ``construct`` maps, writes, or removes a planted inode."""
    try:
        region = construct()
    except Exception:
        region = None
    else:
        try:
            visible = read_prefix(region)
            adopted = visible == marker
            storage = getattr(region, "base_tensor", None)
            if storage is None:
                storage = region.blocks.view(-1)
            storage[poke_offset] = 0x11
            del storage
            del visible
        finally:
            region.cleanup()
        if adopted:
            pytest.fail("mapped a pre-created shm file")
    for before, candidates in planted:
        _assert_inode(before, candidates)


def _offload_region(engine_id: str, **kwargs) -> SharedOffloadRegion:
    page = kwargs.pop("cpu_page_size", _PAGE)
    chunks = kwargs.pop("num_chunks", 4)
    return SharedOffloadRegion(
        engine_id=engine_id,
        num_chunks=chunks,
        rank=0,
        kv_bytes_per_chunk=page,
        cpu_page_size=page,
        **kwargs,
    )


def test_other_user_offload_file_is_not_mapped():
    """A mode 0666 file owned by another uid must not receive KV bytes."""
    engine_id = _engine_id()
    path = _offload_path(engine_id)
    size = 4 * _PAGE
    marker = b"\xa5"
    _plant(path, size, 0o666, marker, nobody=True)
    before = _snapshot(path)
    try:
        _assert_not_adopted(
            [(before, [path])],
            lambda: _offload_region(engine_id),
            lambda region: _mapped_prefix(region, 0, len(marker)),
            marker,
            0,
        )
    finally:
        _remove(path)


def test_offload_symlink_is_not_followed():
    """The joiner open must not follow a symlink onto an attacker file."""
    engine_id = _engine_id()
    path = _offload_path(engine_id)
    target = path + ".target"
    size = 4 * _PAGE
    marker = b"\xa5"
    _plant(target, size, 0o666, marker, nobody=True)
    os.symlink(target, path)
    planted = [
        (_snapshot(path), [path, target]),
        (_snapshot(target), [path, target]),
    ]
    try:
        _assert_not_adopted(
            planted,
            lambda: _offload_region(engine_id),
            lambda region: _mapped_prefix(region, 0, len(marker)),
            marker,
            0,
        )
    finally:
        _remove(path)
        _remove(target)


def test_second_open_cannot_swap_in_another_file(monkeypatch: pytest.MonkeyPatch):
    """A path check and the mapped inode have to be the same file.

    The first file is a regular 0600 file owned by this uid. The second
    open, or a path stat followed by another open, renames a mode 0666
    file into its place. Mapping that replacement fails the test.
    """
    engine_id = _engine_id()
    path = _offload_path(engine_id)
    evil = path + ".evil"
    held = path + ".race-held"
    size = 4 * _PAGE
    marker = b"\xa5"
    _plant(path, size, 0o600, b"\x11", nobody=False)
    _plant(evil, size, 0o666, marker, nobody=False)
    evil_before = _snapshot(evil)
    race = _LookupRace({path: evil})
    race.install(monkeypatch)
    try:
        _assert_not_adopted(
            [(evil_before, [path, evil, held])],
            lambda: _offload_region(engine_id),
            lambda region: _mapped_prefix(region, 0, len(marker)),
            marker,
            0,
        )
    finally:
        for candidate in (path, evil, held):
            _remove(candidate)


def test_undersized_other_user_file_does_not_block_startup():
    """An empty planted file must be refused without the size wait."""
    engine_id = _engine_id()
    path = _offload_path(engine_id)
    _plant(path, 0, 0o666, b"", nobody=True)
    before = _snapshot(path)
    region = None
    started = time.monotonic()
    try:
        try:
            region = _offload_region(engine_id)
        except Exception:
            region = None
        elapsed = time.monotonic() - started
        if region is not None:
            region.cleanup()
        assert elapsed < _REFUSE_WITHIN_S
        _assert_inode(before, [path])
    finally:
        _remove(path)


def test_segment_open_cannot_swap_in_another_file(monkeypatch: pytest.MonkeyPatch):
    """Numbered segment files use the same untrusted-file rule as the first."""
    monkeypatch.setattr(region_module, "_MIN_SEGMENT_BYTES", _PAGE)
    engine_id = _engine_id()
    path = _offload_path(engine_id)
    segment = f"{path}.1"
    evil = segment + ".evil"
    held = segment + ".race-held"
    num_chunks = 4
    row = _PAGE
    num_segments = min(2, (num_chunks * row) // region_module._MIN_SEGMENT_BYTES)
    rows = max(1, cdiv(num_chunks, max(1, num_segments)))
    lengths = [
        min(rows, num_chunks - start) * row
        for start in range(0, max(num_chunks, 1), rows)
    ]
    assert len(lengths) == 2
    marker = b"\xa5"
    _plant(path, lengths[0], 0o600, b"\x11", nobody=False)
    _plant(segment, lengths[1], 0o600, b"\x11", nobody=False)
    _plant(evil, lengths[1], 0o666, marker, nobody=False)
    evil_before = _snapshot(evil)
    race = _LookupRace({segment: evil})
    race.install(monkeypatch)
    offset = lengths[0]
    try:
        _assert_not_adopted(
            [(evil_before, [segment, evil, held])],
            lambda: _offload_region(engine_id, num_segments=2),
            lambda region: _mapped_prefix(region, offset, len(marker)),
            marker,
            offset,
        )
    finally:
        for candidate in (path, segment, evil, held):
            _remove(candidate)


def test_other_user_ec_file_is_not_mapped():
    """EC cache shm joins an existing file with the same missing checks."""
    engine_id = _engine_id()
    path = _ec_path(engine_id)
    marker = b"\xa5"
    _plant(path, _PAGE, 0o666, marker, nobody=True)
    before = _snapshot(path)
    try:
        _assert_not_adopted(
            [(before, [path])],
            lambda: ECSharedRegion(engine_id, 1, _PAGE),
            lambda region: _ec_prefix(region, len(marker)),
            marker,
            0,
        )
    finally:
        _remove(path)


def test_ec_symlink_is_not_followed():
    engine_id = _engine_id()
    path = _ec_path(engine_id)
    target = path + ".target"
    marker = b"\xa5"
    _plant(target, _PAGE, 0o666, marker, nobody=True)
    os.symlink(target, path)
    planted = [
        (_snapshot(path), [path, target]),
        (_snapshot(target), [path, target]),
    ]
    try:
        _assert_not_adopted(
            planted,
            lambda: ECSharedRegion(engine_id, 1, _PAGE),
            lambda region: _ec_prefix(region, len(marker)),
            marker,
            0,
        )
    finally:
        _remove(path)
        _remove(target)


def test_ec_second_open_cannot_swap_in_another_file(monkeypatch: pytest.MonkeyPatch):
    engine_id = _engine_id()
    path = _ec_path(engine_id)
    evil = path + ".evil"
    held = path + ".race-held"
    marker = b"\xa5"
    _plant(path, _PAGE, 0o600, b"\x11", nobody=False)
    _plant(evil, _PAGE, 0o666, marker, nobody=False)
    evil_before = _snapshot(evil)
    race = _LookupRace({path: evil})
    race.install(monkeypatch)
    try:
        _assert_not_adopted(
            [(evil_before, [path, evil, held])],
            lambda: ECSharedRegion(engine_id, 1, _PAGE),
            lambda region: _ec_prefix(region, len(marker)),
            marker,
            0,
        )
    finally:
        for candidate in (path, evil, held):
            _remove(candidate)


def test_same_user_mode_0600_file_stays_shared():
    """A regular 0600 file owned by this uid is the real joiner path.

    Workers of one server still have to map that file. This test passes
    on the current code and must keep passing after the untrusted-file
    checks are added.
    """
    engine_id = _engine_id()
    path = _offload_path(engine_id)
    _plant(path, 4 * _PAGE, 0o600, b"\x11", nobody=False)
    region = _offload_region(engine_id)
    try:
        assert (int(region.base_tensor[0].item()) & 0xFF) == 0x11
        region.base_tensor[0] = 0x22
        with open(path, "rb") as fh:
            assert fh.read(1) == b"\x22"
    finally:
        region.cleanup()
        _remove(path)
