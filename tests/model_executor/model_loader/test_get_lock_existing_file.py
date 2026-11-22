# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A lock acquire must not truncate or chmod a file already at the lock path.

``get_lock`` asks filelock for an explicit mode on a path another local user
can place a file on. These tests fail while that acquire truncates the file
or changes its mode.
"""

import os
import stat
import uuid
from pathlib import Path

import pytest

import vllm.model_executor.model_loader.weight_utils as weight_utils
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    get_lock,
    np_cache_weights_iterator,
)
from vllm.transformers_utils import repo_utils

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

SECRET = b"victim-only-file-contents"
_STOP = "stop-after-lock"


def _secret(directory: Path) -> Path:
    path = directory / "secret"
    path.write_bytes(SECRET)
    path.chmod(0o600)
    return path


def _assert_kept(secret: Path, lock_path: Path) -> None:
    assert secret.read_bytes() == SECRET
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    assert lock_path.stat().st_ino == secret.stat().st_ino
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def _place(secret: Path, lock_path: Path, placement: str) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if placement == "hardlink":
        os.link(secret, lock_path)
        return
    alias = secret.parent / "alias"
    os.link(secret, alias)
    os.rename(alias, lock_path)


def _acquire(model: str, cache_dir: str | None, placement: str, secret: Path) -> None:
    lock = get_lock(model, cache_dir)
    lock_path = Path(lock.lock_file)
    try:
        _place(secret, lock_path, placement)
        with lock:
            assert lock.is_locked
        _assert_kept(secret, lock_path)
    finally:
        lock_path.unlink(missing_ok=True)


@pytest.mark.parametrize("placement", ["hardlink", "rename"])
def test_get_lock_keeps_file_placed_in_download_dir(tmp_path: Path, placement: str):
    """Download dir lock: a file renamed or hard-linked there stays intact."""
    secret = _secret(tmp_path)
    cache = tmp_path / "download"
    cache.mkdir()
    _acquire("org/download-model", str(cache), placement, secret)


def test_get_lock_keeps_file_placed_in_system_temp(tmp_path: Path):
    """Default lock dir is the system temp dir, and a planted file stays intact."""
    secret = _secret(tmp_path)
    model = f"org/temp-model-{uuid.uuid4().hex}"
    _acquire(model, None, "hardlink", secret)


def test_get_lock_keeps_file_swapped_onto_path_at_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Plant the file inside open, after any earlier check of the path."""
    secret = _secret(tmp_path)
    cache = tmp_path / "download"
    cache.mkdir()
    lock = get_lock("org/race-model", str(cache))
    lock_path = Path(lock.lock_file)
    target = os.path.abspath(lock.lock_file)
    real_open = os.open

    def open_and_plant(path, flags, *args, **kwargs):
        if os.path.abspath(path) == target and (
            not lock_path.exists() or lock_path.stat().st_ino != secret.stat().st_ino
        ):
            if lock_path.exists() or lock_path.is_symlink():
                lock_path.unlink()
            os.link(secret, lock_path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_and_plant)
    try:
        with lock:
            assert lock.is_locked
        _assert_kept(secret, lock_path)
    finally:
        monkeypatch.undo()
        lock_path.unlink(missing_ok=True)


def _stop_after_lock(monkeypatch: pytest.MonkeyPatch, module, method: str) -> dict:
    monkeypatch.setattr(weight_utils.huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    seen = {"called": False}

    class _Api:
        def snapshot_download(self, *args, **kwargs):
            seen["called"] = True
            raise RuntimeError(_STOP)

        def hf_hub_download(self, *args, **kwargs):
            seen["called"] = True
            raise RuntimeError(_STOP)

    monkeypatch.setattr(module, "hf_api", lambda: _Api())
    assert method in {"snapshot_download", "hf_hub_download"}
    return seen


def test_download_weights_from_hf_keeps_placed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    secret = _secret(tmp_path)
    cache = tmp_path / "download"
    cache.mkdir()
    model = "org/weights-model"
    lock_path = Path(get_lock(model, str(cache)).lock_file)
    os.link(secret, lock_path)
    seen = _stop_after_lock(monkeypatch, weight_utils, "snapshot_download")
    with pytest.raises(RuntimeError, match=_STOP):
        download_weights_from_hf(model, str(cache), ["*.safetensors"])
    assert seen["called"]
    _assert_kept(secret, lock_path)


def test_safetensors_index_download_keeps_placed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    secret = _secret(tmp_path)
    cache = tmp_path / "download"
    cache.mkdir()
    model = "org/index-model"
    lock_path = Path(get_lock(model, str(cache)).lock_file)
    os.link(secret, lock_path)
    seen = _stop_after_lock(monkeypatch, weight_utils, "hf_hub_download")
    with pytest.raises(RuntimeError, match=_STOP):
        download_safetensors_index_file_from_hf(
            model, "model.safetensors.index.json", str(cache)
        )
    assert seen["called"]
    _assert_kept(secret, lock_path)


def test_non_weight_snapshot_keeps_placed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    secret = _secret(tmp_path)
    cache = tmp_path / "download"
    cache.mkdir()
    model = f"org/snapshot-model-{uuid.uuid4().hex}"
    lock_path = Path(get_lock(model, str(cache)).lock_file)
    os.link(secret, lock_path)
    seen = _stop_after_lock(monkeypatch, repo_utils, "snapshot_download")
    with pytest.raises(RuntimeError, match=_STOP):
        repo_utils._snapshot_non_weight_files(model, "main", str(cache), None)
    assert seen["called"]
    _assert_kept(secret, lock_path)


def test_np_cache_weights_iterator_keeps_placed_file(tmp_path: Path):
    secret = _secret(tmp_path)
    cache = tmp_path / "download"
    hf_folder = tmp_path / "hf"
    np_dir = hf_folder / "np"
    np_dir.mkdir(parents=True)
    (np_dir / "weight_names.json").write_text("[]")
    model = "org/np-cache-model"
    lock_path = Path(get_lock(model, str(cache)).lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    os.link(secret, lock_path)
    assert (
        list(np_cache_weights_iterator(model, str(cache), str(hf_folder), [], False))
        == []
    )
    _assert_kept(secret, lock_path)
