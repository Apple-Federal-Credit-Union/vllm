# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""npcache must not read or write checkpoint names outside its cache directory.

These tests exercise ``DefaultModelLoader`` with ``load_format="npcache"``.
They fail while tensor names are joined onto the cache directory and opened
without a containment check.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

_SENTINEL = b"sentinel-bytes-not-a-numpy-file\n"
_TENSOR = torch.tensor([1.25, 2.5, 3.75], dtype=torch.float32)
_SECRET = np.arange(4, dtype=np.float32)


def _loader() -> DefaultModelLoader:
    return DefaultModelLoader(
        LoadConfig(load_format="npcache", use_tqdm_on_load=False)
    )


def _write_bin(model_dir: Path, names: list[str]) -> None:
    state = {name: _TENSOR.clone() for name in names}
    torch.save(state, model_dir / "pytorch_model.bin")


def _load(model_dir: Path) -> list[tuple[str, torch.Tensor]]:
    source = DefaultModelLoader.Source(
        model_or_path=str(model_dir),
        revision=None,
        fall_back_to_pt=False,
    )
    return list(_loader()._get_weights_iterator(source))


def _assert_rejected_without_touching(
    model_dir: Path,
    protected: Path,
    original: bytes | None,
) -> None:
    """The load must fail, and `protected` must not be created or replaced."""
    rejected = False
    try:
        _load(model_dir)
    except Exception:
        rejected = True
    if original is None:
        assert not protected.exists(), f"{protected} was created"
    else:
        assert protected.is_file()
        assert protected.read_bytes() == original, f"{protected} was replaced"
    assert rejected, "npcache accepted a tensor name that escapes the cache"


def test_parent_directory_tensor_name_does_not_replace_outside_file(tmp_path: Path):
    """A `..` tensor name must not truncate a file outside the np cache.

    The safe name is stored first so a check that only inspects the first key
    still has to reject the later name before opening it.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(_SENTINEL)
    original = victim.read_bytes()
    _write_bin(model_dir, ["layer.weight", "../../victim"])

    _assert_rejected_without_touching(model_dir, victim, original)


def test_absolute_tensor_name_does_not_replace_outside_file(tmp_path: Path):
    """An absolute tensor name must not replace a file outside the np cache."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    victim = tmp_path / "absolute-victim"
    victim.write_bytes(_SENTINEL)
    original = victim.read_bytes()
    _write_bin(model_dir, [str(victim)])

    _assert_rejected_without_touching(model_dir, victim, original)


def test_parent_directory_tensor_name_does_not_create_outside_file(tmp_path: Path):
    """A `..` tensor name must not create a new file outside the np cache."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    victim = tmp_path / "created-victim"
    assert not victim.exists()
    _write_bin(model_dir, ["../../created-victim"])

    _assert_rejected_without_touching(model_dir, victim, None)


def test_symlink_filename_does_not_replace_outside_file(tmp_path: Path):
    """A single path segment must not follow a symlink out of the cache.

    The name has no separator and no parent segment. Checking the string and
    then opening the joined path still truncates the symlink target.
    """
    model_dir = tmp_path / "model"
    np_dir = model_dir / "np"
    np_dir.mkdir(parents=True)
    victim = tmp_path / "symlink-victim"
    victim.write_bytes(_SENTINEL)
    original = victim.read_bytes()
    (np_dir / "alias").symlink_to(victim)
    _write_bin(model_dir, ["alias"])

    _assert_rejected_without_touching(model_dir, victim, original)


def test_symlink_directory_does_not_replace_outside_file(tmp_path: Path):
    """A name with a separator must not follow a directory symlink outside."""
    model_dir = tmp_path / "model"
    np_dir = model_dir / "np"
    np_dir.mkdir(parents=True)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    victim = outside_dir / "placed"
    victim.write_bytes(_SENTINEL)
    original = victim.read_bytes()
    (np_dir / "linkdir").symlink_to(outside_dir, target_is_directory=True)
    _write_bin(model_dir, ["linkdir/placed"])

    _assert_rejected_without_touching(model_dir, victim, original)


def _seed_weight_names(model_dir: Path, names: list[str]) -> None:
    np_dir = model_dir / "np"
    np_dir.mkdir(parents=True, exist_ok=True)
    (np_dir / "weight_names.json").write_text(json.dumps(names), encoding="utf-8")
    # Present so weight discovery succeeds. The iterator skips torch.load
    # once weight_names.json exists.
    _write_bin(model_dir, ["unused.weight"])


def _assert_outside_array_not_loaded(
    model_dir: Path,
    protected: Path,
    original: bytes,
) -> None:
    rejected = False
    loaded: list[tuple[str, torch.Tensor]] = []
    try:
        loaded = _load(model_dir)
    except Exception:
        rejected = True
    assert protected.read_bytes() == original, f"{protected} was replaced"
    assert rejected, f"npcache loaded outside data: {[name for name, _ in loaded]}"
    for _name, tensor in loaded:
        assert not torch.equal(tensor, torch.from_numpy(_SECRET.copy()))


def test_weight_names_parent_reference_does_not_load_outside_array(tmp_path: Path):
    """A cached name with `..` must not np.load a file outside the cache."""
    model_dir = tmp_path / "model"
    outside = tmp_path / "outside.npy"
    np.save(outside, _SECRET)
    original = outside.read_bytes()
    np_dir = model_dir / "np"
    rel = os.path.relpath(outside, np_dir)
    assert ".." in rel
    _seed_weight_names(model_dir, [rel])

    _assert_outside_array_not_loaded(model_dir, outside, original)


def test_weight_names_absolute_path_does_not_load_outside_array(tmp_path: Path):
    """An absolute cached name must not np.load a file outside the cache."""
    model_dir = tmp_path / "model"
    outside = tmp_path / "absolute-outside.npy"
    np.save(outside, _SECRET)
    original = outside.read_bytes()
    _seed_weight_names(model_dir, [str(outside)])

    _assert_outside_array_not_loaded(model_dir, outside, original)


def test_weight_names_symlink_does_not_load_outside_array(tmp_path: Path):
    """A cached single-segment name must not follow a symlink outside."""
    model_dir = tmp_path / "model"
    outside = tmp_path / "symlink-outside.npy"
    np.save(outside, _SECRET)
    original = outside.read_bytes()
    np_dir = model_dir / "np"
    np_dir.mkdir(parents=True)
    (np_dir / "alias").symlink_to(outside)
    _seed_weight_names(model_dir, ["alias"])

    _assert_outside_array_not_loaded(model_dir, outside, original)
