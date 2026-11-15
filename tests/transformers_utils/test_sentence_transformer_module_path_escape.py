# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sentence-Transformers module paths must stay inside the model directory.

``modules.json`` is part of a checkpoint the operator serves. Its module
``path`` values are joined onto that checkpoint and then opened. These tests
load a checkpoint whose module path points at another directory — an absolute
path, a ``..`` path, or a symlink — and require that the outside ``config.json``
and weight files are never opened.

They fail while those files are opened. A fix that checks a path and then
opens it again also fails: the symlink tests retarget the link onto the
outside directory immediately before the open.
"""

import builtins
import ctypes
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import huggingface_hub.constants
import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm.config.model import ModelConfig
from vllm.model_executor.models.adapters import (
    _load_dense_weights,
    _load_st_projector,
)
from vllm.transformers_utils.config import (
    get_pooling_config,
    try_get_dense_modules,
)

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.inotify_init1.argtypes = (ctypes.c_int,)
_LIBC.inotify_init1.restype = ctypes.c_int
_LIBC.inotify_add_watch.argtypes = (
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_uint32,
)
_LIBC.inotify_add_watch.restype = ctypes.c_int

_IN_NONBLOCK = 0x00000800
_IN_OPEN = 0x00000020

_POOLING_TYPES = (
    "sentence_transformers.models.Pooling",
    "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
)
_DENSE_TYPES = (
    "sentence_transformers.models.Dense",
    "sentence_transformers.base.modules.dense.Dense",
    "pylate.models.Dense.Dense",
)
_ESCAPES = ("absolute", "parents", "symlink")

# Not BertModel's default CLS pooling, so a value copied from the outside
# file is visible on the runner config.
_OUTSIDE_POOLING = {"pooling_mode": "lasttoken"}
_OUTSIDE_POOLING_BOOL = {"pooling_mode_mean_tokens": True, "use_cache": True}
_UNRELATED_CONFIG = {"use_cache": True, "tie_word_embeddings": True}
_DENSE_CONFIG = {
    "in_features": 4,
    "out_features": 3,
    "bias": False,
    "activation_function": "",
}
_WEIGHT_SENTINEL = torch.full((3, 4), 7.5)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bert_checkpoint(model: Path) -> None:
    _write_json(
        model / "config.json",
        {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "vocab_size": 128,
            "max_position_embeddings": 32,
        },
    )


def _model_dir(tmp_path: Path, escape: str) -> Path:
    # Several parent segments, as a long "../" module path would use.
    if escape == "parents":
        model = tmp_path / "w" / "x" / "y" / "model"
    else:
        model = tmp_path / "model"
    model.mkdir(parents=True)
    return model


def _module_path(model: Path, outside: Path, escape: str, link_name: str) -> str:
    if escape == "absolute":
        return str(outside)
    if escape == "parents":
        return os.path.relpath(outside, model)
    link = model / link_name
    link.symlink_to(outside, target_is_directory=True)
    return link_name


def _modules(path: str, module_type: str) -> list[dict[str, object]]:
    return [{"idx": 1, "name": "1", "path": path, "type": module_type}]


def _save_weights(path: Path, payload: dict[str, torch.Tensor], kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "safetensors":
        save_file(payload, path)
        return
    torch.save(payload, path)


def _drain_inotify(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        try:
            chunk = os.read(fd, 4096)
        except BlockingIOError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


class _OpenWatch:
    """Fail the test if any watched file is opened."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.fd = _LIBC.inotify_init1(_IN_NONBLOCK)
        if self.fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), "inotify_init1")
        for path in paths:
            watch = _LIBC.inotify_add_watch(self.fd, os.fsencode(path), _IN_OPEN)
            if watch < 0:
                err = ctypes.get_errno()
                os.close(self.fd)
                raise OSError(err, os.strerror(err), str(path))

    def __enter__(self) -> "_OpenWatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        events = _drain_inotify(self.fd)
        os.close(self.fd)
        if events:
            opened = ", ".join(str(path) for path in self.paths)
            failure = AssertionError(
                f"opened a file outside the model directory: {opened}"
            )
            if exc is not None:
                raise failure from exc
            raise failure
        return False


@contextmanager
def _retarget_symlink_before_open(link: Path, new_target: Path) -> Iterator[None]:
    """Point ``link`` at ``new_target`` when a later open walks through it.

    Models the file changing between a containment check and the open. The
    open that still uses the original path follows the new target.
    """
    real_open = builtins.open
    real_os_open = os.open

    def retarget(file: object) -> None:
        if isinstance(file, int):
            return
        try:
            requested = Path(os.fspath(file))  # type: ignore[arg-type]
        except TypeError:
            return
        if requested != link and link not in requested.parents:
            return
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(new_target, target_is_directory=True)

    def wrapped_open(file, *args, **kwargs):
        retarget(file)
        return real_open(file, *args, **kwargs)

    def wrapped_os_open(path, flags, *args, **kwargs):
        retarget(path)
        return real_os_open(path, flags, *args, **kwargs)

    builtins.open = wrapped_open
    os.open = wrapped_os_open
    try:
        yield
    finally:
        builtins.open = real_open
        os.open = real_os_open


def _assert_pooling_ignored(config: dict | None) -> None:
    if config is None:
        return
    assert config.get("seq_pooling_type") != "LAST"
    assert config.get("seq_pooling_type") != "MEAN"
    assert config.get("tok_pooling_type") not in {"ALL", "STEP"}


def _linear() -> nn.Linear:
    linear = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        linear.weight.zero_()
    return linear


def _assert_weight_not_installed(linear: nn.Linear) -> None:
    assert not torch.equal(linear.weight.detach().cpu(), _WEIGHT_SENTINEL)


@pytest.mark.parametrize("module_type", _POOLING_TYPES)
@pytest.mark.parametrize("escape", _ESCAPES)
def test_get_pooling_config_does_not_open_outside_config(
    tmp_path: Path, module_type: str, escape: str
) -> None:
    """Pooling config.json outside the checkpoint is not read or applied."""
    outside = tmp_path / "outside"
    config_path = outside / "config.json"
    _write_json(config_path, _OUTSIDE_POOLING)
    model = _model_dir(tmp_path, escape)
    module_path = _module_path(model, outside, escape, "1_Pooling")
    _write_json(model / "modules.json", _modules(module_path, module_type))

    with _OpenWatch([config_path]):
        config = get_pooling_config(str(model))

    _assert_pooling_ignored(config)


def test_get_pooling_config_boolean_field_does_not_open_outside_config(
    tmp_path: Path,
) -> None:
    """A boolean field in an outside config.json is not a pooling mode."""
    outside = tmp_path / "outside"
    config_path = outside / "config.json"
    _write_json(config_path, _OUTSIDE_POOLING_BOOL)
    model = tmp_path / "model"
    model.mkdir()
    _write_json(
        model / "modules.json",
        _modules(str(outside), "sentence_transformers.models.Pooling"),
    )

    with _OpenWatch([config_path]):
        config = get_pooling_config(str(model))

    _assert_pooling_ignored(config)


def test_get_pooling_config_does_not_open_unrelated_outside_json(
    tmp_path: Path,
) -> None:
    """Any outside config.json is left unread, not only pooling configs."""
    outside = tmp_path / "outside"
    config_path = outside / "config.json"
    _write_json(config_path, _UNRELATED_CONFIG)
    model = tmp_path / "model"
    model.mkdir()
    _write_json(
        model / "modules.json",
        _modules(str(outside), "sentence_transformers.models.Pooling"),
    )

    with _OpenWatch([config_path]):
        config = get_pooling_config(str(model))

    _assert_pooling_ignored(config)


def test_get_pooling_config_symlink_is_not_reread_outside_model(
    tmp_path: Path,
) -> None:
    """Retargeting the module symlink before open still cannot escape."""
    outside = tmp_path / "outside"
    inside = tmp_path / "inside"
    outside_config = outside / "config.json"
    _write_json(outside_config, _OUTSIDE_POOLING)
    _write_json(inside / "config.json", {"pooling_mode": "mean"})
    model = tmp_path / "model"
    model.mkdir()
    link = model / "1_Pooling"
    link.symlink_to(inside, target_is_directory=True)
    _write_json(
        model / "modules.json",
        _modules("1_Pooling", "sentence_transformers.models.Pooling"),
    )

    with (
        _OpenWatch([outside_config]),
        _retarget_symlink_before_open(link, outside),
    ):
        config = get_pooling_config(str(model))

    # MEAN is the inside file. LAST is the outside file. Rejecting the
    # symlink leaves the pooling type unset. Either result is confined.
    if config is not None:
        assert config.get("seq_pooling_type") != "LAST"


def test_hub_repo_pooling_config_does_not_open_absolute_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Hub repo id uses the same join and must not open an absolute path.

    ``modules.json`` is the cached checkpoint file. The module path is an
    absolute directory on this machine. Hugging Face rejects absolute names
    when it looks in the cache; the local join must not open them either.
    """
    outside = tmp_path / "outside"
    config_path = outside / "config.json"
    _write_json(config_path, _OUTSIDE_POOLING)
    repo_id = "published/embedding-escape"
    revision = "0123456789abcdef0123456789abcdef01234567"
    cache = tmp_path / "hub"
    repo_cache = cache / "models--published--embedding-escape"
    snapshot = repo_cache / "snapshots" / revision
    _write_json(
        snapshot / "modules.json",
        _modules(str(outside), "sentence_transformers.models.Pooling"),
    )
    refs = repo_cache / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(revision, encoding="utf-8")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))

    with _OpenWatch([config_path]):
        config = get_pooling_config(repo_id)

    _assert_pooling_ignored(config)


def test_auto_runner_does_not_apply_outside_pooling_config(tmp_path: Path) -> None:
    """Default ``runner="auto"`` reads pooling config while choosing a runner."""
    outside = tmp_path / "outside"
    config_path = outside / "config.json"
    _write_json(config_path, _OUTSIDE_POOLING)
    model = tmp_path / "model"
    model.mkdir()
    _bert_checkpoint(model)
    _write_json(
        model / "modules.json",
        _modules(str(outside), "sentence_transformers.models.Pooling"),
    )

    with _OpenWatch([config_path]):
        model_config = ModelConfig(str(model), runner="auto", enforce_eager=True)

    assert model_config.runner_type == "pooling"
    assert model_config.pooler_config is not None
    assert model_config.pooler_config.seq_pooling_type != "LAST"


@pytest.mark.parametrize("module_type", _DENSE_TYPES)
@pytest.mark.parametrize("escape", _ESCAPES)
def test_try_get_dense_modules_does_not_open_outside_config(
    tmp_path: Path, module_type: str, escape: str
) -> None:
    """Dense config.json outside the checkpoint is not returned."""
    outside = tmp_path / "outside"
    config_path = outside / "config.json"
    _write_json(config_path, _DENSE_CONFIG)
    model = _model_dir(tmp_path, escape)
    module_path = _module_path(model, outside, escape, "2_Dense")
    _write_json(model / "modules.json", _modules(module_path, module_type))

    with _OpenWatch([config_path]):
        modules = try_get_dense_modules(str(model))

    assert not modules


def test_embedding_size_and_projector_do_not_use_outside_dense_module(
    tmp_path: Path,
) -> None:
    """Runner setup and the embedding projector share the dense module path."""
    outside = tmp_path / "outside"
    config_path = outside / "config.json"
    weights_path = outside / "model.safetensors"
    _write_json(config_path, _DENSE_CONFIG)
    _save_weights(weights_path, {"weight": _WEIGHT_SENTINEL}, "safetensors")
    model = tmp_path / "model"
    model.mkdir()
    _bert_checkpoint(model)
    _write_json(
        model / "modules.json",
        _modules(str(outside), "sentence_transformers.models.Dense"),
    )
    model_config = ModelConfig(str(model), runner="auto", enforce_eager=True)

    with _OpenWatch([config_path, weights_path]):
        assert model_config.embedding_size != _DENSE_CONFIG["out_features"]
        projector = _load_st_projector(model_config)

    if projector is not None:
        _assert_weight_not_installed(projector[0])


@pytest.mark.parametrize("escape", _ESCAPES)
@pytest.mark.parametrize(
    ("kind", "filename"),
    (("safetensors", "model.safetensors"), ("pytorch", "pytorch_model.bin")),
)
def test_load_dense_weights_does_not_open_outside_checkpoint(
    tmp_path: Path, escape: str, kind: str, filename: str
) -> None:
    """Weight files named from the dense module path stay in the model dir."""
    outside = tmp_path / "outside"
    weights_path = outside / filename
    _save_weights(weights_path, {"weight": _WEIGHT_SENTINEL}, kind)
    model = _model_dir(tmp_path, escape)
    _bert_checkpoint(model)
    module_path = _module_path(model, outside, escape, "2_Dense")
    model_config = ModelConfig(str(model), runner="auto", enforce_eager=True)
    linear = _linear()

    with _OpenWatch([weights_path]):
        loaded = _load_dense_weights(linear, module_path, model_config)

    assert loaded is False
    _assert_weight_not_installed(linear)


def test_load_dense_weights_symlink_is_not_reread_outside_model(
    tmp_path: Path,
) -> None:
    """A dense weight symlink swapped before open cannot install outside data."""
    outside = tmp_path / "outside"
    inside = tmp_path / "inside"
    outside_weights = outside / "model.safetensors"
    _save_weights(outside_weights, {"weight": _WEIGHT_SENTINEL}, "safetensors")
    _save_weights(
        inside / "model.safetensors",
        {"weight": torch.zeros(3, 4)},
        "safetensors",
    )
    model = tmp_path / "model"
    model.mkdir()
    _bert_checkpoint(model)
    link = model / "2_Dense"
    link.symlink_to(inside, target_is_directory=True)
    model_config = ModelConfig(str(model), runner="auto", enforce_eager=True)
    linear = _linear()

    with (
        _OpenWatch([outside_weights]),
        _retarget_symlink_before_open(link, outside),
    ):
        _load_dense_weights(linear, "2_Dense", model_config)

    # Zeros are the inside checkpoint. The sentinel lives only outside.
    _assert_weight_not_installed(linear)


def test_in_repo_pooling_and_dense_modules_still_load(tmp_path: Path) -> None:
    """A module path that stays inside the checkpoint still loads."""
    model = tmp_path / "model"
    _write_json(
        model / "1_Pooling" / "config.json",
        {"pooling_mode_mean_tokens": True},
    )
    _write_json(model / "2_Dense" / "config.json", _DENSE_CONFIG)
    _save_weights(
        model / "2_Dense" / "model.safetensors",
        {"weight": _WEIGHT_SENTINEL},
        "safetensors",
    )
    _write_json(
        model / "modules.json",
        [
            *_modules("1_Pooling", "sentence_transformers.models.Pooling"),
            *_modules("2_Dense", "sentence_transformers.models.Dense"),
        ],
    )
    _bert_checkpoint(model)

    config = get_pooling_config(str(model))
    assert config is not None
    assert config["seq_pooling_type"] == "MEAN"

    modules = try_get_dense_modules(str(model))
    assert modules is not None
    assert modules[0]["folder"] == "2_Dense"
    assert modules[0]["out_features"] == 3

    model_config = ModelConfig(str(model), runner="auto", enforce_eager=True)
    linear = _linear()
    assert _load_dense_weights(linear, "2_Dense", model_config) is True
    assert torch.equal(linear.weight.detach().cpu(), _WEIGHT_SENTINEL)
