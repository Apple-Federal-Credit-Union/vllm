# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projector dimensions from a DeepSeek-VL2 checkpoint must be bounded.

Published checkpoints that have to keep loading:

* deepseek-ai/deepseek-vl2-tiny and deepseek-vl2 use the default
  ``downsample_mlp_gelu`` projector (``input_dim`` 1152, ``depth`` 2,
  ``mlp_ratio`` 1, ``downsample_ratio`` 2) with ``n_embed`` 1280 and 2560.
  The largest matrix is ``4608 x 2560``.
* deepseek-ai/DeepSeek-OCR uses ``projector_type="linear"`` with
  ``input_dim`` 2048 and ``n_embed`` 1280.

Limits below are inclusive. Every linear weight and every 2D tile marker
(``image_newline`` / ``view_seperator``) must stay within
``MAX_ELEMENTS_PER_TENSOR`` even when each field is at its own maximum.
``MlpProjector`` is also constructed by Deepseek-OCR, Deepseek-OCR2, and
Unlimited-OCR; those constructors are covered here. Vision-tower widths are
not: DeepSeek-VL2's vision module ignores ``vision_config`` and builds a
fixed timm network.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

import vllm.model_executor.models.deepseek_ocr as deepseek_ocr
import vllm.model_executor.models.deepseek_ocr2 as deepseek_ocr2
import vllm.model_executor.models.deepseek_vl2 as deepseek_vl2
from vllm.config.device import DeviceConfig
from vllm.config.model import ModelConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.models.deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.deepseek_ocr2 import DeepseekOCR2ForCausalLM
from vllm.model_executor.models.deepseek_vl2 import (
    DeepseekVLV2ForCausalLM,
    MlpProjector,
)
from vllm.model_executor.models.unlimited_ocr import UnlimitedOCRForCausalLM

MAX_INPUT_DIM = 4096
MAX_N_EMBED = 4096
MAX_MLP_RATIO = 4
MAX_DOWNSAMPLE_RATIO = 4
MAX_DEPTH = 4
# 64 Mi elements is 256 MiB of fp32. The largest published matrix is ~12M.
MAX_ELEMENTS_PER_TENSOR = 67_108_864

_PUBLISHED_DOWNSAMPLE = {
    "projector_type": "downsample_mlp_gelu",
    "input_dim": 1152,
    "n_embed": 2560,
    "depth": 2,
    "mlp_ratio": 1,
    "downsample_ratio": 2,
}
_PUBLISHED_LINEAR = {
    "projector_type": "linear",
    "input_dim": 2048,
    "n_embed": 1280,
    "depth": 2,
    "mlp_ratio": 1,
    "downsample_ratio": 2,
}
_FIELD_AT_MAXIMUM = {
    "input_dim": MAX_INPUT_DIM,
    "n_embed": 1,
    "depth": 2,
    "mlp_ratio": 1,
    "downsample_ratio": 1,
    "projector_type": "downsample_mlp_gelu",
}


class ProjectorStorageRequested(RuntimeError):
    """Construction asked torch for storage outside the projector budget."""


class _Tokenizer:
    vocab = {"<image>": 7}


class _Tower(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class _LanguageModel(nn.Module):
    def make_empty_intermediate_tensors(self, *args, **kwargs):
        return None


def _language_config() -> dict:
    return {
        "architectures": ["DeepseekV2ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 1,
        "first_k_dense_replace": 1,
        "hidden_size": 1280,
        "intermediate_size": 6848,
        "kv_lora_rank": None,
        "max_position_embeddings": 4096,
        "model_type": "deepseek_v2",
        "moe_intermediate_size": 896,
        "n_group": 1,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_attention_heads": 10,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 12,
        "num_key_value_heads": 10,
        "q_lora_rank": None,
        "qk_nope_head_dim": 0,
        "qk_rope_head_dim": 0,
        "torch_dtype": "float32",
        "use_mla": False,
        "v_head_dim": 0,
        "vocab_size": 129280,
    }


def _write_checkpoint(
    directory: Path,
    *,
    architecture: str,
    projector: dict,
    model_type: str = "deepseek_vl_v2",
) -> Path:
    config = {
        "architectures": [architecture],
        "candidate_resolutions": [[384, 384]],
        "global_view_pos": "head",
        "language_config": _language_config(),
        "model_type": model_type,
        "projector_config": {"model_type": "mlp_projector", **projector},
        "tile_tag": "2D",
        "torch_dtype": "float32",
        "vision_config": {
            "heads": 16,
            "image_size": 384,
            "layers": 24,
            "mlp_ratio": 4,
            "patch_size": 16,
            "width": 1152,
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def _load_model_config(directory: Path) -> ModelConfig:
    return ModelConfig(
        model=str(directory),
        tokenizer=str(directory),
        skip_tokenizer_init=True,
        max_model_len=32,
        dtype="float32",
        trust_remote_code=False,
    )


def _requested_numel(args: tuple) -> int:
    if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        dims = tuple(args[0])
    else:
        dims = args
    numel = 1
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ProjectorStorageRequested(dim)
        numel *= dim
    return numel


def _guard_parameter_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject parameter storage outside the budget before the real allocator.

    ``torch.nn.Linear`` allocates its weight with ``torch.empty`` and the 2D
    tile markers use ``torch.randn``. In-budget sizes still go to those
    functions. A later check that runs after the storage request fails this
    guard instead of looking like a clean rejection.
    """
    calls = 0
    real_empty = torch.empty
    real_randn = torch.randn

    def _check(args: tuple) -> None:
        nonlocal calls
        calls += 1
        numel = _requested_numel(args)
        if numel > MAX_ELEMENTS_PER_TENSOR or calls > 64:
            raise ProjectorStorageRequested(numel)

    def empty(*args, **kwargs):
        _check(args)
        return real_empty(*args, **kwargs)

    def randn(*args, **kwargs):
        _check(args)
        return real_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch, "randn", randn)


@pytest.fixture(autouse=True)
def _offline_hub(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _assert_storage(parameter: torch.Tensor, shape: tuple[int, ...]) -> None:
    expected = 1
    for dim in shape:
        expected *= dim
    assert parameter.shape == shape
    assert parameter.numel() == expected
    assert parameter.untyped_storage().nbytes() == expected * parameter.element_size()


def _build_projector(tmp_path: Path, projector: dict) -> MlpProjector:
    directory = _write_checkpoint(
        tmp_path / "checkpoint",
        architecture="DeepseekVLV2ForCausalLM",
        projector=projector,
    )
    model_config = _load_model_config(directory)
    loaded = model_config.hf_config.projector_config
    for key, value in projector.items():
        assert getattr(loaded, key) == value
        assert type(getattr(loaded, key)) is type(value)
    return MlpProjector(loaded)


def test_published_downsample_checkpoint_allocates_projector_matrices(tmp_path: Path):
    projector = _build_projector(tmp_path, _PUBLISHED_DOWNSAMPLE)
    layers = projector.layers
    in_features = 1152 * 2 * 2
    _assert_storage(layers[0].weight, (2560, in_features))
    _assert_storage(layers[0].bias, (2560,))
    _assert_storage(layers[2].weight, (2560, 2560))
    _assert_storage(layers[2].bias, (2560,))


def test_published_linear_checkpoint_allocates_projector_matrix(tmp_path: Path):
    projector = _build_projector(tmp_path, _PUBLISHED_LINEAR)
    _assert_storage(projector.layers.weight, (1280, 2048))
    _assert_storage(projector.layers.bias, (1280,))


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_dim": MAX_INPUT_DIM},
        {"downsample_ratio": MAX_DOWNSAMPLE_RATIO},
        {"mlp_ratio": MAX_MLP_RATIO},
        {"depth": MAX_DEPTH},
        {"n_embed": MAX_N_EMBED},
    ],
)
def test_projector_accepts_each_field_at_its_maximum(tmp_path: Path, overrides: dict):
    spec = dict(_FIELD_AT_MAXIMUM)
    spec.update(overrides)
    # n_embed at the cap with a 1-wide input still builds an n_embed square.
    if overrides.get("n_embed") == MAX_N_EMBED:
        spec["mlp_ratio"] = 1
        spec["downsample_ratio"] = 1
        spec["input_dim"] = 1
    projector = _build_projector(tmp_path, spec)
    assert projector.projector_type == "downsample_mlp_gelu"
    for parameter in projector.parameters():
        assert parameter.numel() <= MAX_ELEMENTS_PER_TENSOR


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("input_dim", 0),
        ("input_dim", -1),
        ("input_dim", MAX_INPUT_DIM + 1),
        ("input_dim", 1.5),
        ("n_embed", 0),
        ("n_embed", -1),
        ("n_embed", MAX_N_EMBED + 1),
        ("mlp_ratio", 0),
        ("mlp_ratio", -1),
        ("mlp_ratio", MAX_MLP_RATIO + 1),
        ("downsample_ratio", 0),
        ("downsample_ratio", -2),
        ("downsample_ratio", MAX_DOWNSAMPLE_RATIO + 1),
        ("depth", 0),
        ("depth", -1),
        ("depth", MAX_DEPTH + 1),
    ],
)
def test_checkpoint_projector_rejects_out_of_range_field(
    tmp_path: Path, field_name: str, value: object
):
    spec = dict(_FIELD_AT_MAXIMUM)
    spec.update({field_name: value, "n_embed": 8, "input_dim": 4, "mlp_ratio": 1})
    spec[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        _build_projector(tmp_path, spec)


@pytest.mark.parametrize(
    "overrides",
    [
        {"depth": MAX_DEPTH + 1},
        {"downsample_ratio": MAX_DOWNSAMPLE_RATIO + 1},
        {"mlp_ratio": MAX_MLP_RATIO + 1},
    ],
)
def test_linear_projector_rejects_fields_it_does_not_read(
    tmp_path: Path, overrides: dict
):
    spec = dict(_PUBLISHED_LINEAR)
    spec.update(overrides)
    field_name = next(iter(overrides))
    with pytest.raises(ValueError, match=field_name):
        _build_projector(tmp_path, spec)


def test_combined_field_maxima_are_rejected_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Each field is in range; the resulting matrix is not."""
    spec = {
        "projector_type": "downsample_mlp_gelu",
        "input_dim": MAX_INPUT_DIM,
        "n_embed": MAX_N_EMBED,
        "depth": MAX_DEPTH,
        "mlp_ratio": MAX_MLP_RATIO,
        "downsample_ratio": MAX_DOWNSAMPLE_RATIO,
    }
    directory = _write_checkpoint(
        tmp_path / "checkpoint",
        architecture="DeepseekVLV2ForCausalLM",
        projector=spec,
    )
    loaded = _load_model_config(directory).hf_config.projector_config
    _guard_parameter_storage(monkeypatch)
    with pytest.raises(ValueError, match="projector|n_embed|input_dim|element"):
        MlpProjector(loaded)


@pytest.mark.parametrize(
    "projector",
    [
        {
            "projector_type": "downsample_mlp_gelu",
            "input_dim": 1,
            "n_embed": 2**28,
            "depth": 2,
            "mlp_ratio": 1,
            "downsample_ratio": 1,
        },
        {
            "projector_type": "downsample_mlp_gelu",
            "input_dim": 1152,
            "n_embed": 1280,
            "depth": 2,
            "mlp_ratio": 1,
            "downsample_ratio": 2**16,
        },
        {
            "projector_type": "linear",
            "input_dim": 2**28,
            "n_embed": 1280,
            "depth": 2,
            "mlp_ratio": 1,
            "downsample_ratio": 2,
        },
    ],
)
def test_exhausting_checkpoint_dims_are_rejected_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, projector: dict
):
    directory = _write_checkpoint(
        tmp_path / "checkpoint",
        architecture="DeepseekVLV2ForCausalLM",
        projector=projector,
    )
    loaded = _load_model_config(directory).hf_config.projector_config
    for key, value in projector.items():
        assert getattr(loaded, key) == value
    _guard_parameter_storage(monkeypatch)
    with pytest.raises(ValueError, match="projector|n_embed|input_dim|downsample"):
        MlpProjector(loaded)


def _vllm_config(model_config: ModelConfig) -> VllmConfig:
    return VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
    )


def _patch_vl2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deepseek_vl2, "cached_tokenizer_from_config", lambda model_config: _Tokenizer()
    )
    monkeypatch.setattr(
        deepseek_vl2,
        "init_vllm_registered_model",
        lambda **kwargs: _LanguageModel(),
    )
    monkeypatch.setattr(
        DeepseekVLV2ForCausalLM,
        "_init_vision_module",
        lambda self, *args, **kwargs: _Tower(),
    )


def _patch_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deepseek_ocr, "cached_tokenizer_from_config", lambda model_config: _Tokenizer()
    )
    monkeypatch.setattr(
        deepseek_ocr,
        "init_vllm_registered_model",
        lambda **kwargs: _LanguageModel(),
    )
    monkeypatch.setattr(deepseek_ocr, "build_sam_vit_b", lambda: _Tower())
    monkeypatch.setattr(deepseek_ocr, "DeepCLIPVisionTransformer", _Tower)


def _patch_ocr2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deepseek_ocr2,
        "cached_tokenizer_from_config",
        lambda model_config: _Tokenizer(),
    )
    monkeypatch.setattr(
        deepseek_ocr2,
        "init_vllm_registered_model",
        lambda **kwargs: _LanguageModel(),
    )
    monkeypatch.setattr(deepseek_ocr2, "ImageEncoderViT", _Tower)
    monkeypatch.setattr(
        deepseek_ocr2, "build_qwen2_decoder_as_encoder", lambda: _Tower()
    )


_MODEL_CASES = [
    pytest.param(
        DeepseekVLV2ForCausalLM,
        "DeepseekVLV2ForCausalLM",
        "deepseek_vl_v2",
        _patch_vl2,
        ("image_newline", "view_seperator"),
        id="deepseek_vl2",
    ),
    pytest.param(
        DeepseekOCRForCausalLM,
        "DeepseekOCRForCausalLM",
        "deepseek_vl_v2",
        _patch_ocr,
        ("image_newline", "view_seperator"),
        id="deepseek_ocr",
    ),
    pytest.param(
        DeepseekOCR2ForCausalLM,
        "DeepseekOCR2ForCausalLM",
        "deepseek_vl_v2",
        _patch_ocr2,
        ("view_seperator",),
        id="deepseek_ocr2",
    ),
    pytest.param(
        UnlimitedOCRForCausalLM,
        "UnlimitedOCRForCausalLM",
        "unlimited-ocr",
        _patch_ocr,
        ("image_newline", "view_seperator"),
        id="unlimited_ocr",
    ),
]


@pytest.mark.parametrize(
    ("model_cls", "architecture", "model_type", "patch", "markers"),
    _MODEL_CASES,
)
def test_model_constructor_rejects_exhausting_n_embed_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[nn.Module],
    architecture: str,
    model_type: str,
    patch: Callable[[pytest.MonkeyPatch], None],
    markers: tuple[str, ...],
):
    directory = _write_checkpoint(
        tmp_path / "checkpoint",
        architecture=architecture,
        model_type=model_type,
        projector={
            "projector_type": "downsample_mlp_gelu",
            "input_dim": 1,
            "n_embed": 2**28,
            "depth": 2,
            "mlp_ratio": 1,
            "downsample_ratio": 1,
        },
    )
    model_config = _load_model_config(directory)
    assert model_config.hf_config.projector_config.n_embed == 2**28
    vllm_config = _vllm_config(model_config)
    patch(monkeypatch)
    _guard_parameter_storage(monkeypatch)
    with pytest.raises(ValueError, match="n_embed|projector"):
        model_cls(vllm_config=vllm_config)


@pytest.mark.parametrize(
    ("model_cls", "architecture", "model_type", "patch", "markers"),
    _MODEL_CASES,
)
def test_model_constructor_allocates_tile_markers_from_n_embed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[nn.Module],
    architecture: str,
    model_type: str,
    patch: Callable[[pytest.MonkeyPatch], None],
    markers: tuple[str, ...],
):
    n_embed = 32
    directory = _write_checkpoint(
        tmp_path / "checkpoint",
        architecture=architecture,
        model_type=model_type,
        projector={
            "projector_type": "downsample_mlp_gelu",
            "input_dim": 8,
            "n_embed": n_embed,
            "depth": 2,
            "mlp_ratio": 1,
            "downsample_ratio": 2,
        },
    )
    patch(monkeypatch)
    model = model_cls(vllm_config=_vllm_config(_load_model_config(directory)))
    assert model.projector.layers[0].weight.shape == (n_embed, 8 * 2 * 2)
    for name in markers:
        parameter = getattr(model, name)
        _assert_storage(parameter, (n_embed,))
