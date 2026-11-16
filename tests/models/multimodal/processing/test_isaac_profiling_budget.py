# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isaac load-time profiling must bound dummy images and reject bad divisors.

A checkpoint ``config.json`` supplies ``vision_max_num_patches``,
``vision_min_num_patches``, ``vision_patch_size`` / ``video_patch_size``, and
``pixel_shuffle_scale``. Engine profiling builds one dummy image from those
fields. These tests fail while that image can exceed the processor pixel
ceiling or while a non-positive divisor raises ``ZeroDivisionError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from vllm.config import ModelConfig
from vllm.model_executor.models import isaac as _isaac  # noqa: F401
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import dummy_inputs
from vllm.transformers_utils.processors.isaac import (
    MAX_PIXELS,
    get_image_size_for_max_num_patches,
    process_vision_for_patches,
)

pytestmark = pytest.mark.skip_global_cleanup

_PROFILE_PIXEL_BUDGET = MAX_PIXELS


def _write_checkpoint(root: Path, **overrides: object) -> Path:
    config = {
        "model_type": "isaac",
        "architectures": ["IsaacForConditionalGeneration"],
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 4,
        "intermediate_size": 128,
        "max_position_embeddings": 2048,
        "vision_max_num_patches": 256,
        "vision_patch_size": 16,
        "pixel_shuffle_scale": 1,
        "vision_token": "<image>",
    }
    config.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))

    vocab = {"<unk>": 0, "<image>": 1, "<|image_pad|>": 2, "hello": 3}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "model_max_length": 2048,
            }
        )
    )
    (root / "special_tokens_map.json").write_text(json.dumps({"unk_token": "<unk>"}))
    return root


def _processor(root: Path, **overrides: object):
    _write_checkpoint(root, **overrides)
    model_config = ModelConfig(
        model=str(root),
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        max_model_len=128,
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _profiling_image_request(
    proc,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple[int, int], object]:
    """Run profiling dummy-input creation and record the PIL size requested.

    Oversized requests are not allocated. Pillow's default allocator builds
    the image as line blocks with no total-byte cap, so the real call exhausts
    the process.
    """
    requested: list[tuple[int, int]] = []
    real_new = dummy_inputs.Image.new

    def _guarded_new(mode, size, color=0):
        width, height = int(size[0]), int(size[1])
        requested.append((width, height))
        if width < 1 or height < 1 or width * height > _PROFILE_PIXEL_BUDGET:
            return real_new(mode, (1, 1), color)
        return real_new(mode, (width, height), color)

    monkeypatch.setattr(dummy_inputs.Image, "new", _guarded_new)
    proc.get_dummy_mm_inputs({"image": 1})
    image = proc.dummy_inputs.get_dummy_mm_data(
        128,
        {"image": 1},
        proc.info.ctx.get_mm_config().limit_per_prompt,
    )["image"][0]
    assert requested, "profiling did not request a dummy image"
    return requested[0], image


def _assert_bounded(size: tuple[int, int]) -> None:
    width, height = size
    assert width > 0 and height > 0
    assert width * height <= _PROFILE_PIXEL_BUDGET, (
        f"profiling image {width}x{height} exceeds {_PROFILE_PIXEL_BUDGET} pixels"
    )


def test_safe_patch_budget_profiling_image_is_allocated(tmp_path, monkeypatch):
    """A normal patch budget still allocates the dummy image profiling asks for."""
    proc = _processor(tmp_path)
    (width, height), image = _profiling_image_request(proc, monkeypatch)
    _assert_bounded((width, height))
    assert image.size == (width, height)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vision_max_num_patches": 10**12},
        {"vision_max_num_patches": 10**12, "pixel_shuffle_scale": -1},
        {"vision_max_num_patches": 10**12, "vision_patch_size": -16},
        {"vision_max_num_patches": 10**12, "video_patch_size": -16},
    ],
)
def test_huge_vision_max_num_patches_profiling_image_is_bounded(
    tmp_path, monkeypatch, overrides
):
    """A huge checkpoint patch budget must not size the profiling image."""
    proc = _processor(tmp_path, **overrides)
    (width, height), image = _profiling_image_request(proc, monkeypatch)
    _assert_bounded((width, height))
    assert image.size == (width, height)


def test_huge_vision_min_num_patches_profiling_image_is_bounded(tmp_path, monkeypatch):
    """An uncapped minimum patch count scales the profiling seed up."""
    proc = _processor(
        tmp_path,
        vision_max_num_patches=256,
        vision_min_num_patches=10**15,
    )
    (width, height), image = _profiling_image_request(proc, monkeypatch)
    _assert_bounded((width, height))
    assert image.size == (width, height)


@pytest.mark.parametrize(
    "overrides",
    [
        {"pixel_shuffle_scale": 0},
        {"vision_patch_size": 0},
        {"video_patch_size": 0},
    ],
)
def test_non_positive_patch_divisor_rejected_before_profiling(
    tmp_path, overrides, monkeypatch
):
    """Non-positive patch size or shuffle scale is rejected before allocation."""
    requested: list[tuple[int, int]] = []
    real_new = dummy_inputs.Image.new

    def _record_new(mode, size, color=0):
        requested.append((int(size[0]), int(size[1])))
        return real_new(mode, (1, 1), color)

    monkeypatch.setattr(dummy_inputs.Image, "new", _record_new)
    with pytest.raises(ValueError):
        proc = _processor(tmp_path, **overrides)
        proc.info.get_mm_max_tokens_per_item(128, {"image": 1})
        proc.info.get_image_size_with_most_features()
        proc.get_dummy_mm_inputs({"image": 1})
    assert not requested, "profiling allocated an image before rejecting the config"


@pytest.mark.parametrize(
    ("patch_size", "pixel_shuffle_scale"),
    [(0, 1), (16, 0)],
)
def test_resize_math_rejects_zero_divisors(patch_size, pixel_shuffle_scale):
    """Both resize callers use the same divisor and must not divide by zero."""
    with pytest.raises(ValueError):
        get_image_size_for_max_num_patches(
            32,
            32,
            patch_size,
            64,
            pixel_shuffle_scale=pixel_shuffle_scale,
        )
    image = torch.zeros(8, 8, 3, dtype=torch.uint8)
    with pytest.raises(ValueError):
        process_vision_for_patches(
            image,
            patch_size=patch_size,
            max_num_patches=64,
            pixel_shuffle_scale=pixel_shuffle_scale,
        )


def test_token_budget_rejects_zero_shuffle_scale(tmp_path):
    """Startup token budgeting divides by the checkpoint shuffle scale."""
    proc = _processor(tmp_path, pixel_shuffle_scale=0, vision_max_num_patches=256)
    with pytest.raises(ValueError):
        proc.info.get_mm_max_tokens_per_item(128, {"image": 1})


def test_loaded_config_divisors_reach_resize_math(tmp_path):
    """The integers in config.json are the divisors used for the profiling seed."""
    proc = _processor(tmp_path, video_patch_size=0, pixel_shuffle_scale=1)
    hf_config = proc.info.get_hf_config()
    with pytest.raises(ValueError):
        get_image_size_for_max_num_patches(
            9999999,
            9999999,
            hf_config.video_patch_size,
            hf_config.vision_max_num_patches,
            min_num_patches=hf_config.vision_min_num_patches,
            pixel_shuffle_scale=hf_config.pixel_shuffle_scale,
        )


@pytest.mark.parametrize(
    ("patch_size", "pixel_shuffle_scale"),
    [(-16, 1), (16, -1)],
)
def test_negative_divisor_does_not_keep_profiling_seed(patch_size, pixel_shuffle_scale):
    """A negative divisor still fits the profiling seed to the patch budget."""
    try:
        height, width = get_image_size_for_max_num_patches(
            9999999,
            9999999,
            patch_size,
            10**12,
            pixel_shuffle_scale=pixel_shuffle_scale,
        )
    except ValueError:
        return
    _assert_bounded((width, height))
