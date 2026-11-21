# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keye profiling must bound checkpoint and processor pixel budgets.

The processor files under ``fixtures/keye_remote_code`` are the Apache-2.0
remote code from ``Kwai-Keye/Keye-VL-8B-Preview`` (configuration, image
processor, and processor). ``SiglipImageProcessor.__init__`` copies
``min_pixels`` and ``max_pixels`` from ``preprocessor_config.json`` into
``image_processor.size``. Startup profiling reads that dict.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from vllm.config import ModelConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.envs import VLLM_MAX_IMAGE_PIXELS
from vllm.model_executor.models.keye_vl1_5 import KeyeVL1_5ProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item
from vllm.multimodal.processing import BaseMultiModalProcessor

pytestmark = pytest.mark.skip_global_cleanup

_FIXTURES = Path(__file__).parent / "fixtures" / "keye_remote_code"
_MAX_PROFILE_PIXELS = VLLM_MAX_IMAGE_PIXELS
_SEQ_LEN = 2048
_MM_COUNTS = {"image": 1, "video": 1}

# Official Keye-VL-8B-Preview preprocessor budgets. Profiling must keep working
# for these values.
_OFFICIAL_MIN_PIXELS = 102400
_OFFICIAL_MAX_PIXELS = 1003520


def _write_checkpoint(
    destination: Path,
    *,
    min_pixels: int,
    max_pixels: int,
    architecture: str = "KeyeForConditionalGeneration",
) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(_FIXTURES, destination)
    config = {
        "architectures": [architecture],
        "model_type": "Keye",
        "auto_map": {"AutoConfig": "configuration_keye.KeyeConfig"},
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 64,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "rope_theta": 1000000,
        "rope_scaling": {"type": "mrope", "mrope_section": [4, 6, 6]},
        "vision_config": {
            "patch_size": 14,
            "spatial_merge_size": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 384,
            "temporal_patch_size": 1,
        },
    }
    preprocessor = {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "patch_size": 14,
        "temporal_patch_size": 1,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "processor_class": "KeyeProcessor",
        "auto_map": {
            "AutoProcessor": "processing_keye.KeyeProcessor",
            "AutoImageProcessor": "image_processing_keye.SiglipImageProcessor",
        },
    }
    processor = {
        "processor_class": "KeyeProcessor",
        "auto_map": {"AutoProcessor": "processing_keye.KeyeProcessor"},
    }
    tokenizer = {
        "tokenizer_class": "Qwen2Tokenizer",
        "model_max_length": 2048,
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
        "add_prefix_space": False,
    }
    vocab = {"<|endoftext|>": 0, "<|image_pad|>": 1, "<|video_pad|>": 2, "a": 3}
    (destination / "config.json").write_text(json.dumps(config))
    (destination / "preprocessor_config.json").write_text(json.dumps(preprocessor))
    (destination / "processor_config.json").write_text(json.dumps(processor))
    (destination / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    (destination / "vocab.json").write_text(json.dumps(vocab))
    (destination / "merges.txt").write_text("#version: 0.2\n")
    (destination / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"})
    )
    return destination


def _model_config(
    model_dir: Path,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> ModelConfig:
    return ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        trust_remote_code=True,
        dtype="float32",
        seed=0,
        max_model_len=_SEQ_LEN,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_processor_kwargs=mm_processor_kwargs,
    )


def _call_within(timeout_s: float, fn: Callable[[], Any]) -> Any:
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        pytest.fail(
            f"{getattr(fn, '__name__', 'call')} did not return within {timeout_s}s"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _assert_positive_budget(counts: Mapping[str, int]) -> None:
    assert set(counts) >= {"image", "video"}
    for modality, count in counts.items():
        assert isinstance(count, int)
        assert count >= 1, modality
        assert count <= _MAX_PROFILE_PIXELS, modality


def _guard_allocations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record image and video buffers and refuse an unbounded allocation."""
    areas: list[int] = []
    original_new = Image.new
    original_full = np.full

    def guarded_new(mode: str, size: tuple[int, int], color: int = 0) -> Image.Image:
        width, height = size
        area = width * height
        areas.append(area)
        assert width > 0 and height > 0
        assert area <= _MAX_PROFILE_PIXELS
        return original_new(mode, size, color)

    def guarded_full(shape, fill_value=0, dtype=None, order="C", *, like=None):
        if like is None and len(shape) >= 3:
            frames, height, width = shape[0], shape[1], shape[2]
            area = int(frames) * int(height) * int(width)
            areas.append(area)
            assert frames > 0 and height > 0 and width > 0
            assert area <= _MAX_PROFILE_PIXELS
        return original_full(shape, fill_value, dtype=dtype, order=order, like=like)

    monkeypatch.setattr(Image, "new", guarded_new)
    monkeypatch.setattr(np, "full", guarded_full)
    return areas


def _processor_for(
    tmp_path: Path,
    *,
    min_pixels: int,
    max_pixels: int,
    architecture: str = "KeyeForConditionalGeneration",
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> BaseMultiModalProcessor:
    model_dir = _write_checkpoint(
        tmp_path / "checkpoint",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        architecture=architecture,
    )
    model_config = _model_config(model_dir, mm_processor_kwargs)
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def test_official_pixel_budget_profiles_inside_cap(tmp_path: Path) -> None:
    processor = _processor_for(
        tmp_path,
        min_pixels=_OFFICIAL_MIN_PIXELS,
        max_pixels=_OFFICIAL_MAX_PIXELS,
    )
    counts = _call_within(
        5,
        lambda: get_mm_max_toks_per_item(processor, _MM_COUNTS, cache=None),
    )
    _assert_positive_budget(counts)
    size = processor.info.get_image_size_with_most_features()
    assert size.width > 0 and size.height > 0
    assert size.width * size.height <= _OFFICIAL_MAX_PIXELS


@pytest.mark.parametrize(
    "architecture",
    ["KeyeForConditionalGeneration", "KeyeVL1_5ForConditionalGeneration"],
)
def test_zero_checkpoint_pixel_budget_startup_finishes(
    tmp_path: Path,
    architecture: str,
) -> None:
    processor = _processor_for(
        tmp_path,
        min_pixels=1,
        max_pixels=1,
        architecture=architecture,
    )
    counts = _call_within(
        2,
        lambda: get_mm_max_toks_per_item(processor, _MM_COUNTS, cache=None),
    )
    _assert_positive_budget(counts)
    size = processor.info.get_image_size_with_most_features()
    assert size.width > 0 and size.height > 0
    assert size.width * size.height <= _MAX_PROFILE_PIXELS

    info = processor.info
    if architecture == "KeyeVL1_5ForConditionalGeneration":
        assert isinstance(info, KeyeVL1_5ProcessingInfo)
    frames = _call_within(
        2,
        lambda: info._get_max_video_frames(_SEQ_LEN),
    )
    assert 1 <= frames <= info.get_max_frame_per_video()


def test_huge_checkpoint_pixel_budget_does_not_allocate_dummy_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    areas = _guard_allocations(monkeypatch)
    processor = _processor_for(
        tmp_path,
        min_pixels=10**18,
        max_pixels=10**18,
    )
    size = _call_within(
        2,
        processor.info.get_image_size_with_most_features,
    )
    assert size.width > 0 and size.height > 0
    assert size.width * size.height <= _MAX_PROFILE_PIXELS
    processor.dummy_inputs.get_dummy_mm_data(
        _SEQ_LEN,
        _MM_COUNTS,
        MultiModalDummyOptions(),
    )
    assert areas
    assert max(areas) <= _MAX_PROFILE_PIXELS


def test_request_pixel_overrides_keep_real_image_token_count(
    tmp_path: Path,
) -> None:
    processor = _processor_for(
        tmp_path,
        min_pixels=_OFFICIAL_MIN_PIXELS,
        max_pixels=_OFFICIAL_MAX_PIXELS,
    )
    info = processor.info
    image_processor = info.get_image_processor()
    overrides = {"min_pixels": 1, "max_pixels": 1}
    image_tokens = info.get_num_image_tokens(
        image_width=224,
        image_height=224,
        image_processor=image_processor,
        mm_kwargs=overrides,
    )
    video_tokens = info.get_num_video_tokens(
        image_width=224,
        image_height=224,
        num_frames=4,
        image_processor=image_processor,
        mm_kwargs=overrides,
    )
    assert image_tokens >= 1
    assert video_tokens > image_tokens


def test_configured_pixel_override_startup_finishes(tmp_path: Path) -> None:
    processor = _processor_for(
        tmp_path,
        min_pixels=_OFFICIAL_MIN_PIXELS,
        max_pixels=_OFFICIAL_MAX_PIXELS,
        mm_processor_kwargs={"min_pixels": 1, "max_pixels": 1},
    )
    counts = _call_within(
        2,
        lambda: get_mm_max_toks_per_item(processor, _MM_COUNTS, cache=None),
    )
    _assert_positive_budget(counts)
