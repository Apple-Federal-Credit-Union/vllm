# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dots3Note pixel-budget bounds.

These tests are expected to fail until preprocessing refuses or clamps
image allocations to ``vllm.envs.VLLM_MAX_IMAGE_PIXELS`` (when that value
is positive). Audio preprocessing does not use this pixel budget and is
not covered.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import vllm.envs as envs
from vllm.config.multimodal import MultiModalConfig, MultiModalDummyOptions
from vllm.entrypoints.generate.base.protocol import validate_request_mm_kwargs
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.processing.context import InputProcessingContext

_NOTE_PKG = "vllm.models.dots3_note"


def _load_note_processor():
    """Import the processor even when the CUDA model package cannot load."""
    try:
        from vllm.models.dots3_note.common import processor as note_processor
    except ImportError:
        for key in list(sys.modules):
            if key == _NOTE_PKG or key.startswith(_NOTE_PKG + "."):
                del sys.modules[key]
        package = types.ModuleType(_NOTE_PKG)
        package.__path__ = [str(Path(vllm_root()) / "vllm/models/dots3_note")]
        package.__package__ = _NOTE_PKG
        sys.modules[_NOTE_PKG] = package
        from vllm.models.dots3_note.common import processor as note_processor

    return note_processor


def vllm_root() -> Path:
    return Path(__file__).resolve().parents[3]


note = _load_note_processor()

# A prime whose only factor is 1, so get_max_image_size walks isqrt(n) steps.
_LARGE_PRIME_PIXELS = 100_000_000_000_031

_MEAN = [0.5, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]


def _budget() -> int:
    budget = int(envs.VLLM_MAX_IMAGE_PIXELS)
    assert budget > 0, "VLLM_MAX_IMAGE_PIXELS must stay enabled for these tests"
    return budget


def _vision_config(
    *,
    min_pixels: int,
    max_pixels: int,
    image_details: dict | None = None,
    patch_size: int = 14,
    merge_size: int = 2,
) -> dict:
    config = {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "patch_size": patch_size,
        "temporal_patch_size": 2,
        "merge_size": merge_size,
        "pre_pixel_shuffle": True,
        "image_mean": _MEAN,
        "image_std": _STD,
    }
    if image_details is not None:
        config["image_details"] = image_details
    return config


def _write_checkpoint(root: Path, vision_config: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"vision_config": {"hidden_size": 16}}),
        encoding="utf-8",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps({"vision_config": vision_config}),
        encoding="utf-8",
    )


class _LocalModel:
    """Model path holder used by the real preprocessor file loader.

    ``Dots3NoteProcessingInfo`` reads ``model`` and ``revision`` and passes
    them to ``get_hf_file_to_dict``. Building a full ``ModelConfig`` pulls in
    the CUDA model stack, which these tests do not execute.
    """

    def __init__(self, path: Path, mm_processor_kwargs: dict | None = None) -> None:
        self.model = str(path)
        self.revision = None
        self.max_model_len = 8192
        self.dtype = torch.float32
        self.multimodal_config = MultiModalConfig(
            mm_processor_kwargs=mm_processor_kwargs
        )

    def get_multimodal_config(self) -> MultiModalConfig:
        return self.multimodal_config


class _Tokenizer:
    """Tokenizer surface used by ``Dots3NoteProcessor`` (encode, call, vocab)."""

    def __init__(self) -> None:
        tokens = [
            note.IMAGE_START,
            note.IMAGE_END,
            note.IMAGE_PAD,
            note.AUDIO_START,
            note.AUDIO_END,
            note.AUDIO_PAD,
            note.VIDEO_PLACEHOLDER,
        ]
        self._vocab = {token: index for index, token in enumerate(tokens, start=1)}
        self._ordered = sorted(tokens, key=len, reverse=True)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        index = 0
        while index < len(text):
            matched = False
            for token in self._ordered:
                if text.startswith(token, index):
                    ids.append(self._vocab[token])
                    index += len(token)
                    matched = True
                    break
            if not matched:
                ids.append(0)
                index += 1
        return ids

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        del kwargs
        return {"input_ids": self.encode(text)}


def _processing_info(
    root: Path,
    vision_config: dict,
    mm_processor_kwargs: dict | None = None,
) -> tuple[InputProcessingContext, note.Dots3NoteProcessingInfo]:
    _write_checkpoint(root, vision_config)
    context = InputProcessingContext(
        model_config=_LocalModel(root, mm_processor_kwargs),
        tokenizer=_Tokenizer(),
    )
    return context, note.Dots3NoteProcessingInfo(context)


def _image_processor(vision_config: dict) -> note.Dots3NoteImageProcessor:
    details = vision_config.get("image_details", {})
    return note.Dots3NoteImageProcessor(vision_config, details)


def _spatial_pixels(grid: torch.Tensor, patch_size: int) -> int:
    if grid.ndim == 1:
        grid = grid.unsqueeze(0)
    areas = []
    for row in grid:
        areas.append(int(row[-2]) * int(row[-1]) * patch_size * patch_size)
    return max(areas)


def _record_resizes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    real_resize = Image.Image.resize

    def wrapped(self, size, *args, **kwargs):
        width, height = int(size[0]), int(size[1])
        sizes.append((width, height))
        return real_resize(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", wrapped)
    return sizes


def _assert_resize_budget(sizes: list[tuple[int, int]], budget: int) -> None:
    oversized = [
        f"{width}x{height}"
        for width, height in sizes
        if width * height > budget
    ]
    assert not oversized, f"image.resize exceeded {budget} pixels: {oversized}"


def test_resized_size_caps_enormous_checkpoint_budget() -> None:
    """A 10**12 min/max budget must not survive into the resize dimensions."""
    budget = _budget()
    processor = _image_processor(
        _vision_config(min_pixels=10**12, max_pixels=10**12)
    )
    height, width = processor.resized_size(32, 32)
    assert height * width <= budget


def test_checkpoint_file_caps_profiling_image_size(tmp_path: Path) -> None:
    """preprocessor_config.json is what profiling reads, with no extra flag."""
    budget = _budget()
    _context, info = _processing_info(
        tmp_path,
        _vision_config(min_pixels=10**12, max_pixels=10**12),
    )
    assert info.image_processor is not None
    assert info.image_processor.max_pixels <= budget
    assert info.image_processor.min_pixels <= budget
    width, height = info.get_max_image_size()
    assert width > 0 and height > 0
    assert width * height <= budget
    tokens = info.get_mm_max_tokens_per_item(8192, {"image": 1})
    factor = info.image_processor.factor
    assert tokens["image"] * factor * factor <= budget


def test_prime_pixel_budget_does_not_scan_unbounded_factors(tmp_path: Path) -> None:
    """A prime max_pixels must not walk isqrt(max_pixels) before returning."""
    budget = _budget()
    _context, info = _processing_info(
        tmp_path,
        _vision_config(
            min_pixels=_LARGE_PRIME_PIXELS,
            max_pixels=_LARGE_PRIME_PIXELS,
            patch_size=1,
            merge_size=1,
        ),
    )
    started = time.perf_counter()
    width, height = info.get_max_image_size()
    elapsed = time.perf_counter() - started
    assert elapsed < 2
    assert width * height <= budget


def test_dummy_profiling_image_stays_within_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker profiling builds the dummy image from get_max_image_size."""
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 4096)
    budget = _budget()
    _context, info = _processing_info(
        tmp_path,
        _vision_config(min_pixels=20_000, max_pixels=20_000),
    )
    builder = note.Dots3NoteDummyInputsBuilder(info)
    data = builder.get_dummy_mm_data(
        seq_len=8192,
        mm_counts={"image": 1},
        mm_options=MultiModalDummyOptions(),
    )
    image = data["image"][0]
    assert image.size[0] * image.size[1] <= budget


def test_min_pixels_upscale_stays_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large min_pixels upscales an ordinary image through Pillow."""
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 4096)
    budget = _budget()
    sizes = _record_resizes(monkeypatch)
    processor = _image_processor(_vision_config(min_pixels=20_000, max_pixels=20_000))
    image = Image.new("RGB", (32, 32), color=(1, 2, 3))
    try:
        _pixels, grid = processor.preprocess(image)
    except ValueError:
        _assert_resize_budget(sizes, budget)
        return
    _assert_resize_budget(sizes, budget)
    assert _spatial_pixels(grid, processor.patch_size) <= budget


def test_image_detail_target_cannot_exceed_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """image_details on the checkpoint supplies target height and width."""
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 4096)
    budget = _budget()
    sizes = _record_resizes(monkeypatch)
    processor = _image_processor(
        _vision_config(
            min_pixels=784,
            max_pixels=3136,
            image_details={
                "auto": {
                    "min_pixels": 20_000,
                    "max_pixels": 10**12,
                    "target_height": 128,
                    "target_width": 128,
                }
            },
        )
    )
    image = Image.new("RGB", (32, 32), color=(4, 5, 6))
    with pytest.raises(ValueError):
        processor.preprocess(image)
    _assert_resize_budget(sizes, budget)


def test_trusted_request_kwargs_cannot_exceed_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--trust-request-mm-kwargs forwards sizes into call_hf_processor."""
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 4096)
    budget = _budget()
    sizes = _record_resizes(monkeypatch)
    context, info = _processing_info(
        tmp_path,
        _vision_config(min_pixels=784, max_pixels=3136),
    )
    request_kwargs = {
        "min_pixels": 20_000,
        "max_pixels": 10**12,
        "target_height": 128,
        "target_width": 128,
    }
    validate_request_mm_kwargs(
        mm_processor_kwargs=request_kwargs,
        media_io_kwargs=None,
        trust_request_mm_kwargs=True,
    )
    image = Image.new("RGB", (32, 32), color=(7, 8, 9))
    with pytest.raises(ValueError):
        context.call_hf_processor(
            info.processor,
            {"text": "describe", "images": [image]},
            request_kwargs,
        )
    _assert_resize_budget(sizes, budget)


def test_configured_mm_processor_kwargs_cannot_exceed_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server mm_processor_kwargs merge into the same processor fields."""
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 4096)
    budget = _budget()
    sizes = _record_resizes(monkeypatch)
    context, info = _processing_info(
        tmp_path,
        _vision_config(min_pixels=784, max_pixels=3136),
        mm_processor_kwargs={
            "min_pixels": 20_000,
            "max_pixels": 10**12,
            "target_height": 128,
            "target_width": 128,
        },
    )
    image = Image.new("RGB", (32, 32), color=(7, 8, 9))
    with pytest.raises(ValueError):
        context.call_hf_processor(
            info.processor,
            {"text": "describe", "images": [image]},
            {},
        )
    _assert_resize_budget(sizes, budget)


def test_video_frames_use_the_same_pixel_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native video frames are resized again by the image processor."""
    # Above the video sampler's own floor (~128 * 28 * 28) and below the
    # explicit target passed through the processor kwargs.
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 200_000)
    budget = _budget()
    sizes = _record_resizes(monkeypatch)
    context, info = _processing_info(
        tmp_path,
        _vision_config(min_pixels=784, max_pixels=3136),
    )
    video = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    request_kwargs = {
        "min_pixels": 300_000,
        "max_pixels": 10**12,
        "target_height": 500,
        "target_width": 500,
        "seq": 8192,
    }
    try:
        context.call_hf_processor(
            info.processor,
            {"text": "describe the video", "videos": [video]},
            request_kwargs,
        )
    except ValueError:
        pass
    _assert_resize_budget(sizes, budget)


def test_within_budget_image_still_preprocesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 4096)
    budget = _budget()
    sizes = _record_resizes(monkeypatch)
    processor = _image_processor(_vision_config(min_pixels=784, max_pixels=3136))
    _pixels, grid = processor.preprocess(Image.new("RGB", (32, 32), color=(1, 1, 1)))
    _assert_resize_budget(sizes, budget)
    assert _spatial_pixels(grid, processor.patch_size) <= budget


def test_untrusted_request_kwargs_are_rejected() -> None:
    with pytest.raises(VLLMValidationError):
        validate_request_mm_kwargs(
            mm_processor_kwargs={"max_pixels": 10**12},
            media_io_kwargs=None,
            trust_request_mm_kwargs=False,
        )
