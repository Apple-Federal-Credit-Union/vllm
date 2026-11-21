# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PaddleOCR-VL profiling must not turn a checkpoint pixel budget into an
unbounded Pillow allocation.

These tests fail until the checkpoint budget is clamped to the documented
image pixel cap before dummy width and height are derived, and until
non-positive vision patch or merge sizes are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from transformers import AutoImageProcessor, PaddleOCRVLConfig

import vllm.envs as envs
from vllm.config.multimodal import MultiModalConfig, MultiModalDummyOptions
from vllm.model_executor.models.paddleocr_vl import (
    PaddleOCRVLDummyInputsBuilder,
    PaddleOCRVLProcessingInfo,
)
from vllm.multimodal.processing import context as processing_context
from vllm.multimodal.processing import dummy_inputs

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Far above VLLM_MAX_IMAGE_PIXELS. A real checkpoint can carry this in a
# few dozen bytes of preprocessor JSON.
_HUGE_PIXELS = 10**12
_MODEST_PIXELS = 28 * 28 * 80
# Refuse to materialize a buffer this process cannot afford. A legal clamp
# may still sit near the documented cap, which is hundreds of megabytes.
_SAFE_ALLOCATION_BYTES = 64 * 1024 * 1024


def _pixel_cap() -> int:
    """Documented raster cap. ``0`` disables it; tests keep the default."""
    configured = envs.VLLM_MAX_IMAGE_PIXELS
    if configured > 0:
        return configured
    return 178_956_970


class _ModelConfig:
    def __init__(self, trust_remote_code: bool) -> None:
        self.trust_remote_code = trust_remote_code
        self._mm_config = MultiModalConfig()

    def get_multimodal_config(self) -> MultiModalConfig:
        return self._mm_config


class _Processor:
    def __init__(self, image_processor: object) -> None:
        self.image_processor = image_processor


class _CheckpointContext:
    """Real processing-info context fed by a checkpoint the test wrote."""

    def __init__(
        self,
        trust_remote_code: bool,
        hf_config: PaddleOCRVLConfig,
        image_processor: object,
    ) -> None:
        self.model_config = _ModelConfig(trust_remote_code)
        self._hf_config = hf_config
        self._processor = _Processor(image_processor)

    def get_hf_config(self) -> PaddleOCRVLConfig:
        return self._hf_config

    def get_hf_processor(self, **kwargs: object) -> _Processor:
        return self._processor

    def get_merged_mm_kwargs(self, kwargs: object, **kw: object) -> dict:
        return processing_context.InputProcessingContext.get_merged_mm_kwargs(
            self, kwargs, **kw
        )


class _FrozenBudget:
    """Reports the checkpoint budget on every read and ignores writes.

    A fix that stores a clamped copy and then reads the processor again
    still sees the original budget and must not allocate it.
    """

    def __init__(
        self,
        *,
        longest_edge: int | None,
        max_pixels: int | None,
        shortest_edge: int = 3136,
        min_pixels: int = 3136,
    ) -> None:
        self._longest_edge = longest_edge
        self._max_pixels = max_pixels
        self._shortest_edge = shortest_edge
        self._min_pixels = min_pixels
        self.size = self

    @property
    def longest_edge(self) -> int | None:
        return self._longest_edge

    @longest_edge.setter
    def longest_edge(self, value: int) -> None:
        return

    @property
    def shortest_edge(self) -> int:
        return self._shortest_edge

    @property
    def max_pixels(self) -> int | None:
        return self._max_pixels

    @max_pixels.setter
    def max_pixels(self, value: int) -> None:
        return

    @property
    def min_pixels(self) -> int:
        return self._min_pixels

    def __getitem__(self, key: str) -> int:
        value = {
            "longest_edge": self._longest_edge,
            "shortest_edge": self._shortest_edge,
            "max_pixels": self._max_pixels,
            "min_pixels": self._min_pixels,
        }[key]
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: int) -> None:
        return

    def __or__(self, other: object) -> _FrozenBudget:
        merged = {
            "longest_edge": self._longest_edge,
            "shortest_edge": self._shortest_edge,
            "max_pixels": self._max_pixels,
            "min_pixels": self._min_pixels,
        }
        merged.update(dict(other))  # type: ignore[arg-type]
        return _FrozenBudget(
            longest_edge=merged["longest_edge"],
            max_pixels=merged["max_pixels"],
            shortest_edge=merged["shortest_edge"],
            min_pixels=merged["min_pixels"],
        )


def _write_config(directory: Path, patch_size: int, merge_size: int) -> None:
    config = {
        "model_type": "paddleocr_vl",
        "vision_config": {
            "model_type": "paddleocr_vl_vision",
            "patch_size": patch_size,
            "spatial_merge_size": merge_size,
        },
    }
    (directory / "config.json").write_text(json.dumps(config))


def _library_checkpoint(
    directory: Path,
    *,
    patch_size: int = 14,
    merge_size: int = 2,
    max_pixels: int | None = None,
    longest_edge: int | None = None,
) -> None:
    _write_config(directory, patch_size, merge_size)
    processor: dict[str, object] = {
        "image_processor_type": "PaddleOCRVLImageProcessor",
        "min_pixels": 3136,
    }
    if max_pixels is not None:
        processor["max_pixels"] = max_pixels
    if longest_edge is not None:
        processor["size"] = {
            "shortest_edge": 3136,
            "longest_edge": longest_edge,
        }
    (directory / "preprocessor_config.json").write_text(json.dumps(processor))


def _remote_checkpoint(directory: Path, max_pixels: int) -> None:
    """Remote processor constructor, matching the published Hub class.

    The published ``PaddleOCRVLImageProcessor.__init__`` stores the JSON
    ``max_pixels`` on the instance and mirrors it into ``size``.
    """
    _write_config(directory, 14, 2)
    source = """
from transformers.image_processing_utils import BaseImageProcessor


class PaddleOCRVLImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        max_pixels: int = 28 * 28 * 1280,
        min_pixels: int = 28 * 28 * 130,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
"""
    (directory / "image_processing_paddleocr_vl.py").write_text(source)
    processor = {
        "image_processor_type": "PaddleOCRVLImageProcessor",
        "auto_map": {
            "AutoImageProcessor": (
                "image_processing_paddleocr_vl.PaddleOCRVLImageProcessor"
            )
        },
        "max_pixels": max_pixels,
        "min_pixels": 3136,
    }
    (directory / "preprocessor_config.json").write_text(json.dumps(processor))


def _load_library(directory: Path, trust_remote_code: bool = False):
    image_processor = AutoImageProcessor.from_pretrained(directory)
    config = PaddleOCRVLConfig.from_pretrained(directory)
    info = PaddleOCRVLProcessingInfo(
        _CheckpointContext(trust_remote_code, config, image_processor)
    )
    return info, image_processor


def _freeze_library(image_processor, longest_edge: int):
    frozen = _FrozenBudget(longest_edge=longest_edge, max_pixels=None)
    image_processor.size = frozen
    return frozen


def _profiling(info: PaddleOCRVLProcessingInfo):
    return PaddleOCRVLDummyInputsBuilder(info)


def _guard_image_new(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    requested: list[tuple[int, int]] = []
    real_new = Image.new
    cap = _pixel_cap()

    def wrapped(mode: str, size: tuple[int, int], *args, **kwargs):
        width, height = size
        requested.append((width, height))
        pixels = width * height
        if pixels > cap:
            raise AssertionError(
                f"dummy image {width}x{height} ({pixels} pixels) exceeds "
                f"the documented cap of {cap}"
            )
        if pixels * 4 > _SAFE_ALLOCATION_BYTES:
            return real_new(mode, (1, 1), 0)
        return real_new(mode, size, *args, **kwargs)

    monkeypatch.setattr(dummy_inputs.Image, "new", wrapped)
    return requested


def test_modest_checkpoint_budget_allocates_rgb_buffer(tmp_path: Path) -> None:
    """An in-range budget is handed to Pillow and the RGB buffer is real."""
    _library_checkpoint(tmp_path, max_pixels=_MODEST_PIXELS)
    info, _processor = _load_library(tmp_path, trust_remote_code=False)
    derived = info.get_image_size_with_most_features()
    images = _profiling(info).get_dummy_mm_data(
        seq_len=128,
        mm_counts={"image": 1},
        mm_options=MultiModalDummyOptions(),
    )
    image = images["image"][0]
    pixels = derived.width * derived.height
    assert image.size == (derived.width, derived.height)
    assert image.mode == "RGB"
    assert len(image.tobytes()) == pixels * 3
    assert pixels <= _pixel_cap()
    assert pixels > _MODEST_PIXELS // 2


@pytest.mark.parametrize(
    "field",
    ["max_pixels", "longest_edge"],
)
def test_library_checkpoint_pixel_budget_is_capped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    _library_checkpoint(tmp_path, **{field: _HUGE_PIXELS})
    info, processor = _load_library(tmp_path, trust_remote_code=False)
    assert processor.size.longest_edge == _HUGE_PIXELS
    _freeze_library(processor, processor.size.longest_edge)

    derived = info.get_image_size_with_most_features()
    requested = _guard_image_new(monkeypatch)
    _profiling(info).get_dummy_mm_data(
        seq_len=128,
        mm_counts={"image": 1},
        mm_options=MultiModalDummyOptions(),
    )

    cap = _pixel_cap()
    assert derived.width * derived.height <= cap
    assert requested == [(derived.width, derived.height)]


def test_library_token_count_follows_capped_budget(tmp_path: Path) -> None:
    _library_checkpoint(tmp_path, max_pixels=_HUGE_PIXELS)
    info, processor = _load_library(tmp_path, trust_remote_code=False)
    _freeze_library(processor, processor.size.longest_edge)
    derived = info.get_image_size_with_most_features()
    tokens = info.get_num_image_tokens(
        image_width=derived.width,
        image_height=derived.height,
        image_processor=processor,
        mm_kwargs={},
    )
    vision = info.get_hf_config().vision_config
    factor = vision.patch_size * vision.spatial_merge_size
    assert tokens * factor * factor <= _pixel_cap()


def test_remote_checkpoint_max_pixels_is_capped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _remote_checkpoint(tmp_path, _HUGE_PIXELS)
    image_processor = AutoImageProcessor.from_pretrained(
        tmp_path, trust_remote_code=True
    )
    assert image_processor.max_pixels == _HUGE_PIXELS
    assert image_processor.size["max_pixels"] == _HUGE_PIXELS
    config = PaddleOCRVLConfig.from_pretrained(tmp_path)
    frozen = _FrozenBudget(
        longest_edge=None,
        max_pixels=image_processor.max_pixels,
    )
    info = PaddleOCRVLProcessingInfo(_CheckpointContext(True, config, frozen))
    derived = info.get_image_size_with_most_features()
    tokens = info.get_num_image_tokens(
        image_width=derived.width,
        image_height=derived.height,
        image_processor=frozen,
        mm_kwargs={},
    )
    requested = _guard_image_new(monkeypatch)
    _profiling(info).get_dummy_mm_data(
        seq_len=128,
        mm_counts={"image": 1},
        mm_options=MultiModalDummyOptions(),
    )
    cap = _pixel_cap()
    factor = 14 * 2
    assert derived.width * derived.height <= cap
    assert tokens * factor * factor <= cap
    assert requested == [(derived.width, derived.height)]


@pytest.mark.parametrize(
    ("patch_size", "merge_size"),
    [(0, 2), (-14, 2), (14, 0), (14, -2)],
)
def test_nonpositive_patch_or_merge_size_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_size: int,
    merge_size: int,
) -> None:
    _library_checkpoint(
        tmp_path,
        patch_size=patch_size,
        merge_size=merge_size,
        max_pixels=_MODEST_PIXELS,
    )
    info, processor = _load_library(tmp_path, trust_remote_code=False)
    requested = _guard_image_new(monkeypatch)

    with pytest.raises(ValueError):
        info.get_image_size_with_most_features()
    with pytest.raises(ValueError):
        info.get_num_image_tokens(
            image_width=28,
            image_height=28,
            image_processor=processor,
            mm_kwargs={},
        )
    with pytest.raises(ValueError):
        _profiling(info).get_dummy_mm_data(
            seq_len=128,
            mm_counts={"image": 1},
            mm_options=MultiModalDummyOptions(),
        )
    assert requested == []
