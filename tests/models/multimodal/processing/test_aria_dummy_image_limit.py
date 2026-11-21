# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling must not allocate an Aria dummy image past the pixel cap.

These tests fail while ``AriaDummyInputsBuilder`` copies
``vision_config.image_size`` into ``Image.new`` with no bound. A fix should
reject that checkpoint, or clamp the size it actually allocates to
``VLLM_MAX_IMAGE_PIXELS``, and it must use the same value it checked.

The same unbounded copy also exists in
``PaliGemmaDummyInputsBuilder``, ``Blip2DummyInputsBuilder``, and
``BagelDummyInputsBuilder``, and ``InternVLDummyInputsBuilder`` passes
``vision_config.image_size`` to ``_get_dummy_videos``. Those paths are not
exercised here. Builders that size dummy images through
``get_image_size_with_most_features`` are not exercised here either.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import vllm.envs as envs
import vllm.multimodal.processing.dummy_inputs as dummy_inputs
from vllm.config.model import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Small enough to allocate safely, large enough to exceed the patched cap.
_OVERSIZED_SIDE = 64
_LEGAL_SIDE = 8
_PIXEL_LIMIT = 32 * 32


def _write_aria_checkpoint(directory: Path, image_size: int) -> None:
    """Write a local Aria checkpoint whose vision config sets ``image_size``."""
    config = {
        "architectures": ["AriaForConditionalGeneration"],
        "model_type": "aria",
        "image_token_index": 9,
        "projector_patch_to_query_dict": {"1225": 128, "4900": 256},
        "vision_config": {
            "model_type": "idefics3_vision",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": image_size,
            "patch_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
        },
        "text_config": {
            "model_type": "aria_text",
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "moe_num_experts": 2,
            "moe_topk": 1,
            "moe_num_shared_experts": 1,
        },
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")

    tokenizer = Tokenizer(
        WordLevel(
            {"<|img|>": 9, "<|endoftext|>": 2, "hello": 10},
            unk_token="hello",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 128,
                "image_token": "<|img|>",
                "extra_special_tokens": {
                    "image_token": "<|img|>",
                    "pad_token": "<|endoftext|>",
                },
                "pad_token": "<|endoftext|>",
                "unk_token": "hello",
            }
        ),
        encoding="utf-8",
    )
    # Present so processor construction can resolve an image processor.
    # Profiling reads ``vision_config.image_size``, not this file, for the
    # dummy image dimensions.
    (directory / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "Idefics3ImageProcessor",
                "processor_class": "AriaProcessor",
                "max_image_size": 64,
                "size": {"longest_edge": 64},
                "do_rescale": False,
                "do_normalize": False,
            }
        ),
        encoding="utf-8",
    )
    (directory / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "AriaProcessor",
                "size_conversion": {"490": 128, "980": 256},
            }
        ),
        encoding="utf-8",
    )


class _RereadVisionConfig:
    """Vision config whose ``image_size`` changes after the first read.

    A check that reads the attribute and then reads it again for ``Image.new``
    allocates the later value.
    """

    def __init__(self, inner: object, first: int, later: int) -> None:
        self._inner = inner
        self._first = first
        self._later = later
        self.reads = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    @property
    def image_size(self) -> int:
        self.reads += 1
        if self.reads == 1:
            return self._first
        return self._later


def _load_aria(directory: Path, monkeypatch: pytest.MonkeyPatch) -> ModelConfig:
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", _PIXEL_LIMIT)
    model_config = ModelConfig(
        str(directory),
        trust_remote_code=False,
        max_model_len=32,
    )
    assert model_config.supports_multimodal_inputs
    return model_config


def _track_dummy_image_new(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int]]:
    recorded: list[tuple[int, int]] = []
    original_new = dummy_inputs.Image.new

    def tracking_new(
        mode: str,
        size: tuple[int, int],
        color: int | tuple[int, ...] | str | None = 0,
    ) -> Image.Image:
        width, height = size
        recorded.append((int(width), int(height)))
        return original_new(mode, size, color)

    monkeypatch.setattr(dummy_inputs.Image, "new", tracking_new)
    return recorded


def _assert_within_limit(recorded: list[tuple[int, int]], limit: int) -> None:
    for width, height in recorded:
        pixels = width * height
        assert pixels <= limit, (
            f"profiling allocated a {width}x{height} image "
            f"({pixels} pixels) above VLLM_MAX_IMAGE_PIXELS={limit}"
        )


def _run_startup_profiling(model_config: ModelConfig) -> None:
    """Call the dummy-input entry point used while the engine is starting.

    ``MultiModalBudget`` calls ``get_dummy_mm_inputs`` because Aria does not
    override ``get_mm_max_tokens_per_item``. Image construction happens inside
    that call, before the HF processor runs.
    """
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    assert processor.info.get_mm_max_tokens_per_item(32, {"image": 1}) is None
    processor.get_dummy_mm_inputs({"image": 1})


def test_aria_startup_profiling_bounds_checkpoint_image_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint image_size above the pixel cap must not be allocated.

    ``get_dummy_mm_inputs`` is the call ``MultiModalBudget`` makes while the
    engine is starting. The HF processor may raise after the dummy image
    already exists; that allocation still counts.
    """
    _write_aria_checkpoint(tmp_path, _OVERSIZED_SIDE)
    recorded = _track_dummy_image_new(monkeypatch)
    try:
        model_config = _load_aria(tmp_path, monkeypatch)
    except Exception:
        _assert_within_limit(recorded, _PIXEL_LIMIT)
        return

    checkpoint_size = json.loads((tmp_path / "config.json").read_text())[
        "vision_config"
    ]["image_size"]
    assert checkpoint_size == _OVERSIZED_SIDE
    assert checkpoint_size * checkpoint_size > envs.VLLM_MAX_IMAGE_PIXELS

    completed = False
    try:
        _run_startup_profiling(model_config)
        completed = True
    except Exception:
        completed = False
    if completed:
        assert recorded, "profiling finished without creating a dummy image"
    _assert_within_limit(recorded, envs.VLLM_MAX_IMAGE_PIXELS)


def test_aria_startup_profiling_keeps_image_size_under_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint image_size inside the cap is still the profiled size."""
    _write_aria_checkpoint(tmp_path, _LEGAL_SIDE)
    model_config = _load_aria(tmp_path, monkeypatch)
    recorded = _track_dummy_image_new(monkeypatch)
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    mm_options = model_config.get_multimodal_config().limit_per_prompt
    inputs = processor.get_dummy_inputs(
        seq_len=model_config.max_model_len,
        mm_counts={"image": 2},
        mm_options=mm_options,
    )

    images = inputs.mm_data_items["image"]
    assert len(images) == 2
    assert recorded == [(_LEGAL_SIDE, _LEGAL_SIDE)]
    for index in range(len(images)):
        image = images.get(index)
        assert image.size == (_LEGAL_SIDE, _LEGAL_SIDE)
        assert len(image.tobytes()) == _LEGAL_SIDE * _LEGAL_SIDE * 3
    _assert_within_limit(recorded, envs.VLLM_MAX_IMAGE_PIXELS)


def test_aria_profiling_uses_the_image_size_it_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The size passed to Pillow is the size that was bounds-checked.

    The first read is inside the pixel cap. Every later read is not. Reading
    the field again between the check and ``Image.new`` allocates the later
    value.
    """
    _write_aria_checkpoint(tmp_path, _LEGAL_SIDE)
    model_config = _load_aria(tmp_path, monkeypatch)
    # Bypass the config object's type check. Profiling only reads this
    # attribute; the replacement returns a different integer on each read.
    object.__setattr__(
        model_config.hf_config,
        "vision_config",
        _RereadVisionConfig(
            model_config.hf_config.vision_config,
            first=_LEGAL_SIDE,
            later=_OVERSIZED_SIDE,
        ),
    )
    recorded = _track_dummy_image_new(monkeypatch)
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    mm_options = model_config.get_multimodal_config().limit_per_prompt

    # Rejection is acceptable only when it happens before an oversized image
    # is created. ``suppress`` still leaves ``recorded`` for that check.
    with contextlib.suppress(Exception):
        processor.get_dummy_inputs(
            seq_len=model_config.max_model_len,
            mm_counts={"image": 1},
            mm_options=mm_options,
        )
    _assert_within_limit(recorded, envs.VLLM_MAX_IMAGE_PIXELS)
    if recorded:
        assert recorded == [(_LEGAL_SIDE, _LEGAL_SIDE)]


def test_tracked_image_new_still_allocates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Image.new wrapper used above calls Pillow's allocator."""
    recorded = _track_dummy_image_new(monkeypatch)
    image = dummy_inputs.Image.new("RGB", (2, 3), color=255)
    assert recorded == [(2, 3)]
    assert image.size == (2, 3)
    assert len(image.tobytes()) == 2 * 3 * 3
