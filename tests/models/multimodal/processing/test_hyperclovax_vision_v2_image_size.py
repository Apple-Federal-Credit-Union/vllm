# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperCLOVAX vision profiling must bound checkpoint ``image_size``.

``HCXVisionV2ProcessingInfo.get_image_size_with_most_features`` reads
``vision_config.image_size`` and, when the field is absent, uses 448.
``HCXVisionV2DummyInputsBuilder.get_dummy_mm_data`` passes that pair of
sides to the shared dummy image and video allocators. Startup profiling
(``get_mm_max_toks_per_item`` -> ``get_dummy_mm_inputs``) calls the same
builder. A checkpoint can set ``image_size`` to any JSON number; nothing
in that path limits it before Pillow and NumPy allocate.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm.config import ModelConfig
from vllm.model_executor.models.hyperclovax_vision_v2 import (
    V2_IMAGE_TOKEN,
    V2_VIDEO_TOKEN,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item
from vllm.multimodal.processing import BaseMultiModalProcessor

# Side used when the checkpoint does not set image_size. Profiling must
# not allocate a wider or taller dummy image or video frame than this.
_MAX_PROFILING_SIDE = 448
_MISSING = object()

pytestmark = pytest.mark.cpu_test


def _checkpoint(directory: Path, image_size: object) -> None:
    vision_config: dict[str, object] = {
        "model_type": "qwen2_5_vl_vision",
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "hidden_size": 64,
        "depth": 2,
        "num_heads": 4,
        "intermediate_size": 64,
        "out_hidden_size": 64,
        "window_size": 112,
        "in_channels": 3,
    }
    if image_size is not _MISSING:
        vision_config["image_size"] = image_size
    config = {
        "architectures": ["HCXVisionV2ForCausalLM"],
        "model_type": "hyperclovax_vision_v2",
        "image_token_id": 10,
        "video_token_id": 11,
        "vision_config": vision_config,
        "text_config": {
            "model_type": "hyperclovax",
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "vocab_size": 32,
            "max_position_embeddings": 256,
            "head_dim": 16,
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    _write_tokenizer(directory)


def _write_tokenizer(directory: Path) -> None:
    vocab = {"<unk>": 0, "<pad>": 1}
    for piece in (
        "<|image_start|>",
        "<|IMAGE_PAD|>",
        "<|image_end|>",
        "<|video_start|>",
        "<|VIDEO_PAD|>",
        "<|video_end|>",
        V2_IMAGE_TOKEN,
        V2_VIDEO_TOKEN,
    ):
        vocab.setdefault(piece, len(vocab))
    model = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=model,
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(directory)


def _processor(directory: Path) -> BaseMultiModalProcessor:
    model_config = ModelConfig(
        model=str(directory),
        tokenizer=str(directory),
        tokenizer_mode="hf",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        max_model_len=64,
        enforce_eager=True,
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _profile(processor: BaseMultiModalProcessor):
    mm_config = processor.info.ctx.model_config.get_multimodal_config()
    mm_options = mm_config.limit_per_prompt
    return processor.dummy_inputs.get_dummy_mm_data(
        seq_len=processor.info.ctx.model_config.max_model_len,
        mm_counts={"image": 1, "video": 1},
        mm_options=mm_options,
    )


@contextmanager
def _record_allocator_calls() -> Iterator[tuple[list[tuple], list[tuple]]]:
    """Call the real Pillow and NumPy allocators and record their shapes."""
    import vllm.multimodal.processing.dummy_inputs as dummy_inputs

    images: list[tuple] = []
    videos: list[tuple] = []
    original_new = dummy_inputs.Image.new
    original_full = dummy_inputs.np.full

    def new(mode, size, *args, **kwargs):
        images.append(tuple(size))
        return original_new(mode, size, *args, **kwargs)

    def full(shape, fill_value, *args, **kwargs):
        videos.append(tuple(shape))
        return original_full(shape, fill_value, *args, **kwargs)

    dummy_inputs.Image.new = new
    dummy_inputs.np.full = full
    try:
        yield images, videos
    finally:
        dummy_inputs.Image.new = original_new
        dummy_inputs.np.full = original_full


def _assert_sides_bounded(width: object, height: object) -> None:
    assert type(width) is int
    assert type(height) is int
    assert 0 < width <= _MAX_PROFILING_SIDE
    assert 0 < height <= _MAX_PROFILING_SIDE


def _assert_profile_bounded(data) -> None:
    image = data["image"][0]
    video = data["video"][0]
    _assert_sides_bounded(image.size[0], image.size[1])
    assert video.shape[0] == 16
    assert video.shape[3] == 3
    _assert_sides_bounded(video.shape[2], video.shape[1])


def test_missing_image_size_profiles_at_fallback(tmp_path: Path) -> None:
    _checkpoint(tmp_path, _MISSING)
    processor = _processor(tmp_path)
    size = processor.info.get_image_size_with_most_features()
    assert (size.width, size.height) == (_MAX_PROFILING_SIDE, _MAX_PROFILING_SIDE)
    _assert_profile_bounded(_profile(processor))


def test_in_range_image_size_is_preserved(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 224)
    processor = _processor(tmp_path)
    size = processor.info.get_image_size_with_most_features()
    assert (size.width, size.height) == (224, 224)
    data = _profile(processor)
    assert data["image"][0].size == (224, 224)
    assert data["video"][0].shape == (16, 224, 224, 3)


def test_oversized_image_size_is_capped_for_image_and_video(tmp_path: Path) -> None:
    """1024 is above the profiling fallback and is what the allocators receive.

    A checkpoint value of 100000 is covered by the size query below. This
    test checks the bytes actually allocated for both modalities.
    """
    _checkpoint(tmp_path, 1024)
    processor = _processor(tmp_path)
    size = processor.info.get_image_size_with_most_features()
    _assert_sides_bounded(size.width, size.height)
    with _record_allocator_calls() as (images, videos):
        data = _profile(processor)
    assert images, "dummy image allocator was not reached"
    assert videos, "dummy video allocator was not reached"
    for width, height in images:
        _assert_sides_bounded(width, height)
    for shape in videos:
        _assert_sides_bounded(shape[2], shape[1])
    _assert_profile_bounded(data)
    # The returned buffers must be the ones just allocated, not a later copy
    # that hides an earlier uncapped allocation.
    assert images[0] == data["image"][0].size
    assert videos[0][1] == data["video"][0].shape[1]
    assert videos[0][2] == data["video"][0].shape[2]


def test_huge_image_size_is_capped_without_allocating_it(tmp_path: Path) -> None:
    """100000 px square RGB video is 16 * 100000**2 * 3 bytes (~447 GiB)."""
    _checkpoint(tmp_path, 100_000)
    processor = _processor(tmp_path)
    with _record_allocator_calls() as (images, videos):
        size = processor.info.get_image_size_with_most_features()
    assert images == []
    assert videos == []
    _assert_sides_bounded(size.width, size.height)


@pytest.mark.parametrize("image_size", [0, -1024, 1.5, "8192", None, True])
def test_non_positive_or_non_int_image_size_does_not_reach_allocators(
    tmp_path: Path, image_size: object
) -> None:
    _checkpoint(tmp_path, image_size)
    try:
        processor = _processor(tmp_path)
    except (TypeError, ValueError, OverflowError):
        return
    with _record_allocator_calls() as (images, videos):
        try:
            data = _profile(processor)
        except (TypeError, ValueError, OverflowError):
            data = None
    for width, height in images:
        _assert_sides_bounded(width, height)
    for shape in videos:
        _assert_sides_bounded(shape[2], shape[1])
    if data is not None:
        _assert_profile_bounded(data)


def test_startup_profiling_allocates_capped_image_and_video(tmp_path: Path) -> None:
    """The engine asks for per-item token ceilings before it serves.

    ``get_mm_max_toks_per_item`` is what ``MultiModalBudget`` calls. This
    model does not override ``get_mm_max_tokens_per_item``, so that call
    builds dummy image and video inputs.
    """
    _checkpoint(tmp_path, 1024)
    processor = _processor(tmp_path)
    with _record_allocator_calls() as (images, videos):
        try:
            get_mm_max_toks_per_item(
                processor,
                {"image": 1, "video": 1},
                cache=None,
            )
        except OSError:
            # Processor files may be absent. The dummy allocation this test
            # watches happens before that load.
            pass
    assert images, "startup profiling did not allocate a dummy image"
    assert videos, "startup profiling did not allocate a dummy video"
    for width, height in images:
        _assert_sides_bounded(width, height)
    for shape in videos:
        assert shape[0] == 16
        _assert_sides_bounded(shape[2], shape[1])


def test_dummy_inputs_entry_uses_the_same_cap(tmp_path: Path) -> None:
    """``HCXVisionV2MultiModalProcessor.get_dummy_inputs`` allocates too."""
    _checkpoint(tmp_path, 1024)
    processor = _processor(tmp_path)
    mm_config = processor.info.ctx.model_config.get_multimodal_config()
    mm_options = mm_config.limit_per_prompt
    with _record_allocator_calls() as (images, videos):
        processor.get_dummy_inputs(
            seq_len=64,
            mm_counts={"image": 1, "video": 1},
            mm_options=mm_options,
        )
    assert images and videos
    for width, height in images:
        _assert_sides_bounded(width, height)
    for shape in videos:
        _assert_sides_bounded(shape[2], shape[1])

