# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profiling must not allocate dummy media at an uncapped checkpoint size.

HyperCLOVAX V2 profiling reads ``vision_config.image_size`` and uses it as
both sides of the dummy image and of every dummy video frame. The vision
tower config does not define that field; the implementation's own fallback
when it is absent is 448. A checkpoint can still set the attribute, and
startup profiling (``get_dummy_inputs``, which ``get_dummy_mm_inputs`` calls
while building the multimodal budget) allocates the image and the 16-frame
video at that size.

These tests load a local checkpoint through ``ModelConfig`` and build dummy
inputs through the registered processor. They fail while that size is used
unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

# CPU-only: global cleanup calls torch.accelerator.empty_cache().
pytestmark = pytest.mark.skip_global_cleanup

# Fallback in HCXVisionV2ProcessingInfo.get_image_size_with_most_features.
_MAX_PROFILE_SIDE = 448
_PROFILE_FRAMES = 16


def _write_checkpoint(path: Path, vision_config: dict) -> None:
    vocab = {
        "<unk>": 0,
        "<|image_start|>": 1,
        "<|IMAGE_PAD|>": 2,
        "<|image_end|>": 3,
        "<|video_start|>": 4,
        "<|VIDEO_PAD|>": 5,
        "<|video_end|>": 6,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "model_max_length": 256,
            }
        )
    )
    config = {
        "model_type": "hyperclovax_vision_v2",
        "architectures": ["HCXVisionV2ForCausalLM"],
        "image_token_id": 2,
        "video_token_id": 5,
        "text_config": {
            "model_type": "hyperclovax",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": len(vocab),
            "max_position_embeddings": 256,
            "tie_word_embeddings": False,
            "bos_token_id": 0,
            "eos_token_id": 0,
        },
        "vision_config": {
            "model_type": "qwen2_5_vl_vision",
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 128,
            "window_size": 112,
            "fullatt_block_indexes": [0],
            "tokens_per_second": 2,
            **vision_config,
        },
    }
    (path / "config.json").write_text(json.dumps(config))


def _profile_media(path: Path) -> tuple[Image.Image, np.ndarray]:
    """Run the dummy-input call that multimodal profiling makes at startup."""
    model_config = ModelConfig(
        str(path),
        max_model_len=256,
        enforce_eager=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    # Startup takes this branch: no precomputed per-item token count.
    assert (
        processor.info.get_mm_max_tokens_per_item(
            seq_len=model_config.max_model_len,
            mm_counts={"image": 1, "video": 1},
        )
        is None
    )
    mm_options = model_config.get_multimodal_config().limit_per_prompt
    inputs = processor.get_dummy_inputs(
        seq_len=model_config.max_model_len,
        mm_counts={"image": 1, "video": 1},
        mm_options=mm_options,
    )
    image = inputs.mm_data_items["image"].get(0)
    video = inputs.mm_data_items["video"].get(0)
    assert isinstance(image, Image.Image)
    assert isinstance(video, np.ndarray)
    return image, video


def _assert_within_profile_bound(image: Image.Image, video: np.ndarray) -> None:
    width, height = image.size
    assert width <= _MAX_PROFILE_SIDE
    assert height <= _MAX_PROFILE_SIDE
    assert width * height * 3 <= _MAX_PROFILE_SIDE * _MAX_PROFILE_SIDE * 3
    assert video.ndim == 4
    frames, frame_h, frame_w, channels = video.shape
    assert channels == 3
    assert frames == _PROFILE_FRAMES
    assert frame_h <= _MAX_PROFILE_SIDE
    assert frame_w <= _MAX_PROFILE_SIDE
    assert video.nbytes <= frames * _MAX_PROFILE_SIDE * _MAX_PROFILE_SIDE * channels


def test_missing_image_size_uses_profile_fallback(tmp_path: Path) -> None:
    """A checkpoint that omits image_size profiles at the 448 fallback."""
    _write_checkpoint(tmp_path, {})
    image, video = _profile_media(tmp_path)
    assert image.size == (_MAX_PROFILE_SIDE, _MAX_PROFILE_SIDE)
    assert video.shape == (
        _PROFILE_FRAMES,
        _MAX_PROFILE_SIDE,
        _MAX_PROFILE_SIDE,
        3,
    )


def test_checkpoint_image_size_cannot_grow_dummy_media(tmp_path: Path) -> None:
    """A large vision_config.image_size must not be the allocated resolution.

    2048 is above the 448 fallback and still small enough to allocate here.
    The same integer is the image side and each video frame's side.
    """
    _write_checkpoint(tmp_path, {"image_size": 2048})
    image, video = _profile_media(tmp_path)
    _assert_within_profile_bound(image, video)
    # The checkpoint value itself must not be what was allocated.
    assert image.size != (2048, 2048)
    assert video.shape[1:3] != (2048, 2048)


@pytest.mark.parametrize("image_size", [-2048, 0])
def test_non_positive_image_size_does_not_allocate_oversized_media(
    tmp_path: Path, image_size: int
) -> None:
    """Zero or negative image_size must not allocate an oversized raster.

    Rejecting the checkpoint and clamping to the profile fallback are both
    acceptable. Taking the absolute value and allocating that raster is not.
    """
    _write_checkpoint(tmp_path, {"image_size": image_size})
    try:
        image, video = _profile_media(tmp_path)
    except (TypeError, ValueError):
        return
    _assert_within_profile_bound(image, video)
