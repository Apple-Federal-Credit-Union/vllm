# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4V profiling must cap the pixel budget before dummy allocation.

``Glm4vProcessingInfo`` reads ``size.longest_edge`` from the checkpoint image
and video processors, and ``max_pixels`` / ``size`` from server
``mm_processor_kwargs``, and passes that value to ``smart_resize`` as
``max_pixels``. Startup profiling
(``MultiModalBudget`` -> ``get_dummy_mm_inputs`` -> ``get_dummy_mm_data``)
then allocates a real image and dummy video frames at the resulting size.

These tests fail until that budget is capped. The cap is the default
``max_pixels`` already used by ``_get_vision_info`` (``28 * 28 * 2 * 30000``),
which is also the video ``longest_edge`` shipped with GLM-4.1V. Official
image (9_633_792) and video (47_040_000) budgets stay within it.

The enormous-budget case asserts the profiled size before calling
``get_dummy_mm_data``. The uncapped size is on the order of
``sqrt(longest_edge)`` and allocating it would exhaust the test process.
A separate test allocates a budget that is over the cap but still fits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vllm.config.model import ModelConfig
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vProcessingInfo,
)
from vllm.multimodal.processing.context import InputProcessingContext

# Default ``max_image_pixels`` on ``Glm4vProcessingInfo._get_vision_info``.
MAX_PROFILED_PIXELS = 28 * 28 * 2 * 30000

# GLM-4.1V processor budgets (zai-org/GLM-4.1V-9B-Thinking).
_OFFICIAL_IMAGE_EDGE = 9_633_792
_OFFICIAL_VIDEO_EDGE = 47_040_000

# Over the cap, and small enough that one RGB image plus two frames fit here.
_OVERSIZED_EDGE = 200_000_000

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _glm4v_config(max_position_embeddings: int) -> dict:
    return {
        "architectures": ["Glm4vForConditionalGeneration"],
        "model_type": "glm4v",
        "image_start_token_id": 151339,
        "image_end_token_id": 151340,
        "video_start_token_id": 151341,
        "video_end_token_id": 151342,
        "image_token_id": 151343,
        "video_token_id": 151344,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "glm4v_text",
            "hidden_size": 4096,
            "intermediate_size": 13696,
            "max_position_embeddings": max_position_embeddings,
            "num_attention_heads": 32,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "vocab_size": 151552,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
        },
        "vision_config": {
            "model_type": "glm4v_vision",
            "hidden_size": 1536,
            "depth": 1,
            "num_heads": 12,
            "intermediate_size": 13696,
            "hidden_act": "silu",
            "image_size": 336,
            "patch_size": 14,
            "out_hidden_size": 4096,
            "rms_norm_eps": 1e-5,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }


def _processor_json(edge: int, *, video: bool) -> dict:
    payload = {
        "size": {"shortest_edge": 12544, "longest_edge": edge},
        "do_rescale": True,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Glm4vProcessor",
    }
    if video:
        payload["video_processor_type"] = "Glm4vVideoProcessor"
    else:
        payload["image_processor_type"] = "Glm4vImageProcessor"
    return payload


def _write_checkpoint(
    directory: Path,
    *,
    image_edge: int,
    video_edge: int,
    max_position_embeddings: int = 65536,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(_glm4v_config(max_position_embeddings))
    )
    (directory / "preprocessor_config.json").write_text(
        json.dumps(_processor_json(image_edge, video=False))
    )
    (directory / "video_preprocessor_config.json").write_text(
        json.dumps(_processor_json(video_edge, video=True))
    )


def _processing_info(
    directory: Path,
    *,
    mm_processor_kwargs: dict | None = None,
    max_model_len: int = 65536,
) -> Glm4vProcessingInfo:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = str(directory)
    model_config = ModelConfig(
        model=model,
        tokenizer=model,
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        max_model_len=max_model_len,
        skip_tokenizer_init=True,
        enforce_eager=True,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    return Glm4vProcessingInfo(InputProcessingContext(model_config, tokenizer=None))


def _allocated_pixels(info: Glm4vProcessingInfo, seq_len: int) -> tuple[int, int]:
    """Run the real dummy builder and return (image pixels, frame pixels)."""
    builder = Glm4vDummyInputsBuilder(info)
    mm = builder.get_dummy_mm_data(
        seq_len,
        {"image": 1, "video": 1},
        info.ctx.get_mm_config().limit_per_prompt,
    )
    image = mm["image"][0]
    video, _metadata = mm["video"][0]
    assert video.ndim == 4
    assert video.shape[-1] == 3
    assert video.shape[0] >= 2
    image_pixels = image.size[0] * image.size[1]
    frame_pixels = video.shape[1] * video.shape[2]
    # Video frames are allocated at the profiled image size.
    assert frame_pixels == image_pixels
    return image_pixels, frame_pixels


def test_official_glm4v_budgets_stay_within_the_cap(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        image_edge=_OFFICIAL_IMAGE_EDGE,
        video_edge=_OFFICIAL_VIDEO_EDGE,
    )
    info = _processing_info(tmp_path)
    assert info._get_image_max_pixels() <= MAX_PROFILED_PIXELS
    assert info._get_video_max_pixels() <= MAX_PROFILED_PIXELS
    image_pixels, frame_pixels = _allocated_pixels(info, seq_len=65536)
    assert 0 < image_pixels <= MAX_PROFILED_PIXELS
    assert frame_pixels <= MAX_PROFILED_PIXELS


def test_checkpoint_longest_edge_dummy_allocation_is_capped(tmp_path: Path) -> None:
    """A published preprocessor ``longest_edge`` is allocated at startup."""
    _write_checkpoint(
        tmp_path,
        image_edge=_OVERSIZED_EDGE,
        video_edge=_OVERSIZED_EDGE,
    )
    info = _processing_info(tmp_path)
    # Glm4vProcessor has no closed-form token count, so profiling builds
    # dummy media instead of skipping it.
    assert info.get_mm_max_tokens_per_item(65536, {"image": 1, "video": 1}) is None
    image_pixels, frame_pixels = _allocated_pixels(info, seq_len=65536)
    assert image_pixels <= MAX_PROFILED_PIXELS
    assert frame_pixels <= MAX_PROFILED_PIXELS
    assert info._get_image_max_pixels() <= MAX_PROFILED_PIXELS
    assert info._get_video_max_pixels() <= MAX_PROFILED_PIXELS


def test_enormous_longest_edge_profiled_size_is_capped(tmp_path: Path) -> None:
    edge = 10**12
    _write_checkpoint(tmp_path, image_edge=edge, video_edge=edge)
    info = _processing_info(tmp_path)
    size = info.get_image_size_with_most_features()
    assert size.width * size.height <= MAX_PROFILED_PIXELS
    assert info._get_image_max_pixels() <= MAX_PROFILED_PIXELS
    assert info._get_video_max_pixels() <= MAX_PROFILED_PIXELS
    image_pixels, frame_pixels = _allocated_pixels(info, seq_len=65536)
    assert image_pixels <= MAX_PROFILED_PIXELS
    assert frame_pixels <= MAX_PROFILED_PIXELS


def test_server_max_pixels_override_is_capped(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        image_edge=_OFFICIAL_IMAGE_EDGE,
        video_edge=_OFFICIAL_VIDEO_EDGE,
    )
    info = _processing_info(
        tmp_path,
        mm_processor_kwargs={"max_pixels": _OVERSIZED_EDGE},
    )
    image_pixels, frame_pixels = _allocated_pixels(info, seq_len=65536)
    assert image_pixels <= MAX_PROFILED_PIXELS
    assert frame_pixels <= MAX_PROFILED_PIXELS
    assert info._get_image_max_pixels() <= MAX_PROFILED_PIXELS
    assert info._get_video_max_pixels() <= MAX_PROFILED_PIXELS


def test_server_size_longest_edge_override_is_capped(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        image_edge=_OFFICIAL_IMAGE_EDGE,
        video_edge=_OFFICIAL_VIDEO_EDGE,
    )
    info = _processing_info(
        tmp_path,
        mm_processor_kwargs={"size": {"longest_edge": _OVERSIZED_EDGE}},
    )
    image_pixels, frame_pixels = _allocated_pixels(info, seq_len=65536)
    assert image_pixels <= MAX_PROFILED_PIXELS
    assert frame_pixels <= MAX_PROFILED_PIXELS
    assert info._get_image_max_pixels() <= MAX_PROFILED_PIXELS
    assert info._get_video_max_pixels() <= MAX_PROFILED_PIXELS


def test_video_processor_longest_edge_is_capped(tmp_path: Path) -> None:
    """Video ``longest_edge`` is a separate read from the image processor."""
    _write_checkpoint(
        tmp_path,
        image_edge=_OFFICIAL_IMAGE_EDGE,
        video_edge=10**12,
    )
    info = _processing_info(tmp_path)
    image_pixels, frame_pixels = _allocated_pixels(info, seq_len=65536)
    assert info._get_image_max_pixels() <= MAX_PROFILED_PIXELS
    assert info._get_video_max_pixels() <= MAX_PROFILED_PIXELS
    assert image_pixels <= MAX_PROFILED_PIXELS
    assert frame_pixels <= MAX_PROFILED_PIXELS
