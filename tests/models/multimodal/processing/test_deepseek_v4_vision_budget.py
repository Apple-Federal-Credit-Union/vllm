# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling must bound DeepSeek vision grids from the checkpoint.

A checkpoint's ``vision_max_n_token`` / ``vision_config.max_image_tokens``,
``vision_patch_size``, and ``vision_downsample_ratio`` are read while the
server builds multimodal dummy inputs. Those inputs are what memory
profiling and the renderer warmup ask for before the API is ready.
"""

import json
import signal
from pathlib import Path

import pytest

from vllm.config import ModelConfig
from vllm.envs import VLLM_MAX_IMAGE_PIXELS
from vllm.exceptions import VLLMValidationError
from vllm.models.deepseek_v4.common.mm_preprocess import (
    IMAGE_PLACEHOLDER as V4_IMAGE_PLACEHOLDER,
)
from vllm.models.deepseek_v4.common.mm_preprocess import (
    IMAGE_SENTINEL_BASE_ID as V4_SENTINEL_BASE_ID,
)
from vllm.models.deepseek_v4.common.mm_preprocess import (
    IMAGE_SENTINEL_TOKEN_NAMES as V4_SENTINEL_NAMES,
)
from vllm.models.deepseek_v4.common.vl_model import (
    DeepseekV4ForConditionalGeneration,
)
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE_PLACEHOLDER as V41_IMAGE_PLACEHOLDER,
)
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE_SENTINEL_BASE_ID as V41_SENTINEL_BASE_ID,
)
from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM
from vllm.multimodal.processing.context import InputProcessingContext

pytestmark = pytest.mark.skip_global_cleanup

_ACCEPTED_REJECTION = (ValueError, VLLMValidationError)
_PROFILE_DEADLINE_S = 2


class _PlaceholderTokenizer:
    """Enough of a tokenizer for dummy-input profiling to run."""

    def __init__(
        self,
        placeholder: str,
        placeholder_id: int,
        extra: dict[str, int] | None = None,
    ) -> None:
        self._placeholder = placeholder
        self._placeholder_id = placeholder_id
        self._extra = extra or {}

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        if not text:
            return []
        return [self._placeholder_id] * text.count(self._placeholder)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._placeholder * len(list(token_ids))

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int] | None:
        if isinstance(tokens, str):
            if tokens == self._placeholder:
                return self._placeholder_id
            return self._extra.get(tokens)
        return [self.convert_tokens_to_ids(token) for token in tokens]


def _call_within_deadline(fn, seconds: int):
    def _handle_alarm(signum, frame):
        raise TimeoutError(f"profiling did not finish within {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handle_alarm)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _model_config(directory: Path) -> ModelConfig:
    return ModelConfig(
        model=str(directory),
        tokenizer=str(directory),
        skip_tokenizer_init=True,
        trust_remote_code=False,
        enforce_eager=True,
        max_model_len=4096,
    )


def _write_v4_checkpoint(
    directory: Path,
    *,
    vision_max_n_token: int,
    vision_patch_size: int,
    vision_downsample_ratio: int,
) -> None:
    # Same flat vision fields as deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.
    # vision_n_layers > 0 is what routes the text architecture onto the
    # multimodal class that profiles images.
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "vocab_size": 1024,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "vision_n_layers": 1,
        "vision_dim": 128,
        "vision_n_heads": 4,
        "vision_inter_dim": 256,
        "vision_patch_size": vision_patch_size,
        "vision_downsample_ratio": vision_downsample_ratio,
        "vision_max_n_token": vision_max_n_token,
        "vision_min_pixels": 1,
        "vision_max_wh_ratio": 8,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_v41_checkpoint(
    directory: Path,
    *,
    max_image_tokens: int,
    patch_size: int,
    downsample_ratio: int,
) -> None:
    config = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "text_config": {
            "model_type": "deepseek_v41",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "vocab_size": 1024,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 64,
        },
        "vision_config": {
            "num_hidden_layers": 1,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "patch_size": patch_size,
            "downsample_ratio": downsample_ratio,
            "max_image_tokens": max_image_tokens,
            "min_pixels": 1,
            "max_wh_ratio": 8,
        },
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _v4_processor(model_config: ModelConfig):
    assert model_config.hf_config.architectures == [
        "DeepseekV4ForConditionalGeneration"
    ]
    assert model_config.multimodal_config is not None
    assert not model_config.multimodal_config.skip_mm_profiling
    assert model_config.multimodal_config.get_limit_per_prompt("image") > 0
    tokenizer = _PlaceholderTokenizer(
        V4_IMAGE_PLACEHOLDER,
        placeholder_id=7,
        extra={
            name: V4_SENTINEL_BASE_ID + index
            for index, name in enumerate(V4_SENTINEL_NAMES)
        },
    )
    ctx = InputProcessingContext(model_config, tokenizer=tokenizer)
    return DeepseekV4ForConditionalGeneration._processor_factory.build_processor(ctx)


def _v41_processor(model_config: ModelConfig):
    assert model_config.hf_config.architectures == ["DeepseekV41ForCausalLM"]
    assert model_config.hf_config.vision_n_layers > 0
    assert model_config.multimodal_config is not None
    assert not model_config.multimodal_config.skip_mm_profiling
    assert model_config.multimodal_config.get_limit_per_prompt("image") > 0
    tokenizer = _PlaceholderTokenizer(
        V41_IMAGE_PLACEHOLDER,
        placeholder_id=V41_SENTINEL_BASE_ID,
    )
    ctx = InputProcessingContext(model_config, tokenizer=tokenizer)
    return DeepseekV41ForCausalLM._processor_factory.build_processor(ctx)


def _assert_dummy_image_within_media_budget(processor) -> None:
    """The image profiling would hand to ``Image.new`` must fit the media cap.

    ``Image.new`` allocates ``width * height * bands`` and does not consult
    ``VLLM_MAX_IMAGE_PIXELS``. Rejection has to happen before that call.
    """
    mm_options = processor.info.ctx.get_mm_config().limit_per_prompt
    try:
        dummy = processor.dummy_inputs.get_dummy_mm_data(
            seq_len=processor.info.ctx.model_config.max_model_len,
            mm_counts={"image": 1},
            mm_options=mm_options,
        )
    except _ACCEPTED_REJECTION:
        return
    images = dummy["image"]
    assert images, "profiling requested an image"
    for image in images:
        pixels = image.size[0] * image.size[1]
        assert pixels <= VLLM_MAX_IMAGE_PIXELS, (
            f"dummy image is {image.size[0]}x{image.size[1]} ({pixels} pixels); "
            f"VLLM_MAX_IMAGE_PIXELS is {VLLM_MAX_IMAGE_PIXELS}"
        )


def _assert_profiling_finishes(processor) -> None:
    def _profile():
        return processor.get_dummy_mm_inputs({"image": 1})

    try:
        _call_within_deadline(_profile, _PROFILE_DEADLINE_S)
    except _ACCEPTED_REJECTION:
        return
    _assert_dummy_image_within_media_budget(processor)


def test_v4_checkpoint_token_budget_does_not_stall_profiling(tmp_path: Path):
    """A huge vision_max_n_token must not make profiling scan that budget."""
    _write_v4_checkpoint(
        tmp_path,
        vision_max_n_token=10**9,
        vision_patch_size=14,
        vision_downsample_ratio=3,
    )
    processor = _v4_processor(_model_config(tmp_path))
    assert processor.info.get_hf_config().vision_max_n_token == 10**9
    _assert_profiling_finishes(processor)


def test_v4_checkpoint_patch_size_dummy_image_respects_media_budget(tmp_path: Path):
    """patch_size scales the dummy image even when the token budget is small."""
    _write_v4_checkpoint(
        tmp_path,
        vision_max_n_token=64,
        vision_patch_size=512,
        vision_downsample_ratio=4,
    )
    processor = _v4_processor(_model_config(tmp_path))
    _assert_dummy_image_within_media_budget(processor)


def test_v4_checkpoint_downsample_ratio_dummy_image_respects_media_budget(
    tmp_path: Path,
):
    """downsample_ratio scales the dummy image with a normal patch size."""
    _write_v4_checkpoint(
        tmp_path,
        vision_max_n_token=32,
        vision_patch_size=14,
        vision_downsample_ratio=208,
    )
    processor = _v4_processor(_model_config(tmp_path))
    _assert_dummy_image_within_media_budget(processor)


def test_v41_checkpoint_max_image_tokens_does_not_stall_profiling(tmp_path: Path):
    """vision_config.max_image_tokens is the V4.1 name of the same budget."""
    _write_v41_checkpoint(
        tmp_path,
        max_image_tokens=10**9,
        patch_size=14,
        downsample_ratio=3,
    )
    processor = _v41_processor(_model_config(tmp_path))
    assert processor.info.get_hf_config().vision_max_n_token == 10**9
    _assert_profiling_finishes(processor)


def test_v41_checkpoint_patch_size_dummy_image_respects_media_budget(tmp_path: Path):
    _write_v41_checkpoint(
        tmp_path,
        max_image_tokens=64,
        patch_size=512,
        downsample_ratio=4,
    )
    processor = _v41_processor(_model_config(tmp_path))
    _assert_dummy_image_within_media_budget(processor)


@pytest.mark.parametrize(
    ("max_image_tokens", "patch_size", "downsample_ratio"),
    [(1024, 14, 3), (64, 14, 3)],
)
def test_v41_modest_checkpoint_still_profiles(
    tmp_path: Path,
    max_image_tokens: int,
    patch_size: int,
    downsample_ratio: int,
):
    """Budgets in the published range still produce a dummy image."""
    _write_v41_checkpoint(
        tmp_path,
        max_image_tokens=max_image_tokens,
        patch_size=patch_size,
        downsample_ratio=downsample_ratio,
    )
    processor = _v41_processor(_model_config(tmp_path))
    output = processor.get_dummy_mm_inputs({"image": 1})
    assert output["mm_kwargs"]
    _assert_dummy_image_within_media_budget(processor)


def test_v4_published_budget_still_profiles(tmp_path: Path):
    """The published Flash-Vision budget (384 tokens, patch 14) still profiles."""
    _write_v4_checkpoint(
        tmp_path,
        vision_max_n_token=384,
        vision_patch_size=14,
        vision_downsample_ratio=3,
    )
    processor = _v4_processor(_model_config(tmp_path))
    output = processor.get_dummy_mm_inputs({"image": 1})
    assert output["mm_kwargs"]
    _assert_dummy_image_within_media_budget(processor)
