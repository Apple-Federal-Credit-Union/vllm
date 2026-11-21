# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profiling must finish for a LLaVA-OneVision checkpoint with a zero patch grid.

``LlavaOnevisionProcessingInfo._get_max_video_frames`` adds frames until the
token count exceeds the context length. ``LlavaNextVideoProcessingInfo`` does
the same with its CLIP grid. The per-frame count comes from the checkpoint
vision grid. A non-positive grid is accepted, each extra frame adds no tokens,
and startup profiling never returns.

These tests load a local checkpoint through ``ModelConfig`` and call the same
profiler ``MultiModalBudget`` uses. They fail while that call does not finish.
"""

import multiprocessing
import os
import queue as queue_mod
import traceback
from collections.abc import Mapping
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    LlavaNextVideoConfig,
    LlavaNextVideoProcessor,
    LlavaOnevisionConfig,
    LlavaOnevisionImageProcessor,
    LlavaOnevisionProcessor,
    LlavaOnevisionVideoProcessor,
    PreTrainedTokenizerFast,
    Qwen2Config,
    SiglipVisionConfig,
)
from transformers.models.llava_next_video.video_processing_llava_next_video import (
    LlavaNextVideoVideoProcessor,
)

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_PROFILE_TIMEOUT_S = 30


def _text_config() -> Qwen2Config:
    return Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        vocab_size=8,
    )


def _vision_config(
    kind: str,
    image_size: int,
    patch_size: int,
) -> SiglipVisionConfig | CLIPVisionConfig:
    kwargs = {
        "image_size": image_size,
        "patch_size": patch_size,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
    }
    if kind == "siglip":
        return SiglipVisionConfig(**kwargs)
    if kind == "clip":
        return CLIPVisionConfig(**kwargs)
    raise ValueError(kind)


def _write_checkpoint(
    directory: Path,
    *,
    vision: str,
    image_size: int,
    patch_size: int,
) -> None:
    """Write a local HF checkpoint whose vision grid is ``image_size // patch_size``."""
    vision_config = _vision_config(vision, image_size, patch_size)
    config = LlavaOnevisionConfig(
        vision_config=vision_config.to_dict(),
        text_config=_text_config().to_dict(),
    )
    config.architectures = ["LlavaOnevisionForConditionalGeneration"]
    config.save_pretrained(directory)
    tokenizer = _save_tokenizer(directory)

    grid = image_size // patch_size if patch_size else 0
    processor = LlavaOnevisionProcessor(
        image_processor=LlavaOnevisionImageProcessor(),
        video_processor=LlavaOnevisionVideoProcessor(),
        tokenizer=tokenizer,
        num_image_tokens=grid * grid if grid > 0 else 1,
    )
    processor.save_pretrained(directory)


def _save_tokenizer(directory: Path) -> PreTrainedTokenizerFast:
    word_level = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<image>": 1, "<video>": 2, "a": 3},
            unk_token="<unk>",
        )
    )
    word_level.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=word_level,
        unk_token="<unk>",
        pad_token="<unk>",
        bos_token="<unk>",
        eos_token="<unk>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<video>"]})
    tokenizer.save_pretrained(directory)
    return tokenizer


def _write_next_video_checkpoint(
    directory: Path,
    *,
    image_size: int,
    patch_size: int,
) -> None:
    """Write a LLaVA-NeXT-Video checkpoint. Its frame search uses the same grid."""
    vision_config = _vision_config("clip", image_size, patch_size)
    config = LlavaNextVideoConfig(
        vision_config=vision_config.to_dict(),
        text_config=_text_config().to_dict(),
    )
    config.architectures = ["LlavaNextVideoForConditionalGeneration"]
    config.save_pretrained(directory)
    tokenizer = _save_tokenizer(directory)
    processor = LlavaNextVideoProcessor(
        video_processor=LlavaNextVideoVideoProcessor(),
        image_processor=CLIPImageProcessor(),
        tokenizer=tokenizer,
        patch_size=patch_size,
    )
    processor.save_pretrained(directory)


def _profile_worker(
    model_dir: str,
    mm_counts: Mapping[str, int],
    expect_zero_grid: bool,
    queue: multiprocessing.Queue,
) -> None:
    try:
        model_config = ModelConfig(
            model_dir,
            runner="generate",
            tokenizer=model_dir,
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float32",
            seed=0,
            max_model_len=128,
            enforce_eager=True,
        )
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
        try:
            grid = processor.info.get_vision_encoder_info().get_patch_grid_length()
        except (ValueError, AssertionError) as exc:
            queue.put(("rejected", f"{type(exc).__name__}: {exc}"))
            return
        if expect_zero_grid and grid > 0:
            queue.put(("error", f"expected a non-positive patch grid, got {grid}"))
            return
        if not expect_zero_grid and grid <= 0:
            queue.put(("error", f"expected a positive patch grid, got {grid}"))
            return

        seq_len = model_config.max_model_len
        # Startup profiling. This is the call that hangs when the grid is zero.
        try:
            profiled = get_mm_max_toks_per_item(
                processor,
                mm_counts,
                cache=None,
            )
            profile_outcome = ("done", repr(sorted(profiled.items())))
        except (ValueError, AssertionError) as exc:
            profile_outcome = ("rejected", f"{type(exc).__name__}: {exc}")

        # Same frame search, reached from the public token-count helper.
        info = processor.info
        if hasattr(info, "get_max_video_tokens"):
            video_tokens = info.get_max_video_tokens(seq_len, mm_counts)
        else:
            video_tokens = info.get_num_frames_with_most_features(seq_len, mm_counts)
        if not isinstance(video_tokens, int) or video_tokens < 0:
            queue.put(("error", f"frame search returned {video_tokens!r}"))
            return
        if not expect_zero_grid and video_tokens <= 0:
            queue.put(("error", f"positive grid produced {video_tokens} video tokens"))
            return
        queue.put(("done", f"profile={profile_outcome!r} video_tokens={video_tokens}"))
    except (ValueError, AssertionError) as exc:
        queue.put(("rejected", f"{type(exc).__name__}: {exc}"))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def _assert_profiling_finishes(
    model_dir: Path,
    mm_counts: Mapping[str, int],
    *,
    expect_zero_grid: bool,
    require_token_counts: bool,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_profile_worker,
        args=(str(model_dir), dict(mm_counts), expect_zero_grid, queue),
    )
    proc.start()
    try:
        try:
            status, payload = queue.get(timeout=_PROFILE_TIMEOUT_S)
        except queue_mod.Empty:
            pytest.fail(
                "multimodal memory profiling did not finish within "
                f"{_PROFILE_TIMEOUT_S}s"
            )
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        else:
            proc.join(5)

    if status == "error":
        pytest.fail(payload)
    if require_token_counts:
        assert status == "done", payload
        assert "video_tokens=" in payload
        return
    assert status in {"done", "rejected"}, payload


@pytest.mark.parametrize(
    ("vision", "image_size", "patch_size", "mm_counts"),
    [
        ("siglip", 0, 14, {"image": 1, "video": 1}),
        ("clip", 0, 14, {"image": 1, "video": 1}),
        ("siglip", 1, 14, {"image": 1, "video": 1}),
        ("siglip", 0, 14, {"image": 1}),
    ],
)
def test_zero_patch_grid_profiling_terminates(
    tmp_path: Path,
    vision: str,
    image_size: int,
    patch_size: int,
    mm_counts: dict[str, int],
) -> None:
    """A published checkpoint with a zero vision grid must not hang profiling."""
    _write_checkpoint(
        tmp_path,
        vision=vision,
        image_size=image_size,
        patch_size=patch_size,
    )
    _assert_profiling_finishes(
        tmp_path,
        mm_counts,
        expect_zero_grid=True,
        require_token_counts=False,
    )


def test_positive_patch_grid_profiling_returns_token_counts(tmp_path: Path) -> None:
    """The same profiler returns token counts when the vision grid is positive."""
    _write_checkpoint(
        tmp_path,
        vision="siglip",
        image_size=224,
        patch_size=14,
    )
    _assert_profiling_finishes(
        tmp_path,
        {"image": 1, "video": 1},
        expect_zero_grid=False,
        require_token_counts=True,
    )


def test_next_video_zero_patch_grid_profiling_terminates(tmp_path: Path) -> None:
    """LLaVA-NeXT-Video uses the same grid and the same nonterminating search."""
    _write_next_video_checkpoint(tmp_path, image_size=0, patch_size=14)
    _assert_profiling_finishes(
        tmp_path,
        {"video": 1},
        expect_zero_grid=True,
        require_token_counts=False,
    )


def test_next_video_positive_patch_grid_profiling_returns(tmp_path: Path) -> None:
    """LLaVA-NeXT-Video profiling finishes when the CLIP grid is positive."""
    _write_next_video_checkpoint(tmp_path, image_size=224, patch_size=14)
    _assert_profiling_finishes(
        tmp_path,
        {"video": 1},
        expect_zero_grid=False,
        require_token_counts=True,
    )
