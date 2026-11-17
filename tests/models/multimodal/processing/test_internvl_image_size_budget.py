# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint vision image_size must not allocate rasters over the pixel budget.

These tests fail while profiling or request preprocessing asks Pillow (or the
shared InternVL-style resize helpers) for a buffer larger than
``VLLM_MAX_IMAGE_PIXELS``. A fix rejects or clamps that size before the
allocation. Refusal has to happen before the oversized call: the guards below
fail the test as soon as an over-budget size is requested.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import dummy_inputs as dummy_inputs_mod
from vllm.transformers_utils.processors.h2ovl import H2OVLImageProcessor
from vllm.transformers_utils.processors.internvl import InternVLChatVideoProcessor
from vllm.transformers_utils.processors.nano_nemotron_vl import dynamic_preprocess
from vllm.transformers_utils.processors.nemotron_vl import (
    LlamaNemotronNanoVLImageProcessor,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_OVER_BUDGET = "over-budget raster"


def _side_over_budget() -> int:
    limit = envs.VLLM_MAX_IMAGE_PIXELS
    assert limit > 1
    side = int(limit**0.5)
    if side * side <= limit:
        side += 1
    return side


def _pixels(width: int, height: int) -> int:
    return width * height


def _write_tokenizer(path: Path) -> Any:
    vocab = {
        "<unk>": 0,
        "<IMG_CONTEXT>": 1,
        "<img>": 2,
        "</img>": 3,
        "<image>": 4,
        "<video>": 5,
        "<|video_pad|>": 6,
    }
    path.mkdir(parents=True, exist_ok=True)
    Tokenizer(WordLevel(vocab, unk_token="<unk>")).save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
            }
        )
    )
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _text_config(model_type: str) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "max_position_embeddings": 2048,
        "rope_theta": 1_000_000,
    }


def _write_qianfan_checkpoint(
    path: Path,
    *,
    image_size: int,
    dynamic_image_size: bool,
    max_dynamic_patch: int,
) -> None:
    config = {
        "model_type": "qianfan_ocr",
        "architectures": ["QianfanOCRForConditionalGeneration"],
        "downsample_ratio": 0.5,
        "dynamic_image_size": dynamic_image_size,
        "force_image_size": 448,
        "min_dynamic_patch": 1,
        "max_dynamic_patch": max_dynamic_patch,
        "use_thumbnail": False,
        "ps_version": "v2",
        "text_config": _text_config("qwen3"),
        "vision_config": {
            "model_type": "qianfan_ocr_vision",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "image_size": image_size,
            "patch_size": 14,
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))


def _write_internvl_checkpoint(path: Path, *, image_size: int) -> None:
    """Official InternVL config classes, with only image_size changed.

    Those modules are what ``trust_remote_code`` executes. They copy
    ``vision_config.image_size`` through with no pixel cap.
    """
    path.mkdir(parents=True, exist_ok=True)
    for name in (
        "configuration_internvl_chat.py",
        "configuration_intern_vit.py",
    ):
        shutil.copy(_FIXTURES / f"{name}.txt", path / name)
    config = {
        "model_type": "internvl_chat",
        "architectures": ["InternVLChatModel"],
        "auto_map": {
            "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
            "AutoModel": "modeling_internvl_chat.InternVLChatModel",
        },
        "downsample_ratio": 0.5,
        "dynamic_image_size": False,
        "force_image_size": image_size,
        "min_dynamic_patch": 1,
        "max_dynamic_patch": 1,
        "use_thumbnail": False,
        "ps_version": "v2",
        "select_layer": -1,
        "template": "Hermes-2",
        "llm_config": _text_config("qwen2"),
        "vision_config": {
            "architectures": ["InternVisionModel"],
            "hidden_size": 64,
            "image_size": image_size,
            "intermediate_size": 128,
            "model_type": "intern_vit_6b",
            "num_attention_heads": 4,
            "num_channels": 3,
            "num_hidden_layers": 1,
            "patch_size": 14,
            "qk_normalization": False,
            "qkv_bias": True,
        },
    }
    (path / "config.json").write_text(json.dumps(config))


def _load_processor(
    model_path: Path,
    tokenizer_path: Path,
    tokenizer: Any,
    *,
    trust_remote_code: bool,
):
    model_config = ModelConfig(
        model=str(model_path),
        tokenizer=str(tokenizer_path),
        trust_remote_code=trust_remote_code,
        max_model_len=128,
        enforce_eager=True,
    )
    return MULTIMODAL_REGISTRY.create_processor(
        model_config,
        tokenizer=tokenizer,
    )


class _RasterGuard:
    """Fail if product code asks for a raster above the pixel budget.

    The real Pillow and NumPy allocators are used only for sizes inside the
    budget, so a clamped fix still builds a real image. An over-budget size
    raises before ``Image.new``, ``resize``, ``np.full``, or
    ``interpolate`` allocates it.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.attempts: list[tuple[str, int, int]] = []
        self._saved: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> _RasterGuard:
        self._patch(Image, "new", self._guard_new(Image.new))
        self._patch(Image.Image, "resize", self._guard_resize(Image.Image.resize))
        self._patch(dummy_inputs_mod.np, "full", self._guard_full(np.full))
        self._patch(
            torch.nn.functional,
            "interpolate",
            self._guard_interpolate(torch.nn.functional.interpolate),
        )
        return self

    def __exit__(self, *exc: object) -> None:
        for obj, name, original in reversed(self._saved):
            setattr(obj, name, original)

    def _patch(self, obj: Any, name: str, wrapped: Any) -> None:
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, wrapped)

    def _check(self, kind: str, width: int, height: int) -> None:
        self.attempts.append((kind, width, height))
        if self.limit > 0 and _pixels(width, height) > self.limit:
            raise AssertionError(
                f"{_OVER_BUDGET}: {kind} {width}x{height} "
                f"({_pixels(width, height)} px) exceeds {self.limit}"
            )

    def _guard_new(self, real: Callable[..., Image.Image]):
        def wrapped(mode: str, size: tuple[int, int], *args: Any, **kwargs: Any):
            self._check("Image.new", int(size[0]), int(size[1]))
            return real(mode, size, *args, **kwargs)

        return wrapped

    def _guard_resize(self, real: Callable[..., Image.Image]):
        def wrapped(image: Image.Image, size: Any, *args: Any, **kwargs: Any):
            self._check("Image.resize", int(size[0]), int(size[1]))
            return real(image, size, *args, **kwargs)

        return wrapped

    def _guard_full(self, real: Callable[..., np.ndarray]):
        def wrapped(shape: Any, *args: Any, **kwargs: Any):
            if isinstance(shape, tuple) and len(shape) == 4:
                _, height, width, _ = shape
                self._check("np.full", int(width), int(height))
            return real(shape, *args, **kwargs)

        return wrapped

    def _guard_interpolate(self, real: Callable[..., torch.Tensor]):
        def wrapped(*args: Any, **kwargs: Any):
            size = kwargs.get("size")
            if size is not None:
                self._check("interpolate", int(size[1]), int(size[0]))
            return real(*args, **kwargs)

        return wrapped


def _run_bounded(fn: Callable[[], Any]) -> Any:
    limit = envs.VLLM_MAX_IMAGE_PIXELS
    with _RasterGuard(limit) as guard:
        try:
            result = fn()
        except AssertionError as exc:
            if str(exc).startswith(_OVER_BUDGET):
                pytest.fail(str(exc))
            raise
        except ValueError as exc:
            over = [
                (kind, width, height)
                for kind, width, height in guard.attempts
                if _pixels(width, height) > limit
            ]
            if over:
                pytest.fail(f"{over} requested before {exc}")
            return None
    assert all(_pixels(width, height) <= limit for _, width, height in guard.attempts)
    return result


def _tiny_image(width: int = 8, height: int = 8) -> Image.Image:
    return Image.new("RGB", (width, height), color=(1, 2, 3))


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("tokenizer")
    tokenizer = _write_tokenizer(path)
    return path, tokenizer


def test_in_budget_checkpoint_still_profiles(tmp_path: Path, tokenizer_dir):
    """A normal tile size still produces a profiling image."""
    tok_path, tokenizer = tokenizer_dir
    model_path = tmp_path / "qianfan-normal"
    _write_qianfan_checkpoint(
        model_path,
        image_size=448,
        dynamic_image_size=False,
        max_dynamic_patch=1,
    )
    processor = _load_processor(
        model_path, tok_path, tokenizer, trust_remote_code=False
    )
    result = _run_bounded(lambda: processor.get_dummy_mm_inputs({"image": 1}))
    assert result is not None
    pixels = result["mm_kwargs"].get_data()["pixel_values_flat"]
    assert pixels.shape[-2] * pixels.shape[-1] == 448 * 448


def test_qianfan_profiling_rejects_uncapped_image_size(
    tmp_path: Path, tokenizer_dir
):
    tok_path, tokenizer = tokenizer_dir
    image_size = _side_over_budget()
    model_path = tmp_path / "qianfan-hostile"
    _write_qianfan_checkpoint(
        model_path,
        image_size=image_size,
        dynamic_image_size=False,
        max_dynamic_patch=1,
    )
    processor = _load_processor(
        model_path, tok_path, tokenizer, trust_remote_code=False
    )
    # Qianfan normalizes a scalar image_size to one int before profiling.
    assert processor.info.get_hf_config().vision_config.image_size == image_size
    _run_bounded(lambda: processor.get_dummy_mm_inputs({"image": 1}))


def test_qianfan_request_resize_rejects_uncapped_image_size(
    tmp_path: Path, tokenizer_dir
):
    tok_path, tokenizer = tokenizer_dir
    image_size = _side_over_budget()
    model_path = tmp_path / "qianfan-request"
    _write_qianfan_checkpoint(
        model_path,
        image_size=image_size,
        dynamic_image_size=False,
        max_dynamic_patch=1,
    )
    processor = _load_processor(
        model_path, tok_path, tokenizer, trust_remote_code=False
    )
    image = _tiny_image()
    mm_items = processor.info.parse_mm_data({"image": [image]})

    def _request():
        return processor("<image>", mm_items=mm_items)

    _run_bounded(_request)


def test_qianfan_profiling_rejects_image_size_times_tiles(
    tmp_path: Path, tokenizer_dir
):
    """image_size itself fits; image_size times the chosen tile counts does not."""
    tok_path, tokenizer = tokenizer_dir
    image_size = _side_over_budget() - 1
    assert image_size * image_size <= envs.VLLM_MAX_IMAGE_PIXELS
    assert image_size * 2 * image_size > envs.VLLM_MAX_IMAGE_PIXELS
    model_path = tmp_path / "qianfan-tiles"
    _write_qianfan_checkpoint(
        model_path,
        image_size=image_size,
        dynamic_image_size=True,
        max_dynamic_patch=2,
    )
    processor = _load_processor(
        model_path, tok_path, tokenizer, trust_remote_code=False
    )
    _run_bounded(lambda: processor.get_dummy_mm_inputs({"image": 1}))


def test_qianfan_wide_request_rejects_image_size_times_tiles(
    tmp_path: Path, tokenizer_dir
):
    tok_path, tokenizer = tokenizer_dir
    image_size = _side_over_budget() - 1
    model_path = tmp_path / "qianfan-wide"
    _write_qianfan_checkpoint(
        model_path,
        image_size=image_size,
        dynamic_image_size=True,
        max_dynamic_patch=2,
    )
    processor = _load_processor(
        model_path, tok_path, tokenizer, trust_remote_code=False
    )
    # Aspect picks a 2-tile ratio, so the resize is image_size by 2*image_size.
    image = _tiny_image(32, 2)
    mm_items = processor.info.parse_mm_data({"image": [image]})
    _run_bounded(lambda: processor("<image>", mm_items=mm_items))


def test_internvl_chat_profiling_and_request_use_checkpoint_image_size(
    tmp_path: Path, tokenizer_dir
):
    tok_path, tokenizer = tokenizer_dir
    image_size = _side_over_budget()
    model_path = tmp_path / "internvl-chat"
    _write_internvl_checkpoint(model_path, image_size=image_size)
    processor = _load_processor(
        model_path, tok_path, tokenizer, trust_remote_code=True
    )
    assert processor.info.get_hf_processor().image_processor.image_size == image_size
    _run_bounded(lambda: processor.get_dummy_mm_inputs({"image": 1}))
    image = _tiny_image()
    mm_items = processor.info.parse_mm_data({"image": [image]})
    _run_bounded(lambda: processor("<image>", mm_items=mm_items))


def test_internvl_video_dummy_uses_checkpoint_image_size(
    tmp_path: Path, tokenizer_dir
):
    tok_path, tokenizer = tokenizer_dir
    image_size = _side_over_budget()
    model_path = tmp_path / "internvl-video"
    _write_internvl_checkpoint(model_path, image_size=image_size)
    processor = _load_processor(
        model_path, tok_path, tokenizer, trust_remote_code=True
    )
    assert processor.info.ctx_video_token == "<|video_pad|>"
    _run_bounded(lambda: processor.get_dummy_mm_inputs({"image": 0, "video": 1}))


def test_internvl_video_frame_resize_uses_checkpoint_image_size(
    tmp_path: Path, tokenizer_dir
):
    tok_path, tokenizer = tokenizer_dir
    image_size = _side_over_budget()
    model_path = tmp_path / "internvl-video-frame"
    _write_internvl_checkpoint(model_path, image_size=image_size)
    processor = _load_processor(
        model_path, tok_path, tokenizer, trust_remote_code=True
    )
    video_processor = processor.info.get_hf_processor().video_processor
    assert isinstance(video_processor, InternVLChatVideoProcessor)
    frame = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    _run_bounded(lambda: video_processor(frame))


def test_h2ovl_resize_uses_loaded_image_size(tmp_path: Path, tokenizer_dir):
    loaded, image = _loaded_internvl_image_size(tmp_path, tokenizer_dir)
    processor = H2OVLImageProcessor(
        image_size=loaded,
        min_dynamic_patch=1,
        max_dynamic_patch=1,
        dynamic_image_size=False,
        use_thumbnail=False,
        use_msac=False,
    )
    _run_bounded(lambda: processor(image))


def test_nemotron_resize_uses_loaded_image_size(tmp_path: Path, tokenizer_dir):
    loaded, image = _loaded_internvl_image_size(tmp_path, tokenizer_dir)
    processor = LlamaNemotronNanoVLImageProcessor(
        image_size=loaded,
        min_dynamic_patch=1,
        max_dynamic_patch=1,
        dynamic_image_size=False,
        use_thumbnail=False,
    )
    _run_bounded(lambda: processor(image))


def test_nano_nemotron_interpolate_uses_loaded_image_size(
    tmp_path: Path, tokenizer_dir
):
    loaded, image = _loaded_internvl_image_size(tmp_path, tokenizer_dir)
    _run_bounded(
        lambda: dynamic_preprocess(
            image,
            image_size=loaded,
            max_num_tiles=1,
            use_thumbnail=False,
        )
    )


def _loaded_internvl_image_size(tmp_path: Path, tokenizer_dir):
    tok_path, tokenizer = tokenizer_dir
    image_size = _side_over_budget()
    model_path = tmp_path / "internvl-shared"
    _write_internvl_checkpoint(model_path, image_size=image_size)
    processor = _load_processor(
        model_path, tok_path, tokenizer, trust_remote_code=True
    )
    loaded = processor.info.get_hf_config().vision_config.image_size
    assert loaded == image_size
    return loaded, _tiny_image()
