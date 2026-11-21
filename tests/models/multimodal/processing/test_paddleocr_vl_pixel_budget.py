# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PaddleOCR-VL profiling must not honor an unbounded checkpoint pixel budget.

``get_image_size_with_most_features`` turns the checkpoint image processor's
pixel budget into dummy width and height, and startup profiling passes those
sides to ``PIL.Image.new``. The budget is ``max_pixels`` when
``trust_remote_code`` is set (Hub processor) and ``size.longest_edge``
otherwise (transformers ``PaddleOCRVLImageProcessor``).

The cap is the live ``vllm.envs.VLLM_MAX_IMAGE_PIXELS`` value, the same
decoded-image maximum documented in ``docs/usage/security.md``. Read it when
the sides are derived. ``0`` disables that limit and is not covered here.
Non-positive ``vision_config.patch_size`` or ``spatial_merge_size`` must be
rejected with ``ValueError`` before ``Image.new`` is called.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import vllm.envs as envs
from vllm.config.model import ModelConfig
from vllm.model_executor.models.paddleocr_vl import PaddleOCRVLProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageProcessorItems

_PATCH = 14
_MERGE = 2
_FACTOR = _PATCH * _MERGE
_REMOTE_FIXTURE = Path(__file__).parents[2] / "fixtures"
_OVER_CAP_TOKENS = 100
_CAP_TOKENS = 4
_UNDER_CAP_TOKENS = 16


def _pixels(num_tokens: int) -> int:
    return num_tokens * _FACTOR * _FACTOR


def _write_checkpoint(
    root: Path,
    *,
    max_pixels: int | None,
    longest_edge: int | None,
    patch_size: int,
    merge_size: int,
    remote: bool,
) -> None:
    vocab = {"<unk>": 0, "<|IMAGE_PLACEHOLDER|>": 1, "a": 2}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<unk>",
                "eos_token": "<unk>",
                "pad_token": "<unk>",
                "image_token": "<|IMAGE_PLACEHOLDER|>",
                "model_max_length": 2048,
                "processor_class": "PaddleOCRVLProcessor",
            }
        )
    )
    preprocessor: dict[str, object] = {
        "image_processor_type": "PaddleOCRVLImageProcessor",
        "processor_class": "PaddleOCRVLProcessor",
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "merge_size": _MERGE,
        "patch_size": _PATCH,
        "resample": 3,
        "rescale_factor": 1 / 255,
        "min_pixels": _pixels(1),
    }
    if max_pixels is not None:
        preprocessor["max_pixels"] = max_pixels
    if longest_edge is not None:
        preprocessor["size"] = {
            "shortest_edge": _pixels(1),
            "longest_edge": longest_edge,
        }
    processor_config: dict[str, object] = {
        "processor_class": "PaddleOCRVLProcessor",
    }
    if remote:
        shutil.copy(
            _REMOTE_FIXTURE / "paddleocr_vl_image_processing.py.fixture",
            root / "image_processing_paddleocr_vl.py",
        )
        shutil.copy(
            _REMOTE_FIXTURE / "paddleocr_vl_processing.py.fixture",
            root / "processing_paddleocr_vl.py",
        )
        preprocessor["auto_map"] = {
            "AutoImageProcessor": (
                "image_processing_paddleocr_vl.PaddleOCRVLImageProcessor"
            ),
            "AutoProcessor": "processing_paddleocr_vl.PaddleOCRVLProcessor",
        }
        processor_config["auto_map"] = {
            "AutoProcessor": "processing_paddleocr_vl.PaddleOCRVLProcessor",
        }
    (root / "preprocessor_config.json").write_text(json.dumps(preprocessor))
    (root / "processor_config.json").write_text(json.dumps(processor_config))
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["PaddleOCRVLForConditionalGeneration"],
                "model_type": "paddleocr_vl",
                "image_token_id": 1,
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 128,
                "vocab_size": 8,
                "max_position_embeddings": 2048,
                "vision_config": {
                    "model_type": "paddleocr_vl_vision",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "patch_size": patch_size,
                    "spatial_merge_size": merge_size,
                    "image_size": 384,
                },
                "text_config": {
                    "model_type": "paddleocr_vl_text",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "vocab_size": 8,
                    "max_position_embeddings": 2048,
                },
            }
        )
    )


def _model_config(root: Path, trust_remote_code: bool) -> ModelConfig:
    return ModelConfig(
        model=str(root),
        tokenizer=str(root),
        tokenizer_mode="hf",
        trust_remote_code=trust_remote_code,
        dtype="float32",
        seed=0,
        max_model_len=128,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    )


def _profile(
    root: Path,
    trust_remote_code: bool,
    cap_pixels: int,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Image.Image, int, int]:
    """Run the startup dummy-input path and return image, pixels, tokens."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("VLLM_MAX_IMAGE_PIXELS", str(cap_pixels))
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", cap_pixels)

    model_config = _model_config(root, trust_remote_code)
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    info = processor.info
    assert isinstance(info, PaddleOCRVLProcessingInfo)

    dummy = processor.get_dummy_inputs(
        seq_len=model_config.max_model_len,
        mm_counts={"image": 1},
        mm_options=model_config.get_multimodal_config().limit_per_prompt,
    )
    images = dummy.mm_data_items.get_items("image", ImageProcessorItems)
    image = images.get(0)
    assert isinstance(image, Image.Image)
    size = images.get_image_size(0)
    pixels = size.width * size.height
    assert len(image.tobytes()) == pixels * 3

    profiled = processor.get_dummy_mm_inputs({"image": 1})
    token_count = sum(
        item.get_num_embeds() for item in profiled["mm_placeholders"]["image"]
    )
    counted = info.get_num_image_tokens(
        image_width=size.width,
        image_height=size.height,
        image_processor=info.get_image_processor(),
        mm_kwargs={},
    )
    assert counted == token_count
    return image, pixels, token_count


def _assert_clamped(pixels: int, token_count: int, cap_pixels: int) -> None:
    assert pixels > 0
    assert pixels <= cap_pixels
    assert token_count * _FACTOR * _FACTOR <= cap_pixels


@pytest.mark.parametrize(
    ("max_pixels", "longest_edge"),
    [
        (_pixels(_OVER_CAP_TOKENS), None),
        (None, _pixels(_OVER_CAP_TOKENS)),
    ],
)
def test_library_checkpoint_pixel_budget_is_clamped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_pixels: int | None,
    longest_edge: int | None,
) -> None:
    """transformers processor: JSON max_pixels or size.longest_edge."""
    _write_checkpoint(
        tmp_path,
        max_pixels=max_pixels,
        longest_edge=longest_edge,
        patch_size=_PATCH,
        merge_size=_MERGE,
        remote=False,
    )
    cap = _pixels(_CAP_TOKENS)
    _image, pixels, token_count = _profile(
        tmp_path, False, cap, monkeypatch
    )
    _assert_clamped(pixels, token_count, cap)


def test_remote_checkpoint_max_pixels_is_clamped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub processor loaded with trust_remote_code exposes max_pixels."""
    budget = _pixels(_OVER_CAP_TOKENS)
    _write_checkpoint(
        tmp_path,
        max_pixels=budget,
        longest_edge=None,
        patch_size=_PATCH,
        merge_size=_MERGE,
        remote=True,
    )
    cap = _pixels(_CAP_TOKENS)
    _image, pixels, token_count = _profile(tmp_path, True, cap, monkeypatch)
    _assert_clamped(pixels, token_count, cap)


def test_checkpoint_budget_under_cap_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget already within the cap stays the profiling size."""
    budget = _pixels(_UNDER_CAP_TOKENS)
    cap = _pixels(64)
    _write_checkpoint(
        tmp_path,
        max_pixels=budget,
        longest_edge=None,
        patch_size=_PATCH,
        merge_size=_MERGE,
        remote=False,
    )
    _image, pixels, token_count = _profile(tmp_path, False, cap, monkeypatch)
    assert pixels == budget
    assert token_count * _FACTOR * _FACTOR == budget


@pytest.mark.parametrize(
    ("patch_size", "merge_size"),
    [(0, _MERGE), (-_PATCH, _MERGE), (_PATCH, 0), (_PATCH, -_MERGE)],
)
def test_non_positive_patch_or_merge_rejected_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_size: int,
    merge_size: int,
) -> None:
    calls: list[tuple[int, int]] = []
    real_new = Image.new

    def _spy(
        mode: str,
        size: tuple[int, int],
        *args: object,
        **kwargs: object,
    ) -> Image.Image:
        calls.append((int(size[0]), int(size[1])))
        return real_new(mode, size, *args, **kwargs)

    monkeypatch.setattr(Image, "new", _spy)
    _write_checkpoint(
        tmp_path,
        max_pixels=_pixels(_OVER_CAP_TOKENS),
        longest_edge=None,
        patch_size=patch_size,
        merge_size=merge_size,
        remote=False,
    )
    model_config = _model_config(tmp_path, trust_remote_code=False)
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    info = processor.info
    assert isinstance(info, PaddleOCRVLProcessingInfo)
    image_processor = info.get_image_processor()
    mm_options = model_config.get_multimodal_config().limit_per_prompt

    with pytest.raises(ValueError):
        info.get_image_size_with_most_features()
    with pytest.raises(ValueError):
        info.get_num_image_tokens(
            image_width=_FACTOR,
            image_height=_FACTOR,
            image_processor=image_processor,
            mm_kwargs={},
        )
    with pytest.raises(ValueError):
        processor.dummy_inputs.get_dummy_mm_data(
            model_config.max_model_len,
            {"image": 1},
            mm_options,
        )
    assert calls == []
