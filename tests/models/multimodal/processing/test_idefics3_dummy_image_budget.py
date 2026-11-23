# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling must not rasterize an uncapped checkpoint image edge.

Idefics3 and SmolVLM both profile through ``Idefics3DummyInputsBuilder``.
That builder reads ``image_processor.max_image_size["longest_edge"]`` from
the checkpoint and passes it to Pillow as both sides of the dummy image.
``MultiModalBudget`` reaches that builder via ``get_dummy_inputs``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm.config import ModelConfig
from vllm.envs import VLLM_MAX_IMAGE_PIXELS
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageProcessorItems
from vllm.multimodal.processing import dummy_inputs

# Square sides whose pixel count stays under the media pixel budget.
_UNDER_BUDGET_EDGES = (364, 1024)
# Square side Pillow will allocate (below INT_MAX / 4) and that exceeds
# VLLM_MAX_IMAGE_PIXELS when used as both width and height.
_OVER_BUDGET_EDGE = 20_000
_REJECTION = (ValueError, VLLMValidationError)

_ARCHITECTURES = (
    (
        "Idefics3ForConditionalGeneration",
        "idefics3",
        "idefics3_vision",
        "Idefics3ImageProcessor",
    ),
    (
        "SmolVLMForConditionalGeneration",
        "smolvlm",
        "smolvlm_vision",
        "SmolVLMImageProcessor",
    ),
)


class _UncappedDummyImage(Exception):
    """Raised when profiling asks Pillow for a non-positive or over-budget size."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        super().__init__(f"{width}x{height}")


def _pixel_budget() -> int:
    assert VLLM_MAX_IMAGE_PIXELS > 0
    return VLLM_MAX_IMAGE_PIXELS


def _write_checkpoint(
    directory: Path, longest_edge: int, spec: tuple[str, ...]
) -> None:
    architecture, model_type, vision_type, image_processor_type = spec
    config = {
        "architectures": [architecture],
        "model_type": model_type,
        "image_token_id": 128257,
        "scale_factor": 2,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-5,
        },
        "vision_config": {
            "model_type": vision_type,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 64,
            "patch_size": 16,
        },
    }
    preprocessor = {
        "image_processor_type": image_processor_type,
        "max_image_size": {"longest_edge": longest_edge},
        "size": {"longest_edge": 4 * 364},
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "preprocessor_config.json").write_text(
        json.dumps(preprocessor), encoding="utf-8"
    )
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "a": 1}, unk_token="<unk>"))
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>"
    )
    fast_tokenizer.save_pretrained(directory)


def _load_processor(directory: Path):
    model_config = ModelConfig(
        str(directory),
        tokenizer=str(directory),
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        max_model_len=128,
        enforce_eager=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    return model_config, processor


def _profile_dummy_image(directory: Path):
    """Run the profiling entry that allocates the dummy image."""
    model_config, processor = _load_processor(directory)
    mm_options = model_config.get_multimodal_config().limit_per_prompt
    inputs = processor.get_dummy_inputs(
        model_config.max_model_len,
        {"image": 1},
        mm_options,
    )
    image = inputs.mm_data_items.get_items("image", ImageProcessorItems).get(0)
    return image


def _guard_image_new(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Record sizes passed to Pillow and refuse ones the budget must stop."""
    requested: list[tuple[int, int]] = []
    real_new = Image.new
    budget = _pixel_budget()

    def wrapped(mode: str, size: tuple[int, int], color: int = 0):
        width, height = int(size[0]), int(size[1])
        requested.append((width, height))
        if width <= 0 or height <= 0 or width * height > budget:
            raise _UncappedDummyImage(width, height)
        return real_new(mode, (width, height), color)

    monkeypatch.setattr(dummy_inputs.Image, "new", wrapped)
    return requested


@pytest.mark.parametrize("edge", _UNDER_BUDGET_EDGES)
@pytest.mark.parametrize("spec", _ARCHITECTURES)
def test_under_budget_checkpoint_edge_is_dummy_image_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: tuple[str, ...], edge: int
) -> None:
    """A checkpoint edge inside the pixel budget is both sides of the dummy image."""
    requested = _guard_image_new(monkeypatch)
    _write_checkpoint(tmp_path, edge, spec)

    image = _profile_dummy_image(tmp_path)

    assert image.size == (edge, edge)
    assert (edge, edge) in requested
    assert all(width * height <= _pixel_budget() for width, height in requested)


@pytest.mark.parametrize("spec", _ARCHITECTURES)
def test_over_budget_checkpoint_edge_is_not_allocated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: tuple[str, ...]
) -> None:
    """Profiling must cap or reject a checkpoint edge before Pillow allocates it."""
    assert _OVER_BUDGET_EDGE * _OVER_BUDGET_EDGE > _pixel_budget()
    requested = _guard_image_new(monkeypatch)
    _write_checkpoint(tmp_path, _OVER_BUDGET_EDGE, spec)

    try:
        image = _profile_dummy_image(tmp_path)
    except _REJECTION:
        image = None
    except _UncappedDummyImage as exc:
        pytest.fail(
            f"profiling passed checkpoint edge to Pillow as {exc.width}x{exc.height}"
        )

    offending = [
        (width, height)
        for width, height in requested
        if width <= 0 or height <= 0 or width * height > _pixel_budget()
    ]
    assert not offending
    if image is not None:
        assert image.size[0] * image.size[1] <= _pixel_budget()
        assert image.size[0] > 0 and image.size[1] > 0


@pytest.mark.parametrize("edge", (0, -1))
@pytest.mark.parametrize("spec", _ARCHITECTURES)
def test_non_positive_checkpoint_edge_rejected_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: tuple[str, ...], edge: int
) -> None:
    """Non-positive checkpoint edges are rejected before Pillow is called."""
    requested = _guard_image_new(monkeypatch)
    _write_checkpoint(tmp_path, edge, spec)

    try:
        _profile_dummy_image(tmp_path)
    except _REJECTION:
        pass
    except _UncappedDummyImage as exc:
        pytest.fail(
            "profiling passed a non-positive edge to Pillow as "
            f"{exc.width}x{exc.height}"
        )
    else:
        pytest.fail("profiling allocated a dummy image for a non-positive edge")

    assert requested == []
