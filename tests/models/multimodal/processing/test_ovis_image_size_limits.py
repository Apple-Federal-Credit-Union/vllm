# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-controlled Ovis image sizes must be rejected before allocation.

Profiling multiplies the image-processor side by ``hidden_stride`` and by 9,
then allocates that dummy image. Each real image then allocates a square tensor
of the uncapped side. Both numbers come from the checkpoint. These tests fail
until those sizes are rejected before Pillow or ``torch.zeros`` allocates.
"""

import json
import os
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

import pytest
import torch
import torchvision.transforms.v2.functional as tv_functional
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.media.image import ImageMediaIO

_REJECTION = (ValueError, VLLMValidationError)
_PIXEL_LIMIT = 1_000


def _write_checkpoint(
    root: Path,
    *,
    size: Mapping[str, int],
    hidden_stride: int,
    image_size: int = 16,
    patch_size: int = 2,
) -> None:
    """Write an Ovis checkpoint whose only attacker-controlled fields vary."""
    config = {
        "architectures": ["Ovis"],
        "model_type": "ovis",
        "hidden_size": 64,
        "multimodal_max_length": 128,
        "llm_config": {
            "model_type": "qwen2",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
        },
        "visual_tokenizer_config": {
            "model_type": "aimv2_visual_tokenizer",
            "vocab_size": 64,
            "hidden_stride": hidden_stride,
            "tokenize_function": "softmax",
            "backbone_config": {
                "model_type": "aimv2",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "image_size": image_size,
                "patch_size": patch_size,
                "num_channels": 3,
            },
        },
    }
    preprocessor = {
        "crop_size": {"height": 8, "width": 8},
        "do_center_crop": False,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_processor_type": "CLIPImageProcessor",
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": dict(size),
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(json.dumps(preprocessor))
    vocab = {"<|image_pad|>": 0, "<image>": 1, "hi": 2}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="hi")),
        unk_token="hi",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|image_pad|>", "<image>"]}
    )
    tokenizer.save_pretrained(root)


def _build_processor(root: Path):
    model_config = ModelConfig(
        model=str(root),
        tokenizer=str(root),
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        max_model_len=32,
        enforce_eager=True,
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _tiny_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


class _AllocationGuard:
    """Record raster allocations and stop any that exceed the pixel limit.

    The real allocators run for sizes inside the limit. Oversized or
    non-positive requests are recorded and refused here so the test process
    does not allocate them. A product fix must raise before making that
    request.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
        self.limit = limit
        self.violations: list[tuple[str, int, int]] = []
        self.accepted: list[tuple[str, int, int]] = []
        self._new = Image.new
        self._resize = Image.Image.resize
        self._zeros = torch.zeros
        self._tv_resize = tv_functional.resize
        monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", limit)
        monkeypatch.setattr(Image, "new", self.new)
        monkeypatch.setattr(Image.Image, "resize", self.resize)
        monkeypatch.setattr(torch, "zeros", self.zeros)
        monkeypatch.setattr(tv_functional, "resize", self.tv_resize)

    def _too_big(self, width: int, height: int, op: str) -> None:
        if width <= 0 or height <= 0 or width * height > self.limit:
            self.violations.append((op, width, height))
            raise RuntimeError(f"{op} requested {width}x{height}")
        self.accepted.append((op, width, height))

    def new(self, mode, size, *args, **kwargs):
        width, height = int(size[0]), int(size[1])
        self._too_big(width, height, "Image.new")
        return self._new(mode, size, *args, **kwargs)

    def resize(self, image, size, *args, **kwargs):
        width, height = _pair(size)
        self._too_big(width, height, "Image.resize")
        return self._resize(image, size, *args, **kwargs)

    def tv_resize(self, image, size, *args, **kwargs):
        width, height = _pair(size)
        self._too_big(width, height, "torchvision.resize")
        return self._tv_resize(image, size, *args, **kwargs)

    def zeros(self, *args, **kwargs):
        shape = _zeros_shape(args, kwargs)
        if len(shape) >= 2:
            self._too_big(int(shape[-1]), int(shape[-2]), "torch.zeros")
        return self._zeros(*args, **kwargs)


def _pair(size) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


def _zeros_shape(args: tuple, kwargs: dict) -> tuple:
    if args and isinstance(args[0], (list, tuple)):
        return tuple(args[0])
    if isinstance(kwargs.get("size"), (list, tuple)):
        return tuple(kwargs["size"])
    dims = []
    for arg in args:
        if isinstance(arg, int):
            dims.append(arg)
        else:
            break
    return tuple(dims)


def _error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    cause = exc.__cause__
    while cause is not None:
        parts.append(str(cause))
        cause = cause.__cause__
    return "\n".join(parts)


def _assert_rejected(guard: _AllocationGuard, exc: BaseException, text: str) -> None:
    assert guard.violations == [], guard.violations
    assert text in _error_text(exc)


def _profile(root: Path) -> None:
    processor = _build_processor(root)
    processor.get_dummy_mm_inputs({"image": 1})


def _preprocess_loaded_image(root: Path) -> None:
    processor = _build_processor(root)
    loaded = ImageMediaIO().load_bytes(_TINY_PNG).media
    mm_items = processor.info.parse_mm_data({"image": loaded})
    processor("<image>", mm_items)


_TINY_PNG = _tiny_png_bytes()


@pytest.mark.parametrize(
    "size",
    [
        {"shortest_edge": 20},
        {"height": 20, "width": 20},
        {"height": 10, "width": 40},
    ],
)
def test_profiling_rejects_oversized_processor_size_before_alloc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: dict[str, int],
) -> None:
    """Startup profiling must not allocate side * hidden_stride * 9."""
    _write_checkpoint(tmp_path, size=size, hidden_stride=1)
    guard = _AllocationGuard(monkeypatch, _PIXEL_LIMIT)
    with pytest.raises(_REJECTION) as exc_info:
        _profile(tmp_path)
    _assert_rejected(guard, exc_info.value, "VLLM_MAX_IMAGE_PIXELS")


def test_profiling_rejects_hidden_stride_multiplier_before_alloc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A modest side times a divisor stride still has to stay inside the cap.

    ``image_size / patch_size`` is divisible by this stride, so segment-length
    setup accepts it. The dummy image is ``8 * 4 * 9`` pixels on a side.
    """
    _write_checkpoint(
        tmp_path,
        size={"shortest_edge": 8},
        hidden_stride=4,
        image_size=16,
        patch_size=2,
    )
    guard = _AllocationGuard(monkeypatch, _PIXEL_LIMIT)
    with pytest.raises(_REJECTION) as exc_info:
        _profile(tmp_path)
    _assert_rejected(guard, exc_info.value, "VLLM_MAX_IMAGE_PIXELS")


def test_profiling_rejects_default_pixel_limit_before_alloc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default pixel cap has to apply to the multiplied dummy image."""
    limit = 178_956_970
    _write_checkpoint(tmp_path, size={"shortest_edge": 20_000}, hidden_stride=1)
    guard = _AllocationGuard(monkeypatch, limit)
    with pytest.raises(_REJECTION) as exc_info:
        _profile(tmp_path)
    _assert_rejected(guard, exc_info.value, "VLLM_MAX_IMAGE_PIXELS")


@pytest.mark.parametrize(
    ("size", "hidden_stride"),
    [
        ({"shortest_edge": 0}, 1),
        ({"shortest_edge": -5}, 1),
        ({"height": 0, "width": 0}, 1),
        ({"height": -8, "width": -8}, 1),
        ({"shortest_edge": 8}, 0),
        ({"shortest_edge": 8}, -4),
    ],
)
def test_profiling_rejects_non_positive_size_and_stride(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: dict[str, int],
    hidden_stride: int,
) -> None:
    _write_checkpoint(tmp_path, size=size, hidden_stride=hidden_stride)
    guard = _AllocationGuard(monkeypatch, 100_000)
    with pytest.raises(_REJECTION) as exc_info:
        _profile(tmp_path)
    _assert_rejected(guard, exc_info.value, "positive")


@pytest.mark.parametrize(
    "size",
    [
        {"shortest_edge": 40},
        {"height": 40, "width": 40},
    ],
)
def test_image_request_rejects_oversized_side_before_square_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: dict[str, int],
) -> None:
    """A 2x2 request image must not allocate a square of the checkpoint side."""
    _write_checkpoint(tmp_path, size=size, hidden_stride=1)
    guard = _AllocationGuard(monkeypatch, _PIXEL_LIMIT)
    loaded = ImageMediaIO().load_bytes(_TINY_PNG)
    assert loaded.media.size == (2, 2)
    with pytest.raises(_REJECTION) as exc_info:
        _preprocess_loaded_image(tmp_path)
    _assert_rejected(guard, exc_info.value, "VLLM_MAX_IMAGE_PIXELS")


@pytest.mark.parametrize("side", [0, -5])
def test_image_request_rejects_non_positive_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    side: int,
) -> None:
    _write_checkpoint(tmp_path, size={"shortest_edge": side}, hidden_stride=1)
    guard = _AllocationGuard(monkeypatch, 100_000)
    with pytest.raises(_REJECTION) as exc_info:
        _preprocess_loaded_image(tmp_path)
    _assert_rejected(guard, exc_info.value, "positive")


def test_within_limit_checkpoint_profiles_and_preprocesses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A side whose multiplied dummy image fits the cap still runs.

    Dummy side is ``8 * 2 * 9``. The square tensor uses the processor side.
    """
    limit = 100_000
    _write_checkpoint(tmp_path, size={"shortest_edge": 8}, hidden_stride=2)
    guard = _AllocationGuard(monkeypatch, limit)
    processor = _build_processor(tmp_path)
    # 8 * hidden_stride 2 * the hardcoded crop factor 9.
    assert processor.info.get_image_size_with_most_features() == (144, 144)
    profiled = processor.get_dummy_mm_inputs({"image": 1})
    assert guard.violations == []
    assert ("Image.new", 144, 144) in guard.accepted
    assert profiled is not None

    loaded = ImageMediaIO().load_bytes(_TINY_PNG).media
    mm_items = processor.info.parse_mm_data({"image": loaded})
    processed = processor("<image>", mm_items)
    assert guard.violations == []
    pixel_values = processed["mm_kwargs"].get_data()["pixel_values"]
    # Batched pixel values keep the checkpoint side, not the 2x2 input.
    tensor = pixel_values[0] if isinstance(pixel_values, list) else pixel_values
    assert tuple(tensor.shape[-2:]) == (8, 8)
