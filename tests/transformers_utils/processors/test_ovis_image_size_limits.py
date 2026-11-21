# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ovis checkpoint image sizes must be bounded before any buffer is allocated.

The image-processor side length and ``visual_tokenizer_config.hidden_stride``
are read from the checkpoint. Profiling does
``Image.new`` at ``side * hidden_stride * 9``, and every image is padded with
``torch.zeros([1, 3, side, side])``. Both have to be rejected, before those
calls, when a side is not a positive integer or the resulting pixel count
exceeds ``VLLM_MAX_IMAGE_PIXELS``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2.functional as tv_functional
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.image_utils import SizeDict

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.model_executor.models.ovis import (
    OvisDummyInputsBuilder,
    OvisProcessingInfo,
)
from vllm.multimodal.processing.context import InputProcessingContext
from vllm.tokenizers import cached_tokenizer_from_config

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

_TINY_EDGE = 32


class OversizedAllocation(Exception):
    """Raised by the test guard when production code asks for a bad buffer."""


class EscalatingSize(SizeDict):
    """First two ``shortest_edge`` reads stay in range; later reads do not.

    ``get_image_size`` checks membership (two reads) and then indexes the
    value (two more reads). A check that looks the size up again therefore
    observes ``100_000`` even if an earlier read was ``32``.
    """

    def __init__(self) -> None:
        super().__init__(shortest_edge=_TINY_EDGE)
        object.__setattr__(self, "_reads", 0)

    def __getattribute__(self, name: str):
        if name == "shortest_edge":
            reads = object.__getattribute__(self, "_reads")
            object.__setattr__(self, "_reads", reads + 1)
            return _TINY_EDGE if reads < 2 else 100_000
        return super().__getattribute__(name)


def _write_checkpoint(
    root: Path,
    *,
    size: dict[str, int],
    hidden_stride: int,
    image_size: int = 224,
    patch_size: int = 14,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    vocab = {"<|image_pad|>": 0, "<image>": 1, "hello": 2}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="hello")),
        unk_token="hello",
        pad_token="hello",
    )
    tokenizer.save_pretrained(root)
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "do_center_crop": False,
                "do_convert_rgb": True,
                "do_normalize": False,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_processor_type": "CLIPImageProcessor",
                "image_std": [0.5, 0.5, 0.5],
                "resample": 3,
                "rescale_factor": 1 / 255,
                "size": size,
                "crop_size": {"height": 16, "width": 16},
            }
        )
    )
    (root / "processor_config.json").write_text(
        json.dumps({"processor_class": "OvisProcessor"})
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Ovis"],
                "model_type": "ovis",
                "hidden_size": 64,
                "multimodal_max_length": 128,
                "llm_config": {
                    "model_type": "qwen2",
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 2,
                    "vocab_size": 8,
                    "max_position_embeddings": 128,
                },
                "visual_tokenizer_config": {
                    "model_type": "aimv2_visual_tokenizer",
                    "hidden_stride": hidden_stride,
                    "vocab_size": 32,
                    "tokenize_function": "softmax",
                    "drop_cls_token": False,
                    "backbone_config": {
                        "model_type": "aimv2",
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "num_attention_heads": 2,
                        "num_hidden_layers": 1,
                        "image_size": image_size,
                        "patch_size": patch_size,
                        "num_channels": 3,
                    },
                },
            }
        )
    )


def _load_info(root: Path) -> OvisProcessingInfo:
    model_config = ModelConfig(
        model=str(root),
        tokenizer=str(root),
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        max_model_len=128,
        enforce_eager=True,
        config_format="hf",
    )
    tokenizer = cached_tokenizer_from_config(model_config)
    return OvisProcessingInfo(
        InputProcessingContext(model_config=model_config, tokenizer=tokenizer)
    )


def _builder(info: OvisProcessingInfo) -> OvisDummyInputsBuilder:
    return OvisDummyInputsBuilder(info)


def _profile(info: OvisProcessingInfo) -> Image.Image:
    data = _builder(info).get_dummy_mm_data(
        seq_len=128,
        mm_counts={"image": 1},
        mm_options=MultiModalDummyOptions(),
    )
    image = data["image"][0]
    assert isinstance(image, Image.Image)
    return image


def _hw_from_size(size) -> tuple[int, int] | None:
    if isinstance(size, int):
        return size, size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    return None


@contextmanager
def _guard_allocations(max_pixels: int) -> Iterator[None]:
    """Fail if Pillow or torch is asked for a non-positive or oversized buffer.

    Legal sizes are forwarded to the real allocators.
    """
    real_new = Image.new
    real_pil_resize = Image.Image.resize
    real_zeros = torch.zeros
    real_tv_resize = tv_functional.resize

    def _check(height: int, width: int, what: str) -> None:
        if height <= 0 or width <= 0 or height * width > max_pixels:
            raise OversizedAllocation(f"{what} requested {width}x{height}")

    def wrapped_new(mode, size, *args, **kwargs):
        hw = _hw_from_size(size)
        if hw is not None:
            width, height = hw
            _check(height, width, "Image.new")
        return real_new(mode, size, *args, **kwargs)

    def wrapped_pil_resize(self, size, *args, **kwargs):
        hw = _hw_from_size(size)
        if hw is not None:
            width, height = hw
            _check(height, width, "Image.resize")
        return real_pil_resize(self, size, *args, **kwargs)

    def wrapped_zeros(*args, **kwargs):
        shape = None
        if args and isinstance(args[0], (tuple, list)):
            shape = tuple(args[0])
        elif args and all(isinstance(arg, int) for arg in args):
            shape = tuple(args)
        elif isinstance(kwargs.get("size"), (tuple, list)):
            shape = tuple(kwargs["size"])
        if shape is not None and len(shape) >= 2:
            _check(int(shape[-2]), int(shape[-1]), "torch.zeros")
        return real_zeros(*args, **kwargs)

    def wrapped_tv_resize(inpt, size, *args, **kwargs):
        hw = _hw_from_size(size)
        if hw is not None:
            height, width = hw
            _check(height, width, "torchvision.resize")
        return real_tv_resize(inpt, size, *args, **kwargs)

    Image.new = wrapped_new
    Image.Image.resize = wrapped_pil_resize
    torch.zeros = wrapped_zeros
    tv_functional.resize = wrapped_tv_resize
    try:
        yield
    finally:
        Image.new = real_new
        Image.Image.resize = real_pil_resize
        torch.zeros = real_zeros
        tv_functional.resize = real_tv_resize


def _expect_rejected(max_pixels: int, call, *, guard: bool) -> None:
    """Require ``ValueError`` before an oversized buffer is requested.

    ``guard=False`` lets the real allocator run. Use that when the uncapped
    size is small enough to allocate and the assertion is that the call is
    still rejected. ``guard=True`` blocks non-positive or over-limit requests
    so a multi-gigabyte ``Image.new`` cannot run, and treats that request as
    a failure even if Pillow would raise afterwards.
    """
    if not guard:
        try:
            call()
        except ValueError:
            return
        raise AssertionError("oversized checkpoint image size was accepted")

    with _guard_allocations(max_pixels):
        try:
            call()
        except OversizedAllocation as exc:
            raise AssertionError(
                "checkpoint size was used for a buffer before it was rejected"
            ) from exc
        except ValueError:
            return
    raise AssertionError("oversized checkpoint image size was accepted")


def test_within_limit_checkpoint_still_profiles(tmp_path: Path) -> None:
    """A normal side under the default pixel limit still allocates that side."""
    root = tmp_path / "model"
    _write_checkpoint(root, size={"shortest_edge": _TINY_EDGE}, hidden_stride=1)
    info = _load_info(root)
    image = _profile(info)
    assert image.size == (_TINY_EDGE * 9, _TINY_EDGE * 9)
    pixels, _, grid = info.get_hf_processor().preprocess_image(
        Image.new("RGB", (8, 8), color=1),
        max_partition=9,
        covering_threshold=0.9,
        do_convert_rgb=True,
        return_tensors="pt",
    )
    assert tuple(pixels.shape) == (1, 3, _TINY_EDGE, _TINY_EDGE)
    assert grid.tolist() == [1, 1]


def test_profiling_rejects_shortest_edge_over_pixel_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MAX_IMAGE_PIXELS", "1000")
    root = tmp_path / "model"
    # 40 * 1 * 9 = 360; 360**2 > 1000. The square side 40**2 is also over.
    _write_checkpoint(root, size={"shortest_edge": 40}, hidden_stride=1)
    info = _load_info(root)
    _expect_rejected(1000, lambda: _profile(info), guard=False)


def test_profiling_rejects_width_and_height_over_pixel_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MAX_IMAGE_PIXELS", "1000")
    root = tmp_path / "model"
    _write_checkpoint(root, size={"height": 80, "width": 80}, hidden_stride=1)
    info = _load_info(root)
    _expect_rejected(1000, lambda: _profile(info), guard=False)


def test_profiling_rejects_nonsquare_sides_before_image_new(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    _write_checkpoint(root, size={"height": 20_000, "width": 10_000}, hidden_stride=1)
    info = _load_info(root)
    _expect_rejected(envs.VLLM_MAX_IMAGE_PIXELS, lambda: _profile(info), guard=True)


def test_profiling_rejects_hidden_stride_that_inflates_dummy_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MAX_IMAGE_PIXELS", "1000")
    root = tmp_path / "model"
    # side 8 is under the limit; side * 8 * 9 = 576 is not.
    # image_size 8 and patch 1 make hidden_stride 8 divide the patch grid.
    _write_checkpoint(
        root,
        size={"shortest_edge": 8},
        hidden_stride=8,
        image_size=8,
        patch_size=1,
    )
    info = _load_info(root)
    _expect_rejected(1000, lambda: _profile(info), guard=False)


@pytest.mark.parametrize(
    ("size", "hidden_stride"),
    [
        ({"shortest_edge": 0}, 1),
        ({"shortest_edge": -5}, 1),
        ({"shortest_edge": _TINY_EDGE}, 0),
        ({"shortest_edge": _TINY_EDGE}, -1),
    ],
)
def test_profiling_rejects_non_positive_side_and_stride(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: dict[str, int],
    hidden_stride: int,
) -> None:
    monkeypatch.setenv("VLLM_MAX_IMAGE_PIXELS", "1000")
    root = tmp_path / "model"
    _write_checkpoint(root, size=size, hidden_stride=hidden_stride)
    info = _load_info(root)
    _expect_rejected(1000, lambda: _profile(info), guard=True)


def test_profiling_reread_of_hidden_stride_cannot_allocate(
    tmp_path: Path,
) -> None:
    """Stride is used again after processor setup. A later read must not allocate."""
    root = tmp_path / "model"
    _write_checkpoint(root, size={"shortest_edge": 200}, hidden_stride=1)
    info = _load_info(root)
    # Processor setup reads hidden_stride while it is still 1.
    info.get_hf_processor()
    info.get_hf_config().visual_tokenizer_config.hidden_stride = 16
    limit = envs.VLLM_MAX_IMAGE_PIXELS

    def _call() -> Image.Image:
        return _profile(info)

    with _guard_allocations(limit):
        try:
            image = _call()
        except OversizedAllocation as exc:
            raise AssertionError(
                "hidden_stride was reread and used for Image.new"
            ) from exc
        except ValueError:
            return
    # The value captured while it was 1 profiles a 200 * 9 dummy image.
    assert image.size == (200 * 9, 200 * 9)


def test_preprocess_rejects_square_tensor_over_pixel_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MAX_IMAGE_PIXELS", "1000")
    root = tmp_path / "model"
    _write_checkpoint(root, size={"shortest_edge": 40}, hidden_stride=1)
    processor = _load_info(root).get_hf_processor()
    tiny = Image.new("RGB", (4, 4), color=1)

    def _call() -> None:
        processor(
            images=[tiny],
            max_partition=9,
            covering_threshold=0.9,
        )

    _expect_rejected(1000, _call, guard=False)


def test_preprocess_rejects_width_height_square_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MAX_IMAGE_PIXELS", "1000")
    root = tmp_path / "model"
    _write_checkpoint(root, size={"height": 48, "width": 48}, hidden_stride=1)
    processor = _load_info(root).get_hf_processor()
    tiny = Image.new("RGB", (4, 4), color=1)
    _expect_rejected(
        1000,
        lambda: processor.preprocess_image(
            tiny,
            max_partition=9,
            covering_threshold=0.9,
            do_convert_rgb=True,
            return_tensors="pt",
        ),
        guard=False,
    )


def test_preprocess_reread_of_side_cannot_allocate(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _write_checkpoint(root, size={"shortest_edge": _TINY_EDGE}, hidden_stride=1)
    processor = _load_info(root).get_hf_processor()
    processor.image_processor.size = EscalatingSize()
    tiny = Image.new("RGB", (4, 4), color=1)
    limit = envs.VLLM_MAX_IMAGE_PIXELS

    with _guard_allocations(limit):
        try:
            pixels, _, _ = processor.preprocess_image(
                tiny,
                max_partition=9,
                covering_threshold=0.9,
                do_convert_rgb=True,
                return_tensors="pt",
            )
        except OversizedAllocation as exc:
            raise AssertionError(
                "image size was reread and used for the square tensor"
            ) from exc
        except ValueError:
            return
    assert pixels.shape[-1] == _TINY_EDGE
    assert pixels.shape[-2] == _TINY_EDGE


def test_huge_shortest_edge_is_rejected_before_pillow_allocates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    _write_checkpoint(root, size={"shortest_edge": 20_000}, hidden_stride=1)
    info = _load_info(root)
    limit = envs.VLLM_MAX_IMAGE_PIXELS
    _expect_rejected(limit, lambda: _profile(info), guard=True)
    processor = info.get_hf_processor()
    _expect_rejected(
        limit,
        lambda: processor.preprocess_image(
            Image.new("RGB", (2, 2), color=1),
            max_partition=9,
            covering_threshold=0.9,
            do_convert_rgb=True,
            return_tensors="pt",
        ),
        guard=True,
    )
