# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling must reject an uncapped checkpoint pixel budget.

Qwen2-VL-family processors, and the copied Cohere Compass search, turn
``size.longest_edge`` (or the ``max_pixels`` alias) into a patch count and
trial-divide downward from its integer square root on the engine startup
thread. ``MultiModalBudget`` runs that search while the server is still
initializing. These tests fail while that search is unbounded: a prime patch
count does not finish, and it is not rejected. A fix must reject an over-cap
processor config quickly, before that search runs.
"""

from __future__ import annotations

import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    CohereCompassConfig,
    CohereCompassImageProcessor,
    CohereCompassProcessor,
    CohereCompassVideoProcessor,
    PreTrainedTokenizerFast,
    Qwen2_5_VLConfig,
    Qwen2_5_VLProcessor,
    Qwen2VLConfig,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
    Qwen3VLConfig,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from vllm.config import ModelConfig, VllmConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget

# Prime near 10**18 (Miller-Rabin). isqrt is 10**9, so one trial division
# from that root takes far longer than BUDGET_TIMEOUT_S.
_PRIME_PATCH_COUNT = 1_000_000_000_000_000_079
_BENIGN_LONGEST_EDGE = 28 * 28 * 1280
_BUDGET_TIMEOUT_S = 8.0
_REJECTION_TOKENS = ("pixel", "longest_edge", "max_pixels", "patch count")


@dataclass(frozen=True)
class _VisionFamily:
    name: str
    config_cls: type
    architecture: str
    processor_cls: type
    image_processor_cls: type
    video_processor_cls: type
    patch_size: int
    merge_size: int
    temporal_patch_size: int


_QWEN2 = _VisionFamily(
    name="qwen2_vl",
    config_cls=Qwen2VLConfig,
    architecture="Qwen2VLForConditionalGeneration",
    processor_cls=Qwen2VLProcessor,
    image_processor_cls=Qwen2VLImageProcessor,
    video_processor_cls=Qwen2VLVideoProcessor,
    patch_size=14,
    merge_size=2,
    temporal_patch_size=2,
)
_QWEN2_5 = _VisionFamily(
    name="qwen2_5_vl",
    config_cls=Qwen2_5_VLConfig,
    architecture="Qwen2_5_VLForConditionalGeneration",
    processor_cls=Qwen2_5_VLProcessor,
    image_processor_cls=Qwen2VLImageProcessor,
    video_processor_cls=Qwen2VLVideoProcessor,
    patch_size=14,
    merge_size=2,
    temporal_patch_size=2,
)
_QWEN3 = _VisionFamily(
    name="qwen3_vl",
    config_cls=Qwen3VLConfig,
    architecture="Qwen3VLForConditionalGeneration",
    processor_cls=Qwen3VLProcessor,
    image_processor_cls=Qwen2VLImageProcessor,
    video_processor_cls=Qwen3VLVideoProcessor,
    patch_size=16,
    merge_size=2,
    temporal_patch_size=2,
)
_OPENCUA = _VisionFamily(
    name="opencua",
    config_cls=Qwen2_5_VLConfig,
    architecture="OpenCUAForConditionalGeneration",
    processor_cls=Qwen2_5_VLProcessor,
    image_processor_cls=Qwen2VLImageProcessor,
    video_processor_cls=Qwen2VLVideoProcessor,
    patch_size=14,
    merge_size=2,
    temporal_patch_size=2,
)
_COMPASS = _VisionFamily(
    name="cohere_compass",
    config_cls=CohereCompassConfig,
    architecture="CohereCompassForConditionalGeneration",
    processor_cls=CohereCompassProcessor,
    image_processor_cls=CohereCompassImageProcessor,
    video_processor_cls=CohereCompassVideoProcessor,
    patch_size=16,
    merge_size=2,
    temporal_patch_size=2,
)


def _unit_area(family: _VisionFamily) -> int:
    unit = family.patch_size * family.merge_size
    return unit * unit


def _write_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
    vocab = {
        "<unk>": 0,
        "<|image_pad|>": 1,
        "<|video_pad|>": 2,
        "<|vision_start|>": 3,
        "<|vision_end|>": 4,
        "hello": 5,
    }
    inner = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer_path = model_dir / "tokenizer.json"
    inner.save(str(tokenizer_path))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="<unk>",
        model_max_length=2048,
    )
    tokenizer.save_pretrained(model_dir)
    return tokenizer


def _write_checkpoint(
    model_dir: Path,
    family: _VisionFamily,
    *,
    image_longest_edge: int,
    video_longest_edge: int | None = None,
    image_as_max_pixels: bool = False,
) -> None:
    """Write a local checkpoint whose processor JSON carries the pixel budget.

    The processor is first saved with a normal size so the surrounding files
    match what ``from_pretrained`` loads. The attacker's integer is then
    written into that JSON, which is the file the server reads.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    config = family.config_cls(
        num_hidden_layers=1,
        vision_config={
            "depth": 1,
            "patch_size": family.patch_size,
            "spatial_merge_size": family.merge_size,
            "temporal_patch_size": family.temporal_patch_size,
        },
    )
    config.save_pretrained(model_dir)
    config_path = model_dir / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict["architectures"] = [family.architecture]
    vision = config_dict.get("vision_config")
    if isinstance(vision, dict):
        vision["patch_size"] = family.patch_size
        vision["spatial_merge_size"] = family.merge_size
        vision["temporal_patch_size"] = family.temporal_patch_size
    config_path.write_text(json.dumps(config_dict))

    tokenizer = _write_tokenizer(model_dir)
    image_processor = family.image_processor_cls(
        size={
            "shortest_edge": 56 * 56,
            "longest_edge": _BENIGN_LONGEST_EDGE,
        },
        patch_size=family.patch_size,
        merge_size=family.merge_size,
        temporal_patch_size=family.temporal_patch_size,
    )
    video_processor = family.video_processor_cls(
        size={
            "shortest_edge": 56 * 56,
            "longest_edge": _BENIGN_LONGEST_EDGE,
        },
        patch_size=family.patch_size,
        merge_size=family.merge_size,
        temporal_patch_size=family.temporal_patch_size,
    )
    processor = family.processor_cls(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
    )
    processor.save_pretrained(model_dir)

    processor_path = model_dir / "processor_config.json"
    processor_dict = json.loads(processor_path.read_text())
    image_dict = processor_dict["image_processor"]
    if image_as_max_pixels:
        image_dict.pop("size", None)
        image_dict["min_pixels"] = 56 * 56
        image_dict["max_pixels"] = image_longest_edge
    else:
        image_dict["size"]["longest_edge"] = image_longest_edge
        image_dict["size"]["shortest_edge"] = 56 * 56
    if video_longest_edge is not None:
        video_dict = processor_dict["video_processor"]
        video_dict["size"]["longest_edge"] = video_longest_edge
        video_dict["size"]["shortest_edge"] = 56 * 56
        video_dict["temporal_patch_size"] = family.temporal_patch_size
    processor_path.write_text(json.dumps(processor_dict))


def _profile_budget(model_dir: str) -> dict[str, int]:
    """Build the multimodal budget the engine constructs during startup."""
    model_config = ModelConfig(
        model=model_dir,
        tokenizer=model_dir,
        tokenizer_mode="hf",
        dtype="float32",
        max_model_len=2048,
        enforce_eager=True,
        seed=0,
        trust_remote_code=False,
    )
    scheduler_config = SchedulerConfig(
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
    )
    budget = MultiModalBudget(
        vllm_config,
        MULTIMODAL_REGISTRY,
        enable_cache=False,
    )
    return dict(budget.mm_max_toks_per_item)


def _profile_budget_worker(model_dir: str, queue) -> None:
    try:
        queue.put(("ok", _profile_budget(model_dir)))
    except BaseException as exc:
        queue.put(("err", type(exc).__name__, f"{type(exc).__name__}: {exc}"))


def _assert_budget_rejected(model_dir: Path) -> None:
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_profile_budget_worker,
        args=(str(model_dir), queue),
        daemon=True,
    )
    proc.start()
    proc.join(_BUDGET_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        pytest.fail(
            "MultiModalBudget did not finish within "
            f"{_BUDGET_TIMEOUT_S}s. The checkpoint pixel budget is still "
            "trial-divided on the startup path."
        )
    if queue.empty():
        pytest.fail(
            f"startup profiling exited without a result (exitcode={proc.exitcode})"
        )
    status, *payload = queue.get_nowait()
    assert status == "err", payload
    message = payload[-1].lower()
    assert any(token in message for token in _REJECTION_TOKENS), payload


def test_qwen2_vl_official_pixel_budget_profiles(tmp_path: Path) -> None:
    """A published-size longest_edge still produces a finite token budget."""
    model_dir = tmp_path / "benign"
    _write_checkpoint(
        model_dir,
        _QWEN2,
        image_longest_edge=_BENIGN_LONGEST_EDGE,
    )
    budget = _profile_budget(str(model_dir))
    assert 1 <= budget["image"] <= 8192
    assert 1 <= budget["video"] <= 8192


@pytest.mark.parametrize(
    "family",
    [_QWEN2, _QWEN2_5, _QWEN3, _OPENCUA, _COMPASS],
    ids=lambda family: family.name,
)
def test_prime_longest_edge_rejected_at_startup(
    tmp_path: Path, family: _VisionFamily
) -> None:
    """``size.longest_edge`` that factorizes to a large prime is rejected."""
    longest_edge = _PRIME_PATCH_COUNT * _unit_area(family)
    model_dir = tmp_path / family.name
    _write_checkpoint(
        model_dir,
        family,
        image_longest_edge=longest_edge,
    )
    _assert_budget_rejected(model_dir)


def test_qwen2_vl_prime_max_pixels_alias_rejected(tmp_path: Path) -> None:
    """The ``max_pixels`` alias is the same uncapped budget as longest_edge."""
    longest_edge = _PRIME_PATCH_COUNT * _unit_area(_QWEN2)
    model_dir = tmp_path / "max-pixels"
    _write_checkpoint(
        model_dir,
        _QWEN2,
        image_longest_edge=longest_edge,
        image_as_max_pixels=True,
    )
    _assert_budget_rejected(model_dir)


def test_qwen3_vl_prime_video_longest_edge_rejected(tmp_path: Path) -> None:
    """Qwen3-VL profiles video pixels through the same factor search."""
    video_edge = _PRIME_PATCH_COUNT * _unit_area(_QWEN3) * _QWEN3.temporal_patch_size
    model_dir = tmp_path / "qwen3-video"
    _write_checkpoint(
        model_dir,
        _QWEN3,
        image_longest_edge=_BENIGN_LONGEST_EDGE,
        video_longest_edge=video_edge,
    )
    _assert_budget_rejected(model_dir)
