# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cos/sin caches must not be sized from an unbounded checkpoint integer.

``LlamaDecoderLayer`` passes the checkpoint ``max_position_embeddings`` and
``rope_parameters`` into ``get_rope``. Several rotary implementations then
``torch.arange`` that length, or that length times the checkpoint rope
factor, while building the cosine/sine table. ``max_model_len`` has to bound
every table that is actually materialized:

* a normal checkpoint window larger than ``max_model_len`` still loads, and
  every materialized table has at most ``max_model_len`` rows (``2 *
  max_model_len`` when long-RoPE keeps a short and a long table, and ``4 *
  max_model_len`` for M-RoPE's fixed position multiplier);
* a non-positive ``max_position_embeddings`` or rope factor is rejected with
  ``ValueError``;
* a length at or above ``_ABSURD_POSITIONS`` is rejected with ``ValueError``
  before ``torch.arange`` is asked for that many positions.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers import rotary_embedding as rope_module
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.models.llama import LlamaDecoderLayer
from vllm.utils.network_utils import get_open_port

_MAX_MODEL_LEN = 32
_CHECKPOINT_MAX_POS = 4096
# Far past any context window this tree's configs declare (the largest
# in-tree default found while writing this test is 524288).
_ABSURD_POSITIONS = 1 << 24
_MROPE_POSITIONS_FACTOR = 4


def _llama_config(
    directory: Path,
    max_position_embeddings: int,
    rope_parameters: dict[str, Any] | None,
) -> None:
    config: dict[str, Any] = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "vocab_size": 128,
        "max_position_embeddings": max_position_embeddings,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
    }
    if rope_parameters is not None:
        config["rope_parameters"] = rope_parameters
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _cache_rows(rotary_emb: torch.nn.Module) -> int:
    if hasattr(rotary_emb, "long_short_cos_sin_cache"):
        cache = rotary_emb.long_short_cos_sin_cache
    else:
        cache = rotary_emb.cos_sin_cache
    return int(cache.shape[0])


def _row_limit(rotary_emb: torch.nn.Module, max_model_len: int) -> int:
    if isinstance(rotary_emb, MRotaryEmbedding):
        return max_model_len * _MROPE_POSITIONS_FACTOR
    if hasattr(rotary_emb, "long_short_cos_sin_cache"):
        return max_model_len * 2
    return max_model_len


def _position_span(args: tuple[Any, ...]) -> int | None:
    if not args:
        return None
    if len(args) == 1:
        start: Any = 0
        end: Any = args[0]
    else:
        start, end = args[0], args[1]
    if isinstance(start, torch.Tensor) or isinstance(end, torch.Tensor):
        return None
    try:
        return int(end) - int(start)
    except (TypeError, ValueError, OverflowError):
        return None


@contextmanager
def _forbid_position_arange(limit: int) -> Iterator[list[int]]:
    """Fail the test if a position ``torch.arange`` exceeds ``limit``.

    The real ``arange`` still runs for every smaller span, including the
    rotary-dimension vector. A huge span raises before the tensor exists.
    """

    requested: list[int] = []
    real_arange = torch.arange

    def guarded(*args: Any, **kwargs: Any) -> torch.Tensor:
        span = _position_span(args)
        if span is not None and span > limit:
            requested.append(span)
            raise AssertionError(
                f"torch.arange requested {span} positions, above {limit}"
            )
        return real_arange(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "arange", guarded)
        yield requested


def _load_llama_layer(
    directory: Path,
    max_position_embeddings: int,
    rope_parameters: dict[str, Any] | None,
    max_model_len: int | None = _MAX_MODEL_LEN,
) -> LlamaDecoderLayer:
    rope_module._ROPE_DICT.clear()
    _llama_config(directory, max_position_embeddings, rope_parameters)
    model_config = ModelConfig(
        model=str(directory),
        skip_tokenizer_init=True,
        max_model_len=max_model_len,
        tokenizer_mode="hf",
        dtype="float32",
        enforce_eager=True,
    )
    vllm_config = VllmConfig(model_config=model_config)
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        local_rank=0,
        backend="gloo",
    )
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(1, 1)
        return LlamaDecoderLayer(vllm_config, prefix="model.layers.0")


_BOUNDED_ROPE: list[dict[str, Any] | None] = [
    None,
    {"rope_type": "linear", "factor": 4.0, "rope_theta": 10000.0},
    {
        "rope_type": "deepseek_yarn",
        "factor": 8,
        "original_max_position_embeddings": 512,
        "beta_fast": 32,
        "beta_slow": 1,
        "rope_theta": 10000.0,
    },
    {
        "rope_type": "deepseek_llama_scaling",
        "factor": 8.0,
        "original_max_position_embeddings": 512,
        "beta_fast": 32,
        "beta_slow": 1,
        "rope_theta": 10000.0,
    },
    {
        "rope_type": "deepseek_yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 512,
        "beta_fast": 32,
        "beta_slow": 1,
        "rope_theta": 10000.0,
        "is_deepseek_v4": True,
    },
    {
        "rope_type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 512,
        "rope_theta": 10000.0,
    },
    {
        "rope_type": "llama3",
        "factor": 8.0,
        "original_max_position_embeddings": 512,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "rope_theta": 10000.0,
    },
    {"rope_type": "ntk", "factor": 4.0, "rope_theta": 10000.0},
    {"rope_type": "dynamic", "factor": 4.0, "rope_theta": 10000.0},
    {
        "rope_type": "dynamic",
        "factor": 2.0,
        "alpha": 2.0,
        "rope_theta": 10000.0,
    },
    {
        "rope_type": "longrope",
        "short_factor": [1.0, 1.0, 1.0, 1.0],
        "long_factor": [1.0, 1.1, 1.2, 1.3],
        "original_max_position_embeddings": 128,
        "factor": 32.0,
        "rope_theta": 10000.0,
    },
    {"rope_type": "proportional", "rope_theta": 10000.0},
    {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "mrope_section": [1, 1, 2],
    },
    {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "use_fope": True,
        "num_key_value_heads": 4,
        "num_inv_freq": 4,
        "fope_sep_head": True,
        "fope_init_factor": 1.0,
    },
    {
        "rope_type": "openpangu",
        "rope_theta": 10000.0,
        "mrope_section": [1, 1, 2],
        "mrope_interleaved": True,
    },
]

_BOUNDED_IDS = [
    "default",
    "linear",
    "deepseek-yarn",
    "deepseek-llama-scaling",
    "deepseek-v4",
    "yarn",
    "llama3",
    "ntk",
    "dynamic",
    "dynamic-alpha",
    "longrope",
    "proportional",
    "mrope",
    "fope",
    "openpangu",
]


@pytest.mark.parametrize("rope_parameters", _BOUNDED_ROPE, ids=_BOUNDED_IDS)
def test_checkpoint_window_does_not_size_the_cache(
    tmp_path: Path,
    rope_parameters: dict[str, Any] | None,
) -> None:
    layer = _load_llama_layer(tmp_path, _CHECKPOINT_MAX_POS, rope_parameters)
    rotary_emb = layer.self_attn.rotary_emb
    rows = _cache_rows(rotary_emb)
    limit = _row_limit(rotary_emb, _MAX_MODEL_LEN)
    assert 0 < rows <= limit, (
        f"{type(rotary_emb).__name__} materialized {rows} cos/sin rows "
        f"from max_position_embeddings={_CHECKPOINT_MAX_POS} with "
        f"max_model_len={_MAX_MODEL_LEN}; limit is {limit}"
    )


@pytest.mark.parametrize("max_position_embeddings", [0, -1])
def test_nonpositive_max_position_embeddings_rejected(
    tmp_path: Path,
    max_position_embeddings: int,
) -> None:
    with pytest.raises(ValueError):
        _load_llama_layer(tmp_path, max_position_embeddings, None)


@pytest.mark.parametrize(
    "rope_parameters",
    [
        {"rope_type": "linear", "factor": 0.0, "rope_theta": 10000.0},
        {"rope_type": "linear", "factor": -2.0, "rope_theta": 10000.0},
        {
            "rope_type": "deepseek_yarn",
            "factor": 0.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32,
            "beta_slow": 1,
            "rope_theta": 10000.0,
        },
        {
            "rope_type": "deepseek_yarn",
            "factor": -2.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32,
            "beta_slow": 1,
            "rope_theta": 10000.0,
        },
        {
            "rope_type": "yarn",
            "factor": 0.0,
            "original_max_position_embeddings": 128,
            "rope_theta": 10000.0,
        },
    ],
    ids=[
        "linear-zero",
        "linear-negative",
        "deepseek-zero",
        "deepseek-negative",
        "yarn-zero",
    ],
)
def test_nonpositive_rope_factor_rejected(
    tmp_path: Path,
    rope_parameters: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        _load_llama_layer(tmp_path, _CHECKPOINT_MAX_POS, rope_parameters)


def test_absurd_max_position_embeddings_rejected_before_allocation(
    tmp_path: Path,
) -> None:
    with _forbid_position_arange(_ABSURD_POSITIONS - 1):
        with pytest.raises(ValueError):
            _load_llama_layer(tmp_path, _ABSURD_POSITIONS, None)


def test_absurd_max_position_embeddings_rejected_without_max_model_len(
    tmp_path: Path,
) -> None:
    with _forbid_position_arange(_ABSURD_POSITIONS - 1):
        with pytest.raises(ValueError):
            _load_llama_layer(
                tmp_path,
                _ABSURD_POSITIONS,
                None,
                max_model_len=None,
            )


@pytest.mark.parametrize(
    "rope_parameters",
    [
        {
            "rope_type": "linear",
            "factor": float(1 << 20),
            "rope_theta": 10000.0,
        },
        {
            "rope_type": "deepseek_yarn",
            "factor": float(1 << 20),
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "rope_theta": 10000.0,
        },
        {
            "rope_type": "yarn",
            "factor": float(1 << 20),
            "original_max_position_embeddings": 4096,
            "rope_theta": 10000.0,
        },
        {
            "rope_type": "deepseek_yarn",
            "factor": float(1 << 20),
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "rope_theta": 10000.0,
            "is_deepseek_v4": True,
        },
    ],
    ids=["linear", "deepseek-yarn", "yarn", "deepseek-v4"],
)
def test_absurd_rope_factor_rejected_before_allocation(
    tmp_path: Path,
    rope_parameters: dict[str, Any],
) -> None:
    with _forbid_position_arange(_ABSURD_POSITIONS - 1):
        with pytest.raises(ValueError):
            _load_llama_layer(tmp_path, _CHECKPOINT_MAX_POS, rope_parameters)
