# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""param_sink_number from a checkpoint must be capped before device storage.

These tests are the spec for the missing cap. They fail while model
construction still allocates sink storage from the raw config integer.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.attention.static_sink_attention import (
    create_static_sink_attention_backend,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.flash_attn_diffkv import FlashAttentionDiffKVBackend
from vllm.v1.kv_cache_interface import FullAttentionSpec

# A handful of sink tokens still has to load. Anything past this ceiling is
# the uncapped config integer and must be rejected first.
MAX_PARAM_SINK_NUMBER = 64
ACCEPTED_SINK_NUMBER = 16
OVER_CAP_SINK_NUMBER = MAX_PARAM_SINK_NUMBER + 1
ENORMOUS_SINK_NUMBER = 100_003

_ARCHITECTURES = (
    "PanguEmbeddedForCausalLM",
    "PanguProMoEV2ForCausalLM",
    "PanguUltraMoEForCausalLM",
)


def _checkpoint(
    directory: Path,
    architecture: str,
    param_sink_number: int,
    *,
    with_value: bool,
    num_hidden_layers: int,
) -> None:
    """Write a tiny in-tree Pangu config and nothing else.

    ``model_type`` is a class transformers already knows, so loading does not
    need ``trust_remote_code``. The MLA fields are omitted, which selects the
    sink attention path. No weight file is present.
    """
    config = {
        "architectures": [architecture],
        "model_type": "llama",
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "attention_bias": False,
        "param_sink_number": param_sink_number,
        "param_sink_with_value": with_value,
        "qk_nope_dim": 8,
        "qk_rope_dim": 8,
        "v_channels": 8,
    }
    if architecture != "PanguEmbeddedForCausalLM":
        config.update(
            {
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 32,
                "norm_topk_prob": True,
                "routed_scaling_factor": 1.0,
                "first_k_dense_replace": 0,
            }
        )
    (directory / "config.json").write_text(json.dumps(config))
    assert sorted(path.name for path in directory.iterdir()) == ["config.json"]


def _model_config(directory: Path) -> ModelConfig:
    config = ModelConfig(
        model=str(directory),
        skip_tokenizer_init=True,
        max_model_len=32,
        dtype="float32",
        trust_remote_code=False,
        enforce_eager=True,
    )
    assert config.trust_remote_code is False
    return config


def _shape_from_factory_args(args: tuple) -> tuple[int, ...] | None:
    if not args:
        return None
    first = args[0]
    if isinstance(first, (tuple, list)) and all(isinstance(dim, int) for dim in first):
        return tuple(first)
    dims: list[int] = []
    for arg in args:
        if not isinstance(arg, int):
            break
        dims.append(arg)
    return tuple(dims) if dims else None


@contextmanager
def _record_storage(
    recorded: list[tuple[str, tuple[int, ...] | int]],
) -> Iterator[None]:
    """Record tensor factories the sink path uses to size its storage."""
    original_empty = torch.empty
    original_zeros = torch.zeros
    original_arange = torch.arange

    def empty(*args, **kwargs):
        shape = _shape_from_factory_args(args)
        if shape is not None:
            recorded.append(("empty", shape))
        return original_empty(*args, **kwargs)

    def zeros(*args, **kwargs):
        shape = _shape_from_factory_args(args)
        if shape is not None:
            recorded.append(("zeros", shape))
        return original_zeros(*args, **kwargs)

    def arange(*args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], int) and isinstance(args[1], int):
            recorded.append(("arange", args[1] - args[0]))
        elif len(args) == 1 and isinstance(args[0], int):
            recorded.append(("arange", args[0]))
        return original_arange(*args, **kwargs)

    torch.empty = empty
    torch.zeros = zeros
    torch.arange = arange
    try:
        yield
    finally:
        torch.empty = original_empty
        torch.zeros = original_zeros
        torch.arange = original_arange


def _sink_derived_storage(
    recorded: list[tuple[str, tuple[int, ...] | int]],
    sink_number: int,
    *,
    block_size: int,
    max_model_len: int,
) -> list[tuple[str, tuple[int, ...] | int]]:
    """Storage whose size is the config integer or the sink block-table width."""
    banned = {sink_number}
    num_sink_blocks = sink_number // block_size
    # Small quotients collide with ordinary dims such as the head count.
    if num_sink_blocks > MAX_PARAM_SINK_NUMBER:
        banned.add(num_sink_blocks)
        banned.add(cdiv(max_model_len, block_size) + num_sink_blocks)
    hits: list[tuple[str, tuple[int, ...] | int]] = []
    for kind, shape in recorded:
        if kind == "arange":
            if shape in banned:
                hits.append((kind, shape))
            continue
        if isinstance(shape, tuple) and any(dim in banned for dim in shape):
            hits.append((kind, shape))
    return hits


def _create_model(vllm_config: VllmConfig):
    """Construct the module the loader builds before it reads weight bytes."""
    loader = get_model_loader(vllm_config.load_config)
    return loader.create_model(
        vllm_config=vllm_config,
        model_config=vllm_config.model_config,
    )


@pytest.mark.parametrize("architecture", _ARCHITECTURES)
def test_handful_of_sink_tokens_still_load(tmp_path, dist_init, architecture):
    """A short sink length is real configuration and must still allocate it."""
    _checkpoint(
        tmp_path,
        architecture,
        ACCEPTED_SINK_NUMBER,
        with_value=False,
        num_hidden_layers=1,
    )
    model_config = _model_config(tmp_path)
    assert model_config.architecture == architecture
    vllm_config = VllmConfig(model_config=model_config)
    model = _create_model(vllm_config)

    attention = model.model.layers[0].self_attn
    assert tuple(attention.param_sink_key.shape) == (ACCEPTED_SINK_NUMBER, 1, 16)
    assert tuple(attention.param_sink_value.shape) == (ACCEPTED_SINK_NUMBER, 1, 8)
    assert attention.param_sink_key.device.type == current_platform.device_type
    assert attention.attn.sink_len == ACCEPTED_SINK_NUMBER


@pytest.mark.parametrize("architecture", _ARCHITECTURES)
@pytest.mark.parametrize("with_value", [False, True])
@pytest.mark.parametrize(
    "param_sink_number",
    [OVER_CAP_SINK_NUMBER, ENORMOUS_SINK_NUMBER],
)
def test_param_sink_number_rejected_before_sink_storage(
    tmp_path,
    dist_init,
    architecture,
    with_value,
    param_sink_number,
):
    """Reject an over-cap sink length before key or value storage exists.

    Both value forms are covered: a parameter when ``param_sink_with_value``
    is set, and the non-persistent zero buffer when it is not. Two decoder
    layers make a check that runs after the first layer's allocation fail.
    """
    _checkpoint(
        tmp_path,
        architecture,
        param_sink_number,
        with_value=with_value,
        num_hidden_layers=2,
    )
    model_config = _model_config(tmp_path)
    assert model_config.architecture == architecture
    assert model_config.hf_config.param_sink_number == param_sink_number
    vllm_config = VllmConfig(model_config=model_config)
    recorded: list[tuple[str, tuple[int, ...] | int]] = []
    with _record_storage(recorded):
        with pytest.raises(ValueError, match="param_sink_number"):
            _create_model(vllm_config)
    derived = _sink_derived_storage(
        recorded,
        param_sink_number,
        block_size=vllm_config.cache_config.block_size,
        max_model_len=model_config.max_model_len,
    )
    assert derived == []


def test_static_sink_block_table_rejects_config_length(tmp_path, dist_init):
    """The sink block table is sized from the same config integer.

    OpenPangu passes that integer to StaticSinkAttention, whose builder
    allocates ``(max_num_seqs, max_num_blocks + param_sink_number // block_size)``
    on the worker device. The cap has to apply before that ``torch.zeros``.
    """
    _checkpoint(
        tmp_path,
        "PanguEmbeddedForCausalLM",
        ENORMOUS_SINK_NUMBER,
        with_value=False,
        num_hidden_layers=1,
    )
    model_config = _model_config(tmp_path)
    sink_number = model_config.hf_config.param_sink_number
    assert sink_number == ENORMOUS_SINK_NUMBER
    vllm_config = VllmConfig(model_config=model_config)
    spec = FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float32,
    )
    recorded: list[tuple[str, tuple[int, ...] | int]] = []
    with _record_storage(recorded):
        with pytest.raises(ValueError, match="sink"):
            # The factory closes over the length; the builder then allocates.
            create_static_sink_attention_backend.cache_clear()
            backend = create_static_sink_attention_backend(
                FlashAttentionDiffKVBackend,
                sink_len=sink_number,
            )
            backend.get_builder_cls()(
                spec,
                ["model.layers.0.self_attn.attn"],
                vllm_config,
                torch.device(current_platform.device_type),
            )
    derived = _sink_derived_storage(
        recorded,
        sink_number,
        block_size=vllm_config.cache_config.block_size,
        max_model_len=model_config.max_model_len,
    )
    assert derived == []
