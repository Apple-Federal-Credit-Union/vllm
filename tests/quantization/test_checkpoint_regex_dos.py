# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint regex patterns must not pin model initialization.

Quantization configs loaded from a checkpoint are matched against layer names
with the ``regex`` engine and no time limit. A pattern that fails on a normal
layer name can force exponential backtracking, so these tests fail until every
checkpoint pattern is rejected, escaped, or matched within ``LIMIT_S``.
"""

import json
import signal
from collections.abc import Callable
from pathlib import Path

import pytest
import regex
import torch

from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig
from vllm.config.model import ModelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.config.vllm import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors as compressed_tensors_quant,
)
from vllm.model_executor.layers.quantization.humming import HummingConfig
from vllm.model_executor.layers.quantization.inc.inc import INCConfig
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    find_matching_patterns,
    is_shared_expert_quant_fse_compatible,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
)
from vllm.model_executor.layers.quantization.utils.humming.schema import (
    humming_is_layer_skipped,
)
from vllm.model_executor.model_loader.weight_utils import get_quant_config
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

LIMIT_S = 1.0
LAYER_NAME = "model.layers.0.self_attn.qkv_proj"
# ``regex`` 2026.2.28 rejects a pattern whose minimum length exceeds the
# subject. Keep the failing pattern shorter than the layer name so the
# engine actually attempts the match. 30 ambiguous characters of this
# 33-character name do not finish within LIMIT_S.
_PATTERN_CHARS = 30
_LLAMA_PACKED = LlamaForCausalLM.packed_modules_mapping


class _MatchTimeout(Exception):
    """Raised when a checkpoint pattern match exceeds ``LIMIT_S``."""


def _on_alarm(signum: int, frame: object) -> None:
    raise _MatchTimeout


def _ambiguous_body(text: str) -> str:
    """Build a failing pattern whose minimum length still fits ``text``."""
    take = min(_PATTERN_CHARS, max(len(text) - 3, 1))
    return "".join(f"(?:{regex.escape(ch)}|{regex.escape(ch)})" for ch in text[:take])


def _re_pattern(text: str) -> str:
    return "re:" + _ambiguous_body(text) + "!"


def assert_bounded(fn: Callable[[], object]) -> None:
    """Fail if ``fn`` is still inside a checkpoint match after ``LIMIT_S``.

    Returning or raising is acceptable: a fix may reject the pattern or
    match it as a literal. Hanging is not.
    """
    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, LIMIT_S)
    try:
        fn()
    except _MatchTimeout:
        pytest.fail(
            f"checkpoint regex match was still running after {LIMIT_S} seconds"
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _write_llama_checkpoint(model_dir: Path, quantization_config: dict) -> None:
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 4,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "quantization_config": quantization_config,
    }
    (model_dir / "config.json").write_text(json.dumps(config))


def _compressed_tensors_config(ignore: list[str], targets: list[str]) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": ignore,
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "channel",
                    "symmetric": True,
                    "dynamic": False,
                },
            }
        },
    }


def _load_compressed_tensors(model_dir: Path, quantization_config: dict):
    """Load a local checkpoint the way ``vllm serve`` does, without remote code."""
    _write_llama_checkpoint(model_dir, quantization_config)
    try:
        model_config = ModelConfig(
            model=str(model_dir),
            trust_remote_code=False,
            max_model_len=32,
        )
    except (ValueError, regex.error):
        return None
    assert model_config.trust_remote_code is False
    try:
        quant_config = get_quant_config(model_config, LoadConfig())
    except (ValueError, regex.error, NotImplementedError):
        return None
    quant_config.packed_modules_mapping.update(_LLAMA_PACKED)
    return quant_config


def _build_linear(quant_config, prefix: str = LAYER_NAME) -> LinearBase:
    return LinearBase(
        8,
        8,
        quant_config=quant_config,
        prefix=prefix,
        disable_tp=True,
    )


def test_compressed_tensors_ignore_does_not_pin_layer_init(tmp_path: Path) -> None:
    """``quantization_config.ignore`` entries that start with ``re:`` are
    matched while each linear layer is constructed.
    """
    pattern = _re_pattern(LAYER_NAME)
    quant_config = _load_compressed_tensors(
        tmp_path,
        _compressed_tensors_config(ignore=[pattern], targets=["Linear"]),
    )
    if quant_config is None:
        return
    assert isinstance(
        quant_config, compressed_tensors_quant.CompressedTensorsConfig
    )
    assert pattern in quant_config.ignore
    assert_bounded(lambda: _build_linear(quant_config))


def test_compressed_tensors_target_does_not_pin_layer_init(tmp_path: Path) -> None:
    """``config_groups`` targets that start with ``re:`` use the same matcher."""
    pattern = _re_pattern(LAYER_NAME)
    quant_config = _load_compressed_tensors(
        tmp_path,
        _compressed_tensors_config(ignore=[], targets=[pattern]),
    )
    if quant_config is None:
        return
    assert_bounded(lambda: _build_linear(quant_config))


def test_fused_shard_patterns_do_not_pin() -> None:
    """Fused qkv/gate projections compare the pattern with every shard name."""
    pattern = _re_pattern(LAYER_NAME)
    quant_config = compressed_tensors_quant.CompressedTensorsConfig.from_config(
        _compressed_tensors_config(ignore=[pattern], targets=["Linear"])
    )
    quant_config.packed_modules_mapping.update(_LLAMA_PACKED)
    assert_bounded(
        lambda: find_matching_patterns(
            LAYER_NAME,
            quant_config.ignore,
            quant_config.packed_modules_mapping,
        )
    )


def test_quark_exclude_pattern_does_not_pin() -> None:
    """Quark ``exclude`` entries that start with ``re:`` are matched per layer."""
    pattern = _re_pattern(LAYER_NAME)
    quant_config = QuarkConfig.from_config(
        {
            "quant_method": "quark",
            "exclude": [pattern],
            "export": {"kv_cache_group": [], "pack_method": "reorder"},
        }
    )
    assert_bounded(lambda: quant_config.get_quant_method_target(LAYER_NAME, LinearBase))


def test_humming_ignore_pattern_does_not_pin() -> None:
    """Humming matches ``re:`` ignore entries while resolving a layer scheme."""
    pattern = _re_pattern(LAYER_NAME)
    quant_config = HummingConfig.from_config(
        {"quant_method": "humming", "ignore": [pattern]}
    )
    schema = quant_config.get_layer_weight_schema
    full_config = quant_config.full_config
    assert_bounded(lambda: schema(full_config, LAYER_NAME))


def test_humming_dynamic_patterns_do_not_pin() -> None:
    """Humming ``dynamic`` keys are regular expressions, not literals."""
    body = _ambiguous_body(LAYER_NAME) + "!"
    negative = HummingConfig.from_config(
        {"quant_method": "humming", "dynamic": {"-:" + body: {}}}
    )
    positive = HummingConfig.from_config(
        {
            "quant_method": "humming",
            "dynamic": {"+:" + body: {"quant_method": "humming"}},
        }
    )
    moe_skip = {"dynamic": {"-:" + body: {}}}
    assert_bounded(
        lambda: negative.get_layer_weight_schema(negative.full_config, LAYER_NAME)
    )
    assert_bounded(
        lambda: positive.get_layer_weight_schema(positive.full_config, LAYER_NAME)
    )
    assert_bounded(lambda: humming_is_layer_skipped(moe_skip, LAYER_NAME))


def test_gptq_dynamic_pattern_does_not_pin() -> None:
    """GPTQ ``dynamic`` keys are compiled and matched for every module."""
    body = _ambiguous_body(LAYER_NAME) + "!"
    quant_config = AutoGPTQConfig.from_config(
        {
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
            "dynamic": {"-:" + body: {}},
        }
    )
    assert_bounded(lambda: get_dynamic_override(quant_config, LAYER_NAME))


def test_inc_extra_config_pattern_does_not_pin() -> None:
    """INC ``extra_config`` keys that contain regex metacharacters are searched."""
    pattern = _ambiguous_body(LAYER_NAME) + "!"
    quant_config = INCConfig.from_config(
        {
            "quant_method": "inc",
            "bits": 4,
            "group_size": 128,
            "extra_config": {pattern: {"bits": 4}},
        }
    )
    assert_bounded(
        lambda: quant_config.get_quant_method(torch.nn.Linear(8, 8), LAYER_NAME)
    )


def test_online_ignore_and_targets_do_not_pin() -> None:
    """Online quantization matches ``re:`` ignore entries and target keys."""
    pattern = _re_pattern(LAYER_NAME)
    ignored = OnlineQuantizationConfig(
        QuantizationConfigArgs(ignore=[pattern], linear="fp8_per_tensor")
    )
    targeted = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={pattern: "fp8_per_tensor"})
    )
    assert_bounded(lambda: ignored.resolve_quant_method_cls(LinearBase, LAYER_NAME))
    assert_bounded(lambda: targeted.resolve_quant_method_cls(LinearBase, LAYER_NAME))


def test_deepseek_v4_layer_quant_pattern_does_not_pin(tmp_path: Path) -> None:
    """DeepSeek-V4 shared-expert setup ``fullmatch``es ``re:`` layer keys."""
    subject = "layers.0.ffn.shared_experts.w1"
    pattern = _re_pattern(subject)
    quantization_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "fmt": "e4m3",
        "global_quant_config": {
            "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32}
        },
        "layer_quant_config": {pattern: {"weight": {"dtype": "fp8"}}},
    }
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "expert_dtype": "fp4",
        "quantization_config": quantization_config,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    try:
        model_config = ModelConfig(
            model=str(tmp_path),
            trust_remote_code=False,
            max_model_len=32,
        )
    except (ValueError, regex.error):
        return
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
    )
    quant_config = DeepseekV4FP8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    )

    def match() -> None:
        with set_current_vllm_config(vllm_config):
            is_shared_expert_quant_fse_compatible(
                quant_config,
                "model.layers.0.mlp.experts",
                "model.layers.0.mlp.shared_experts",
            )

    assert_bounded(match)
