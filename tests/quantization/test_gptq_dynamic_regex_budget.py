# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic checkpoint patterns must not stall layer construction.

GPTQ ``dynamic`` keys are copied from the checkpoint and matched with
``regex`` while each linear or MoE layer is built. The third-party ``regex``
engine shipped with vLLM (2026.2.28 in the CUDA and CPU test locks) returns
quickly for a nested quantifier such as ``(a+)+``. It still takes exponential
time on a chain of ambiguous alternations. These tests build that chain from
a prefix of a real layer name, so ``re.match`` explores the prefix and then
fails. They fail while that match is unbounded and pass when loading either
rejects the pattern or finishes the match within a few seconds.

The same unbounded ``regex`` match is also covered for Humming dynamic rules
and for ``re:`` targets. These callers are not covered here:

- ``TorchAOConfig.get_quant_method`` (``re.fullmatch`` on ``re:`` module keys)
- ``INCConfigParser`` (``re.search`` over ``extra_config`` keys)
"""

import json
import signal
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import regex
import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, ReplicatedLinear
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    get_moe_quant_method,
)
from vllm.model_executor.layers.quantization.humming import HummingConfig
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_equal_or_regex_match,
)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    get_linear_quant_method,
    override_config,
)
from vllm.model_executor.layers.quantization.utils.humming.schema import (
    humming_is_layer_skipped,
    resolve_humming_layer_config,
)

LAYER = "model.layers.0.self_attn.q_proj"
# 3**22 candidate splits. On this class of CPU the match of 19 characters
# already exceeds 8s, so 22 characters stays over a 3s budget on a much
# faster machine, while a bounded or literal matcher returns immediately.
_AMBIGUOUS_CHARS = 22
_MATCH_BUDGET_S = 3.0


class _MatchBudgetExceeded(Exception):
    """Raised when a checkpoint pattern match exceeds the time budget."""


def _on_alarm(signum: int, frame: object) -> None:
    raise _MatchBudgetExceeded


class _alarm_budget:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._previous: signal.Handlers = signal.SIG_DFL

    def __enter__(self) -> None:
        self._previous = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, self._seconds)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._previous)


def _ambiguous_pattern(layer_name: str = LAYER) -> str:
    """Alternation chain that matches ``layer_name`` until a failing char."""
    parts: list[str] = []
    for char in layer_name[:_AMBIGUOUS_CHARS]:
        escaped = regex.escape(char)
        parts.append(f"(?:{escaped}|{escaped}|{escaped})")
    return "".join(parts) + "!"


def _gptq_config(dynamic: dict[str, dict[str, int | bool]]) -> dict[str, object]:
    return {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "lm_head": False,
        "quant_method": "gptq",
        "dynamic": dynamic,
    }


def _load_gptq(dynamic: dict[str, dict[str, int | bool]]) -> AutoGPTQConfig:
    # quantize_config.json is json.load'd and passed to from_config.
    payload = json.loads(json.dumps(_gptq_config(dynamic)))
    return AutoGPTQConfig.from_config(payload)


def assert_finishes(fn: Callable[[], object]) -> None:
    """Pass if ``fn`` returns or raises before the match budget expires."""
    try:
        with _alarm_budget(_MATCH_BUDGET_S):
            fn()
    except _MatchBudgetExceeded:
        pytest.fail(
            f"checkpoint pattern match did not return within {_MATCH_BUDGET_S} seconds"
        )


class _BareLinear(LinearBase):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)


class _ShapedExperts(RoutedExperts):
    """RoutedExperts shape that Marlin accepts, without building experts."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.moe_config = SimpleNamespace(
            hidden_dim=128,
            intermediate_size_per_partition_unpadded=128,
        )


def test_short_positive_dynamic_rule_still_overrides_bits() -> None:
    """A documented ``+:`` rule must still apply once matching is bounded."""
    config = _load_gptq({r"+:model\.layers\.0\..*": {"bits": 8}})
    with _alarm_budget(_MATCH_BUDGET_S):
        bits = get_dynamic_override(config, LAYER, "bits", config.weight_bits)
    assert bits == 8


def test_short_negative_dynamic_rule_still_skips_layer() -> None:
    config = _load_gptq({r"-:model\.layers\.0\..*": {}})
    with _alarm_budget(_MATCH_BUDGET_S):
        skipped = get_dynamic_override(config, LAYER)
    assert skipped is False


@pytest.mark.parametrize(
    "key_prefix",
    ["", "+:", "-:"],
    ids=["unprefixed", "positive", "negative"],
)
def test_get_dynamic_override_bounds_checkpoint_pattern(key_prefix: str) -> None:
    pattern = key_prefix + _ambiguous_pattern()

    def run() -> None:
        config = _load_gptq({pattern: {"bits": 8}})
        get_dynamic_override(config, LAYER, "bits", config.weight_bits)

    assert_finishes(run)


def test_override_config_bounds_checkpoint_pattern() -> None:
    pattern = _ambiguous_pattern()

    def run() -> None:
        config = _load_gptq({pattern: {"bits": 8}})
        override_config(config, LAYER)

    assert_finishes(run)


def test_get_linear_quant_method_bounds_checkpoint_pattern() -> None:
    pattern = _ambiguous_pattern()

    def run() -> None:
        config = _load_gptq({pattern: {"bits": 8}})
        get_linear_quant_method(config, _BareLinear(), LAYER, lambda cfg: cfg)

    assert_finishes(run)


def test_replicated_linear_construction_bounds_checkpoint_pattern() -> None:
    """Layer init resolves the quant method and matches ``dynamic`` there."""
    pattern = _ambiguous_pattern()

    def run() -> None:
        config = _load_gptq({pattern: {"bits": 8}})
        ReplicatedLinear(
            128,
            128,
            bias=False,
            quant_config=config,
            prefix=LAYER,
            disable_tp=True,
            params_dtype=torch.float16,
        )

    assert_finishes(run)


def test_moe_quant_method_bounds_checkpoint_pattern() -> None:
    pattern = _ambiguous_pattern()

    def run() -> None:
        config = _load_gptq({pattern: {"bits": 8}})
        get_moe_quant_method(config, _ShapedExperts(), LAYER, lambda cfg, moe: cfg)

    assert_finishes(run)


def test_auto_gptq_expert_quant_method_bounds_checkpoint_pattern() -> None:
    pattern = _ambiguous_pattern()

    def run() -> None:
        config = _load_gptq({pattern: {"bits": 8}})
        config.get_quant_method(_ShapedExperts(), LAYER)

    assert_finishes(run)


def test_moe_wna16_linear_delegation_bounds_checkpoint_pattern() -> None:
    pattern = _ambiguous_pattern()

    def run() -> None:
        config = MoeWNA16Config.from_config(_gptq_config({pattern: {"bits": 8}}))
        config.get_quant_method(_BareLinear(), LAYER)

    assert_finishes(run)


def test_humming_positive_dynamic_rule_is_bounded() -> None:
    pattern = "+:" + _ambiguous_pattern()

    def run() -> None:
        resolve_humming_layer_config({"dynamic": {pattern: {"bits": 8}}}, LAYER)

    assert_finishes(run)


def test_humming_negative_dynamic_rule_is_bounded() -> None:
    pattern = "-:" + _ambiguous_pattern()

    def run() -> None:
        humming_is_layer_skipped({"dynamic": {pattern: {}}}, LAYER)

    assert_finishes(run)


def test_humming_config_negative_dynamic_rule_is_bounded() -> None:
    pattern = "-:" + _ambiguous_pattern()
    quant_config = HummingConfig()

    def run() -> None:
        quant_config.is_layer_skipped({"dynamic": {pattern: {}}}, LAYER)

    assert_finishes(run)


def test_humming_ignored_regex_layer_is_bounded() -> None:
    pattern = "re:" + _ambiguous_pattern()
    quant_config = HummingConfig()

    def run() -> None:
        quant_config.is_layer_skipped({"ignored_layers": [pattern]}, LAYER)

    assert_finishes(run)


def test_re_prefix_target_match_is_bounded() -> None:
    pattern = "re:" + _ambiguous_pattern()

    def run() -> None:
        is_equal_or_regex_match(LAYER, pattern)

    assert_finishes(run)
