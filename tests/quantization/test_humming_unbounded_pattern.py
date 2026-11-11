# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming checkpoint patterns must return while a layer scheme is resolved.

``re:`` ignore entries and ``dynamic`` keys that start with ``+:`` or ``-:``
are passed to ``regex.match`` with no timeout. A nested pattern that does not
match a layer name never returns, so layer construction does not finish.

``ignored_layers`` and ``modules_to_not_convert`` share the ``ignore`` loop in
``HummingConfig.is_layer_skipped``. ``humming_is_layer_skipped`` does not
compile ``re:`` ignore entries (it uses a substring check), so that ignore
path is not a regex match and is not covered here. ``moe.py`` calls
``humming_is_layer_skipped`` and ``resolve_humming_layer_config`` directly;
those two functions are covered.
"""

import multiprocessing
from typing import Any

import pytest

from vllm.model_executor.layers.quantization.humming import HummingConfig
from vllm.model_executor.layers.quantization.utils.humming.schema import (
    humming_is_layer_skipped,
    resolve_humming_layer_config,
)

# Real layer-prefix shape. At this length, ``^((.|.)+)+x$`` does not return
# from regex 2026.2.28 (the pin in requirements/test/cuda.txt).
LAYER_NAME = "language_model.model.layers.3.mlp.experts.383.up_proj"
NESTED_UNMATCHED = r"^((.|.)+)+x$"
BENIGN_EXACT = r"^language_model\.model\.layers\.3\.mlp\.experts\.383\.up_proj$"
RETURN_DEADLINE_S = 1.0


def _checkpoint(extra: dict[str, Any]) -> dict[str, Any]:
    return {"quant_method": "humming", **extra}


def _invoke(case: str) -> Any:
    """Run one Humming matcher the checkpoint load path calls."""
    ignore = _checkpoint({"ignore": [f"re:{NESTED_UNMATCHED}"]})
    minus = _checkpoint({"dynamic": {f"-:{NESTED_UNMATCHED}": {}}})
    plus = _checkpoint(
        {"dynamic": {f"+:{NESTED_UNMATCHED}": {"quant_method": "humming"}}}
    )
    benign = _checkpoint({"ignore": [f"re:{BENIGN_EXACT}"]})
    if case == "benign_ignore":
        cfg = HummingConfig.from_config(benign)
        return cfg.is_layer_skipped(benign, LAYER_NAME)
    if case == "quant_config_ignore":
        cfg = HummingConfig.from_config(ignore)
        return cfg.get_quant_config_for_layer(LAYER_NAME, "linear")
    if case == "quant_config_minus":
        cfg = HummingConfig.from_config(minus)
        return cfg.get_quant_config_for_layer(LAYER_NAME, "linear")
    if case == "quant_config_plus":
        cfg = HummingConfig.from_config(plus)
        return cfg.get_quant_config_for_layer(LAYER_NAME, "linear")
    if case == "weight_schema_ignore":
        cfg = HummingConfig.from_config(ignore)
        return cfg.get_layer_weight_schema(ignore, LAYER_NAME)
    if case == "weight_schema_plus":
        cfg = HummingConfig.from_config(plus)
        return cfg.get_layer_weight_schema(plus, LAYER_NAME)
    if case == "input_config_ignore":
        cfg = HummingConfig.from_config(ignore)
        return cfg.get_layer_input_config(ignore, LAYER_NAME)
    if case == "input_config_plus":
        cfg = HummingConfig.from_config(plus)
        return cfg.get_layer_input_config(plus, LAYER_NAME)
    if case == "skipped_ignore":
        cfg = HummingConfig.from_config(ignore)
        return cfg.is_layer_skipped(ignore, LAYER_NAME)
    if case == "skipped_minus":
        cfg = HummingConfig.from_config(minus)
        return cfg.is_layer_skipped(minus, LAYER_NAME)
    if case == "resolve_plus":
        return resolve_humming_layer_config(plus, LAYER_NAME)
    if case == "schema_skipped_minus":
        return humming_is_layer_skipped(minus, LAYER_NAME)
    raise AssertionError(case)


def _worker(case: str, conn: multiprocessing.connection.Connection) -> None:
    try:
        _invoke(case)
    except Exception:
        conn.send("returned")
    else:
        conn.send("returned")
    finally:
        conn.close()


def _assert_returns(case: str) -> None:
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(case, child))
    proc.start()
    child.close()
    proc.join(RETURN_DEADLINE_S)
    if proc.is_alive():
        proc.kill()
        proc.join(5)
        pytest.fail(
            f"{case} was still matching a nested Humming pattern after "
            f"{RETURN_DEADLINE_S}s"
        )
    assert parent.poll(1)
    assert parent.recv() == "returned"


STALL_CASES = [
    "quant_config_ignore",
    "quant_config_minus",
    "quant_config_plus",
    "weight_schema_ignore",
    "weight_schema_plus",
    "input_config_ignore",
    "input_config_plus",
    "skipped_ignore",
    "skipped_minus",
    "resolve_plus",
    "schema_skipped_minus",
]


@pytest.mark.parametrize("case", STALL_CASES)
def test_humming_nested_pattern_returns(case: str) -> None:
    """Nested unmatched Humming patterns must not stall layer setup."""
    _assert_returns(case)


def test_humming_exact_ignore_regex_still_matches() -> None:
    """A normal ``re:`` ignore entry still selects that layer."""
    assert _invoke("benign_ignore") is True
