# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INC extra_config keys must not stall per-layer resolution.

A checkpoint can select the INC quantizer directly, or select auto-round so
``INCConfig.override_quantization_method`` rewrites it to INC. ``from_config``
keeps ``extra_config`` as supplied. For each layer, ``get_quant_method``
resolves that map. Keys that contain a regex metacharacter are compiled and
searched against the layer name, both in the default lookup and again when a
fused module name is expanded through ``packed_modules_mapping``.

Those searches have to return. An unmatched key leaves the checkpoint's
default bits and group size in place. An exact layer-name key still applies.
"""

from __future__ import annotations

import multiprocessing
from typing import Any

import torch

from vllm.model_executor.layers.quantization.inc.inc import INCConfig

# Long enough that a bounded match finishes, short enough that an unbounded
# backtracking search on the layer name below is still running.
_RESOLUTION_DEADLINE_S = 3.0
_REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
# Ambiguous one-character alternation, then a lookahead for a token that only
# occurs before the quantifier can stop. The match fails, but the search does
# not return.
_UNMATCHED_KEY = r"^(?:.|.)+(?=model)"
_LAYER = "model.layers.0.self_attn.q_proj"
_FUSED_LAYER = "model.layers.0.self_attn.qkv_proj"
# LlamaForCausalLM.packed_modules_mapping. Model load copies this onto the
# quant config before get_quant_method runs for each layer.
_LLAMA_PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _checkpoint(quant_method: str, extra_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "quant_method": quant_method,
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "extra_config": extra_config,
    }


def _unmatched_entry() -> dict[str, int | bool]:
    # Distinct from the checkpoint defaults so a search that applies this
    # entry cannot be confused with "no key matched".
    return {"bits": 2, "group_size": 32, "sym": True}


def _resolve_layer(
    quant_method: str,
    prefix: str,
    packed_modules: dict[str, list[str]] | None,
    result: multiprocessing.Queue,
) -> None:
    try:
        raw = _checkpoint(quant_method, {_UNMATCHED_KEY: _unmatched_entry()})
        if quant_method == "auto-round":
            overridden = INCConfig.override_quantization_method(raw, None)
            if overridden != INCConfig.get_name():
                result.put(("err", f"override returned {overridden!r}"))
                return
        config = INCConfig.from_config(raw)
        if _UNMATCHED_KEY not in (config.extra_config or {}):
            result.put(("err", "from_config dropped extra_config key"))
            return
        if packed_modules is not None:
            config.packed_modules_mapping = packed_modules
        layer = torch.nn.Linear(8, 8)
        config.get_quant_method(layer, prefix)
        bits, group_size, sym = config.get_layer_config(layer, prefix)
        result.put(("ok", bits, group_size, sym))
    except Exception as exc:
        result.put(("err", f"{type(exc).__name__}: {exc}"))


def _assert_unmatched_key_resolves(
    quant_method: str,
    prefix: str,
    packed_modules: dict[str, list[str]] | None = None,
) -> None:
    """Fail when layer resolution does not return the default config in time."""
    assert any(char in _REGEX_SPECIAL_CHARS for char in _UNMATCHED_KEY)
    ctx = multiprocessing.get_context("fork")
    result: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_resolve_layer,
        args=(quant_method, prefix, packed_modules, result),
    )
    proc.start()
    proc.join(_RESOLUTION_DEADLINE_S)
    if proc.is_alive():
        proc.kill()
        proc.join()
        raise AssertionError(
            f"{quant_method} layer resolution for {prefix!r} did not return "
            f"within {_RESOLUTION_DEADLINE_S}s"
        )
    assert proc.exitcode == 0, proc.exitcode
    status, *payload = result.get(timeout=1)
    assert status == "ok", payload
    bits, group_size, sym = payload
    assert (bits, group_size, sym) == (4, 128, True)


def test_inc_extra_config_key_does_not_stall_get_quant_method() -> None:
    """quant_method inc must resolve a metacharacter extra_config key."""
    _assert_unmatched_key_resolves("inc", _LAYER)


def test_auto_round_extra_config_key_does_not_stall_get_quant_method() -> None:
    """auto-round is rewritten to inc and must resolve the same key."""
    _assert_unmatched_key_resolves("auto-round", _LAYER)


def test_fused_module_extra_config_key_does_not_stall_resolution() -> None:
    """Fused-name expansion must resolve the key before the default lookup.

    ``qkv_proj`` occurs once in the layer name, so resolution builds the
    unpacked names and searches extra_config for each of them.
    """
    fusion_key = "qkv_proj"
    assert fusion_key in _FUSED_LAYER
    assert _FUSED_LAYER.count(fusion_key) == 1
    assert not _FUSED_LAYER.endswith(f".{_UNMATCHED_KEY}")
    assert _UNMATCHED_KEY not in _LLAMA_PACKED_MODULES
    _assert_unmatched_key_resolves("inc", _FUSED_LAYER, _LLAMA_PACKED_MODULES)


def test_exact_extra_config_layer_name_still_applies() -> None:
    """An exact layer-name key still overrides bits and group size."""
    raw = _checkpoint(
        "inc",
        {_LAYER: {"bits": 2, "group_size": 32, "sym": False}},
    )
    config = INCConfig.from_config(raw)
    layer = torch.nn.Linear(8, 8)
    bits, group_size, sym = config.get_layer_config(layer, _LAYER)
    assert (bits, group_size, sym) == (2, 32, False)
    config.get_quant_method(layer, _LAYER)
