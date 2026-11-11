# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint TorchAO module patterns must not stall quantizer selection.

``TorchAOConfig`` keeps ``module_fqn_to_config`` keys from the checkpoint and
``get_quant_method`` applies every ``re:`` key with ``regex.fullmatch`` and no
timeout. A nested pattern makes that call fail to return while a linear
layer's quantizer is chosen.
"""

import json
import os
import resource
import select
import subprocess
import sys
import tempfile

import pytest

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

pytest.importorskip("torchao")

from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.torchao import (
    TorchAOConfig,
    TorchAOLinearMethod,
)

_LAYER = "model.layers.0.self_attn.q_proj"
_OTHER_LAYER = "model.layers.0.self_attn.k_proj"
_NESTED_DEPTH = 22
_SELECTION_BUDGET_S = 1.5
_IMPORT_BUDGET_S = 90.0
_GROWTH_LIMIT_BYTES = 512 * 1024 * 1024

_INT8_WEIGHT_ONLY = {
    "_type": "Int8WeightOnlyConfig",
    "_version": 2,
    "_data": {
        "group_size": None,
        "granularity": {
            "_type": "PerRow",
            "_version": 1,
            "_data": {"dim": -1},
        },
        "set_inductor_config": True,
    },
}


def _nested_pattern() -> str:
    """Nested quantifiers. ``regex`` 2026.2.28 does not finish this quickly."""
    return "(" * _NESTED_DEPTH + "a" + ")+" * _NESTED_DEPTH + "!"


def _evil_module_map() -> dict:
    # A fast non-match precedes the nested key so selection still reaches it.
    return {
        "re:^$": None,
        "re:" + _nested_pattern(): _INT8_WEIGHT_ONLY,
    }


def _benign_module_map() -> dict:
    return {
        "re:^$": None,
        r"re:model\.layers\..*\.q_proj": _INT8_WEIGHT_ONLY,
    }


def _quant_type(module_map: dict) -> dict:
    return {
        "_type": "ModuleFqnToConfig",
        "_version": 1,
        "_data": {"module_fqn_to_config": module_map},
    }


def _checkpoint_config(module_map: dict) -> dict:
    return {
        "quant_method": "torchao",
        "quant_type": {"default": _quant_type(module_map)},
    }


def _linear(prefix: str) -> LinearBase:
    return LinearBase(
        8,
        8,
        bias=False,
        quant_config=None,
        prefix=prefix,
        disable_tp=True,
    )


def _load(entry: str, module_map: dict) -> TorchAOConfig:
    quant_type = _quant_type(module_map)
    if entry == "from_config":
        return TorchAOConfig.from_config(_checkpoint_config(module_map))
    if entry == "from_config_dict_json":
        return TorchAOConfig.from_config_dict_json(json.dumps(quant_type))
    if entry == "from_config_file":
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as handle:
            json.dump(quant_type, handle)
            path = handle.name
        try:
            return TorchAOConfig.from_config_file(path)
        finally:
            os.unlink(path)
    raise AssertionError(entry)


def _vm_kb(field: str) -> int:
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith(field):
                return int(line.split()[1])
    raise RuntimeError(field)


def _cap_address_space() -> None:
    """Keep a nested compile from taking the machine down with it."""
    limit = (_vm_kb("VmSize:") * 1024) + (1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _select(entry: str, module_map: dict, prefix: str):
    before = _vm_kb("VmHWM:")
    config = _load(entry, module_map)
    method = config.get_quant_method(_linear(prefix), prefix)
    growth = (_vm_kb("VmHWM:") - before) * 1024
    return method, growth


def _worker(entry: str) -> int:
    print("READY", flush=True)
    sys.stdin.readline()
    _cap_address_space()
    try:
        _method, growth = _select(entry, _evil_module_map(), _LAYER)
    except MemoryError:
        print("RESULT memory", flush=True)
        return 1
    except Exception as exc:
        print(f"RESULT rejected {type(exc).__name__}", flush=True)
        return 0
    if growth > _GROWTH_LIMIT_BYTES:
        print(f"RESULT grew {growth}", flush=True)
        return 1
    print(f"RESULT done {type(_method).__name__}", flush=True)
    return 0


def _selection_finishes(entry: str) -> None:
    env = os.environ.copy()
    env["VLLM_TARGET_DEVICE"] = "cpu"
    env["PYTHONUNBUFFERED"] = "1"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(item for item in (repo_root, previous) if item)
    proc = subprocess.Popen(
        [sys.executable, __file__, entry],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    ready = ""
    result = ""
    try:
        readable, _, _ = select.select([proc.stdout], [], [], _IMPORT_BUDGET_S)
        if not readable:
            pytest.fail(f"{entry} did not start")
        ready = proc.stdout.readline().strip()
        proc.stdin.write("\n")
        proc.stdin.flush()
        readable, _, _ = select.select([proc.stdout], [], [], _SELECTION_BUDGET_S)
        if not readable:
            pytest.fail(
                f"{entry} did not finish selecting a quantizer within "
                f"{_SELECTION_BUDGET_S}s"
            )
        result = proc.stdout.readline().strip()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{entry} did not exit after reporting a quantizer")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert ready == "READY", ready
    if (
        proc.returncode != 0
        or result.startswith("RESULT memory")
        or result.startswith("RESULT grew")
    ):
        pytest.fail(
            f"{entry} did not finish matching the nested checkpoint "
            f"pattern ({result or 'no result'}, exit {proc.returncode})"
        )


@pytest.mark.parametrize(
    "entry",
    ["from_config", "from_config_dict_json", "from_config_file"],
)
def test_nested_checkpoint_pattern_does_not_stall_quantizer_selection(entry):
    """A nested ``re:`` module key must not block ``get_quant_method``.

    The checkpoint is loaded through each public TorchAO config entry, then
    a real linear layer is asked for its quantizer. The nested key is not the
    first map entry, so a check that only inspects one pattern still has to
    deal with the pattern that is matched.
    """
    _selection_finishes(entry)


def test_bounded_checkpoint_pattern_still_selects_matching_linear():
    """A short ``re:`` key still selects the checkpoint quantizer."""
    config = TorchAOConfig.from_config(_checkpoint_config(_benign_module_map()))
    matched = config.get_quant_method(_linear(_LAYER), _LAYER)
    unmatched = config.get_quant_method(_linear(_OTHER_LAYER), _OTHER_LAYER)
    assert isinstance(matched, TorchAOLinearMethod)
    assert isinstance(unmatched, UnquantizedLinearMethod)


if __name__ == "__main__":
    sys.exit(_worker(sys.argv[1]))
