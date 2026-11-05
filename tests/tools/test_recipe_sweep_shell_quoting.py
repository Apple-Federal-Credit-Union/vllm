# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep runners must not execute recipe model names.

Recipe ``model`` and ``served-model-name`` values are copied into generated
concurrency, scheduler, and parallel-layout sweep scripts. The runners are
executed here. A command substitution or backtick in those names must not run.

``run_scheduler_sweep.sh`` and ``run_full_sweep.sh`` do not embed the benchmark
command; they delegate to the runners covered below.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "tools" / "recipes" / "recipe_json_to_vllm_config.py"
RECIPES = ROOT / "tools" / "recipes"
SAFE_MODEL = "org/safe-model"
SAFE_SERVED = "safe-served-name"


def _payload(kind: str, marker: Path) -> str:
    command = f"touch {marker}"
    if kind == "dollar":
        return f"$({command})"
    if kind == "backtick":
        return f"`{command}`"
    if kind == "dquote":
        return f'x"$({command})'
    raise AssertionError(kind)


def _recipe(
    model: str,
    served: str | list[str] | None = None,
) -> dict[str, Any]:
    argv: list[str] = ["vllm", "serve", model]
    if isinstance(served, str):
        argv.extend(["--served-model-name", served])
    elif isinstance(served, list):
        argv.append("--served-model-name")
        argv.extend(served)
    return {"hardware": "xeon6", "deploy_type": "single_node", "argv": argv}


def _offline_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.pop("VLLM_USE_MODELSCOPE", None)
    return env


def _converter_env() -> dict[str, str]:
    env = _offline_env()
    parts = [str(ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run_converter(
    tmp_path: Path,
    recipe: dict[str, Any],
    extra_args: list[str],
) -> subprocess.CompletedProcess[str]:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    command = [
        sys.executable,
        str(CONVERTER),
        str(recipe_path),
        "--input-tokens",
        "16",
        "--output-tokens",
        "8",
        "--concurrency",
        "2",
        "--config-out",
        str(tmp_path / "config.yml"),
        "--env-out",
        str(tmp_path / "env.sh"),
        "--sweep-out-dir",
        str(tmp_path / "sweep"),
        *extra_args,
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        env=_converter_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _install_vllm_stub(bin_dir: Path, log_path: Path) -> None:
    bin_dir.mkdir()
    stub = bin_dir / "vllm"
    stub.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {shlex.quote(str(log_path))}\nexit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)


def _benchmark_runners(sweep_dir: Path) -> list[Path]:
    if not sweep_dir.is_dir():
        return []
    runners = []
    for path in sorted(sweep_dir.glob("run_*.sh")):
        if "--bench-cmd" in path.read_text(encoding="utf-8"):
            runners.append(path)
    return runners


def _assert_runners_do_not_execute(
    tmp_path: Path,
    sweep_dir: Path,
    marker: Path,
    payload: str,
) -> None:
    assert not marker.exists()
    runners = _benchmark_runners(sweep_dir)
    if not runners:
        return

    blob = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sweep_dir.rglob("*")
        if path.is_file()
    )
    assert payload in blob

    log_path = tmp_path / "vllm-invocations.log"
    bin_dir = tmp_path / "bin"
    _install_vllm_stub(bin_dir, log_path)
    env = _offline_env()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["VLLM_RECIPE_SWEEP_RETRIES"] = "0"
    env.pop("PYTHONPATH", None)
    executed: list[str] = []
    for runner in runners:
        marker.unlink(missing_ok=True)
        completed = subprocess.run(
            ["bash", str(runner), "--dry-run"],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if marker.exists():
            executed.append(f"{runner.name} (exit {completed.returncode})")
            marker.unlink()
    assert not executed


def _names_for(
    kind: str,
    field: str,
    marker: Path,
) -> tuple[str, str | list[str] | None]:
    payload = _payload(kind, marker)
    if field == "model":
        return payload, None
    if field == "tokenizer":
        return payload, SAFE_SERVED
    if field == "served":
        return SAFE_MODEL, payload
    if field == "served-list":
        return SAFE_MODEL, [payload, "other-served-name"]
    raise AssertionError(field)


@pytest.mark.parametrize(
    ("extra_args", "field", "kind"),
    [
        (["--generate-sweep"], "model", "dollar"),
        (["--generate-scheduler-sweep"], "served", "dollar"),
        (["--generate-concurrency-sweep"], "tokenizer", "backtick"),
        (
            ["--generate-concurrency-sweep", "--ttft-sla-ms", "50"],
            "served-list",
            "dollar",
        ),
        (["--sweep-stage", "concurrency"], "model", "dquote"),
        (["--sweep-stage", "scheduler"], "served", "backtick"),
    ],
    ids=[
        "generate-sweep-model-dollar",
        "scheduler-sweep-served-dollar",
        "concurrency-tokenizer-backtick",
        "concurrency-adaptive-served-list",
        "sweep-stage-concurrency-dquote",
        "sweep-stage-scheduler-served-backtick",
    ],
)
def test_converter_sweep_runners_do_not_execute_model_names(
    tmp_path: Path,
    extra_args: list[str],
    field: str,
    kind: str,
) -> None:
    marker = tmp_path / "marker"
    model, served = _names_for(kind, field, marker)
    payload = _payload(kind, marker)
    completed = _run_converter(tmp_path, _recipe(model, served), extra_args)
    assert not marker.exists(), completed.stderr
    assert completed.returncode == 0, completed.stderr
    assert _benchmark_runners(tmp_path / "sweep")
    _assert_runners_do_not_execute(
        tmp_path,
        tmp_path / "sweep",
        marker,
        payload,
    )


def _import_recipe_tools() -> tuple[Any, Any, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.pop("VLLM_USE_MODELSCOPE", None)
    for entry in (str(ROOT), str(RECIPES)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import recipe_json_to_vllm_config as converter
    from sweep.runtime_tuning import (
        WorkloadHints,
        finetune_runtime_config,
        get_runtime_tuning_policies,
    )
    from sweep.sweep_generation import (
        write_full_sweep_files,
        write_parallel_layout_sweep_files,
    )

    return (
        converter,
        (
            WorkloadHints,
            finetune_runtime_config,
            get_runtime_tuning_policies,
        ),
        (
            write_full_sweep_files,
            write_parallel_layout_sweep_files,
        ),
    )


def _config_from_recipe(recipe: dict[str, Any], workload: Any) -> dict[str, Any]:
    converter, tuning_api, _writers = _import_recipe_tools()
    _workload_hints, finetune_runtime_config, get_runtime_tuning_policies = tuning_api
    from hardware_detection import detect_hardware

    config = converter.argv_to_config(converter.recipe_argv(recipe))
    policies = get_runtime_tuning_policies(recipe["hardware"], tune_scheduler=False)
    tuning = finetune_runtime_config(
        config,
        hardware=detect_hardware(),
        workload=workload,
        policies=policies,
    )
    config.update(tuning.overrides)
    return config


@pytest.mark.parametrize("writer_name", ["parallel", "full"])
def test_parallel_and_full_sweep_runners_do_not_execute_model_names(
    tmp_path: Path,
    writer_name: str,
) -> None:
    """Generate the runners that need two NUMA nodes.

    This host's hardware detector reports a single NUMA node, and
    ``build_parallel_layout_plan`` refuses that count before writing scripts.
    The converter passes the detected count through, so these cases call the
    same writers with a two-node count.
    """
    marker = tmp_path / "marker"
    payload = _payload("dollar", marker)
    recipe = _recipe(payload)
    converter, tuning_api, writers = _import_recipe_tools()
    workload_cls, _finetune, _policies = tuning_api
    write_full, write_parallel = writers
    workload = workload_cls(input_tokens=16, output_tokens=8, concurrency=2)
    config = _config_from_recipe(recipe, workload)
    config_path = tmp_path / "config.yml"
    env_path = tmp_path / "env.sh"
    sweep_dir = tmp_path / "sweep"
    source = str(tmp_path / "recipe.json")
    converter.write_config(str(config_path), source, recipe, config)
    converter.write_env(str(env_path), source, recipe)
    kwargs = {
        "output_dir": str(sweep_dir),
        "config_path": str(config_path),
        "env_path": str(env_path),
        "config": config,
        "workload": workload,
        "numa_node_count": 2,
    }
    if writer_name == "parallel":
        write_parallel(**kwargs)
    else:
        write_full(**kwargs)
    _assert_runners_do_not_execute(tmp_path, sweep_dir, marker, payload)
