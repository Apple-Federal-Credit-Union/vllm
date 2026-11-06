# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter rank_pattern and alpha_pattern keys must not stall loading.

PEFTHelper interpolates those keys into a regular expression and matches
them, with no deadline, against every adapter tensor name. These tests
drive the checkpoint load that WorkerLoRAManager._load_adapter uses.
"""

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.lora import LoRAConfig
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

_READY = "PEFT_PATTERN_WORKER_READY"
_DONE = "PEFT_PATTERN_WORKER_DONE"
_START_DEADLINE_S = 180.0
# Long enough that a linear match finishes, short enough that the
# exponential cases below cannot.
_MATCH_DEADLINE_S = 1.0
_BACKTRACK_CHARS = 30
_DEFAULT_RANK = 8
_DEFAULT_ALPHA = 16

_EVIL_CASES = [
    ("rank_pattern", "alternation"),
    ("alpha_pattern", "alternation"),
    ("rank_pattern", "repeat"),
    ("alpha_pattern", "repeat"),
]


def _pattern_key(kind: str) -> str:
    if kind == "literal":
        return "q_proj"
    if kind == "alternation":
        # Quantified alternation. regex 2026.2.28 still backtracks on this;
        # it finishes the nested form `(a+)+b` immediately.
        return "(a|a)+b"
    if kind == "repeat":
        return "(a+){1,20}b"
    raise ValueError(kind)


def _module_name(kind: str) -> str:
    if kind == "literal":
        return "model.layers.0.self_attn.q_proj"
    return ("a" * _BACKTRACK_CHARS) + ".q_proj"


def _pattern_value(field: str) -> int:
    if field == "rank_pattern":
        return 4
    if field == "alpha_pattern":
        return 8
    raise ValueError(field)


def _load_adapter(field: str, kind: str) -> tuple[str, int, float]:
    """Load one adapter the way the worker does, and classify the result."""
    module_name = _module_name(kind)
    tensors = {
        f"base_model.model.{module_name}.lora_A.weight": torch.zeros(8, 4),
        f"base_model.model.{module_name}.lora_B.weight": torch.zeros(4, 8),
    }
    config = {
        "r": _DEFAULT_RANK,
        "lora_alpha": _DEFAULT_ALPHA,
        "target_modules": ["q_proj"],
        "bias": "none",
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    config[field] = {_pattern_key(kind): _pattern_value(field)}

    with tempfile.TemporaryDirectory() as lora_dir:
        save_file(tensors, os.path.join(lora_dir, "adapter_model.safetensors"))
        config_path = os.path.join(lora_dir, "adapter_config.json")
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file)
        peft_helper = PEFTHelper.from_local_dir(
            lora_dir, max_position_embeddings=4096
        )
        peft_helper.validate_legal(
            LoRAConfig(max_lora_rank=16, max_cpu_loras=2, max_loras=2)
        )
        model = LoRAModel.from_local_checkpoint(
            lora_dir,
            {"q_proj"},
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            dtype=torch.float32,
        )

    loaded = list(model.loras.values())
    if len(loaded) != 1:
        raise AssertionError(f"expected one LoRA module, got {list(model.loras)}")
    rank = loaded[0].rank
    scaling = float(loaded[0].scaling)
    if kind == "literal":
        if field == "rank_pattern":
            applied = rank == 4 and scaling == 4.0
        else:
            applied = rank == _DEFAULT_RANK and scaling == 1.0
        status = "applied" if applied else "mismatch"
    else:
        untouched = rank == _DEFAULT_RANK and scaling == 2.0
        status = "default" if untouched else "matched"
    return status, rank, scaling


def _worker_main(field: str, kind: str) -> None:
    print(_READY, flush=True)
    try:
        status, rank, scaling = _load_adapter(field, kind)
    except Exception as exc:
        print(
            f"{_DONE} rejected {type(exc).__name__}: {exc}",
            flush=True,
        )
        return
    print(f"{_DONE} {status} rank={rank} scaling={scaling}", flush=True)


def _done_status(line: str) -> str:
    prefix = f"{_DONE} "
    if not line.startswith(prefix):
        raise AssertionError(line)
    return line[len(prefix) :].split(" ", 1)[0]


def _await_line(
    messages: queue.Queue[str | None],
    token: str,
    timeout: float,
    seen: list[str],
) -> str | None:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            line = messages.get(timeout=remaining)
        except queue.Empty:
            return None
        if line is None:
            return None
        stripped = line.rstrip("\n")
        seen.append(stripped)
        if token in stripped:
            return stripped


def _run_worker(field: str, kind: str) -> str:
    env = os.environ.copy()
    # Loading always targets CPU. Selecting the CPU platform keeps pinned
    # host memory off when this process has no accelerator.
    env["VLLM_TARGET_DEVICE"] = "cpu"
    proc = subprocess.Popen(
        [sys.executable, "-u", __file__, "--worker", field, kind],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    messages: queue.Queue[str | None] = queue.Queue()

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=_pump, daemon=True).start()
    seen: list[str] = []
    try:
        ready = _await_line(messages, _READY, _START_DEADLINE_S, seen)
        if ready is None:
            pytest.fail("pattern worker did not start:\n" + "\n".join(seen[-40:]))
        done = _await_line(messages, _DONE, _MATCH_DEADLINE_S, seen)
        if done is None:
            pytest.fail(
                "adapter load did not finish within "
                f"{_MATCH_DEADLINE_S}s:\n" + "\n".join(seen[-40:])
            )
        return done
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.parametrize(("field", "kind"), _EVIL_CASES)
def test_pattern_key_does_not_stall_adapter_load(field: str, kind: str) -> None:
    """Nested rank_pattern and alpha_pattern keys must not hang the loader.

    The load has to finish within a second and leave the module on the
    adapter-wide rank and alpha. Rejecting the config is also acceptable.
    """
    done = _run_worker(field, kind)
    status = _done_status(done)
    assert status in {"default", "rejected"}, done


@pytest.mark.parametrize(
    ("field", "rank", "scaling"),
    [
        ("rank_pattern", 4, 4.0),
        ("alpha_pattern", 8, 1.0),
    ],
)
def test_literal_suffix_pattern_still_applies(
    field: str, rank: int, scaling: float
) -> None:
    """A literal module suffix in either pattern map still selects that value."""
    done = _run_worker(field, "literal")
    assert _done_status(done) == "applied", done
    assert f"rank={rank} " in done
    assert done.endswith(f"scaling={scaling}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: test_peft_pattern_regex.py --worker FIELD KIND")
