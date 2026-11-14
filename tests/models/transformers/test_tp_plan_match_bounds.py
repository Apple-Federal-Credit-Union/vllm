# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint tensor-parallel plan keys must not stall model construction.

`Base.recursive_replace` matches every linear-layer qualname against each
`base_model_tp_plan` key with the third-party `regex` engine (`import regex
as re`). Those keys are loaded from `config.json` with `trust_remote_code`
left off. A key the engine does not finish matching leaves the worker inside
that call, so startup never reaches attention creation.

This file is the regression spec. It fails while that match is unbounded and
passes once a plan key is either rejected or matched within the time limit.
`pipeline_parallel` does not compile plan keys as regular expressions, so it
is not covered here. The only match of plan keys is `recursive_replace`.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# Nested fuzzy pattern. On regex==2026.2.28 (the version pinned for CUDA
# tests) this does not return against an ordinary GPT-NeoX linear name.
# A trailing "!" is absent from every layer qualname.
_STALLING_TP_PLAN_KEY = (
    r"(?b)(\w+){e<=3}(\s+\w+){e<=3}(\s+\w+){e<=3}"
    r"(\s+\w+){e<=3}(\s+\w+){e<=3}(\s+\w+){e<=3}!"
)
# Documented whole-segment wildcard. A fix must still let this through.
_WILDCARD_TP_PLAN_KEY = "layers.*.attention.query_key_value"

_LOAD_TIMEOUT_S = 45
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _config(plan_key: str) -> dict:
    return {
        "architectures": ["GPTNeoXForCausalLM"],
        "model_type": "gpt_neox",
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 128,
        "max_position_embeddings": 32,
        "rotary_emb_base": 10000,
        "use_parallel_residual": True,
        "base_model_tp_plan": {plan_key: "colwise"},
    }


def _load_worker(plan_key: str, *, accept_rejection: bool) -> int:
    """Build the Transformers backend from a local checkpoint config.

    Exit 0 once construction has passed the plan match, or, when
    `accept_rejection` is set, once load fails inside that match. Exit 1
    when setup fails earlier.
    """
    import traceback

    from vllm.config import (
        DeviceConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.models.transformers import TransformersForCausalLM
    from vllm.platforms import current_platform

    class _PassedPlanMatch(Exception):
        pass

    class _Probe(TransformersForCausalLM):
        def _create_attention_instances(self, prefix: str = "") -> None:
            # `recursive_replace` has returned. Stop before attention setup,
            # which depends on a device backend this test does not need.
            raise _PassedPlanMatch

    directory = tempfile.mkdtemp()
    config_path = Path(directory, "config.json")
    config_path.write_text(json.dumps(_config(plan_key)), encoding="utf-8")
    fd, init_file = tempfile.mkstemp()
    os.close(fd)
    try:
        model_config = ModelConfig(
            model=directory,
            tokenizer=directory,
            skip_tokenizer_init=True,
            trust_remote_code=False,
            dtype="float32",
            seed=0,
            enforce_eager=True,
            max_model_len=16,
        )
        if current_platform.device_type:
            device_config = DeviceConfig()
        else:
            device_config = DeviceConfig(device="cpu")
        vllm_config = VllmConfig(
            model_config=model_config,
            device_config=device_config,
        )
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{init_file}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
            try:
                _Probe(vllm_config=vllm_config)
            except _PassedPlanMatch:
                return 0
            except Exception:
                formatted = traceback.format_exc()
                if accept_rejection and "recursive_replace" in formatted:
                    return 0
                sys.stderr.write(formatted)
                return 1
        return 1
    finally:
        config_path.unlink(missing_ok=True)
        os.rmdir(directory)
        with_suppress = Path(init_file)
        if with_suppress.exists():
            with_suppress.unlink()


def _run(plan_key: str, *, accept_rejection: bool) -> subprocess.CompletedProcess:
    worker = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_REPO_ROOT)!r})
        from tests.models.transformers.test_tp_plan_match_bounds import (
            _load_worker,
        )
        raise SystemExit(
            _load_worker({plan_key!r}, accept_rejection={accept_rejection})
        )
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", worker],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=_LOAD_TIMEOUT_S,
        check=False,
    )


def _output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def test_checkpoint_tp_plan_key_cannot_stall_construction():
    """A config.json plan key must not keep `recursive_replace` from returning.

    GPT-NeoX is loaded by the Transformers backend. The plan key is data in
    the checkpoint, not remote code. The worker must exit before the timeout.
    """
    try:
        completed = _run(_STALLING_TP_PLAN_KEY, accept_rejection=True)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "Transformers backend did not finish matching the checkpoint "
            f"tensor-parallel plan key within {_LOAD_TIMEOUT_S}s.\n"
            f"stdout:\n{_output(exc.stdout)}\nstderr:\n{_output(exc.stderr)}"
        )
    assert completed.returncode == 0, (
        "Load failed before the tensor-parallel plan match.\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_documented_tp_plan_wildcard_still_reaches_attention_setup():
    """A single-star layer wildcard from config.json still gets past the match."""
    try:
        completed = _run(_WILDCARD_TP_PLAN_KEY, accept_rejection=False)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "A documented tensor-parallel wildcard did not finish matching "
            f"within {_LOAD_TIMEOUT_S}s.\nstdout:\n{_output(exc.stdout)}\n"
            f"stderr:\n{_output(exc.stderr)}"
        )
    assert completed.returncode == 0, (
        "Documented tensor-parallel wildcard did not get past plan matching.\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
