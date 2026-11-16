# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject an unbounded ``index_topk`` before the indexer scratch allocation.

A checkpoint ``config.json`` can set ``index_topk`` on a DeepSeek-family
architecture. Model construction then does ``torch.empty(max_num_batched_tokens,
index_topk, dtype=torch.int32)`` inside ``initialize_model``, which
``get_model`` runs before ``load_weights``. These tests fail until that
value is rejected with ``ValueError`` before the allocation.

The in-tree indexer selectors accept 512, 1024, 2048, and (AITER) 4096.
Anything larger, or a non-positive value, must be refused. A check that
accepts an object equal to 2048 and then allocates ``operator.index`` of
that same object still fails.

Not executed here, same unbounded ``config.index_topk`` allocation:

* ``vllm/models/deepseek_v4/amd/model.py`` (and AMD MTP / DSpark copies)
  import ``vllm/platforms/rocm.py``, which calls
  ``torch.cuda.get_device_properties`` at import and aborts when this
  build has no CUDA.
* ``vllm/models/deepseek_v41/nvidia/model.py`` and
  ``vllm/models/deepseek_v41/amd/model.py``.
* ``vllm/models/dots3_note/nvidia/model.py`` (import requires the CUDA
  flash-attention extension).
* ``vllm/models/glm5next/common/model.py`` (import requires torchvision).
* ``vllm/models/deepseek_v4/nvidia/dspark.py`` and the V4.1 DSpark copies
  (same shape, not reached by the causal-LM constructors below).
"""

from __future__ import annotations

import json
import operator
import os
import resource
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import torch

from vllm.config import (
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.glm4_moe_lite_mtp import Glm4MoeLiteMTP
from vllm.models.deepseek_v32.amd.model import (
    DeepseekV32ForCausalLM as AmdDeepseekV32ForCausalLM,
)
from vllm.models.deepseek_v32.amd.mtp import DeepseekV32MTP as AmdDeepseekV32MTP
from vllm.models.deepseek_v32.nvidia.model import (
    DeepseekV32ForCausalLM as NvidiaDeepseekV32ForCausalLM,
)
from vllm.models.deepseek_v32.nvidia.mtp import (
    DeepseekV32MTP as NvidiaDeepseekV32MTP,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM as NvidiaDeepseekV4ForCausalLM,
)
from vllm.models.deepseek_v4.nvidia.mtp import DeepSeekV4MTP as NvidiaDeepSeekV4MTP
from vllm.models.deepseek_v4.xpu.model import (
    DeepseekV4ForCausalLM as XpuDeepseekV4ForCausalLM,
)
from vllm.models.deepseek_v4.xpu.mtp import DeepSeekV4MTP as XpuDeepSeekV4MTP
from vllm.models.hy_v4.nvidia.mtp import HYV4MTP

MAX_INDEX_TOPK = 4096
HUGE_INDEX_TOPK = 2**30
# 2048 batched tokens * 131072 * 4 bytes = 1 GiB, above a 256 MiB headroom.
LIMITED_INDEX_TOPK = 131072


class OversizedIndexerAllocation(RuntimeError):
    """The constructor asked torch.empty for an unbounded indexer buffer."""


class _ClaimedTopk:
    """Compares equal to 2048; ``int`` / ``__index__`` are unbounded."""

    def __eq__(self, other: object) -> bool:
        return other == 2048

    def __int__(self) -> int:
        return HUGE_INDEX_TOPK

    def __index__(self) -> int:
        return HUGE_INDEX_TOPK


def _dims(args: tuple) -> list:
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return list(args[0])
    return list(args)


def _as_dim(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            return operator.index(value)  # type: ignore[arg-type]
        except TypeError:
            return None
    return value


@pytest.fixture
def indexer_allocs(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    real_empty = torch.empty
    seen: list[tuple] = []

    def guarded(*args, **kwargs):
        resolved = [_as_dim(dim) for dim in _dims(args)]
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        seen.append((tuple(resolved), dtype, kwargs.get("device")))
        if dtype == torch.int32 and any(
            isinstance(dim, int) and dim > MAX_INDEX_TOPK for dim in resolved
        ):
            raise OversizedIndexerAllocation(
                f"indexer scratch buffer dims={resolved} device={kwargs.get('device')}"
            )
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", guarded)
    return seen


def _base_config(model_type: str, architecture: str, index_topk: object) -> dict:
    return {
        "architectures": [architecture],
        "model_type": model_type,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "max_position_embeddings": 32,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "first_k_dense_replace": 0,
        "kv_lora_rank": 16,
        "q_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "v_head_dim": 16,
        "head_dim": 16,
        "n_group": 1,
        "topk_group": 1,
        "index_head_dim": 16,
        "index_n_heads": 1,
        "num_nextn_predict_layers": 1,
        "index_topk": index_topk,
    }


def _write_checkpoint(directory: Path, config: dict) -> None:
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _vllm_config(directory: Path) -> VllmConfig:
    model_config = ModelConfig(
        model=str(directory),
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=32,
        dtype="float32",
        trust_remote_code=False,
    )
    return VllmConfig(model_config=model_config)


def _load(
    vllm_config: VllmConfig,
    model_cls: type | None,
    speculative: bool,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> None:
    if speculative:
        vllm_config.speculative_config = SpeculativeConfig(
            target_model_config=vllm_config.model_config,
            target_parallel_config=vllm_config.parallel_config,
            method="mtp",
            num_speculative_tokens=1,
        )
    if model_cls is None:
        model_config = vllm_config.model_config
        if speculative:
            assert vllm_config.speculative_config is not None
            model_config = vllm_config.speculative_config.draft_model_config
        get_model(vllm_config=vllm_config, model_config=model_config)
        return
    # NVIDIA V4 creates CUDA streams and then allocates the scratch buffer.
    # The stream stand-in only gets execution to that allocation; the buffer
    # itself still goes through torch.empty.
    if model_cls is NvidiaDeepseekV4ForCausalLM:
        assert monkeypatch is not None
        monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: object())
    with set_current_vllm_config(vllm_config, check_compile=True):
        model_cls(vllm_config=vllm_config, prefix="")


def _assert_rejected(exc: BaseException, directory: Path) -> None:
    assert isinstance(exc, ValueError), type(exc).__name__ + ": " + str(exc)
    assert not isinstance(exc, OversizedIndexerAllocation)
    assert "index_topk" in str(exc)
    assert sorted(path.name for path in directory.iterdir()) == ["config.json"]


# Registry architectures. On CPU, DeepseekV32 and GlmMoeDsa alias to
# DeepseekV2ForCausalLM, and DeepseekV4 uses the CPU model. Each still has
# to enforce the bound in the class the registry actually constructs.
_REGISTRY_CASES = [
    ("deepseek_v2", "DeepseekV2ForCausalLM"),
    ("deepseek_v3", "DeepseekV3ForCausalLM"),
    ("deepseek_v32", "DeepseekV32ForCausalLM"),
    ("deepseek_v4", "DeepseekV4ForCausalLM"),
    ("glm4_moe_lite", "Glm4MoeLiteForCausalLM"),
    ("hy_v4", "HYV4ForCausalLM"),
    ("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
]

# Constructors the CPU registry does not select, but CUDA/XPU workers do.
_DIRECT_CASES = [
    ("deepseek_v32", "DeepseekV32ForCausalLM", NvidiaDeepseekV32ForCausalLM),
    ("deepseek_v32", "DeepseekV32ForCausalLM", AmdDeepseekV32ForCausalLM),
    ("deepseek_v4", "DeepseekV4ForCausalLM", NvidiaDeepseekV4ForCausalLM),
    ("deepseek_v4", "DeepseekV4ForCausalLM", XpuDeepseekV4ForCausalLM),
    ("deepseek_v4", "DeepseekV4ForCausalLM", NvidiaDeepSeekV4MTP),
    ("deepseek_v4", "DeepseekV4ForCausalLM", XpuDeepSeekV4MTP),
    ("deepseek_v32", "DeepseekV32ForCausalLM", NvidiaDeepseekV32MTP),
    ("deepseek_v32", "DeepseekV32ForCausalLM", AmdDeepseekV32MTP),
    ("deepseek_v3", "DeepseekV3ForCausalLM", DeepSeekMTP),
    ("glm4_moe_lite", "Glm4MoeLiteForCausalLM", Glm4MoeLiteMTP),
    ("hy_v4", "HYV4ForCausalLM", HYV4MTP),
]


@pytest.mark.parametrize(("model_type", "architecture"), _REGISTRY_CASES)
def test_registry_load_rejects_unbounded_index_topk(
    tmp_path: Path,
    indexer_allocs: list[tuple],
    model_type: str,
    architecture: str,
) -> None:
    _write_checkpoint(
        tmp_path, _base_config(model_type, architecture, HUGE_INDEX_TOPK)
    )
    vllm_config = _vllm_config(tmp_path)
    with pytest.raises(Exception) as exc_info:
        _load(vllm_config, model_cls=None, speculative=False)
    _assert_rejected(exc_info.value, tmp_path)
    assert indexer_allocs == [] or all(
        dim is None or dim <= MAX_INDEX_TOPK
        for dims, _dtype, _device in indexer_allocs
        for dim in dims
    )


@pytest.mark.parametrize(
    ("model_type", "architecture", "model_cls"),
    _DIRECT_CASES,
    ids=[
        f"{cls.__module__}.{cls.__name__}"
        for _model_type, _arch, cls in _DIRECT_CASES
    ],
)
def test_platform_constructor_rejects_unbounded_index_topk(
    tmp_path: Path,
    indexer_allocs: list[tuple],
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    architecture: str,
    model_cls: type,
) -> None:
    del indexer_allocs
    _write_checkpoint(
        tmp_path, _base_config(model_type, architecture, HUGE_INDEX_TOPK)
    )
    vllm_config = _vllm_config(tmp_path)
    needs_spec = model_cls in (
        DeepSeekMTP,
        Glm4MoeLiteMTP,
        HYV4MTP,
        NvidiaDeepseekV32MTP,
        AmdDeepseekV32MTP,
    )
    with pytest.raises(Exception) as exc_info:
        _load(
            vllm_config,
            model_cls=model_cls,
            speculative=needs_spec,
            monkeypatch=monkeypatch,
        )
    _assert_rejected(exc_info.value, tmp_path)


def test_mtp_draft_load_rejects_unbounded_index_topk(
    tmp_path: Path,
    indexer_allocs: list[tuple],
) -> None:
    """Speculative MTP rewrites the draft architecture and loads that class."""
    del indexer_allocs
    _write_checkpoint(
        tmp_path,
        _base_config("deepseek_v3", "DeepseekV3ForCausalLM", HUGE_INDEX_TOPK),
    )
    vllm_config = _vllm_config(tmp_path)
    with pytest.raises(Exception) as exc_info:
        _load(vllm_config, model_cls=None, speculative=True)
    _assert_rejected(exc_info.value, tmp_path)
    draft = vllm_config.speculative_config.draft_model_config
    assert draft.architectures == ["DeepSeekMTPModel"]


@pytest.mark.parametrize("index_topk", [0, -1])
def test_non_positive_index_topk_rejected(
    tmp_path: Path,
    indexer_allocs: list[tuple],
    index_topk: int,
) -> None:
    del indexer_allocs
    _write_checkpoint(
        tmp_path,
        _base_config("deepseek_v3", "DeepseekV3ForCausalLM", index_topk),
    )
    with pytest.raises(Exception) as exc_info:
        _load(_vllm_config(tmp_path), model_cls=None, speculative=False)
    _assert_rejected(exc_info.value, tmp_path)


def test_equality_check_cannot_diverge_from_allocated_index(
    tmp_path: Path,
    indexer_allocs: list[tuple],
) -> None:
    """A value that compares equal to 2048 must not allocate a huge index."""
    _write_checkpoint(
        tmp_path,
        _base_config("deepseek_v3", "DeepseekV3ForCausalLM", 2048),
    )
    vllm_config = _vllm_config(tmp_path)
    object.__setattr__(vllm_config.model_config.hf_config, "index_topk", _ClaimedTopk())
    with pytest.raises(Exception) as exc_info:
        _load(vllm_config, model_cls=None, speculative=False)
    _assert_rejected(exc_info.value, tmp_path)
    for dims, dtype, _device in indexer_allocs:
        if dtype == torch.int32:
            assert HUGE_INDEX_TOPK not in dims


def test_supported_index_topk_is_not_rejected(
    tmp_path: Path,
    indexer_allocs: list[tuple],
) -> None:
    """2048 is a supported width and must reach the scratch allocation."""
    _write_checkpoint(
        tmp_path,
        _base_config("deepseek_v3", "DeepseekV3ForCausalLM", 2048),
    )
    with pytest.raises(Exception) as exc_info:
        _load(_vllm_config(tmp_path), model_cls=None, speculative=False)
    assert "index_topk" not in str(exc_info.value)
    assert any(
        dtype == torch.int32 and 2048 in dims
        for dims, dtype, _device in indexer_allocs
    )


def test_real_allocator_refuses_unbounded_index_topk(tmp_path: Path) -> None:
    """No stand-in: the CPU allocator must not be asked for the scratch buffer.

    The child sets RLIMIT_AS just above its own virtual size. A 1 GiB
    ``torch.empty`` fails that limit inside PyTorch's ``posix_memalign``.
    A fixed loader raises ``ValueError`` before that call.
    """
    _write_checkpoint(
        tmp_path,
        _base_config("deepseek_v3", "DeepseekV3ForCausalLM", LIMITED_INDEX_TOPK),
    )
    script = r"""
import os, resource, sys
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, os.environ["REPO_ROOT"])
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader import get_model

model = sys.argv[1]
model_config = ModelConfig(
    model=model,
    skip_tokenizer_init=True,
    enforce_eager=True,
    max_model_len=32,
    dtype="float32",
    trust_remote_code=False,
)
vllm_config = VllmConfig(model_config=model_config)
status = open("/proc/self/status", encoding="utf-8").read()
virtual_kb = int(status.split("VmSize:")[1].split()[0])
limit = (virtual_kb + 256 * 1024) * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
try:
    get_model(vllm_config=vllm_config)
except Exception as exc:
    print("RESULT", type(exc).__name__, str(exc).splitlines()[0])
    raise SystemExit(0)
print("RESULT success")
"""
    env = os.environ.copy()
    env["REPO_ROOT"] = str(Path(__file__).resolve().parents[2])
    env["VLLM_TARGET_DEVICE"] = "cpu"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    output = completed.stdout + "\n" + completed.stderr
    assert "RESULT" in output, output[-2000:]
    result_lines = [line for line in output.splitlines() if line.startswith("RESULT")]
    assert result_lines, output[-2000:]
    result = result_lines[-1]
    assert "can't allocate memory" not in result
    assert "OversizedIndexerAllocation" not in result
    assert "ValueError" in result
    assert "index_topk" in result
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.json"]
