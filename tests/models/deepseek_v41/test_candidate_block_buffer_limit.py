# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 candidate-block scratch buffer is sized from the checkpoint.

``deepseek-ai/DeepSeek-V4.1-Flash`` publishes ``candidate_topk_blocks`` of
2048. ``DeepseekV4Model`` on both the NVIDIA and AMD paths passes that field
straight to ``torch.empty(max_num_batched_tokens, candidate_topk_blocks)``
during construction. These tests are the cap: a checkpoint value above 2048
must be rejected before any tensor of that width is allocated, and 2048 must
still allocate exactly that width.

The same uncapped field also sizes
``DeepseekV41SparseIndexerMetadataBuilder.sparse_indices_buffer`` when
``attention_config.indexer_sparse_logits`` is enabled. That builder is not
covered here: its hardware checks run before the allocation and this host
cannot reach that ``torch.zeros``.
"""

from __future__ import annotations

import json
import sys

import pytest
import torch

from vllm.config.kernel import KernelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.transformers_utils.config import get_config

# Published text_config.candidate_topk_blocks on DeepSeek-V4.1-Flash.
DOCUMENTED_MAX_CANDIDATE_TOPK_BLOCKS = 2048
# 2048 rows (scheduler default) * 131072 * 4 bytes == 1 GiB.
OVERSIZED_CANDIDATE_TOPK_BLOCKS = 131072


def _write_config(directory, candidate_topk_blocks: int, *, placement: str) -> None:
    text = {
        "model_type": "deepseek_v41_text",
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "rms_norm_eps": 1e-6,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "index_topk": 4,
        "candidate_source_layer_id": 0,
    }
    payload = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "vision_config": {"num_hidden_layers": 0},
    }
    if placement == "text_config":
        text["candidate_topk_blocks"] = candidate_topk_blocks
    elif placement == "top_level":
        payload["candidate_topk_blocks"] = candidate_topk_blocks
        payload["candidate_source_layer_id"] = 0
    else:
        raise AssertionError(placement)
    payload["text_config"] = text
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def _vllm_config(hf_config):
    class _ModelConfig:
        def __init__(self):
            self.hf_config = hf_config

    class _VllmConfig:
        def __init__(self):
            self.model_config = _ModelConfig()
            self.quant_config = None
            self.parallel_config = ParallelConfig()
            self.kernel_config = KernelConfig()
            self.scheduler_config = SchedulerConfig(
                max_model_len=128,
                is_encoder_decoder=False,
            )
            self.speculative_config = None

    return _VllmConfig()


class _GcnArch:
    gcnArchName = "gfx942"
    name = "stub"


def _import_amd_model() -> None:
    """Import the AMD class, whose module import reads the GCN arch.

    ``vllm/platforms/rocm.py`` calls ``torch.cuda.get_device_properties`` while
    the module is loading. That call does not read the checkpoint. On a host
    with no driver it raises, so the properties object is stubbed only then.
    """
    if "vllm.platforms.rocm" in sys.modules:
        return
    try:
        torch.cuda.get_device_properties(0)
    except Exception:
        torch.cuda.get_device_properties = lambda *_args, **_kwargs: _GcnArch()
    import vllm.platforms.rocm  # noqa: F401


def _model_cls(backend: str):
    if backend == "nvidia":
        from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model

        return DeepseekV4Model
    if backend == "amd":
        _import_amd_model()
        from vllm.models.deepseek_v41.amd.model import DeepseekV4Model

        return DeepseekV4Model
    raise AssertionError(backend)


@pytest.fixture
def cpu_streams(monkeypatch):
    """``torch.cuda.Stream`` only allocates a device stream.

    ``Stream.__new__`` (torch/cuda/streams.py) checks that CUDA is built and
    forwards the device and priority. It does not read ``candidate_topk_blocks``.
    This host has no driver, so the real constructor raises before the scratch
    buffer. The replacement exists only so ``DeepseekV4Model.__init__`` can
    reach ``torch.empty``.
    """

    class _Stream:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(torch.cuda, "Stream", _Stream)


@pytest.fixture
def recorded_empty(monkeypatch):
    calls: list[tuple[tuple, dict, torch.Tensor]] = []
    real_empty = torch.empty

    def _empty(*args, **kwargs):
        tensor = real_empty(*args, **kwargs)
        calls.append((args, kwargs, tensor))
        return tensor

    monkeypatch.setattr(torch, "empty", _empty)
    return calls


def _candidate_widths(calls) -> list[int]:
    widths = []
    for args, kwargs, _tensor in calls:
        if kwargs.get("dtype") != torch.int32 or len(args) < 2:
            continue
        widths.append(args[1])
    return widths


def _load(tmp_path, candidate_topk_blocks: int, placement: str):
    _write_config(
        tmp_path,
        candidate_topk_blocks,
        placement=placement,
    )
    hf_config = get_config(str(tmp_path), trust_remote_code=False)
    assert hf_config.candidate_topk_blocks == candidate_topk_blocks
    assert hf_config.candidate_source_layer_id == 0
    return _vllm_config(hf_config)


def _construct(model_cls, vllm_config):
    # Model loading builds the class inside ``with target_device``
    # (base_loader.load_model). CPU stands in for that context here.
    with torch.device("cpu"):
        model_cls(vllm_config=vllm_config)


@pytest.mark.parametrize("backend", ["nvidia", "amd"])
@pytest.mark.parametrize("placement", ["text_config", "top_level"])
def test_oversized_candidate_topk_blocks_rejected_before_allocation(
    backend,
    placement,
    tmp_path,
    cpu_streams,
    recorded_empty,
):
    """A checkpoint integer above the published width must not be allocated."""
    del cpu_streams
    vllm_config = _load(tmp_path, OVERSIZED_CANDIDATE_TOPK_BLOCKS, placement)
    model_cls = _model_cls(backend)
    rows = vllm_config.scheduler_config.max_num_batched_tokens

    try:
        _construct(model_cls, vllm_config)
    except ValueError as exc:
        if "candidate_topk_blocks" not in str(exc):
            pytest.fail(f"rejection did not name candidate_topk_blocks: {exc}")
    except Exception as exc:
        widths = _candidate_widths(recorded_empty)
        allocated = [
            tensor
            for args, _kwargs, tensor in recorded_empty
            if args[:2] == (rows, OVERSIZED_CANDIDATE_TOPK_BLOCKS)
        ]
        pytest.fail(
            "candidate_topk_blocks="
            f"{OVERSIZED_CANDIDATE_TOPK_BLOCKS} was allocated before "
            f"construction stopped ({type(exc).__name__}: {exc}). "
            f"int32 widths={widths} "
            f"nbytes={[t.nbytes for t in allocated]}"
        )
    else:
        pytest.fail("model construction accepted the oversized checkpoint field")

    widths = _candidate_widths(recorded_empty)
    assert OVERSIZED_CANDIDATE_TOPK_BLOCKS not in widths
    assert all(width <= DOCUMENTED_MAX_CANDIDATE_TOPK_BLOCKS for width in widths)


@pytest.mark.parametrize("backend", ["nvidia", "amd"])
def test_documented_candidate_topk_blocks_allocates_that_width(
    backend,
    tmp_path,
    cpu_streams,
    recorded_empty,
):
    """2048, the published DeepSeek-V4.1-Flash width, still allocates."""
    del cpu_streams
    vllm_config = _load(
        tmp_path,
        DOCUMENTED_MAX_CANDIDATE_TOPK_BLOCKS,
        "text_config",
    )
    rows = vllm_config.scheduler_config.max_num_batched_tokens
    model_cls = _model_cls(backend)

    try:
        _construct(model_cls, vllm_config)
    except ValueError as exc:
        if "candidate_topk_blocks" in str(exc):
            raise
    except Exception:
        # Construction continues into pipeline-parallel init after the buffer.
        pass

    matched = [
        tensor
        for args, kwargs, tensor in recorded_empty
        if args[:2] == (rows, DOCUMENTED_MAX_CANDIDATE_TOPK_BLOCKS)
        and kwargs.get("dtype") is torch.int32
    ]
    assert matched, recorded_empty
    tensor = matched[0]
    assert tensor.shape == (rows, DOCUMENTED_MAX_CANDIDATE_TOPK_BLOCKS)
    assert tensor.dtype == torch.int32
    assert tensor.device.type == "cpu"
    assert tensor.nbytes == rows * DOCUMENTED_MAX_CANDIDATE_TOPK_BLOCKS * 4
