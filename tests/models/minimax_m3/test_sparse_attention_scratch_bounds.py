# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax M3 must reject unbounded sparse-attention sizes before allocating.

A checkpoint ``config.json`` supplies ``sparse_num_index_heads`` and
``sparse_topk_blocks``. Model startup multiplies those by the scheduler's
max batched token count and passes the product to ``torch.empty`` for a
shared top-k index buffer. The buffer is scratch space: no weight tensor
has to match it.

These tests fail while that allocation still uses the raw JSON values.
They pass once those fields are rejected first.

Required bounds (stock MiniMax M3 uses 4 heads and 16 top-k blocks):

* ``sparse_num_index_heads``: integer in ``[1, 64]``
* ``sparse_topk_blocks``: integer in ``[1, 256]``

Bools are not integers. Rejection must be a ``ValueError`` that names the
field, and it must happen before ``torch.empty`` is asked for the buffer.

Covered constructors:

* NVIDIA ``MiniMaxM3Model`` through ``initialize_model``
* AMD ``MiniMaxM3Model``
* AMD ``MiniMaxM3MTP`` (draft config loaded by ``SpeculativeConfig``)

Not covered, because they do not perform this unbounded allocation:

* NVIDIA ``MiniMaxM3MTP`` never builds the shared buffer
  (``vllm/models/minimax_m3/nvidia/mtp.py``).
* AMD ``sparse_bt_buffer`` multiplies ``sparse_block_size`` only after
  ``select_msa_indexer_impl_cls`` has required that size to be 128, so a
  checkpoint cannot choose an unbounded block size for that call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.model_loader.utils import initialize_model

pytestmark = pytest.mark.skip_global_cleanup

MAX_SPARSE_NUM_INDEX_HEADS = 64
MAX_SPARSE_TOPK_BLOCKS = 256
_HUGE = 1 << 20

# field, raw JSON value
_REJECTED: tuple[tuple[str, object], ...] = (
    ("sparse_topk_blocks", _HUGE),
    ("sparse_num_index_heads", _HUGE),
    ("sparse_topk_blocks", MAX_SPARSE_TOPK_BLOCKS + 1),
    ("sparse_num_index_heads", MAX_SPARSE_NUM_INDEX_HEADS + 1),
    ("sparse_topk_blocks", "999999"),
    ("sparse_num_index_heads", "999999"),
    ("sparse_topk_blocks", 16.5),
    ("sparse_num_index_heads", 4.5),
    ("sparse_topk_blocks", True),
    ("sparse_num_index_heads", True),
)


class ScratchBufferRequested(Exception):
    """``torch.empty`` was asked for the shared top-k index buffer."""

    def __init__(self, shape: tuple[object, ...]):
        super().__init__(repr(shape))
        self.shape = shape


def _token_axis(max_num_batched_tokens: int) -> set[int]:
    padded = ((max_num_batched_tokens + 3) // 4) * 4
    return {max_num_batched_tokens, padded}


def _shape_of(args: tuple[object, ...]) -> tuple[object, ...] | None:
    if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
        return tuple(args[0])
    if args and all(isinstance(dim, (int, float, str, bool)) for dim in args):
        return args
    return None


def _is_shared_topk_buffer(
    shape: tuple[object, ...],
    dtype: object,
    token_axis: set[int],
    forbidden: set[object],
) -> bool:
    if dtype is not torch.int32 or len(shape) != 3:
        return False
    if not any(dim in token_axis for dim in shape):
        return False
    return any(type(dim) is not int or dim in forbidden for dim in shape)


@pytest.fixture(scope="module")
def parallel_ready(tmp_path_factory: pytest.TempPathFactory):
    store = tmp_path_factory.mktemp("dist") / "store"
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{store}",
        local_rank=0,
        backend="gloo",
    )
    # initialize_model_parallel reads the current config. A placeholder is
    # enough; each case installs its own config before constructing a model.
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(1, 1)
    yield


def _write_checkpoint(directory: Path, field: str, value: object) -> None:
    sparse = {
        "use_sparse_attention": True,
        "sparse_index_dim": 16,
        "sparse_num_index_heads": 4,
        "sparse_topk_blocks": 16,
        "sparse_block_size": 128,
        "sparse_disable_index_value": [1],
        "sparse_score_type": "max",
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_attention_freq": [1],
    }
    sparse[field] = value
    config = {
        "model_type": "minimax_m3_vl",
        "architectures": ["MiniMaxM3SparseForCausalLM"],
        "text_config": {
            "model_type": "minimax_m3_text",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 64,
            "dense_intermediate_size": 64,
            "shared_intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "max_position_embeddings": 64,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "moe_layer_freq": [0],
            "num_mtp_modules": 1,
            "sparse_attention_config": sparse,
        },
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _load_model_config(directory: Path) -> ModelConfig:
    return ModelConfig(
        model=str(directory),
        skip_tokenizer_init=True,
        trust_remote_code=False,
        max_model_len=32,
        enforce_eager=True,
        dtype="float32",
        generation_config="vllm",
    )


def _construct_registered_model(vllm_config: VllmConfig, _model_dir: Path) -> None:
    initialize_model(vllm_config)


def _construct_amd_model(vllm_config: VllmConfig, _model_dir: Path) -> None:
    from vllm.models.minimax_m3.amd.model import MiniMaxM3Model

    MiniMaxM3Model(vllm_config=vllm_config)


def _construct_amd_mtp(vllm_config: VllmConfig, model_dir: Path) -> None:
    vllm_config.speculative_config = SpeculativeConfig(
        method="minimax_m3_mtp",
        model=str(model_dir),
        num_speculative_tokens=1,
        target_model_config=vllm_config.model_config,
        target_parallel_config=vllm_config.parallel_config,
    )
    from vllm.models.minimax_m3.amd.mtp import MiniMaxM3MTP

    MiniMaxM3MTP(vllm_config=vllm_config)


_CONSTRUCTORS: tuple[tuple[str, Callable[[VllmConfig, Path], None]], ...] = (
    ("nvidia-initialize-model", _construct_registered_model),
    ("amd-model", _construct_amd_model),
    ("amd-mtp", _construct_amd_mtp),
)


def _assert_rejected(
    model_config: ModelConfig,
    model_dir: Path,
    field: str,
    value: object,
    construct: Callable[[VllmConfig, Path], None],
) -> str | None:
    """Return a failure message, or None when the field was rejected in time."""
    from vllm.config import set_current_vllm_config

    vllm_config = VllmConfig(model_config=model_config)
    token_axis = _token_axis(vllm_config.scheduler_config.max_num_batched_tokens)
    forbidden = {value}
    real_empty = torch.empty

    def guarded(*args, **kwargs):
        shape = _shape_of(args)
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        if shape is not None and _is_shared_topk_buffer(
            shape, dtype, token_axis, forbidden
        ):
            raise ScratchBufferRequested(shape)
        return real_empty(*args, **kwargs)

    torch.empty = guarded
    try:
        with set_current_vllm_config(vllm_config):
            construct(vllm_config, model_dir)
    except ScratchBufferRequested as exc:
        return (
            f"{field}={value!r} reached torch.empty{exc.shape} before rejection"
        )
    except ValueError as exc:
        if field not in str(exc):
            return f"{field}={value!r} raised ValueError without the field name: {exc}"
        return None
    except Exception as exc:
        return f"{field}={value!r} failed open: {type(exc).__name__}: {exc}"
    else:
        return f"{field}={value!r} was accepted"
    finally:
        torch.empty = real_empty


@pytest.mark.parametrize(
    ("field", "value"),
    _REJECTED,
    ids=[f"{field}-{value!r}" for field, value in _REJECTED],
)
def test_sparse_sizes_rejected_before_topk_buffer(
    field: str,
    value: object,
    parallel_ready,
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path, field, value)
    try:
        model_config = _load_model_config(tmp_path)
    except ValueError as exc:
        assert field in str(exc), exc
        return

    failures: list[str] = []
    for name, construct in _CONSTRUCTORS:
        message = _assert_rejected(
            model_config, tmp_path, field, value, construct
        )
        if message is not None:
            failures.append(f"{name}: {message}")
    if failures:
        pytest.fail("\n".join(failures))


def test_stock_sparse_sizes_still_load(tmp_path: Path) -> None:
    """The published defaults are inside the required bounds."""
    _write_checkpoint(tmp_path, "sparse_topk_blocks", 16)
    model_config = _load_model_config(tmp_path)
    sparse = model_config.hf_text_config.sparse_attention_config
    assert sparse["sparse_num_index_heads"] == 4
    assert sparse["sparse_topk_blocks"] == 16
    assert sparse["sparse_block_size"] == 128
