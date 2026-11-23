# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DFlash mask-embedding files must be bounded before they are materialized.

These tests fail while ``mask_embedding.pt`` is read and deserialized without a
size, rank, or dtype limit. A later change should make them pass.
"""

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.model_executor.models.lilicorr import LiLiCorrForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
from vllm.platforms import current_platform
from vllm.transformers_utils import repo_utils

_HIDDEN = 32
_DTYPE = torch.float32
_MASK_TOKEN_ID = 7
# 8 MiB of float32. Deflate stores that record in about 9 KiB.
_BOMB_ELEMENTS = 2_000_000
# A small multiple of the 128-byte embedding. The bomb file is larger.
_MAX_FILE_BYTES = _HIDDEN * _DTYPE.itemsize * 32

_DRAFT_CLASSES = (
    DFlashQwen3ForCausalLM,
    DFlash2Qwen3ForCausalLM,
    LiLiCorrForCausalLM,
)


def _compressed_wide_tensor(num_elements: int) -> bytes:
    """Zip of ``num_elements`` float32 zeros, stored compressed."""
    saved = io.BytesIO()
    torch.save(torch.arange(4, dtype=torch.float32), saved)
    archive = zipfile.ZipFile(io.BytesIO(saved.getvalue()))
    pickle_bytes = archive.read("archive/data.pkl")
    encoded = b"J" + int(num_elements).to_bytes(4, "little")
    storage = b"cpuq\x04K\x04t"
    shape = b"QK\x00K\x04\x85"
    if pickle_bytes.count(storage) != 1 or pickle_bytes.count(shape) != 1:
        raise AssertionError("torch.save pickle layout changed")
    pickle_bytes = pickle_bytes.replace(storage, b"cpuq\x04" + encoded + b"t", 1)
    pickle_bytes = pickle_bytes.replace(shape, b"QK\x00" + encoded + b"\x85", 1)
    zeros = bytes(num_elements * 4)
    out = io.BytesIO()
    with zipfile.ZipFile(
        out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as rewritten:
        for info in archive.infolist():
            if info.filename == "archive/data.pkl":
                payload = pickle_bytes
            elif info.filename == "archive/data/0":
                payload = zeros
            else:
                payload = archive.read(info.filename)
            rewritten.writestr(info.filename, payload)
    del zeros
    return out.getvalue()


def _draft_config() -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=128,
        hidden_size=_HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        intermediate_size=64,
        hidden_act="silu",
        draft_vocab_size=128,
        dflash_config={
            "mask_token_id": _MASK_TOKEN_ID,
            "target_layer_ids": [0],
            "selector_rank": 4,
            "selector_top_k": 2,
            "conv_kernel_size": 1,
            "conv_group_size": 8,
            "block_size": 8,
            "lilicorr_enabled": True,
            "lilicorr_hidden_size": _HIDDEN,
            "lilicorr_candidate_topk": 4,
            "lilicorr_num_heads": 2,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": 4,
            "lilicorr_vector_eps": 1e-6,
            "lilicorr_logit_scale": 1.0,
            "lilicorr_num_layers": 1,
        },
    )


def _build_draft(model_dir: str, model_cls: type):
    hf_config = _draft_config()
    compilation_config = SimpleNamespace(
        compile_custom_ops=False,
        custom_ops="all",
        enabled_custom_ops=set(),
        static_forward_context={},
        mode=0,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
                model=model_dir,
                revision=None,
            ),
            num_speculative_tokens=3,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=_DTYPE,
            is_mm_prefix_lm=False,
            get_total_num_hidden_layers=lambda: 1,
            get_vocab_size=lambda: hf_config.vocab_size,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
        compilation_config=compilation_config,
    )
    current = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=vllm_config.speculative_config,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=compilation_config,
        model_config=SimpleNamespace(
            dtype=_DTYPE,
            is_mm_prefix_lm=False,
            rswa_window=None,
            head_dtype=None,
        ),
        kernel_config=SimpleNamespace(linear_backend="auto"),
    )
    with set_current_vllm_config(current):
        return model_cls(vllm_config=vllm_config)


def _write_mask(tmp_path: Path, payload: bytes | torch.Tensor | dict) -> str:
    path = tmp_path / "mask_embedding.pt"
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        torch.save(payload, path)
    return str(tmp_path)


@pytest.fixture
def parallel_state(monkeypatch):
    class _Group:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr("vllm.distributed.parallel_state._TP", _Group())
    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _load_while_watching(monkeypatch, model):
    """Run ``load_weights`` and record the mask file read and ``torch.load`` result.

    The real ``torch.load(..., weights_only=True)`` still runs. The records are
    how the test sees a tensor that was materialized and then discarded.
    """
    read_sizes: list[int] = []
    loaded: list[object] = []
    real_open = open
    real_load = torch.load

    def spy_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if not (
            isinstance(file, (str, Path))
            and str(file).endswith("mask_embedding.pt")
            and "b" in mode
        ):
            return handle
        original_read = handle.read

        def read(*args, **kwargs):
            data = original_read(*args, **kwargs)
            if isinstance(data, (bytes, bytearray)):
                read_sizes.append(len(data))
            return data

        handle.read = read
        return handle

    def spy_load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        loaded.append(result)
        return result

    monkeypatch.setattr(repo_utils, "open", spy_open, raising=False)
    monkeypatch.setattr(torch, "load", spy_load)
    caught: list[BaseException] = []
    try:
        model.load_weights([])
    except Exception as exc:
        caught.append(exc)
    return read_sizes, loaded, caught


def _assert_rejected(model, read_sizes, loaded, caught) -> None:
    for value in loaded:
        for tensor in _iter_tensors(value):
            assert tensor.dtype == _DTYPE
            assert tensor.dim() == 1
            assert tensor.shape == (_HIDDEN,)
    assert sum(read_sizes) <= _MAX_FILE_BYTES, (
        f"read {sum(read_sizes)} bytes of mask_embedding.pt before rejecting it"
    )
    assert caught, "oversized mask embedding was accepted"
    assert isinstance(caught[0], ValueError)
    assert model.model.has_separate_mask_embedding is False
    assert torch.count_nonzero(model.model.mask_embedding.detach()) == 0


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("model_cls", _DRAFT_CLASSES)
def test_compressed_mask_embedding_is_rejected_before_allocation(
    parallel_state, tmp_path, model_cls, monkeypatch
):
    """A few-kilobyte zip must not expand to the tensor it declares."""
    payload = _compressed_wide_tensor(_BOMB_ELEMENTS)
    assert _MAX_FILE_BYTES < len(payload) < 32 * 1024
    model = _build_draft(_write_mask(tmp_path, payload), model_cls)
    _assert_rejected(model, *_load_while_watching(monkeypatch, model))


@pytest.mark.skip_global_cleanup
def test_uncompressed_mask_embedding_is_not_read_before_rejection(
    parallel_state, tmp_path, monkeypatch
):
    """A file far larger than the embedding width is rejected before it is read."""
    path = tmp_path / "mask_embedding.pt"
    torch.save(torch.zeros(_BOMB_ELEMENTS, dtype=_DTYPE), path)
    assert path.stat().st_size > _MAX_FILE_BYTES
    model = _build_draft(str(tmp_path), DFlashQwen3ForCausalLM)
    _assert_rejected(model, *_load_while_watching(monkeypatch, model))


@pytest.mark.skip_global_cleanup
def test_mask_embedding_must_match_hidden_size_and_dtype(parallel_state, tmp_path):
    """Rank, length, and dtype are enforced, and a rejected file is not installed."""
    cases = (
        torch.arange(_HIDDEN * 2, dtype=_DTYPE),
        torch.arange(_HIDDEN, dtype=torch.float64),
        torch.arange(_HIDDEN, dtype=_DTYPE).reshape(1, _HIDDEN),
        {
            "mask_token_id": _MASK_TOKEN_ID,
            "embedding": torch.arange(_HIDDEN * 2, dtype=_DTYPE),
        },
    )
    for payload in cases:
        model_dir = _write_mask(tmp_path, payload)
        model = _build_draft(model_dir, DFlashQwen3ForCausalLM)
        with pytest.raises(ValueError):
            model.load_weights([])
        assert model.model.has_separate_mask_embedding is False
        assert torch.count_nonzero(model.model.mask_embedding.detach()) == 0


@pytest.mark.skip_global_cleanup
def test_matching_mask_embedding_is_installed(parallel_state, tmp_path):
    expected = torch.arange(_HIDDEN, dtype=_DTYPE)
    wrapped = {"mask_token_id": _MASK_TOKEN_ID, "embedding": expected.clone()}
    for payload in (expected, wrapped):
        model_dir = _write_mask(tmp_path, payload)
        model = _build_draft(model_dir, DFlashQwen3ForCausalLM)
        model.load_weights([])
        assert model.model.has_separate_mask_embedding is True
        torch.testing.assert_close(model.model.mask_embedding.detach(), expected)


@pytest.mark.skip_global_cleanup
def test_absent_mask_embedding_leaves_the_placeholder(parallel_state, tmp_path):
    model = _build_draft(str(tmp_path), DFlashQwen3ForCausalLM)
    model.load_weights([])
    assert model.model.has_separate_mask_embedding is False
    assert torch.count_nonzero(model.model.mask_embedding.detach()) == 0
