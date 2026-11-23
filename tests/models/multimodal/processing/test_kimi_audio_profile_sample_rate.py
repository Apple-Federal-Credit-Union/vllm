# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio startup profiling must cap the dummy waveform.

``whisper-large-v3/preprocessor_config.json`` supplies ``sampling_rate``.
``MultiModalBudget`` profiling multiplies ``min(chunk_length, 30)`` by that
rate and passes the product to the dummy-audio allocator. The waveform has
to stay within 30 seconds at 16 kHz, the rate ``KimiAudioProcessor`` passes
to the feature extractor. These tests fail while a checkpoint can raise
that product.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Resolve the CPU platform before the first ``current_platform`` use so
# ``VllmConfig`` can build the same profiling object the scheduler builds.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm.config import ModelConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.processing import dummy_inputs as dummy_inputs_module
from vllm.transformers_utils.processor import cached_get_feature_extractor

# 30 s at 16 kHz. ``KimiAudioProcessor`` always calls the extractor with
# this rate, and profiling already clamps chunk length to 30 seconds.
_MAX_PROFILE_SAMPLES = 30 * 16_000

pytestmark = pytest.mark.skip_global_cleanup


class _OversizedProfileAllocation(Exception):
    def __init__(self, n_samples: int) -> None:
        super().__init__(n_samples)
        self.n_samples = n_samples


def _n_samples(shape: object) -> int:
    if isinstance(shape, tuple):
        return int(shape[0]) if shape else 0
    return int(shape)  # type: ignore[arg-type]


def _write_kimi_checkpoint(
    directory: Path,
    *,
    chunk_length: int,
    sampling_rate: int,
) -> None:
    """Write a MoonshotKimia checkpoint with the given extractor fields."""
    config = {
        "architectures": ["MoonshotKimiaForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 64,
        "vocab_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }
    (directory / "config.json").write_text(json.dumps(config))
    whisper = directory / "whisper-large-v3"
    whisper.mkdir()
    preprocessor = {
        "chunk_length": chunk_length,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 80,
        "hop_length": 160,
        "n_fft": 400,
        "padding_value": 0.0,
        "return_attention_mask": False,
        "sampling_rate": sampling_rate,
    }
    (whisper / "preprocessor_config.json").write_text(json.dumps(preprocessor))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(directory)


def _profile_shapes(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    guard_oversized: bool,
) -> list[int]:
    """Run scheduler profiling and return every zeros length it requests.

    The feature extractor is whatever ``from_pretrained`` loaded from the
    checkpoint. A later re-read of that file still sees the same rate, so
    capping a value and then looking the extractor up again still fails
    when the allocator receives the uncapped product.
    """
    cached_get_feature_extractor.cache_clear()
    requested: list[int] = []
    real_zeros = dummy_inputs_module.np.zeros

    def zeros(shape, *args, **kwargs):
        n_samples = _n_samples(shape)
        requested.append(n_samples)
        if guard_oversized and n_samples > _MAX_PROFILE_SAMPLES:
            raise _OversizedProfileAllocation(n_samples)
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(dummy_inputs_module.np, "zeros", zeros)
    model_config = ModelConfig(
        model=str(directory),
        tokenizer=str(directory),
        # ``auto`` selects the tiktoken Kimi tokenizer. Audio length does
        # not depend on it; ``hf`` lets profiling construct the registered
        # Kimi processor from this local directory.
        tokenizer_mode="hf",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        enforce_eager=True,
        max_model_len=128,
    )
    vllm_config = VllmConfig(model_config=model_config)
    try:
        MultiModalBudget(vllm_config, MULTIMODAL_REGISTRY)
    except _OversizedProfileAllocation:
        raise
    except Exception:
        # A rate other than 16 kHz makes ``KimiAudioProcessor`` raise after
        # the waveform exists. The samples already requested are the bug.
        pass
    return requested


def test_legitimate_kimi_checkpoint_profiles_30s_at_16khz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 16 kHz extractor with a long chunk still profiles exactly 30 seconds."""
    _write_kimi_checkpoint(tmp_path, chunk_length=90, sampling_rate=16_000)
    requested = _profile_shapes(tmp_path, monkeypatch, guard_oversized=False)
    assert _MAX_PROFILE_SAMPLES in requested
    assert max(requested) <= _MAX_PROFILE_SAMPLES


def test_checkpoint_sample_rate_is_capped_before_dummy_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiling must allocate the capped waveform, not 30 times the file's rate.

    ``chunk_length`` is already clamped to 30. ``sampling_rate`` is not.
    32 kHz is small enough to allocate for real and larger than 16 kHz, so
    an uncapped product is observable without exhausting the machine.
    """
    _write_kimi_checkpoint(tmp_path, chunk_length=90, sampling_rate=32_000)
    requested = _profile_shapes(tmp_path, monkeypatch, guard_oversized=False)
    assert requested, "startup profiling did not allocate a dummy waveform"
    assert max(requested) <= _MAX_PROFILE_SAMPLES, (
        f"profiling allocated {max(requested)} samples from the checkpoint rate"
    )


def test_enormous_checkpoint_sample_rate_is_rejected_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A huge rate must not reach the zeros allocator.

    The guard allocates for real only at or under the cap. An uncapped
    product raises ``_OversizedProfileAllocation`` instead of building a
    multi-gigabyte buffer, and that request fails the test.
    """
    _write_kimi_checkpoint(tmp_path, chunk_length=30, sampling_rate=50_000_000)
    try:
        requested = _profile_shapes(tmp_path, monkeypatch, guard_oversized=True)
    except _OversizedProfileAllocation as exc:
        pytest.fail(
            "profiling passed an uncapped sample count to the dummy-audio "
            f"allocator: {exc.n_samples}"
        )
    assert requested, "startup profiling did not allocate a dummy waveform"
    assert max(requested) <= _MAX_PROFILE_SAMPLES
