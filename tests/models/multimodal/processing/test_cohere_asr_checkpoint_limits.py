# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint fields must be rejected before they size allocations.

Cohere ASR profiling waveform length is ``max_audio_clip_s * sample_rate``
and the first filterbank build sizes the FFT window and the mel matrix from
``window_size``, ``n_fft``, and ``features``. Those values are copied out of
the checkpoint. The bounds below are the ranges documented by
``transformers`` 5.20.2 ``CohereAsrFeatureExtractor`` (``max_audio_clip_s``
35s, ``sampling_rate`` 16000, ``feature_size`` 128, ``n_fft`` 512,
``win_length`` 400) and the defaults ``CohereASRProcessingInfo`` already uses
when a preprocessor key is absent (20 ms window, 64 features).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from vllm.config import ModelConfig
from vllm.model_executor.models.cohere_asr import (
    CohereAsrForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import InputProcessingContext
from vllm.multimodal.processing import dummy_inputs as dummy_inputs_mod
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processors import cohere_asr as fe_mod

# Documented envelope. A checkpoint outside it must be rejected.
MAX_AUDIO_CLIP_S = 35
SAMPLE_RATE = 16_000
MAX_FEATURES = 128
MAX_N_FFT = 512
MAX_WINDOW_SAMPLES = 400  # 0.025 s at 16 kHz
MAX_AUDIO_SAMPLES = MAX_AUDIO_CLIP_S * SAMPLE_RATE
MAX_MEL_FREQS = MAX_N_FFT // 2 + 1

_ENCODER = {
    "n_layers": 1,
    "d_model": 64,
    "feat_out": 64,
    "causal_downsampling": False,
    "subsampling": "striding",
    "subsampling_factor": 4,
    "subsampling_conv_channels": 8,
    "ff_expansion_factor": 2,
    "self_attention_model": "rel_pos",
    "n_heads": 4,
    "att_context_size": [-1, -1],
    "xscaling": True,
    "untie_biases": True,
    "pos_emb_max_len": 128,
    "conv_kernel_size": 3,
    "conv_norm_type": "batch_norm",
    "conv_context_size": None,
}


def _write_checkpoint(
    root: Path,
    *,
    max_audio_clip_s: int = 30,
    preproc: dict | None = None,
) -> None:
    preprocessor = {
        "sample_rate": SAMPLE_RATE,
        "window_size": 0.02,
        "window_stride": 0.01,
        "features": 64,
        "dither": 0.0,
    }
    if preproc:
        preprocessor.update(preproc)
    features = preprocessor["features"]
    config = {
        "model_type": "cohere_asr",
        "architectures": ["CohereAsrForConditionalGeneration"],
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "vocab_size": 32,
        "is_encoder_decoder": True,
        "max_position_embeddings": 128,
        "sample_rate": SAMPLE_RATE,
        "max_audio_clip_s": max_audio_clip_s,
        "overlap_chunk_second": 1,
        "preprocessor": preprocessor,
        "encoder": {"feat_in": features, **_ENCODER},
        "transf_decoder": {
            "config_dict": {
                "num_attention_heads": 4,
                "hidden_size": 64,
            }
        },
        "head": {"num_classes": 32, "hidden_size": 64},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "a": 1}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
    ).save_pretrained(root)


def _load(root: Path) -> ModelConfig:
    return ModelConfig(
        model=str(root),
        tokenizer=str(root),
        tokenizer_mode="hf",
        trust_remote_code=False,
        dtype="float32",
        max_model_len=64,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
    )


def _processor(model_config: ModelConfig):
    tokenizer = cached_tokenizer_from_config(model_config)
    model_cls = MULTIMODAL_REGISTRY._get_model_cls(model_config)
    ctx = InputProcessingContext(model_config, tokenizer=tokenizer)
    return model_cls._processor_factory.build_processor(ctx)


class _Allocations:
    def __init__(self) -> None:
        self.audio_lengths: list[int] = []
        self.mel_freqs: list[int] = []
        self.mel_bins: list[int] = []
        self.windows: list[int] = []

    def assert_within_limits(self) -> None:
        assert self.audio_lengths == [] or max(self.audio_lengths) <= (
            MAX_AUDIO_SAMPLES
        )
        assert self.mel_freqs == [] or max(self.mel_freqs) <= MAX_MEL_FREQS
        assert self.mel_bins == [] or max(self.mel_bins) <= MAX_FEATURES
        assert self.windows == [] or max(self.windows) <= MAX_WINDOW_SAMPLES


@pytest.fixture
def allocations(monkeypatch: pytest.MonkeyPatch) -> _Allocations:
    """Record sizes, then perform the real allocation."""
    seen = _Allocations()
    real_zeros = dummy_inputs_mod.np.zeros
    real_mel = fe_mod.melscale_fbanks
    real_hann = fe_mod.torch.hann_window

    def zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and shape:
            seen.audio_lengths.append(int(shape[0]))
        return real_zeros(shape, *args, **kwargs)

    def melscale_fbanks(n_freqs, *args, **kwargs):
        seen.mel_freqs.append(int(n_freqs))
        n_mels = kwargs.get("n_mels")
        if n_mels is None and args:
            n_mels = args[2] if len(args) > 2 else None
        if n_mels is not None:
            seen.mel_bins.append(int(n_mels))
        return real_mel(n_freqs, *args, **kwargs)

    def hann_window(window_length, *args, **kwargs):
        seen.windows.append(int(window_length))
        return real_hann(window_length, *args, **kwargs)

    monkeypatch.setattr(dummy_inputs_mod.np, "zeros", zeros)
    monkeypatch.setattr(fe_mod, "melscale_fbanks", melscale_fbanks)
    monkeypatch.setattr(fe_mod.torch, "hann_window", hann_window)
    return seen


def _profile(root: Path):
    """Run the load-time profiling path that sizes the waveform and filterbank.

    ``InputProcessor`` builds this processor and ``MultiModalBudget`` calls
    ``get_dummy_mm_inputs``, which allocates the dummy waveform and then
    builds ``FilterbankFeatures`` on the first forward.
    """
    model_config = _load(root)
    processor = _processor(model_config)
    mm_options = model_config.get_multimodal_config().limit_per_prompt
    processor.get_dummy_inputs(
        seq_len=model_config.max_model_len,
        mm_counts={"audio": 1},
        mm_options=mm_options,
    )
    filterbank = processor.info.get_feature_extractor().filterbank
    CohereAsrForConditionalGeneration.get_speech_to_text_config(
        model_config, "transcribe"
    )
    return processor, filterbank


def test_in_range_checkpoint_profiles_at_documented_sizes(
    tmp_path: Path, allocations: _Allocations
):
    """A checkpoint inside the documented envelope still profiles that audio."""
    root = tmp_path / "in_range"
    _write_checkpoint(
        root,
        max_audio_clip_s=MAX_AUDIO_CLIP_S,
        preproc={
            "sample_rate": SAMPLE_RATE,
            "window_size": 0.025,
            "window_stride": 0.01,
            "features": MAX_FEATURES,
            "n_fft": MAX_N_FFT,
            "dither": 0.0,
        },
    )
    processor, filterbank = _profile(root)
    mm_config = processor.info.ctx.model_config.get_multimodal_config()
    audio = processor.dummy_inputs.get_dummy_mm_data(
        seq_len=64,
        mm_counts={"audio": 1},
        mm_options=mm_config.limit_per_prompt,
    )["audio"][0]

    assert audio.shape == (MAX_AUDIO_SAMPLES,)
    assert audio.dtype == np.float64
    assert filterbank.n_fft == MAX_N_FFT
    assert tuple(filterbank.fb.shape) == (1, MAX_FEATURES, MAX_MEL_FREQS)
    assert tuple(filterbank.window.shape) == (MAX_WINDOW_SAMPLES,)
    allocations.assert_within_limits()


def test_omitted_n_fft_stays_within_the_documented_fft_size(
    tmp_path: Path, allocations: _Allocations
):
    """Missing ``n_fft`` derives the next power of two of the 20 ms window."""
    root = tmp_path / "default_fft"
    _write_checkpoint(root, max_audio_clip_s=30)
    _processor_obj, filterbank = _profile(root)
    assert filterbank.n_fft <= MAX_N_FFT
    assert filterbank.window.numel() <= MAX_WINDOW_SAMPLES
    allocations.assert_within_limits()


def test_profiling_entry_builds_features_from_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``get_dummy_mm_inputs`` is the call ``MultiModalBudget`` makes at load."""
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    root = tmp_path / "profile"
    _write_checkpoint(
        root,
        max_audio_clip_s=1,
        preproc={"features": 8, "n_fft": 512, "dither": 0.0},
    )
    model_config = _load(root)
    processor = _processor(model_config)
    profiled = processor.get_dummy_mm_inputs({"audio": 1})
    features = profiled["mm_kwargs"]["audio"][0].get_data()["input_features"]
    assert tuple(features.shape)[0] == 8
    assert 0 < tuple(features.shape)[1] < 400


@pytest.mark.parametrize(
    ("field", "max_audio_clip_s", "preproc"),
    [
        ("max_audio_clip_s", 36, None),
        ("sample_rate", 1, {"sample_rate": 32_000}),
        ("window_size", 1, {"window_size": 0.1, "window_stride": 0.01}),
        ("n_fft", 1, {"n_fft": 1024}),
        ("features", 1, {"features": 256}),
    ],
)
def test_checkpoint_outside_documented_range_is_rejected_before_allocation(
    tmp_path: Path,
    allocations: _Allocations,
    field: str,
    max_audio_clip_s: int,
    preproc: dict | None,
):
    root = tmp_path / field
    _write_checkpoint(
        root,
        max_audio_clip_s=max_audio_clip_s,
        preproc=preproc,
    )
    with pytest.raises(ValueError, match=field):
        _profile(root)
    # Refusal has to happen before the profiling waveform, the FFT window,
    # and the mel matrix exist. A check that runs after any of those
    # allocations is still the bug.
    assert allocations.audio_lengths == []
    assert allocations.mel_freqs == []
    assert allocations.windows == []
