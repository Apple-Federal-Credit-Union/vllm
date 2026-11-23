# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 ASR must not allocate profiling audio from an uncapped sample rate.

The checkpoint feature extractor supplies ``sampling_rate``. Profiling
multiplies ``min(chunk_length, 30)`` by that rate and allocates the product
with ``np.zeros``. Realtime streaming preallocates 60 seconds at the same
rate with ``np.empty``. The acoustic frontend is 16 kHz.

These tests fail while that product is uncapped. The same dummy builder is
registered for ``Qwen3ASRForConditionalGeneration`` and
``Qwen3ASRRealtimeGeneration``. Other audio models that multiply an uncapped
``sampling_rate`` into a dummy waveform (Whisper, Qwen2-Audio, Kimi-Audio,
Ultravox, FunASR, FireRedASR2, GLM-ASR, AudioFlamingo3, and the generic
Transformers multimodal backend) are not covered here.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

import vllm.model_executor.models.qwen3_asr_realtime as qwen3_asr_realtime
import vllm.multimodal.processing.dummy_inputs as dummy_inputs
from vllm.config import ModelConfig
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRProcessingInfo,
)
from vllm.model_executor.models.qwen3_asr_realtime import (
    Qwen3ASRRealtimeGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.context import InputProcessingContext
from vllm.tokenizers import cached_tokenizer_from_config

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Qwen3 ASR's mel frontend and processor kwargs are built for 16 kHz.
_FRONTEND_HZ = 16_000
_MAX_PROFILE_SECONDS = 30
_REALTIME_BUFFER_SECONDS = 60


def _max_profile_samples(chunk_length: int) -> int:
    return min(chunk_length, _MAX_PROFILE_SECONDS) * _FRONTEND_HZ


def _write_checkpoint(
    directory: Path,
    *,
    sampling_rate: int,
    chunk_length: int,
    architecture: str,
) -> None:
    """Write a local Qwen3 ASR repo whose preprocessor sets ``sampling_rate``."""
    config = {
        "model_type": "qwen3_asr",
        "architectures": [architecture],
        "thinker_config": {
            "text_config": {
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 32,
                "max_position_embeddings": 128,
            },
        },
    }
    preprocessor = {
        "feature_extractor_type": "Qwen3ASRFeatureExtractor",
        "processor_class": "Qwen3ASRProcessor",
        "sampling_rate": sampling_rate,
        "chunk_length": chunk_length,
        "feature_size": 128,
        "hop_length": 160,
        "n_fft": 400,
    }
    vocab = {
        "<|endoftext|>": 0,
        "<|audio_pad|>": 1,
        "<|audio_start|>": 2,
        "<|audio_end|>": 3,
        "a": 4,
    }
    tokenizer = {
        "tokenizer_class": "Qwen2Tokenizer",
        "processor_class": "Qwen3ASRProcessor",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "audio_token": "<|audio_pad|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "model_max_length": 128,
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "preprocessor_config.json").write_text(json.dumps(preprocessor))
    (directory / "vocab.json").write_text(json.dumps(vocab))
    (directory / "merges.txt").write_text("")
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer))


def _model_config(directory: Path) -> ModelConfig:
    return ModelConfig(
        model=str(directory),
        max_model_len=128,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
    )


def _dummy_builder(model_config: ModelConfig) -> Qwen3ASRDummyInputsBuilder:
    tokenizer = cached_tokenizer_from_config(model_config)
    info = Qwen3ASRProcessingInfo(InputProcessingContext(model_config, tokenizer))
    return Qwen3ASRDummyInputsBuilder(info)


def _profile_waveform(
    builder: Qwen3ASRDummyInputsBuilder,
) -> np.ndarray:
    mm_options = builder.info.ctx.get_mm_config().limit_per_prompt
    data = builder.get_dummy_mm_data(
        seq_len=builder.info.ctx.model_config.max_model_len,
        mm_counts={"audio": 1},
        mm_options=mm_options,
    )
    audios = data["audio"]
    assert isinstance(audios, list) and len(audios) == 1
    return audios[0]


def _guard_numpy_size(
    monkeypatch: pytest.MonkeyPatch,
    module,
    name: str,
    requested: list[int],
    *,
    ceiling: int,
):
    """Record allocator lengths and skip materializing a multi-gigabyte buffer."""
    real = getattr(module.np, name)

    def wrapped(shape, *args, **kwargs):
        length = shape[0] if isinstance(shape, tuple) else shape
        length = int(length)
        requested.append(length)
        if length > ceiling:
            shape = (1, *shape[1:]) if isinstance(shape, tuple) else 1
        return real(shape, *args, **kwargs)

    monkeypatch.setattr(module.np, name, wrapped)


def test_supported_sampling_rate_profiles_thirty_seconds(tmp_path: Path):
    """A 16 kHz checkpoint still profiles a 30 second waveform."""
    _write_checkpoint(
        tmp_path,
        sampling_rate=_FRONTEND_HZ,
        chunk_length=30,
        architecture="Qwen3ASRForConditionalGeneration",
    )
    waveform = _profile_waveform(_dummy_builder(_model_config(tmp_path)))
    assert waveform.shape == (_max_profile_samples(30),)
    assert waveform.nbytes <= _max_profile_samples(30) * np.dtype(np.float64).itemsize


def test_inflated_sampling_rate_dummy_waveform_is_capped(tmp_path: Path):
    """Profiling must not allocate min(chunk, 30) * an inflated sampling_rate."""
    sampling_rate = 64_000
    chunk_length = 30
    _write_checkpoint(
        tmp_path,
        sampling_rate=sampling_rate,
        chunk_length=chunk_length,
        architecture="Qwen3ASRForConditionalGeneration",
    )
    waveform = _profile_waveform(_dummy_builder(_model_config(tmp_path)))
    assert waveform.shape[0] <= _max_profile_samples(chunk_length)
    assert waveform.nbytes <= _max_profile_samples(chunk_length) * 8


def test_long_chunk_does_not_restore_the_uncapped_rate(tmp_path: Path):
    """chunk_length above 30 is already clamped; sampling_rate still must be."""
    sampling_rate = 64_000
    chunk_length = 1_000_000
    _write_checkpoint(
        tmp_path,
        sampling_rate=sampling_rate,
        chunk_length=chunk_length,
        architecture="Qwen3ASRForConditionalGeneration",
    )
    waveform = _profile_waveform(_dummy_builder(_model_config(tmp_path)))
    assert waveform.shape[0] <= _max_profile_samples(chunk_length)


def test_short_chunk_scales_the_frontend_rate_only(tmp_path: Path):
    """A shorter chunk stays at the frontend rate, not the checkpoint rate."""
    sampling_rate = 64_000
    chunk_length = 10
    _write_checkpoint(
        tmp_path,
        sampling_rate=sampling_rate,
        chunk_length=chunk_length,
        architecture="Qwen3ASRForConditionalGeneration",
    )
    waveform = _profile_waveform(_dummy_builder(_model_config(tmp_path)))
    assert waveform.shape[0] <= _max_profile_samples(chunk_length)


def test_enormous_sampling_rate_is_not_passed_to_the_allocator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The uncapped product must not be the length requested from np.zeros.

    The allocator is replaced only after the length is recorded, so a fix that
    checks the rate and then multiplies the original value still fails.
    """
    sampling_rate = 2**26
    chunk_length = 30
    _write_checkpoint(
        tmp_path,
        sampling_rate=sampling_rate,
        chunk_length=chunk_length,
        architecture="Qwen3ASRForConditionalGeneration",
    )
    builder = _dummy_builder(_model_config(tmp_path))
    requested: list[int] = []
    _guard_numpy_size(
        monkeypatch,
        dummy_inputs,
        "zeros",
        requested,
        ceiling=_max_profile_samples(chunk_length),
    )
    waveform = _profile_waveform(builder)
    assert requested, "profiling did not allocate a dummy waveform"
    assert max(requested) <= _max_profile_samples(chunk_length)
    assert waveform.shape[0] <= _max_profile_samples(chunk_length)


def test_profiling_entry_point_caps_the_dummy_waveform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Scheduler profiling calls get_dummy_mm_inputs, which allocates first."""
    sampling_rate = 64_000
    chunk_length = 30
    _write_checkpoint(
        tmp_path,
        sampling_rate=sampling_rate,
        chunk_length=chunk_length,
        architecture="Qwen3ASRForConditionalGeneration",
    )
    model_config = _model_config(tmp_path)
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    requested: list[int] = []
    _guard_numpy_size(
        monkeypatch,
        dummy_inputs,
        "zeros",
        requested,
        ceiling=_max_profile_samples(chunk_length),
    )
    profiling_error: Exception | None = None
    try:
        processor.get_dummy_mm_inputs({"audio": 1})
    except Exception as exc:
        profiling_error = exc
    assert requested, "profiling did not allocate a dummy waveform"
    assert max(requested) <= _max_profile_samples(chunk_length)
    if profiling_error is not None:
        raise profiling_error


def test_realtime_buffer_caps_preallocated_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Opening a realtime stream must not preallocate 60s at the checkpoint rate."""
    sampling_rate = 64_000
    _write_checkpoint(
        tmp_path,
        sampling_rate=sampling_rate,
        chunk_length=30,
        architecture="Qwen3ASRRealtimeGeneration",
    )
    model_config = _model_config(tmp_path)
    requested: list[int] = []
    _guard_numpy_size(
        monkeypatch,
        qwen3_asr_realtime,
        "empty",
        requested,
        ceiling=_REALTIME_BUFFER_SECONDS * _FRONTEND_HZ,
    )

    async def _no_audio():
        if False:
            yield np.zeros(1, dtype=np.float32)

    async def _open_buffer():
        queue: asyncio.Queue[list[int]] = asyncio.Queue()
        stream = Qwen3ASRRealtimeGeneration.buffer_realtime_audio(
            _no_audio(),
            queue,
            model_config,
        )
        try:
            await stream.__anext__()
        except StopAsyncIteration:
            return

    asyncio.run(_open_buffer())

    assert requested, "realtime startup did not preallocate a buffer"
    assert max(requested) <= _REALTIME_BUFFER_SECONDS * _FRONTEND_HZ
