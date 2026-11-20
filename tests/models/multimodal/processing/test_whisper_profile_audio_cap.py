# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling must not allocate a dummy waveform from uncapped
checkpoint ``chunk_length`` and ``sampling_rate`` values.

These tests fail while that product is passed to the zeros allocator.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import transformers.feature_extraction_sequence_utils as sequence_utils
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, WhisperConfig, WhisperFeatureExtractor

from vllm.config import ModelConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.model_executor.models.fireredasr2 import FireRedASR2DummyInputsBuilder
from vllm.model_executor.models.funasr import FunASRDummyInputsBuilder
from vllm.model_executor.models.kimi_audio import KimiAudioDummyInputsBuilder
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
)
from vllm.model_executor.models.qwen2_audio import Qwen2AudioDummyInputsBuilder
from vllm.model_executor.models.qwen3_asr import Qwen3ASRDummyInputsBuilder
from vllm.model_executor.models.transformers.multimodal import (
    MultiModalDummyInputsBuilder,
)
from vllm.model_executor.models.ultravox import (
    _MAX_ENCODER_BATCH_SIZE,
    UltravoxDummyInputsBuilder,
)
from vllm.model_executor.models.whisper import WhisperDummyInputsBuilder
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import dummy_inputs as dummy_inputs_module
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.transformers_utils.processors.fireredasr2 import FireRedASR2FeatureExtractor
from vllm.transformers_utils.processors.funasr import FunASRFeatureExtractor

# 30 s at Whisper's 16 kHz encoder window. Neither checkpoint field may
# raise the profiled buffer above this, including models that already
# clamp only ``chunk_length``.
_MAX_PROFILE_SAMPLES = 30 * 16_000
_MAX_PROFILE_FRAMES = _MAX_PROFILE_SAMPLES // 160

pytestmark = pytest.mark.skip_global_cleanup


class _OversizedProfileAllocation(Exception):
    def __init__(self, n_samples: int) -> None:
        super().__init__(n_samples)
        self.n_samples = n_samples


def _write_checkpoint(
    directory: Path,
    *,
    chunk_length: int,
    sampling_rate: int,
) -> None:
    WhisperConfig(
        num_hidden_layers=1,
        encoder_layers=1,
        decoder_layers=1,
        vocab_size=8,
        d_model=32,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=1500,
        max_target_positions=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    ).save_pretrained(directory)
    preprocessor = {
        "chunk_length": chunk_length,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 80,
        "hop_length": 160,
        "n_fft": 400,
        "padding_value": 0.0,
        "sampling_rate": sampling_rate,
        "processor_class": "WhisperProcessor",
    }
    (directory / "preprocessor_config.json").write_text(json.dumps(preprocessor))
    vocab = {
        "<pad>": 0,
        "<|startoftranscript|>": 1,
        "<|endoftext|>": 2,
        "a": 3,
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<pad>")),
        unk_token="<pad>",
        pad_token="<pad>",
        bos_token="<|startoftranscript|>",
        eos_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(directory)


def _whisper_processor(directory: Path):
    model_config = ModelConfig(
        model=str(directory),
        tokenizer=str(directory),
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        enforce_eager=True,
        max_model_len=32,
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _profile_features(directory: Path) -> torch.Tensor:
    processor = _whisper_processor(directory)
    # Same entry encoder profiling uses via
    # MultiModalBudget.get_dummy_encoder_profile_inputs.
    profiled = processor.get_dummy_mm_inputs({"audio": 1})
    return profiled["mm_kwargs"].get_data()["input_features"]


def _guard_profile_allocators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse the zeros and pad allocators once the waveform exceeds the cap.

    The moderate cases allocate for real. This guard only keeps an
    unbounded checkpoint from killing the runner, and it fails the test
    when that unbounded size is requested.
    """
    real_zeros = dummy_inputs_module.np.zeros
    real_pad = sequence_utils.np.pad

    def _n_samples(shape: object) -> int:
        if isinstance(shape, tuple):
            return int(shape[0]) if shape else 0
        return int(shape)  # type: ignore[arg-type]

    def zeros(shape, *args, **kwargs):
        n_samples = _n_samples(shape)
        if n_samples > _MAX_PROFILE_SAMPLES:
            raise _OversizedProfileAllocation(n_samples)
        return real_zeros(shape, *args, **kwargs)

    def pad(array, pad_width, *args, **kwargs):
        width_is_pairs = (
            isinstance(pad_width, tuple)
            and pad_width
            and isinstance(pad_width[0], tuple)
        )
        if width_is_pairs:
            extra = sum(pad_width[0])
        elif isinstance(pad_width, tuple):
            extra = sum(pad_width)
        else:
            extra = 0
        n_samples = len(array) + int(extra)
        if n_samples > _MAX_PROFILE_SAMPLES:
            raise _OversizedProfileAllocation(n_samples)
        return real_pad(array, pad_width, *args, **kwargs)

    monkeypatch.setattr(dummy_inputs_module.np, "zeros", zeros)
    monkeypatch.setattr(sequence_utils.np, "pad", pad)


def _assert_capped(n_samples: int, *, multiplier: int = 1) -> None:
    assert n_samples <= _MAX_PROFILE_SAMPLES * multiplier, (
        f"profiled dummy audio is {n_samples} samples"
    )


class _ExtractorInfo:
    def __init__(self, feature_extractor: object) -> None:
        self._feature_extractor = feature_extractor

    def get_feature_extractor(self) -> object:
        return self._feature_extractor


def _builder_audio_len(
    builder_cls: type[BaseDummyInputsBuilder],
    feature_extractor: object,
) -> int:
    data = builder_cls(_ExtractorInfo(feature_extractor)).get_dummy_mm_data(
        32,
        {"audio": 1},
        MultiModalDummyOptions(),
    )
    return int(data["audio"][0].shape[0])


def _generic_mm_audio_len(feature_extractor: object) -> int:
    class _Info:
        _is_audio_model = True
        _is_image_model = False
        _is_video_model = False

        def _get_audio_sampling_rate(self) -> int:
            return feature_extractor.sampling_rate

        def _get_audio_processor(self) -> object:
            return feature_extractor

    data = MultiModalDummyInputsBuilder(_Info()).get_dummy_mm_data(
        32,
        {"audio": 1},
        MultiModalDummyOptions(),
    )
    return int(data["audio"][0].shape[0])


def _omni_audio_len(feature_extractor: object) -> int:
    class _Processor:
        pass

    processor = _Processor()
    processor.feature_extractor = feature_extractor

    class _Info:
        def get_hf_processor(self) -> object:
            return processor

        def get_image_size_with_most_features(self) -> tuple[int, int]:
            return (8, 8)

        def get_num_frames_with_most_features(
            self,
            seq_len: int,
            mm_counts: dict[str, int],
        ) -> int:
            return 1

    data = Qwen2_5OmniThinkerDummyInputsBuilder(_Info()).get_dummy_mm_data(
        32,
        {"audio": 1},
        MultiModalDummyOptions(),
    )
    return int(data["audio"][0].shape[0])


def _load_extractor(extractor_cls: type, directory: Path) -> object:
    payload = json.loads((directory / "preprocessor_config.json").read_text())
    payload.pop("processor_class", None)
    (directory / "preprocessor_config.json").write_text(json.dumps(payload))
    return extractor_cls.from_pretrained(directory)


@pytest.mark.parametrize(
    ("chunk_length", "sampling_rate"),
    [(60, 16_000), (30, 32_000)],
)
def test_whisper_profile_dummy_audio_is_capped(
    tmp_path: Path,
    chunk_length: int,
    sampling_rate: int,
) -> None:
    """Encoder profiling of a local Whisper checkpoint must cap the waveform.

    ``trust_remote_code`` stays off. The integers come only from
    ``preprocessor_config.json``.
    """
    _write_checkpoint(
        tmp_path,
        chunk_length=chunk_length,
        sampling_rate=sampling_rate,
    )
    processor = _whisper_processor(tmp_path)
    audio = processor.dummy_inputs.get_dummy_mm_data(
        32,
        {"audio": 1},
        processor.info.ctx.get_mm_config().limit_per_prompt,
    )["audio"][0]
    _assert_capped(int(audio.shape[0]))

    features = _profile_features(tmp_path)
    assert features.shape[-1] <= _MAX_PROFILE_FRAMES


@pytest.mark.parametrize(
    ("chunk_length", "sampling_rate"),
    [(10**9, 16_000), (30, 10**8)],
)
def test_whisper_profile_refuses_unbounded_checkpoint_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunk_length: int,
    sampling_rate: int,
) -> None:
    _write_checkpoint(
        tmp_path,
        chunk_length=chunk_length,
        sampling_rate=sampling_rate,
    )
    _guard_profile_allocators(monkeypatch)
    try:
        features = _profile_features(tmp_path)
    except _OversizedProfileAllocation as exc:
        pytest.fail(f"profiling requested {exc.n_samples} samples")
    assert features.shape[-1] <= _MAX_PROFILE_FRAMES


@pytest.mark.parametrize(
    ("chunk_length", "sampling_rate"),
    [(60, 16_000), (30, 32_000)],
)
@pytest.mark.parametrize(
    ("builder_cls", "extractor_cls", "multiplier"),
    [
        (WhisperDummyInputsBuilder, WhisperFeatureExtractor, 1),
        (Qwen2AudioDummyInputsBuilder, WhisperFeatureExtractor, 1),
        (FunASRDummyInputsBuilder, FunASRFeatureExtractor, 1),
        (FireRedASR2DummyInputsBuilder, FireRedASR2FeatureExtractor, 1),
        (UltravoxDummyInputsBuilder, WhisperFeatureExtractor, _MAX_ENCODER_BATCH_SIZE),
    ],
)
def test_chunk_length_dummy_builders_cap_profile_audio(
    tmp_path: Path,
    builder_cls: type[BaseDummyInputsBuilder],
    extractor_cls: type,
    multiplier: int,
    chunk_length: int,
    sampling_rate: int,
) -> None:
    _write_checkpoint(
        tmp_path,
        chunk_length=chunk_length,
        sampling_rate=sampling_rate,
    )
    feature_extractor = _load_extractor(extractor_cls, tmp_path)
    _assert_capped(
        _builder_audio_len(builder_cls, feature_extractor),
        multiplier=multiplier,
    )


@pytest.mark.parametrize(
    ("chunk_length", "sampling_rate"),
    [(60, 16_000), (30, 32_000)],
)
def test_transformers_backend_dummy_audio_is_capped(
    tmp_path: Path,
    chunk_length: int,
    sampling_rate: int,
) -> None:
    _write_checkpoint(
        tmp_path,
        chunk_length=chunk_length,
        sampling_rate=sampling_rate,
    )
    feature_extractor = _load_extractor(WhisperFeatureExtractor, tmp_path)
    _assert_capped(_generic_mm_audio_len(feature_extractor))


@pytest.mark.parametrize("sampling_rate", [32_000, 10**6])
@pytest.mark.parametrize(
    "measure",
    [
        pytest.param(
            lambda fe: _builder_audio_len(Qwen3ASRDummyInputsBuilder, fe),
            id="qwen3_asr",
        ),
        pytest.param(
            lambda fe: _builder_audio_len(KimiAudioDummyInputsBuilder, fe),
            id="kimi_audio",
        ),
        pytest.param(_omni_audio_len, id="qwen2_5_omni"),
    ],
)
def test_chunk_capped_builders_still_cap_sampling_rate(
    tmp_path: Path,
    sampling_rate: int,
    measure: Callable[[object], int],
) -> None:
    """Builders that clamp ``chunk_length`` to 30 still multiply by
    ``sampling_rate``. A hostile rate must not enlarge the zeros buffer.
    """
    _write_checkpoint(tmp_path, chunk_length=30, sampling_rate=sampling_rate)
    feature_extractor = _load_extractor(WhisperFeatureExtractor, tmp_path)
    if sampling_rate > _MAX_PROFILE_SAMPLES:
        # chunk_length * this rate would exhaust the runner. Record the
        # size requested from the allocator instead of materializing it.
        requested: list[int] = []
        real_zeros = dummy_inputs_module.np.zeros

        def zeros(shape, *args, **kwargs):
            n_samples = int(shape[0]) if isinstance(shape, tuple) else int(shape)
            requested.append(n_samples)
            if n_samples > _MAX_PROFILE_SAMPLES:
                return real_zeros((1,), dtype=kwargs.get("dtype", None))
            return real_zeros(shape, *args, **kwargs)

        dummy_inputs_module.np.zeros = zeros
        try:
            measure(feature_extractor)
        finally:
            dummy_inputs_module.np.zeros = real_zeros
        assert requested, "dummy builder did not allocate audio"
        _assert_capped(max(requested))
        return

    _assert_capped(measure(feature_extractor))
