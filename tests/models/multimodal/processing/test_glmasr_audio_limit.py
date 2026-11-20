# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profiling must not size dummy audio from an uncapped checkpoint value.

GLM-ASR and AudioFlamingo3 read ``max_audio_len`` and ``sampling_rate`` from
the published processor config and pass the product to ``np.zeros`` while
building the startup profile batch. The same GLM-ASR value is also the
speech-to-text clip limit and the chunk-window cap.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AudioFlamingo3Config,
    BatchFeature,
    GlmAsrConfig,
    PreTrainedTokenizerFast,
)

from vllm.config import ModelConfig
from vllm.config.multimodal import AudioDummyOptions, MultiModalDummyOptions
from vllm.model_executor.models.audioflamingo3 import (
    AudioFlamingo3DummyInputsBuilder,
    AudioFlamingo3MultiModalProcessor,
    AudioFlamingo3ProcessingInfo,
)
from vllm.model_executor.models.glmasr import (
    GlmAsrDummyInputsBuilder,
    GlmAsrForConditionalGeneration,
    GlmAsrMultiModalProcessor,
    GlmAsrProcessingInfo,
)
from vllm.model_executor.models.glmasr_utils import DEFAULT_MAX_AUDIO_LEN_S
from vllm.multimodal.parse import AudioProcessorItems
from vllm.multimodal.processing import InputProcessingContext
from vllm.multimodal.processing import dummy_inputs as dummy_inputs_mod
from vllm.tokenizers import cached_tokenizer_from_config

# WhisperFeatureExtractor's default, and the rate in the published GLM-ASR
# processor config. A checkpoint may set a larger rate; profiling must not.
STANDARD_SAMPLING_RATE = 16_000
# AudioFlamingo3Processor's own default duration. vLLM has no tighter constant.
AF3_MAX_AUDIO_S = 600
# One profiling item. Encoder profiling calls get_dummy_mm_inputs with this.
PROFILE_AUDIO_COUNT = {"audio": 1}

pytestmark = pytest.mark.skip_global_cleanup


def _save_tokenizer(root: Path, audio_token: str) -> None:
    vocab = {"<unk>": 0, audio_token: 1}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token=audio_token,
    ).save_pretrained(root)


def _write_processor_config(
    root: Path,
    *,
    processor_class: str,
    audio_token: str,
    sampling_rate: int,
    max_audio_len: int | None,
) -> None:
    # Same layout as a published GLM-ASR repo: one processor_config.json whose
    # top-level max_audio_len and nested feature_extractor are what
    # from_pretrained applies. There is no separate preprocessor_config.json.
    payload: dict[str, object] = {
        "processor_class": processor_class,
        "audio_token": audio_token,
        "feature_extractor": {
            "chunk_length": 30,
            "dither": 0.0,
            "feature_extractor_type": "WhisperFeatureExtractor",
            "feature_size": 128,
            "hop_length": 160,
            "n_fft": 400,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": False,
            "sampling_rate": sampling_rate,
        },
    }
    if max_audio_len is not None:
        payload["max_audio_len"] = max_audio_len
    (root / "processor_config.json").write_text(json.dumps(payload))


def _load_model(root: Path) -> tuple[ModelConfig, InputProcessingContext]:
    model_config = ModelConfig(
        model=str(root),
        tokenizer=str(root),
        tokenizer_mode="hf",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        enforce_eager=True,
        max_model_len=128,
    )
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )
    return model_config, ctx


def _expected_profile_samples(
    max_audio_len: int | None,
    sampling_rate: int,
    max_duration_s: int,
) -> int:
    duration_s = max_duration_s if max_audio_len is None else max_audio_len
    duration_s = min(duration_s, max_duration_s)
    rate = min(sampling_rate, STANDARD_SAMPLING_RATE)
    return int(duration_s * rate)


def _spy_profile_allocation(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the length passed to the profiling allocator.

    Lengths above the fixed sample ceiling are not materialized, so a
    checkpoint that asks for a huge buffer fails the assertion instead of
    killing the test process. A fix that clamps only after allocating still
    fails, because the length passed to ``np.zeros`` is what is recorded.
    """
    requested: list[int] = []
    real_zeros = dummy_inputs_mod.np.zeros
    ceiling = DEFAULT_MAX_AUDIO_LEN_S * STANDARD_SAMPLING_RATE

    def spy(shape, *args, **kwargs):
        length = shape[0] if isinstance(shape, tuple) else shape
        length = int(length)
        requested.append(length)
        if length > ceiling:
            return real_zeros((0,), *args, **kwargs)
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(dummy_inputs_mod.np, "zeros", spy)
    return requested


def _profile_audio_length(mm_processor, model_config, mm_options) -> int:
    # get_dummy_inputs is what profiling's get_dummy_mm_inputs calls before
    # the multimodal encoder runs. Count 1 matches get_dummy_encoder_profile_inputs.
    inputs = mm_processor.get_dummy_inputs(
        seq_len=model_config.max_model_len,
        mm_counts=PROFILE_AUDIO_COUNT,
        mm_options=mm_options,
    )
    items = inputs.mm_data_items.get_items("audio", AudioProcessorItems)
    assert items.get_count() == 1
    return items.get_audio_length(0)


def _glmasr_repo(
    root: Path,
    *,
    max_audio_len: int | None,
    sampling_rate: int,
) -> tuple[ModelConfig, GlmAsrMultiModalProcessor]:
    _save_tokenizer(root, "<|pad|>")
    _write_processor_config(
        root,
        processor_class="GlmAsrProcessor",
        audio_token="<|pad|>",
        sampling_rate=sampling_rate,
        max_audio_len=max_audio_len,
    )
    (root / "config.json").write_text(GlmAsrConfig().to_json_string())
    model_config, ctx = _load_model(root)
    info = GlmAsrProcessingInfo(ctx)
    processor = GlmAsrMultiModalProcessor(info, GlmAsrDummyInputsBuilder(info))
    return model_config, processor


def _af3_repo(
    root: Path,
    *,
    max_audio_len: int,
    sampling_rate: int,
) -> tuple[ModelConfig, AudioFlamingo3MultiModalProcessor]:
    _save_tokenizer(root, "<sound>")
    _write_processor_config(
        root,
        processor_class="AudioFlamingo3Processor",
        audio_token="<sound>",
        sampling_rate=sampling_rate,
        max_audio_len=max_audio_len,
    )
    (root / "config.json").write_text(AudioFlamingo3Config().to_json_string())
    model_config, ctx = _load_model(root)
    info = AudioFlamingo3ProcessingInfo(ctx)
    processor = AudioFlamingo3MultiModalProcessor(
        info, AudioFlamingo3DummyInputsBuilder(info)
    )
    return model_config, processor


@pytest.mark.parametrize(
    ("max_audio_len", "sampling_rate"),
    [
        (800, STANDARD_SAMPLING_RATE),
        (10, STANDARD_SAMPLING_RATE),
        (None, STANDARD_SAMPLING_RATE),
        (DEFAULT_MAX_AUDIO_LEN_S, 1_000_000),
        (10**9, STANDARD_SAMPLING_RATE),
    ],
)
def test_glmasr_profiling_audio_is_capped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_audio_len: int | None,
    sampling_rate: int,
):
    model_config, processor = _glmasr_repo(
        tmp_path,
        max_audio_len=max_audio_len,
        sampling_rate=sampling_rate,
    )
    loaded = processor.info.get_hf_processor()
    if max_audio_len is None:
        assert loaded.max_audio_len == DEFAULT_MAX_AUDIO_LEN_S
    else:
        assert loaded.max_audio_len == max_audio_len
    assert loaded.feature_extractor.sampling_rate == sampling_rate

    requested = _spy_profile_allocation(monkeypatch)
    expected = _expected_profile_samples(
        max_audio_len, sampling_rate, DEFAULT_MAX_AUDIO_LEN_S
    )
    mm_options = model_config.get_multimodal_config().limit_per_prompt
    length = _profile_audio_length(processor, model_config, mm_options)

    assert requested == [expected], requested
    assert length == expected

    stt = GlmAsrForConditionalGeneration.get_speech_to_text_config(
        model_config, "transcribe"
    )
    expected_clip = DEFAULT_MAX_AUDIO_LEN_S if max_audio_len is None else max_audio_len
    expected_clip = min(expected_clip, DEFAULT_MAX_AUDIO_LEN_S)
    assert stt.max_audio_clip_s == expected_clip
    assert stt.sample_rate == sampling_rate


def test_glmasr_profile_override_cannot_exceed_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_config, processor = _glmasr_repo(
        tmp_path,
        max_audio_len=10**9,
        sampling_rate=STANDARD_SAMPLING_RATE,
    )
    processor.info.get_hf_processor()
    requested = _spy_profile_allocation(monkeypatch)
    mm_options = MultiModalDummyOptions(
        {"audio": AudioDummyOptions(count=1, length=10**12)}
    )
    length = _profile_audio_length(processor, model_config, mm_options)
    expected = DEFAULT_MAX_AUDIO_LEN_S * STANDARD_SAMPLING_RATE
    assert requested == [expected], requested
    assert length == expected


def test_glmasr_chunk_windows_use_the_same_cap(tmp_path: Path):
    _, processor = _glmasr_repo(
        tmp_path,
        max_audio_len=10**9,
        sampling_rate=STANDARD_SAMPLING_RATE,
    )
    feature_extractor = processor.info.get_feature_extractor()
    hf_processor = processor.info.get_hf_processor()
    window = int(feature_extractor.sampling_rate * feature_extractor.chunk_length)
    capped_windows = int(DEFAULT_MAX_AUDIO_LEN_S // feature_extractor.chunk_length)
    # Longer than the cap, still small enough to build in the test itself.
    audio = np.zeros((capped_windows + 8) * window, dtype=np.float32)
    counts = processor._calculate_chunk_counts([audio], feature_extractor, hf_processor)
    assert counts == [capped_windows]


@pytest.mark.parametrize(
    ("max_audio_len", "sampling_rate"),
    [
        (800, STANDARD_SAMPLING_RATE),
        (10, STANDARD_SAMPLING_RATE),
        (AF3_MAX_AUDIO_S, 1_000_000),
        (10**9, STANDARD_SAMPLING_RATE),
    ],
)
def test_audioflamingo3_profiling_audio_is_capped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_audio_len: int,
    sampling_rate: int,
):
    model_config, processor = _af3_repo(
        tmp_path,
        max_audio_len=max_audio_len,
        sampling_rate=sampling_rate,
    )
    loaded = processor.info.get_hf_processor()
    assert loaded.max_audio_len == max_audio_len
    assert loaded.feature_extractor.sampling_rate == sampling_rate

    requested = _spy_profile_allocation(monkeypatch)
    expected = _expected_profile_samples(max_audio_len, sampling_rate, AF3_MAX_AUDIO_S)
    # The shared allocator spy allows up to the GLM ceiling, which is above
    # AudioFlamingo3's own cap. The assertion is the tighter of the two.
    mm_options = model_config.get_multimodal_config().limit_per_prompt
    length = _profile_audio_length(processor, model_config, mm_options)
    assert requested == [expected], requested
    assert length == expected


def test_audioflamingo3_chunk_windows_use_the_same_cap(tmp_path: Path):
    _, processor = _af3_repo(
        tmp_path,
        max_audio_len=10**9,
        sampling_rate=STANDARD_SAMPLING_RATE,
    )
    feature_extractor = processor.info.get_feature_extractor()
    window = int(feature_extractor.sampling_rate * feature_extractor.chunk_length)
    capped_windows = int(AF3_MAX_AUDIO_S // feature_extractor.chunk_length)
    audio = np.zeros((capped_windows + 8) * window, dtype=np.float32)
    processed = processor._postprocess_hf_mm_data(
        {"audio": [audio]},
        {},
        BatchFeature({"input_features": np.zeros((1, 1))}),
    )
    assert processed["chunk_counts"].tolist() == [capped_windows]
