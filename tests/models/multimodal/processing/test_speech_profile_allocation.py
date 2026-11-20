# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound speech profiling buffers built from checkpoint preprocessor fields.

``chunk_length`` and ``sampling_rate`` are read from the checkpoint
preprocessor config. Profiling then allocates ``chunk_length * sampling_rate``
samples (Whisper, Qwen2-Audio, FireRedASR2, and FunASR). A checkpoint can set
either field large enough to exhaust memory while the worker builds dummy
multimodal inputs, which happens before the server accepts requests.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2AudioConfig,
    Qwen2Config,
    WhisperConfig,
)

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseMultiModalProcessor

# 30 seconds at 48 kHz. A normal 16 kHz / 30 s speech checkpoint stays under
# this, and so does a checkpoint whose fields have been clamped to those caps.
MAX_PROFILE_SAMPLES = 30 * 48_000
LEGITIMATE_CHUNK_LENGTH = 30
LEGITIMATE_SAMPLING_RATE = 16_000
HOSTILE_CHUNK_LENGTH = 120
HOSTILE_SAMPLING_RATE = 96_000


def _write_tokenizer(model_dir: Path, tokens: list[str], processor_class: str | None):
    vocab = {token: index for index, token in enumerate(tokens)}
    vocab.setdefault("<unk>", len(vocab))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(model_dir)
    if processor_class is None:
        return
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["processor_class"] = processor_class
    config_path.write_text(json.dumps(config))


def _write_checkpoint(
    model_dir: Path,
    *,
    model_config: Mapping[str, object],
    chunk_length: int,
    sampling_rate: int,
    processor_class: str | None,
    feature_extractor_type: str,
    extra_extractor: Mapping[str, object] | None = None,
) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(model_config))
    extractor = {
        "feature_extractor_type": feature_extractor_type,
        "chunk_length": chunk_length,
        "sampling_rate": sampling_rate,
        "feature_size": 80,
        "padding_value": 0.0,
    }
    if extra_extractor:
        extractor.update(extra_extractor)
    (model_dir / "preprocessor_config.json").write_text(json.dumps(extractor))
    _write_tokenizer(
        model_dir,
        ["<|AUDIO|>", "<|startoftranscript|>", "<unk>"],
        processor_class,
    )


def _firered_config() -> dict[str, object]:
    return {
        "architectures": ["FireRedASR2ForConditionalGeneration"],
        "model_type": "qwen2",
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "max_position_embeddings": 128,
        "max_window_layers": 1,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 32,
        "encoder_downsample_rate": 2,
        "audio_encoder_conf": {
            "idim": 80,
            "n_layers_enc": 1,
            "n_head": 2,
            "d_model": 32,
            "pe_maxlen": 16,
            "kernel_size": 3,
        },
    }


def _funasr_config() -> dict[str, object]:
    return {
        "architectures": ["FunASRForConditionalGeneration"],
        "model_type": "qwen3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
    }


def _whisper_config() -> dict[str, object]:
    config = WhisperConfig(
        encoder_layers=1,
        decoder_layers=1,
        d_model=64,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        vocab_size=51865,
        max_source_positions=1500,
        max_target_positions=448,
    ).to_dict()
    config["architectures"] = ["WhisperForConditionalGeneration"]
    return config


def _qwen2_audio_config() -> dict[str, object]:
    config = Qwen2AudioConfig(
        audio_config={
            "encoder_layers": 1,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 128,
            "d_model": 64,
            "num_mel_bins": 80,
            "max_source_positions": 1500,
        },
        text_config=Qwen2Config(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            vocab_size=32,
            max_position_embeddings=128,
        ).to_dict(),
        audio_token_index=0,
    ).to_dict()
    config["architectures"] = ["Qwen2AudioForConditionalGeneration"]
    return config


SPEECH_PROFILE_CASES = [
    pytest.param(
        _firered_config,
        "FireRedASR2Processor",
        "FireRedASR2FeatureExtractor",
        {
            "dim": 80,
            "means": [0.0] * 80,
            "inverse_std_variences": [1.0] * 80,
            "num_mel_bins": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "dither": 0.0,
            "max_length": 3000,
            "downsample_rate": 2,
            "left_context": 3,
            "right_context": 3,
        },
        id="fireredasr2",
    ),
    pytest.param(
        _funasr_config,
        "FunASRProcessor",
        "FunASRFeatureExtractor",
        {"hop_length": 160, "n_fft": 400, "max_length": 1000, "dither": 0.0},
        id="funasr",
    ),
    pytest.param(
        _whisper_config,
        None,
        "WhisperFeatureExtractor",
        {"hop_length": 160, "n_fft": 400, "dither": 0.0},
        id="whisper",
    ),
    pytest.param(
        _qwen2_audio_config,
        None,
        "WhisperFeatureExtractor",
        {"hop_length": 160, "n_fft": 400, "dither": 0.0},
        id="qwen2_audio",
    ),
]


def _load_processor(
    model_dir: Path,
) -> tuple[ModelConfig, BaseMultiModalProcessor]:
    model_config = ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        tokenizer_mode="hf",
        dtype="float32",
        max_model_len=64,
        enforce_eager=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    return model_config, processor


def _profile_audio_length(
    model_config: ModelConfig,
    processor: BaseMultiModalProcessor,
) -> int:
    # Engine startup reaches this through MultiModalBudget ->
    # get_dummy_mm_inputs, which allocates the waveform here before the
    # feature extractor runs.
    inputs = processor.get_dummy_inputs(
        model_config.max_model_len,
        {"audio": 1},
        model_config.get_multimodal_config().limit_per_prompt,
    )
    audio = inputs.mm_data_items["audio"].get(0)
    if isinstance(audio, tuple):
        audio = audio[0]
    return int(audio.shape[-1])


@pytest.mark.parametrize(
    ("config_factory", "processor_class", "feature_extractor_type", "extra"),
    SPEECH_PROFILE_CASES,
)
@pytest.mark.parametrize(
    ("chunk_length", "sampling_rate"),
    [
        (HOSTILE_CHUNK_LENGTH, LEGITIMATE_SAMPLING_RATE),
        (LEGITIMATE_CHUNK_LENGTH, HOSTILE_SAMPLING_RATE),
    ],
    ids=["hostile_chunk_length", "hostile_sampling_rate"],
)
def test_checkpoint_preprocessor_cannot_inflate_profile_audio(
    tmp_path: Path,
    config_factory,
    processor_class: str | None,
    feature_extractor_type: str,
    extra: Mapping[str, object],
    chunk_length: int,
    sampling_rate: int,
):
    model_dir = tmp_path / "checkpoint"
    _write_checkpoint(
        model_dir,
        model_config=config_factory(),
        chunk_length=chunk_length,
        sampling_rate=sampling_rate,
        processor_class=processor_class,
        feature_extractor_type=feature_extractor_type,
        extra_extractor=extra,
    )
    model_config, processor = _load_processor(model_dir)
    length = _profile_audio_length(model_config, processor)
    assert length <= MAX_PROFILE_SAMPLES, (
        f"profiling allocated {length} samples for chunk_length={chunk_length} "
        f"sampling_rate={sampling_rate}"
    )
    assert length > 0


def test_legitimate_firered_profile_audio_stays_one_chunk(tmp_path: Path):
    model_dir = tmp_path / "checkpoint"
    _write_checkpoint(
        model_dir,
        model_config=_firered_config(),
        chunk_length=LEGITIMATE_CHUNK_LENGTH,
        sampling_rate=LEGITIMATE_SAMPLING_RATE,
        processor_class="FireRedASR2Processor",
        feature_extractor_type="FireRedASR2FeatureExtractor",
        extra_extractor={
            "dim": 80,
            "means": [0.0] * 80,
            "inverse_std_variences": [1.0] * 80,
            "num_mel_bins": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "dither": 0.0,
            "max_length": 3000,
            "downsample_rate": 2,
            "left_context": 3,
            "right_context": 3,
        },
    )
    model_config, processor = _load_processor(model_dir)
    length = _profile_audio_length(model_config, processor)
    assert length == LEGITIMATE_CHUNK_LENGTH * LEGITIMATE_SAMPLING_RATE
