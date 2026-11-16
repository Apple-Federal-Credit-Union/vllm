# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunASR preprocessor sizes must stay inside published checkpoint ranges.

The published FunASR checkpoint
``FunAudioLLM/Fun-ASR-Nano-2512-vllm`` ships ``chunk_length=30``,
``sampling_rate=16000``, ``hop_length=160``, and a ``frontend_conf`` of
``fs=16000``, ``n_mels=80``, ``frame_length=25``, ``frame_shift=10``,
``lfr_m=7``, ``lfr_n=6``. Those fields are read from
``preprocessor_config.json`` with no upper bound and are then used to size
profiling audio and mel features.

A value outside the ranges below must fail while the checkpoint is loaded,
before dummy-audio allocation, mel-filterbank construction, low-frame-rate
stacking, or a ``cmvn_file`` open. A later read of the same field must not
be able to substitute a different value for the one that was checked.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm.config.model import ModelConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.model_executor.models.funasr import (
    FunASRDummyInputsBuilder,
    FunASRForConditionalGeneration,
    FunASRProcessingInfo,
)
from vllm.multimodal.processing.context import InputProcessingContext

_MAX_CHUNK_LENGTH_S = 30
_MIN_CHUNK_LENGTH_S = 1
_SAMPLING_RATE = 16_000
_HOP_LENGTH = 160
_FS = 16_000
_MIN_N_MELS = 4
_MAX_N_MELS = 80
_FRAME_LENGTH_MS = 25
_FRAME_SHIFT_MS = 10
_MIN_LFR_M = 1
_MAX_LFR_M = 7
_MIN_LFR_N = 1

_PUBLISHED_FRONTEND = {
    "fs": _FS,
    "window": "hamming",
    "n_mels": _MAX_N_MELS,
    "frame_length": _FRAME_LENGTH_MS,
    "frame_shift": _FRAME_SHIFT_MS,
    "lfr_m": _MAX_LFR_M,
    "lfr_n": 6,
    "cmvn_file": None,
}

_MODEL_CONFIG = {
    "architectures": ["FunASRForConditionalGeneration"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_hidden_layers": 1,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 32,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
    "head_dim": 16,
}


class _SecondRead(Mapping):
    """Mapping whose selected key changes after the first lookup.

    This is not a ``dict`` subclass: ``**mapping`` copies a dict subclass
    from its storage and never calls ``__getitem__``. A check that accepts
    the first value and then unpacks the mapping again observes the second.
    """

    def __init__(self, data: Mapping[str, object], key: str, first, second):
        self._data = dict(data)
        self._key = key
        self._first = first
        self._second = second
        self.reads = 0

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str):
        value = self._data[key]
        if key != self._key:
            return value
        self.reads += 1
        return self._first if self.reads == 1 else self._second


class _BoundExtractor:
    """Processing-info stand-in that returns one already loaded extractor."""

    def __init__(self, extractor):
        self._extractor = extractor

    def get_feature_extractor(self, **kwargs):
        return self._extractor


def _tokenizer() -> PreTrainedTokenizerFast:
    raw = Tokenizer(WordLevel({"<|AUDIO|>": 0, "a": 1}, unk_token="a"))
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="a")


def _write_checkpoint(root: Path, **overrides) -> Path:
    frontend = dict(_PUBLISHED_FRONTEND)
    top = {
        "chunk_length": _MAX_CHUNK_LENGTH_S,
        "feature_extractor_type": "FunASRFeatureExtractor",
        "feature_size": _MAX_N_MELS,
        "hop_length": _HOP_LENGTH,
        "n_fft": 400,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "FunASRProcessor",
        "return_attention_mask": False,
        "sampling_rate": _SAMPLING_RATE,
        "frontend": "WavFrontend",
    }
    for key, value in overrides.items():
        if key in frontend:
            frontend[key] = value
        else:
            top[key] = value
    top["frontend_conf"] = frontend
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_MODEL_CONFIG), encoding="utf-8")
    (root / "preprocessor_config.json").write_text(
        json.dumps(top),
        encoding="utf-8",
    )
    return root


def _processing_info(root: Path, **overrides) -> FunASRProcessingInfo:
    model_dir = _write_checkpoint(root, **overrides)
    model_config = ModelConfig(
        model=str(model_dir),
        skip_tokenizer_init=True,
        trust_remote_code=False,
        max_model_len=32,
        enforce_eager=True,
    )
    return FunASRProcessingInfo(InputProcessingContext(model_config, _tokenizer()))


def _dummy_length(info) -> int:
    data = FunASRDummyInputsBuilder(info).get_dummy_mm_data(
        seq_len=32,
        mm_counts={"audio": 1},
        mm_options=MultiModalDummyOptions(),
    )
    return int(data["audio"][0].shape[0])


def _flip_attribute(obj, name: str, first, second) -> None:
    reads = {"n": 0}

    def getter(self):
        reads["n"] += 1
        return first if reads["n"] == 1 else second

    flipped = type(f"Flip_{name}", (type(obj),), {name: property(getter)})
    obj.__class__ = flipped


def _short_audio() -> list[np.ndarray]:
    return [np.zeros(1600, dtype=np.float32)]


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_length": _MAX_CHUNK_LENGTH_S + 1},
        {"chunk_length": 0},
        {"chunk_length": -1},
        {"sampling_rate": _SAMPLING_RATE + 1},
        {"sampling_rate": 0},
        {"sampling_rate": -1},
        {"hop_length": 0},
        {"hop_length": -1},
        {"n_mels": _MAX_N_MELS + 1},
        {"n_mels": _MIN_N_MELS - 1},
        {"n_mels": 0},
        {"lfr_m": _MAX_LFR_M + 1},
        {"lfr_m": 0},
        {"lfr_n": 0},
        {"lfr_n": -1},
        {"frame_shift": 0},
        {"frame_shift": _FRAME_SHIFT_MS - 1},
        {"frame_shift": _FRAME_SHIFT_MS + 1},
        {"frame_length": 0},
        {"frame_length": _FRAME_LENGTH_MS - 1},
        {"frame_length": _FRAME_LENGTH_MS + 1},
        {"frame_length": 10**6},
        {"fs": 0},
        {"fs": -1},
        {"fs": _FS - 1},
        {"fs": _FS + 1},
    ],
    ids=lambda overrides: ",".join(
        f"{key}={value}" for key, value in overrides.items()
    ),
)
def test_preprocessor_outside_published_range_rejected_at_load(tmp_path, overrides):
    """Checkpoint load must refuse the value before any feature allocation."""
    info = _processing_info(tmp_path, **overrides)
    with pytest.raises(ValueError):
        info.get_feature_extractor()


def test_published_preprocessor_sizes_are_accepted(tmp_path):
    info = _processing_info(tmp_path)
    extractor = info.get_feature_extractor()
    assert extractor.chunk_length == _MAX_CHUNK_LENGTH_S
    assert extractor.sampling_rate == _SAMPLING_RATE
    assert extractor.hop_length == _HOP_LENGTH
    assert _dummy_length(info) == _MAX_CHUNK_LENGTH_S * _SAMPLING_RATE

    features = extractor(_short_audio())["input_features"]
    assert features.shape[-1] == _MAX_N_MELS * _MAX_LFR_M


def test_lfr_identity_sizes_are_accepted(tmp_path):
    info = _processing_info(tmp_path, lfr_m=_MIN_LFR_M, lfr_n=_MIN_LFR_N)
    extractor = info.get_feature_extractor()
    features = extractor(_short_audio())["input_features"]
    assert features.shape[-1] == _MAX_N_MELS


def test_minimum_chunk_length_is_accepted(tmp_path):
    info = _processing_info(tmp_path, chunk_length=_MIN_CHUNK_LENGTH_S)
    assert _dummy_length(info) == _MIN_CHUNK_LENGTH_S * _SAMPLING_RATE


def test_speech_to_text_config_rejects_oversize_chunk_length(tmp_path):
    info = _processing_info(tmp_path, chunk_length=_MAX_CHUNK_LENGTH_S + 1)
    with pytest.raises(ValueError):
        FunASRForConditionalGeneration.get_speech_to_text_config(
            info.ctx.model_config,
            "transcribe",
        )


def test_num_audio_tokens_rejects_zero_hop_length(tmp_path):
    info = _processing_info(tmp_path, hop_length=0)
    with pytest.raises(ValueError):
        FunASRForConditionalGeneration.get_num_audio_tokens(
            1.0,
            SpeechToTextConfig(sample_rate=_SAMPLING_RATE, max_audio_clip_s=30),
            info.ctx.model_config,
        )


def test_dummy_audio_rechecks_chunk_length(tmp_path):
    info = _processing_info(tmp_path)
    extractor = info.get_feature_extractor()
    _flip_attribute(
        extractor,
        "chunk_length",
        _MAX_CHUNK_LENGTH_S + 1,
        _MAX_CHUNK_LENGTH_S,
    )
    with pytest.raises(ValueError):
        _dummy_length(_BoundExtractor(extractor))


def test_dummy_audio_rechecks_sampling_rate(tmp_path):
    info = _processing_info(tmp_path)
    extractor = info.get_feature_extractor()
    _flip_attribute(extractor, "sampling_rate", _SAMPLING_RATE + 1, _SAMPLING_RATE)
    with pytest.raises(ValueError):
        _dummy_length(_BoundExtractor(extractor))


@pytest.mark.parametrize(
    ("key", "first", "second"),
    [
        ("n_mels", _MAX_N_MELS + 1, _MAX_N_MELS),
        ("lfr_m", _MAX_LFR_M + 1, _MAX_LFR_M),
        ("lfr_n", 0, 6),
        ("frame_shift", 0, _FRAME_SHIFT_MS),
        ("frame_length", 10**6, _FRAME_LENGTH_MS),
        ("fs", 0, _FS),
    ],
)
def test_feature_extraction_rechecks_frontend_conf(tmp_path, key, first, second):
    info = _processing_info(tmp_path)
    extractor = info.get_feature_extractor()
    extractor.frontend_conf = _SecondRead(
        dict(extractor.frontend_conf),
        key,
        first,
        second,
    )
    with pytest.raises(ValueError):
        extractor(_short_audio())


def test_illegal_n_mels_does_not_open_cmvn_file(tmp_path):
    cmvn_path = tmp_path / "cmvn.ark"
    info = _processing_info(
        tmp_path / "checkpoint",
        n_mels=_MAX_N_MELS + 1,
        cmvn_file=str(cmvn_path),
    )
    with pytest.raises(ValueError):
        extractor = info.get_feature_extractor()
        extractor(_short_audio())
    assert not cmvn_path.exists()
