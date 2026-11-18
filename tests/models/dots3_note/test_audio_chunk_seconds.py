# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dots3Note audio chunk_seconds must be a finite, positive, bounded duration.

A checkpoint ``audio_config.chunk_seconds`` is copied into the audio tower and
used as the stride of ``encode_waveforms``. These tests fail while that value
can be zero, negative, non-finite, or unbounded.
"""

import json
import signal
from pathlib import Path

import pytest
import torch

import vllm.models.dots3_note.nvidia.audio as audio_mod
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.models.dots3_note.common.processor import (
    Dots3NoteDummyInputsBuilder,
    Dots3NoteProcessingInfo,
    load_note_config_section,
)
from vllm.models.dots3_note.nvidia.audio import (
    SAMPLE_RATE,
    Dots3NoteAudioConfig,
    Dots3NoteAudioModel,
    pad_or_trim,
)

# Same order of magnitude as the audio decode duration limit: a few minutes,
# not an arbitrary multi-hour allocation.
MAX_CHUNK_SECONDS = 600

_WHISPER = {
    "d_model": 32,
    "encoder_layers": 1,
    "encoder_attention_heads": 4,
    "encoder_ffn_dim": 64,
    "max_source_positions": 128,
}


def _audio_config(**overrides: object) -> Dots3NoteAudioConfig:
    fields: dict[str, object] = {
        "chunk_seconds": 30,
        "whisper_config": _WHISPER,
        "whisper_adapter_in_dim": 32,
        "whisper_adapter_out_dim": 32,
        "use_conv2d_stem": False,
        "use_rope": False,
        "use_rms_norm": False,
        "merge_factor": 1,
    }
    fields.update(overrides)
    return Dots3NoteAudioConfig(**fields)


def _run_bounded(fn, seconds: int = 2):
    """Run ``fn`` and raise TimeoutError if it is still going after ``seconds``."""

    def _handle(signum, frame):
        del signum, frame
        raise TimeoutError("audio encode did not finish")

    previous = signal.signal(signal.SIGALRM, _handle)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _processing_info(audio_config: dict) -> Dots3NoteProcessingInfo:
    info = Dots3NoteProcessingInfo.__new__(Dots3NoteProcessingInfo)
    info.__dict__["audio_config"] = audio_config
    info.__dict__["image_processor"] = None
    info.__dict__["vision_config"] = None
    return info


def _dummy_builder(audio_config: dict) -> Dots3NoteDummyInputsBuilder:
    builder = Dots3NoteDummyInputsBuilder.__new__(Dots3NoteDummyInputsBuilder)
    builder.info = _processing_info(audio_config)
    return builder


@pytest.fixture
def audio_model() -> Dots3NoteAudioModel:
    return Dots3NoteAudioModel(_audio_config())


def test_default_chunk_seconds_stays_one_minute():
    config = Dots3NoteAudioConfig(
        whisper_config=_WHISPER,
        whisper_adapter_in_dim=32,
        whisper_adapter_out_dim=32,
    )
    assert config.chunk_seconds == 60
    assert config.chunk_samples == 60 * SAMPLE_RATE
    assert config.chunk_mel_frames == 60 * 100


def test_pad_or_trim_length_is_the_allocated_size():
    padded = pad_or_trim(torch.zeros(4), length=32)
    assert padded.shape[-1] == 32


@pytest.mark.parametrize("chunk_seconds", [0, 0.0, -1, -0.5])
def test_audio_config_rejects_non_positive_chunk_seconds(chunk_seconds):
    with pytest.raises(ValueError, match="chunk_seconds"):
        _audio_config(chunk_seconds=chunk_seconds)


@pytest.mark.parametrize("chunk_seconds", [float("nan"), float("inf"), float("-inf")])
def test_audio_config_rejects_non_finite_chunk_seconds(chunk_seconds):
    with pytest.raises(ValueError, match="chunk_seconds"):
        _audio_config(chunk_seconds=chunk_seconds)


def test_audio_config_bounds_huge_chunk_seconds():
    try:
        config = _audio_config(chunk_seconds=10**7)
    except ValueError as exc:
        assert "chunk_seconds" in str(exc)
        return
    assert 0 < config.chunk_seconds <= MAX_CHUNK_SECONDS
    assert config.chunk_samples <= MAX_CHUNK_SECONDS * SAMPLE_RATE
    assert config.chunk_mel_frames <= MAX_CHUNK_SECONDS * 100


def test_from_json_rejects_zero_chunk_seconds(tmp_path: Path):
    path = tmp_path / "audio_config.json"
    path.write_text(json.dumps({"chunk_seconds": 0, "encoder_type": "dots"}))
    with pytest.raises(ValueError, match="chunk_seconds"):
        Dots3NoteAudioConfig.from_json(path)


def test_checkpoint_section_rejects_zero_chunk_seconds(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "dots3_note",
                "architectures": ["Dots3NoteForCausalLM"],
                "audio_config": {"chunk_seconds": 0, "encoder_type": "dots"},
            }
        )
    )
    loaded = load_note_config_section(str(tmp_path), None, "audio_config")
    assert loaded is not None
    with pytest.raises(ValueError, match="chunk_seconds"):
        Dots3NoteAudioConfig(**loaded)


def test_checkpoint_section_keeps_a_normal_chunk(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"audio_config": {"chunk_seconds": 30, "encoder_type": "dots"}})
    )
    loaded = load_note_config_section(str(tmp_path), None, "audio_config")
    config = Dots3NoteAudioConfig(**loaded)
    assert config.chunk_seconds == 30
    assert config.chunk_samples == 30 * SAMPLE_RATE


def _force_chunk(model: Dots3NoteAudioModel, chunk_seconds: float) -> None:
    """Set the tower fields a checkpoint already stored, after construction."""
    encoder = model.dots_encoder
    encoder.chunk_seconds = chunk_seconds
    try:
        encoder.chunk_samples = int(chunk_seconds * SAMPLE_RATE)
        encoder.chunk_mel_frames = int(chunk_seconds * 100)
    except (OverflowError, ValueError):
        encoder.chunk_samples = 0
        encoder.chunk_mel_frames = 0


def _forward_waveform(model: Dots3NoteAudioModel, *, as_list: bool) -> None:
    wave = torch.zeros(800, dtype=torch.float32)
    lengths = torch.tensor([wave.numel()], dtype=torch.long)
    audio_inputs = [wave] if as_list else wave
    model.forward(audio_inputs, lengths)


@pytest.mark.parametrize("chunk_seconds", [0, 0.0, -1])
@pytest.mark.parametrize("as_list", [False, True])
def test_forward_rejects_non_advancing_chunk(audio_model, chunk_seconds, as_list):
    _force_chunk(audio_model, chunk_seconds)
    try:
        _run_bounded(
            lambda: _forward_waveform(audio_model, as_list=as_list),
        )
    except TimeoutError:
        pytest.fail("Dots3NoteAudioModel.forward did not stop")
    except ValueError as exc:
        assert "chunk" in str(exc).lower()
        return
    except Exception as exc:
        pytest.fail(
            f"forward did not reject chunk_seconds: {type(exc).__name__}: {exc}"
        )
    pytest.fail("forward accepted a chunk_seconds that does not advance")


def test_forward_caps_pad_length(audio_model, monkeypatch):
    seen: list[int] = []
    original = audio_mod.pad_or_trim

    def _recording_pad(array, length=audio_mod.N_SAMPLES, axis=-1):
        seen.append(int(length))
        if length > MAX_CHUNK_SECONDS * SAMPLE_RATE or length < 1:
            raise AssertionError(f"pad length {length} is outside the chunk cap")
        return original(array, length=length, axis=axis)

    monkeypatch.setattr(audio_mod, "pad_or_trim", _recording_pad)
    _force_chunk(audio_model, 10**6)
    try:
        _run_bounded(lambda: _forward_waveform(audio_model, as_list=False))
    except AssertionError:
        raise
    except ValueError as exc:
        assert "chunk" in str(exc).lower()
        return
    except Exception:
        assert seen, "encode did not pad and did not reject the chunk"
        assert max(seen) <= MAX_CHUNK_SECONDS * SAMPLE_RATE
        return
    assert seen
    assert max(seen) <= MAX_CHUNK_SECONDS * SAMPLE_RATE


@pytest.mark.parametrize(
    "chunk_seconds",
    [0, 0.0, -5, float("nan"), float("inf")],
)
def test_token_budget_rejects_bad_chunk_seconds(chunk_seconds):
    info = _processing_info(
        {"chunk_seconds": chunk_seconds, "sampling_rate": SAMPLE_RATE}
    )
    with pytest.raises(ValueError, match="chunk_seconds"):
        info.get_mm_max_tokens_per_item(1024, {"audio": 1})


@pytest.mark.parametrize("chunk_seconds", [0, -5, float("nan"), float("inf")])
def test_profiling_dummy_audio_rejects_bad_chunk_seconds(chunk_seconds):
    builder = _dummy_builder(
        {
            "chunk_seconds": chunk_seconds,
            "sampling_rate": SAMPLE_RATE,
            "use_conv2d_stem": True,
            "merge_factor": 1,
        }
    )
    with pytest.raises(ValueError, match="chunk_seconds"):
        builder.get_dummy_mm_data(
            128,
            {"audio": 1},
            MultiModalDummyOptions(),
        )
