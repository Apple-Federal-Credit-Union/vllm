# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCM byte budget must be enforced while torchcodec is still decoding.

``load_audio_torchcodec`` compares ``samples.data.nbytes`` with
``max_decode_bytes`` only after ``AudioDecoder.get_samples_played_in_range``
returns. Torchcodec 0.14 accumulates every frame in that window and
concatenates them before returning, so a container whose STREAMINFO
duration is missing or under-reported still materializes the whole clip.

These tests fail until decoding stops once the running PCM size exceeds
the budget. Re-reading the same container metadata does not satisfy them:
the headers below stay under the budget while the packets do not.
"""

from collections.abc import Callable
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import pybase64
import pytest
import soundfile as sf

import vllm.envs as envs
from vllm.multimodal.media import AudioMediaIO
from vllm.multimodal.media import audio as audio_module
from vllm.multimodal.media.audio import (
    load_audio,
    load_audio_pyav,
    load_audio_soundfile,
    load_audio_torchcodec,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# 4 s at 48 kHz. Declared STREAMINFO length is overwritten below; the
# packets still contain this many samples per channel.
_FULL_FRAMES = 192_000
_CHANNELS = 8
_SAMPLE_RATE = 48_000
_FULL_PCM_BYTES = _FULL_FRAMES * _CHANNELS * np.dtype(np.float32).itemsize

# Above one decoded frame (64 KiB for this FLAC) and below the real PCM.
_DECODE_LIMIT = 1_000_000
_FRAME_SLACK = 256 * 1024

# FLAC STREAMINFO: 20-bit rate, 3-bit channels-1, 5-bit bps-1, 36-bit samples.
_STREAMINFO_SAMPLE_SHIFT = 36


def _flac_with_declared_samples(declared_samples: int) -> bytes:
    """Build 8-channel silence and replace the STREAMINFO sample count."""
    buffer = BytesIO()
    silence = np.zeros((_FULL_FRAMES, _CHANNELS), dtype=np.int16)
    sf.write(buffer, silence, _SAMPLE_RATE, format="FLAC")
    data = bytearray(buffer.getvalue())
    assert data[:4] == b"fLaC"
    info = 8
    packed = int.from_bytes(data[info + 10 : info + 18], "big")
    sample_rate = packed >> 44
    channels_m1 = (packed >> 41) & 0x7
    bps_m1 = (packed >> _STREAMINFO_SAMPLE_SHIFT) & 0x1F
    rewritten = (
        (sample_rate << 44)
        | (channels_m1 << 41)
        | (bps_m1 << _STREAMINFO_SAMPLE_SHIFT)
        | declared_samples
    )
    data[info + 10 : info + 18] = rewritten.to_bytes(8, "big")
    assert len(data) < 25 * (1 << 20)
    return bytes(data)


def _assert_metadata_understates_pcm(payload: bytes, declared_samples: int) -> None:
    """The pre-check must be looking at a lie, not at the real frame count."""
    metadata = audio_module.AudioDecoder(BytesIO(payload)).metadata
    assert metadata.sample_rate == _SAMPLE_RATE
    assert metadata.num_channels == _CHANNELS
    if declared_samples == 0:
        assert metadata.duration_seconds is None
        return
    assert metadata.duration_seconds == pytest.approx(0.01)
    estimated = (
        int(metadata.duration_seconds * metadata.sample_rate)
        * metadata.num_channels
        * np.dtype(np.float32).itemsize
    )
    assert estimated < _DECODE_LIMIT
    assert _FULL_PCM_BYTES > _DECODE_LIMIT + _FRAME_SLACK


@contextmanager
def _capture_torchcodec_pcm(recorded: list[int]):
    """Record PCM bytes returned by the real torchcodec range decoder."""
    decoder_cls = audio_module.AudioDecoder
    assert decoder_cls is not None
    original = decoder_cls.get_samples_played_in_range

    def _wrapped(self, *args, **kwargs):
        samples = original(self, *args, **kwargs)
        tensor = samples.data
        recorded.append(max(int(tensor.nbytes), int(tensor.untyped_storage().size())))
        return samples

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(decoder_cls, "get_samples_played_in_range", _wrapped)
        yield


def _assert_refused_within_budget(decode: Callable[[], object]) -> None:
    recorded: list[int] = []
    with _capture_torchcodec_pcm(recorded), pytest.raises(ValueError):
        decode()
    materialized = sum(recorded)
    assert materialized <= _DECODE_LIMIT + _FRAME_SLACK, (
        f"torchcodec materialized {materialized} PCM bytes before refusing "
        f"a {_DECODE_LIMIT}-byte decode budget"
    )


@pytest.fixture(params=[0, 480], ids=["missing-duration", "under-reported-duration"])
def forged_flac(request: pytest.FixtureRequest) -> bytes:
    pytest.importorskip("torchcodec")
    payload = _flac_with_declared_samples(request.param)
    _assert_metadata_understates_pcm(payload, request.param)
    return payload


def test_torchcodec_stops_before_full_pcm(forged_flac: bytes):
    """Native-rate decode, the path AudioMediaIO uses (``sr=None``)."""

    def decode():
        load_audio_torchcodec(
            BytesIO(forged_flac),
            sr=None,
            mono=True,
            max_duration_s=600,
            max_decode_bytes=_DECODE_LIMIT,
        )

    _assert_refused_within_budget(decode)


def test_auto_backend_stops_before_full_pcm(forged_flac: bytes):
    """``backend='auto'`` prefers torchcodec when it is installed."""

    def decode():
        load_audio(
            BytesIO(forged_flac),
            sr=None,
            mono=True,
            max_duration_s=600,
            max_decode_bytes=_DECODE_LIMIT,
            backend="auto",
        )

    _assert_refused_within_budget(decode)


def test_explicit_torchcodec_backend_stops_before_full_pcm(forged_flac: bytes):
    def decode():
        load_audio(
            BytesIO(forged_flac),
            sr=None,
            mono=True,
            max_duration_s=600,
            max_decode_bytes=_DECODE_LIMIT,
            backend="torchcodec",
        )

    _assert_refused_within_budget(decode)


def test_unbounded_duration_stops_before_full_pcm(forged_flac: bytes):
    """``max_duration_s=None`` decodes through ``get_all_samples``."""

    def decode():
        load_audio_torchcodec(
            BytesIO(forged_flac),
            sr=None,
            mono=True,
            max_duration_s=None,
            max_decode_bytes=_DECODE_LIMIT,
        )

    _assert_refused_within_budget(decode)


def test_speech_sample_rate_stops_before_full_pcm(forged_flac: bytes):
    """Speech-to-text calls ``load_audio`` at ``SpeechToTextConfig.sample_rate``.

    Resampling inside torchcodec still returns the whole window before the
    byte check. 16 kHz is the config default.
    """

    def decode():
        load_audio(
            BytesIO(forged_flac),
            sr=16_000,
            mono=True,
            max_duration_s=600,
            max_decode_bytes=_DECODE_LIMIT,
            backend="torchcodec",
        )

    _assert_refused_within_budget(decode)


def test_audio_media_io_load_bytes_stops_before_full_pcm(
    forged_flac: bytes, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_BYTES", _DECODE_LIMIT)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 600)

    def decode():
        AudioMediaIO().load_bytes(forged_flac)

    _assert_refused_within_budget(decode)


def test_audio_media_io_load_base64_stops_before_full_pcm(
    forged_flac: bytes, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_BYTES", _DECODE_LIMIT)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 600)
    encoded = pybase64.b64encode(forged_flac).decode("ascii")

    def decode():
        AudioMediaIO().load_base64("audio/flac", encoded)

    _assert_refused_within_budget(decode)


def test_audio_media_io_load_file_stops_before_full_pcm(
    forged_flac: bytes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_BYTES", _DECODE_LIMIT)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 600)
    path = tmp_path / "clip.flac"
    path.write_bytes(forged_flac)

    def decode():
        AudioMediaIO().load_file(path)

    _assert_refused_within_budget(decode)


def test_honest_metadata_rejects_before_torchcodec_allocates():
    """A true over-budget duration is refused from metadata, with no decode."""
    pytest.importorskip("torchcodec")
    buffer = BytesIO()
    frames = 100_000
    sf.write(
        buffer,
        np.zeros((frames, _CHANNELS), dtype=np.int16),
        _SAMPLE_RATE,
        format="FLAC",
    )
    payload = buffer.getvalue()
    recorded: list[int] = []
    with (
        _capture_torchcodec_pcm(recorded),
        pytest.raises(ValueError, match="would allocate"),
    ):
        load_audio_torchcodec(
            BytesIO(payload),
            sr=None,
            mono=True,
            max_duration_s=600,
            max_decode_bytes=_DECODE_LIMIT,
        )
    assert recorded == []


def test_small_clip_still_decodes():
    pytest.importorskip("torchcodec")
    buffer = BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.int16), 16_000, format="FLAC")
    audio, sample_rate = load_audio_torchcodec(
        BytesIO(buffer.getvalue()),
        sr=None,
        mono=True,
        max_duration_s=600,
        max_decode_bytes=_DECODE_LIMIT,
    )
    assert sample_rate == 16_000
    assert audio.nbytes <= _DECODE_LIMIT
    assert audio.shape[-1] == 1600


@pytest.mark.parametrize("declared_samples", [0, 480])
def test_pyav_stops_during_decode(declared_samples: int):
    """PyAV already aborts inside the frame loop. Keep that property."""
    payload = _flac_with_declared_samples(declared_samples)
    with pytest.raises(ValueError, match=r"decoded so far") as exc_info:
        load_audio_pyav(
            BytesIO(payload),
            sr=None,
            mono=False,
            max_duration_s=600,
            max_decode_bytes=_DECODE_LIMIT,
        )
    message = str(exc_info.value)
    assert "6 MiB" not in message


@pytest.mark.parametrize("declared_samples", [0, 480])
def test_soundfile_does_not_expand_past_declared_frames(declared_samples: int):
    """Libsndfile follows the STREAMINFO frame count.

    A zero count makes the duration guard reject the file. A short count
    returns only those frames. Neither path materializes the undeclared
    packets, so this is not the torchcodec bug.
    """
    payload = _flac_with_declared_samples(declared_samples)
    try:
        audio, _sample_rate = load_audio_soundfile(
            BytesIO(payload),
            sr=None,
            mono=False,
            max_duration_s=600,
            max_decode_bytes=_DECODE_LIMIT,
        )
    except ValueError:
        return
    assert audio.nbytes <= _DECODE_LIMIT
