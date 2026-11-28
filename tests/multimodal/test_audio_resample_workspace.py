# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client sample rates must not build unbounded resample workspaces.

Audio models resample a decoded clip to the feature-extractor rate, 16000 Hz
for Whisper-style models, with the torchaudio sinc kernel by default and with
``scipy.signal.resample_poly`` for Phi-4-multimodal. The source rate is the
rate in the uploaded WAV header. These tests send a short PCM WAV through
``MediaConnector.fetch_audio`` (the data-URL path used by chat ``input_audio``
and ``audio_url``) and then through ``MultiModalDataParser``.

A resample may reject the clip, or it may return audio at the target rate.
It must not request a single workspace larger than ``_WORKSPACE_LIMIT_BYTES``.
That limit sits well above the few-hundred-kilobyte kernel of a 44100 Hz to
16000 Hz resample and well below the tables built when the rate ratio does
not reduce. The same construction is what ``_get_torchaudio_resampler``
caches, so a workspace over the limit would also be pinned in that cache.
"""

import base64
import io
import math
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import soundfile
import torch
from torch.overrides import TorchFunctionMode

from vllm.exceptions import VLLMClientError
from vllm.multimodal.audio import _get_torchaudio_resampler
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.parse import MultiModalDataParser

_TARGET_SR = 16_000
_N_SAMPLES = 32
# 44.1 kHz -> 16 kHz builds a ~300 KiB float32 kernel. Coprime rates near
# 16 kHz build multi-gigabyte float64 workspaces.
_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024
_CLIENT_ERRORS = (ValueError, VLLMClientError)


class OversizedAudioWorkspace(RuntimeError):
    """A resample implementation asked for a workspace over the limit."""

    def __init__(self, nbytes: int, op: str) -> None:
        super().__init__(f"{op} requested {nbytes} bytes")
        self.nbytes = nbytes
        self.op = op


def _broadcast_nbytes(left: torch.Tensor, right: torch.Tensor) -> int | None:
    left_shape = (1,) * (right.ndim - left.ndim) + tuple(left.shape)
    right_shape = (1,) * (left.ndim - right.ndim) + tuple(right.shape)
    if len(left_shape) != len(right_shape):
        return None
    numel = 1
    for left_dim, right_dim in zip(left_shape, right_shape):
        if left_dim != right_dim and left_dim != 1 and right_dim != 1:
            return None
        numel *= max(left_dim, right_dim)
    return numel * max(left.element_size(), right.element_size())


class _TorchWorkspaceMode(TorchFunctionMode):
    """Run real torchaudio ops, but stop before a huge broadcast allocates."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        name = getattr(func, "__name__", "")
        if name in {"add", "mul", "sub"}:
            tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
            if len(tensors) >= 2:
                nbytes = _broadcast_nbytes(tensors[0], tensors[1])
                if nbytes is not None and nbytes > _WORKSPACE_LIMIT_BYTES:
                    raise OversizedAudioWorkspace(nbytes, name)
        return func(*args, **kwargs)


def _arange_count(args: tuple) -> int | None:
    if not args or len(args) > 3:
        return None
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        start, stop, step = args[0], args[1], 1
    else:
        start, stop, step = args[0], args[1], args[2]
    try:
        start_f, stop_f, step_f = float(start), float(stop), float(step)
    except (TypeError, ValueError):
        return None
    if step_f == 0:
        return None
    count = math.ceil((stop_f - start_f) / step_f)
    return max(int(count), 0)


def _arange_nbytes(args: tuple, kwargs: dict) -> int | None:
    count = _arange_count(args)
    if count is None:
        return None
    dtype = kwargs.get("dtype")
    if dtype is None:
        dtype = np.result_type(*args) if args else np.int_
    return count * np.dtype(dtype).itemsize


@contextmanager
def _bounded_workspace() -> Iterator[None]:
    original_arange = np.arange

    def _arange(*args, **kwargs):
        nbytes = _arange_nbytes(args, kwargs)
        if nbytes is not None and nbytes > _WORKSPACE_LIMIT_BYTES:
            raise OversizedAudioWorkspace(nbytes, "numpy.arange")
        return original_arange(*args, **kwargs)

    np.arange = _arange
    try:
        with _TorchWorkspaceMode():
            yield
    finally:
        np.arange = original_arange


def _wav_bytes(sample_rate: int) -> bytes:
    pcm = np.zeros(_N_SAMPLES, dtype=np.float32)
    pcm[0] = 0.1
    buffer = io.BytesIO()
    soundfile.write(buffer, pcm, sample_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _fetch_and_parse(
    sample_rate: int,
    method: str,
    *,
    target_channels: int | None,
):
    payload = base64.b64encode(_wav_bytes(sample_rate)).decode("ascii")
    # Same data-URL shape as MultiModalContentParser.parse_input_audio.
    audio_url = f"data:audio/wav;base64,{payload}"
    connector = MediaConnector(
        media_io_kwargs={"audio": {"audio_backend": "soundfile"}}
    )
    parser = MultiModalDataParser(
        target_sr=_TARGET_SR,
        target_channels=target_channels,
        audio_resample_method=method,
    )
    with _bounded_workspace():
        waveform, decoded_sr = connector.fetch_audio(audio_url)
        items = parser.parse_mm_data({"audio": [(waveform, decoded_sr)]})
    return items


def _assert_bounded_resample(
    sample_rate: int,
    method: str,
    *,
    target_channels: int | None,
) -> None:
    assert math.gcd(sample_rate, _TARGET_SR) == 1
    oversized: OversizedAudioWorkspace | None = None
    try:
        items = _fetch_and_parse(
            sample_rate,
            method,
            target_channels=target_channels,
        )
    except OversizedAudioWorkspace as exc:
        oversized = exc
        items = None
    except _CLIENT_ERRORS:
        return

    if oversized is not None:
        pytest.fail(
            f"{method} resample of a {sample_rate} Hz clip requested "
            f"a {oversized.nbytes}-byte workspace ({oversized.op})",
            pytrace=False,
        )

    assert items is not None
    audio = items["audio"].get(0)
    expected = _N_SAMPLES * _TARGET_SR / sample_rate
    # A no-op that returns the original PCM is not a resample to the target
    # rate. Four samples of slack covers backend length differences.
    assert audio.nbytes <= _WORKSPACE_LIMIT_BYTES
    assert abs(len(audio) - expected) <= 4


@pytest.fixture(autouse=True)
def _clear_torchaudio_resampler_cache():
    _get_torchaudio_resampler.cache_clear()
    yield
    _get_torchaudio_resampler.cache_clear()


@pytest.mark.parametrize("sample_rate", [16001, 44101])
def test_client_wav_does_not_build_oversized_torchaudio_kernel(sample_rate: int):
    # Default AudioResampler method. Whisper, Qwen2-Audio, Gemma3n and
    # Cohere ASR pass the feature-extractor rate and mono channel count.
    _assert_bounded_resample(
        sample_rate,
        "torchaudio",
        target_channels=1,
    )


@pytest.mark.parametrize("sample_rate", [500003, 1000003])
def test_scipy_resampler_does_not_build_oversized_fir(sample_rate: int):
    # Phi-4-multimodal selects this method. The FIR length is
    # 20 * max(reduced rates) + 1, which is unbounded in the source rate.
    _assert_bounded_resample(
        sample_rate,
        "scipy",
        target_channels=None,
    )


@pytest.mark.parametrize("sample_rate", [44100, 48000])
@pytest.mark.parametrize("method", ["torchaudio", "scipy"])
def test_standard_rates_still_resample(sample_rate: int, method: str):
    items = _fetch_and_parse(
        sample_rate,
        method,
        target_channels=1,
    )
    audio = items["audio"].get(0)
    expected = math.ceil(_N_SAMPLES * _TARGET_SR / sample_rate)
    assert abs(len(audio) - expected) <= 1
    assert np.isfinite(audio).all()
