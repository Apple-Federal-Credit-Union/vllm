# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenCV must not reserve a header-sized frame buffer before the request budget.

``_read_frames_no_recovery`` and ``Glm5NextVideoBackend.read_frames`` size a
dense uint8 array from the container width, height, and the sampled frame
count. ``VideoMediaIO`` charges ``VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES`` only
for the array it gets back. A short container whose header overstates that
sample keeps the full allocation in the returned view, and a clip that does
decode still reserves the buffer before the budget raises.

These tests fail while that order stands. They pass when a sample whose
``width * height * sample_count * 3`` exceeds the request decode budget is
rejected before any buffer that large exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pybase64
import pytest

import vllm.envs as envs
from vllm.multimodal.media.base import MediaDecodedBudget
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.media.video import VideoMediaIO
from vllm.multimodal.video import (
    Glm5NextVideoBackend,
    Qwen3VLVideoBackend,
    VideoBackend,
)
from vllm.multimodal.video_decoders.base import VideoTargetMetadata
from vllm.multimodal.video_decoders.opencv import OpenCVVideoBackendMixin

pytestmark = pytest.mark.cpu_test

_BOX_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}
_BUDGET_ERROR = "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"


def _walk_boxes(buf: bytearray, start: int, end: int):
    index = start
    while index + 8 <= end:
        size = int.from_bytes(buf[index : index + 4], "big")
        box_type = bytes(buf[index + 4 : index + 8])
        header = 8
        if size == 1:
            if index + 16 > end:
                return
            size = int.from_bytes(buf[index + 8 : index + 16], "big")
            header = 16
        if size < header or index + size > end:
            return
        yield box_type, index, size, header
        if box_type in _BOX_CONTAINERS:
            yield from _walk_boxes(buf, index + header, index + size)
        index += size


def _write_clip(path: Path, *, width: int, height: int, frames: int) -> None:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 180)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create {path}")
    try:
        for _ in range(frames):
            writer.write(frame)
    finally:
        writer.release()


def _inflate_header(payload: bytes, *, width: int, height: int, frames: int) -> bytes:
    """Point tkhd, the visual sample entry, stts, and stsz at ``frames``.

    The stsz sample-count field is updated without adding sample sizes, which
    is the container OpenCV reports as ``frames`` wide samples while ``grab``
    returns nothing. Metadata seeking does not correct that count.
    """
    buf = bytearray(payload)
    patched_stts = patched_stsz = patched_tkhd = patched_stsd = False
    for box_type, index, _size, header in _walk_boxes(buf, 0, len(buf)):
        if box_type == b"tkhd":
            if buf[index + header] != 0:
                raise AssertionError("expected a version-0 tkhd")
            buf[index + 84 : index + 88] = (width << 16).to_bytes(4, "big")
            buf[index + 88 : index + 92] = (height << 16).to_bytes(4, "big")
            patched_tkhd = True
        elif box_type == b"stsd":
            entry = index + header + 8
            buf[entry + 32 : entry + 34] = width.to_bytes(2, "big")
            buf[entry + 34 : entry + 36] = height.to_bytes(2, "big")
            patched_stsd = True
        elif box_type == b"stts":
            entry_count = int.from_bytes(
                buf[index + header + 4 : index + header + 8], "big"
            )
            if entry_count != 1:
                raise AssertionError(f"expected one stts entry, found {entry_count}")
            buf[index + header + 8 : index + header + 12] = frames.to_bytes(4, "big")
            patched_stts = True
        elif box_type == b"stsz":
            buf[index + header + 8 : index + header + 12] = frames.to_bytes(4, "big")
            patched_stsz = True
    if not (patched_tkhd and patched_stsd and patched_stts and patched_stsz):
        raise AssertionError("mp4 is missing tkhd, stsd, stts, or stsz")
    return bytes(buf)


def _header_clip(
    tmp_path: Path, *, width: int, height: int, claimed_frames: int
) -> bytes:
    path = tmp_path / "clip.mp4"
    _write_clip(path, width=64, height=64, frames=2)
    payload = _inflate_header(
        path.read_bytes(), width=width, height=height, frames=claimed_frames
    )
    source = OpenCVVideoBackendMixin.get_video_metadata(payload)
    if source.total_frames_num != claimed_frames:
        raise AssertionError(
            f"header frame count {source.total_frames_num} != {claimed_frames}"
        )
    return payload


def _honest_clip(tmp_path: Path, *, width: int, height: int, frames: int) -> bytes:
    path = tmp_path / "honest.mp4"
    _write_clip(path, width=width, height=height, frames=frames)
    return path.read_bytes()


def _data_url(payload: bytes) -> str:
    encoded = pybase64.b64encode(payload).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def _backing_nbytes(frames: np.ndarray) -> int:
    owner = frames
    for _ in range(8):
        base = owner.base
        if not isinstance(base, np.ndarray):
            return int(owner.nbytes)
        owner = base
    return int(owner.nbytes)


@contextmanager
def _record_numpy_buffers() -> Iterator[list[tuple[tuple, int]]]:
    """Record buffers NumPy reserves while the decoder runs."""
    reserved: list[tuple[tuple, int]] = []
    real_empty = np.empty
    real_zeros = np.zeros
    real_stack = np.stack

    def empty(shape, *args, **kwargs):
        array = real_empty(shape, *args, **kwargs)
        reserved.append((tuple(array.shape), int(array.nbytes)))
        return array

    def zeros(shape, *args, **kwargs):
        array = real_zeros(shape, *args, **kwargs)
        reserved.append((tuple(array.shape), int(array.nbytes)))
        return array

    def stack(arrays, *args, **kwargs):
        array = real_stack(arrays, *args, **kwargs)
        reserved.append((tuple(array.shape), int(array.nbytes)))
        return array

    np.empty = empty
    np.zeros = zeros
    np.stack = stack
    try:
        yield reserved
    finally:
        np.empty = real_empty
        np.zeros = real_zeros
        np.stack = real_stack


def _assert_budget_rejects_before_reserve(
    load: Callable[[], object],
    limit: int,
) -> None:
    with _record_numpy_buffers() as reserved:
        try:
            result = load()
        except ValueError as exc:
            message = str(exc)
            assert _BUDGET_ERROR in message, message
        else:
            frames = result[0]
            assert isinstance(frames, np.ndarray)
            retained = _backing_nbytes(frames)
            raise AssertionError(
                f"video decode returned a {frames.nbytes}-byte view backed by "
                f"{retained} bytes; the request decode budget is {limit} bytes"
            )
        too_big = [item for item in reserved if item[1] > limit]
    assert not too_big, (
        f"reserved {too_big} before rejecting the {_limit_phrase(limit)} budget"
    )


def _limit_phrase(limit: int) -> str:
    return f"{limit}-byte"


def _planned_bytes(indices: list[int], width: int, height: int) -> int:
    return len(indices) * width * height * 3


def test_data_url_rejects_opencv_header_sample_before_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Default OpenCV loader, data URL, header sample larger than the budget."""
    width, height, claimed = 320, 240, 32
    limit = 1_000_000
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", limit)
    monkeypatch.setattr(envs, "VLLM_VIDEO_LOADER_BACKEND", "opencv")
    payload = _header_clip(tmp_path, width=width, height=height, claimed_frames=claimed)
    source = OpenCVVideoBackendMixin.get_video_metadata(payload)
    indices = VideoBackend.compute_frames_index_to_sample(
        source, VideoTargetMetadata(num_frames=claimed, fps=-1, max_duration=300)
    )
    assert _planned_bytes(indices, width, height) > limit
    assert width * height <= envs.VLLM_MAX_IMAGE_PIXELS

    connector = MediaConnector(decoded_budget=MediaDecodedBudget())
    _assert_budget_rejects_before_reserve(
        lambda: connector.fetch_video(_data_url(payload)),
        limit,
    )


def test_async_data_url_rejects_opencv_header_sample_before_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Async fetch uses the same OpenCV reader as the chat parser."""
    width, height, claimed = 320, 240, 32
    limit = 1_000_000
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", limit)
    monkeypatch.setattr(envs, "VLLM_VIDEO_LOADER_BACKEND", "opencv")
    payload = _header_clip(tmp_path, width=width, height=height, claimed_frames=claimed)
    connector = MediaConnector(decoded_budget=MediaDecodedBudget())

    async def load():
        return await connector.fetch_video_async(_data_url(payload))

    _assert_budget_rejects_before_reserve(
        lambda: asyncio.run(load()),
        limit,
    )


def test_qwen3_vl_opencv_rejects_header_sample_before_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Qwen3-VL sampling asks for up to 768 header frames and uses OpenCV."""
    width, height, claimed = 64, 64, 768
    limit = 1_000_000
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", limit)
    payload = _header_clip(tmp_path, width=width, height=height, claimed_frames=claimed)
    source = OpenCVVideoBackendMixin.get_video_metadata(payload)
    indices = Qwen3VLVideoBackend.compute_frames_index_to_sample(
        source, VideoTargetMetadata(num_frames=-1, fps=2, max_duration=300)
    )
    assert len(indices) == claimed
    assert _planned_bytes(indices, width, height) > limit

    video_io = VideoMediaIO(
        ImageMediaIO(decoded_budget=MediaDecodedBudget()),
        video_backend="qwen3_vl",
    )
    _assert_budget_rejects_before_reserve(
        lambda: video_io.load_bytes(payload),
        limit,
    )


def test_glm5next_opencv_rejects_header_sample_before_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """GLM-5 reader preallocates its own header-sized buffer before decode."""
    width, height, claimed = 320, 240, 32
    limit = 1_000_000
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", limit)
    payload = _header_clip(tmp_path, width=width, height=height, claimed_frames=claimed)
    source = OpenCVVideoBackendMixin.get_video_metadata(payload)
    indices = Glm5NextVideoBackend.compute_frames_index_to_sample(
        source, VideoTargetMetadata(num_frames=-1, fps=2, max_duration=300)
    )
    assert _planned_bytes(indices, width, height) > limit

    video_io = VideoMediaIO(
        ImageMediaIO(decoded_budget=MediaDecodedBudget()),
        video_backend="glm5next",
        fps=2,
    )
    _assert_budget_rejects_before_reserve(
        lambda: video_io.load_bytes(payload),
        limit,
    )


def test_decoded_clip_over_budget_is_rejected_before_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A clip that fully decodes still has to fail before the sample buffer."""
    width, height, frames = 80, 80, 8
    limit = 8_192
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", limit)
    monkeypatch.setattr(envs, "VLLM_VIDEO_LOADER_BACKEND", "opencv")
    payload = _honest_clip(tmp_path, width=width, height=height, frames=frames)
    assert width * height * frames * 3 > limit

    connector = MediaConnector(decoded_budget=MediaDecodedBudget())
    _assert_budget_rejects_before_reserve(
        lambda: connector.fetch_video(_data_url(payload)),
        limit,
    )


def test_decoded_clip_within_budget_keeps_only_its_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A sample inside the budget still decodes and does not hide a larger base."""
    width, height, frames = 64, 64, 2
    limit = 1_000_000
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", limit)
    monkeypatch.setattr(envs, "VLLM_VIDEO_LOADER_BACKEND", "opencv")
    payload = _honest_clip(tmp_path, width=width, height=height, frames=frames)
    budget = MediaDecodedBudget()
    connector = MediaConnector(decoded_budget=budget)

    frames_out = connector.fetch_video(_data_url(payload))[0]
    assert isinstance(frames_out, np.ndarray)
    assert frames_out.shape[0] == frames
    assert _backing_nbytes(frames_out) == int(frames_out.nbytes)
    assert int(frames_out.nbytes) <= limit
    assert budget.used == int(frames_out.nbytes)
