# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec for the OpenCV video loader's unbounded frame walk.

These tests fail until the loader refuses a container whose claimed length
exceeds ``MAX_CONTAINER_FRAME_WALK`` and stops grabbing once that bound is
hit. A refusal that happens only after the walk is still a failure.

``ClefOmniVideoBackend`` is not covered: its sampler raises once the index
list passes 768 frames, before ``read_frames``. TorchCodec, PyNvVideoCodec,
and DeepStream do not use this OpenCV reader. ``fetch_video_async`` and
``VideoMediaIO.load_file`` call the same ``load_bytes`` as the data-URL test.

``MiniMaxM3VideoBackend``, ``MuseSpark1xOmniVideoBackend``, and
``LlavaOnevision2VideoBackend`` are not separate cases. Each keeps the
inherited OpenCV reader and builds indexes with ``linspace`` through the last
claimed frame (``mm_preprocess.py``, ``processor.py``, and
``llava_onevision2.py``). The ``opencv`` case is that reader.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from io import BytesIO

import cv2
import pybase64
import pytest

import vllm.envs as envs
import vllm.multimodal.video_decoders.opencv as opencv_decoder
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.media.video import VideoMediaIO
from vllm.multimodal.video import (
    DynamicVideoBackend,
    EmbeddingGemma2VideoBackend,
    Glm5NextVideoBackend,
    GLM46VVideoBackend,
    GLMGAVideoBackend,
    Molmo2VideoBackend,
    NemotronVLVideoBackend,
    OpenCVDynamicOpenPanguVideoBackend,
    Qwen2VLVideoBackend,
    Qwen3VLVideoBackend,
    VideoBackend,
)

pytestmark = pytest.mark.cpu_test

# Largest frame cap already used by an OpenCV sampler in this tree
# (Glm5NextVideoBackend._DEFAULT_MAX_FRAMES). A claimed length above this
# must be refused, and the grab walk must stop at the cap.
MAX_CONTAINER_FRAME_WALK = 2048
CLAIMED_FRAMES = 8192
HEADERLESS_FRAMES = 2500

_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}
Loader = Callable[[bytes], object]


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.fail("ffmpeg is required to build the video fixtures")
    return ffmpeg


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.run(
        [_require_ffmpeg(), "-y", *args],
        check=True,
        capture_output=True,
    )


def _find_box(buf: bytearray, typ: bytes) -> int | None:
    def walk(start: int, end: int) -> int | None:
        pos = start
        while pos + 8 <= end:
            size = int.from_bytes(buf[pos : pos + 4], "big")
            name = bytes(buf[pos + 4 : pos + 8])
            header = 8
            if size == 1:
                size = int.from_bytes(buf[pos + 8 : pos + 16], "big")
                header = 16
            if size < header:
                return None
            if name == typ:
                return pos
            if name in _CONTAINER_BOXES:
                found = walk(pos + header, pos + size)
                if found is not None:
                    return found
            pos += size
        return None

    return walk(0, len(buf))


def _set_u32(buf: bytearray, offset: int, value: int) -> None:
    buf[offset : offset + 4] = int(value).to_bytes(4, "big")


def _claim_frame_count(data: bytes, claim: int, *, zero_duration: bool) -> bytes:
    """Point the stts sample count at ``claim``.

    With ``zero_duration``, also zero the container duration fields. OpenCV
    still reports ``claim`` from ``nb_frames``, and an end-of-stream seek then
    reports position 0, which skips the loader's presentable-frame check.
    """
    buf = bytearray(data)
    stts = _find_box(buf, b"stts")
    assert stts is not None
    assert buf[stts + 8] == 0  # version 0
    assert int.from_bytes(buf[stts + 12 : stts + 16], "big") == 1
    _set_u32(buf, stts + 16, claim)
    if zero_duration:
        for name, duration_at in (
            (b"mvhd", 24),
            (b"tkhd", 28),
            (b"mdhd", 24),
            (b"elst", 16),
        ):
            start = _find_box(buf, name)
            assert start is not None
            assert buf[start + 8] == 0
            _set_u32(buf, start + duration_at, 0)
    return bytes(buf)


def _opencv_frame_count(data: bytes) -> int:
    cap = cv2.VideoCapture(BytesIO(data), cv2.CAP_FFMPEG, [])
    assert cap.isOpened()
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def _end_seek_keeps_claim(data: bytes, claim: int) -> bool:
    """True when get_video_metadata's presentable-frame check would not shrink."""
    cap = cv2.VideoCapture(BytesIO(data), cv2.CAP_FFMPEG, [])
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == claim
    if not cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0):
        return True
    visible = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    at_end = not cap.grab()
    return not (at_end and 0 < visible < claim - 1)


class _GrabCounter:
    def __init__(self) -> None:
        self.n = 0

    def add(self) -> None:
        self.n += 1


class _counting_grabs:
    def __init__(self) -> None:
        self.counter = _GrabCounter()
        self._original = opencv_decoder.cv2.VideoCapture

    def __enter__(self) -> _GrabCounter:
        counter = self.counter
        original = self._original

        class _Bound:
            def __init__(self, *args, **kwargs) -> None:
                self._cap = original(*args, **kwargs)

            def grab(self) -> bool:
                counter.add()
                return self._cap.grab()

            def read(self, *args, **kwargs):
                counter.add()
                return self._cap.read(*args, **kwargs)

            def __getattr__(self, name: str):
                return getattr(self._cap, name)

        opencv_decoder.cv2.VideoCapture = _Bound
        return counter

    def __exit__(self, *exc) -> None:
        opencv_decoder.cv2.VideoCapture = self._original


@pytest.fixture(scope="module")
def short_mp4(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("video") / "short.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=10:d=0.3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    data = path.read_bytes()
    count = _opencv_frame_count(data)
    assert 1 <= count < 32
    assert len(data) < 64 * 1024
    return data


@pytest.fixture(scope="module")
def inflated_zero_duration(short_mp4: bytes) -> bytes:
    data = _claim_frame_count(short_mp4, CLAIMED_FRAMES, zero_duration=True)
    assert len(data) < envs.VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB * 1024 * 1024
    assert _opencv_frame_count(data) == CLAIMED_FRAMES
    assert _end_seek_keeps_claim(data, CLAIMED_FRAMES)
    return data


@pytest.fixture(scope="module")
def headerless_h264(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("video") / "stream.h264"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=25:d=100",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-f",
            "h264",
            str(path),
        ]
    )
    data = path.read_bytes()
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(probe.stdout.strip()) == HEADERLESS_FRAMES
    assert _opencv_frame_count(data) <= 0
    assert len(data) < envs.VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB * 1024 * 1024
    return data


def _assert_refused_within_cap(loader: Loader, data: bytes) -> None:
    raised: ValueError | None = None
    with _counting_grabs() as grabs:
        try:
            loader(data)
        except ValueError as exc:
            raised = exc
    assert grabs.n <= MAX_CONTAINER_FRAME_WALK, (
        f"loader issued {grabs.n} grab/read calls; cap is {MAX_CONTAINER_FRAME_WALK}"
    )
    assert raised is not None, "oversized frame claim was decoded instead of refused"


def _load_media_io(data: bytes):
    return VideoMediaIO(ImageMediaIO(), num_frames=4).load_bytes(data)


def _load_data_url(data: bytes):
    encoded = pybase64.b64encode(data).decode("ascii")
    return MediaConnector().fetch_video(f"data:video/mp4;base64,{encoded}")


def _opencv_loaders() -> list[tuple[str, Loader]]:
    return [
        ("opencv", lambda data: VideoBackend.load_bytes(data, num_frames=4)),
        (
            "opencv_frame_recovery",
            lambda data: VideoBackend.load_bytes(
                data, num_frames=4, frame_recovery=True
            ),
        ),
        ("video_media_io", _load_media_io),
        ("data_url", _load_data_url),
        (
            "qwen3_vl",
            lambda data: Qwen3VLVideoBackend.load_bytes(data, num_frames=4, fps=2),
        ),
        (
            "qwen2_vl",
            lambda data: Qwen2VLVideoBackend.load_bytes(data, num_frames=4, fps=2),
        ),
        (
            "opencv_dynamic",
            lambda data: DynamicVideoBackend.load_bytes(
                data, num_frames=4, fps=2, max_duration=300
            ),
        ),
        (
            "glm46v",
            lambda data: GLM46VVideoBackend.load_bytes(data, num_frames=4, fps=2),
        ),
        (
            "glmga",
            lambda data: GLMGAVideoBackend.load_bytes(data, num_frames=4, fps=2),
        ),
        (
            "embedding_gemma2",
            lambda data: EmbeddingGemma2VideoBackend.load_bytes(
                data, fps=1, max_frames=8
            ),
        ),
        (
            "molmo2",
            lambda data: Molmo2VideoBackend.load_bytes(
                data,
                num_frames=8,
                frame_sample_mode="uniform_last_frame",
                max_fps=2,
            ),
        ),
        (
            "openpangu",
            lambda data: OpenCVDynamicOpenPanguVideoBackend.load_bytes(
                data, num_frames=4, fps=2
            ),
        ),
        (
            "nemotron_vl",
            lambda data: NemotronVLVideoBackend.load_bytes(data, num_frames=4),
        ),
        ("glm5next", lambda data: Glm5NextVideoBackend.load_bytes(data, fps=2)),
        (
            "glm5next_frame_recovery",
            lambda data: Glm5NextVideoBackend.load_bytes(
                data, fps=2, frame_recovery=True
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("name", "loader"),
    _opencv_loaders(),
    ids=[name for name, _ in _opencv_loaders()],
)
def test_inflated_frame_count_is_refused_without_walking_the_claim(
    name: str, loader: Loader, inflated_zero_duration: bytes
) -> None:
    del name
    _assert_refused_within_cap(loader, inflated_zero_duration)


def test_headerless_stream_demux_is_capped(headerless_h264: bytes) -> None:
    _assert_refused_within_cap(
        lambda data: VideoBackend.load_bytes(data, num_frames=4),
        headerless_h264,
    )


def test_short_video_still_samples(short_mp4: bytes) -> None:
    with _counting_grabs() as grabs:
        frames, metadata = VideoBackend.load_bytes(short_mp4, num_frames=2)
    assert frames.shape[0] == 2
    assert metadata["total_num_frames"] == _opencv_frame_count(short_mp4)
    assert grabs.n <= 32


def test_duration_bearing_overclaim_does_not_walk_the_claim(short_mp4: bytes) -> None:
    """A positive container duration lets the end-seek shrink the claim.

    This shape is already bounded. Zeroing the duration is what keeps the
    claim, and that case is covered above.
    """
    data = _claim_frame_count(short_mp4, CLAIMED_FRAMES, zero_duration=False)
    assert _opencv_frame_count(data) == CLAIMED_FRAMES
    assert not _end_seek_keeps_claim(data, CLAIMED_FRAMES)
    with _counting_grabs() as grabs:
        frames, _metadata = VideoBackend.load_bytes(data, num_frames=2)
    assert frames.shape[0] == 2
    assert grabs.n <= 32
