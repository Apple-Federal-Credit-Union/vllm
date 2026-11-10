# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Image and video data URLs must honor the encoded media size cap.

HTTP fetches stop at ``VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB`` (or a tighter
``MediaIO.get_max_bytes()``). ``data:`` image and video URLs must reject the
same payloads before base64-decoding them. These tests fail while decoding
still runs on an oversized body.
"""

import inspect
from collections.abc import Awaitable, Callable
from io import BytesIO
from unittest.mock import patch

import pybase64
import pytest
from PIL import Image

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import ImageMediaIO, MediaConnector, VideoMediaIO
from vllm.utils.mem_constants import MiB_bytes

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

_SIZE_LIMIT_ERROR = (ValueError, VLLMValidationError)
_SIZE_LIMIT_MATCH = r"VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB|Maximum file size exceeded"
_Loader = Callable[[], object | Awaitable[object]]


def _base64_at_least(decoded_size: int) -> str:
    """Unpadded base64 whose decoded length is at least ``decoded_size``."""
    chars = ((decoded_size + 2) // 3) * 4
    return "A" * chars


def _data_url(media_type: str, payload: str, *, scheme: str = "data") -> str:
    return f"{scheme}:{media_type};charset=utf-8;base64,{payload}"


async def _assert_rejected_before_base64_decode(load: _Loader) -> None:
    """The size error must be raised without calling the base64 decoder.

    The spy delegates to the real decoder, so a check that decodes and then
    raises still fails this assertion.
    """
    calls = 0
    real_decode = pybase64.b64decode

    def spy(data, *args, **kwargs):
        nonlocal calls
        calls += 1
        return real_decode(data, *args, **kwargs)

    with (
        patch("vllm.multimodal.media.image.pybase64.b64decode", spy),
        pytest.raises(_SIZE_LIMIT_ERROR, match=_SIZE_LIMIT_MATCH),
    ):
        result = load()
        if inspect.isawaitable(result):
            await result
    assert calls == 0, "base64 decode ran before the encoded-size limit"


async def _load(connector: MediaConnector, method: str, url: str):
    loader = getattr(connector, method)
    result = loader(url)
    if inspect.isawaitable(result):
        return await result
    return result


@pytest.fixture
def one_mebibyte_download_cap(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 1, raising=False)
    return MiB_bytes


def _tiny_png_data_url() -> str:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(buffer, format="PNG")
    encoded = pybase64.b64encode(buffer.getvalue()).decode("ascii")
    return _data_url("image/png", encoded)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["fetch_image", "fetch_image_async"])
async def test_image_data_url_within_download_cap_still_loads(
    one_mebibyte_download_cap: int,
    method: str,
):
    """A data URL under the configured cap is still decoded."""
    image = await _load(MediaConnector(), method, _tiny_png_data_url())
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (10, 20, 30)
    assert len(image.original_bytes) < one_mebibyte_download_cap


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["fetch_image", "fetch_image_async"])
async def test_image_data_url_over_download_cap_rejected_before_decode(
    one_mebibyte_download_cap: int,
    method: str,
):
    """Mixed-case data URLs over the download cap never reach base64 decode."""
    url = _data_url(
        "image/png",
        _base64_at_least(one_mebibyte_download_cap + 1),
        scheme="DaTa",
    )

    def load():
        return _load(MediaConnector(), method, url)

    await _assert_rejected_before_base64_decode(load)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["fetch_video", "fetch_video_async"])
async def test_video_data_url_over_download_cap_rejected_before_decode(
    one_mebibyte_download_cap: int,
    method: str,
):
    """A video container data URL is capped before it is base64-decoded."""
    url = _data_url(
        "video/mp4",
        _base64_at_least(one_mebibyte_download_cap + 1),
        scheme="DaTa",
    )

    def load():
        return _load(MediaConnector(), method, url)

    await _assert_rejected_before_base64_decode(load)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["fetch_video", "fetch_video_async"])
@pytest.mark.parametrize("large_frame_first", [True, False])
async def test_video_jpeg_data_url_total_over_cap_rejected_before_decode(
    one_mebibyte_download_cap: int,
    method: str,
    large_frame_first: bool,
):
    """video/jpeg rejects before any frame is decoded.

    Each frame is under the cap. Their decoded sizes together are over it.
    Checking only the first or only the last comma-separated frame must not
    pass, because both frames are what this data URL decodes.
    """
    cap = one_mebibyte_download_cap
    smaller = _base64_at_least((cap * 2) // 5 + 3)
    larger = _base64_at_least((cap * 3) // 5 + 3)
    frames = (larger, smaller) if large_frame_first else (smaller, larger)
    assert len(pybase64.b64decode(smaller)) < cap
    assert len(pybase64.b64decode(larger)) < cap
    decoded_total = len(pybase64.b64decode(smaller)) + len(pybase64.b64decode(larger))
    assert decoded_total > cap
    url = _data_url("video/jpeg", ",".join(frames))

    def load():
        return _load(MediaConnector(), method, url)

    await _assert_rejected_before_base64_decode(load)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "media_io_cls"),
    [
        ("fetch_image", ImageMediaIO),
        ("fetch_image_async", ImageMediaIO),
        ("fetch_video", VideoMediaIO),
        ("fetch_video_async", VideoMediaIO),
    ],
)
async def test_data_url_honors_tighter_modality_cap_before_decode(
    one_mebibyte_download_cap: int,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    media_io_cls: type[ImageMediaIO] | type[VideoMediaIO],
):
    """A per-modality cap below the download cap applies to the whole data URL."""
    modality_cap = 32
    assert modality_cap < one_mebibyte_download_cap
    monkeypatch.setattr(
        media_io_cls,
        "get_max_bytes",
        lambda self: modality_cap,
        raising=False,
    )
    media_type = "image/png" if media_io_cls is ImageMediaIO else "video/mp4"
    url = _data_url(media_type, _base64_at_least(modality_cap + 1))

    def load():
        return _load(MediaConnector(), method, url)

    await _assert_rejected_before_base64_decode(load)
