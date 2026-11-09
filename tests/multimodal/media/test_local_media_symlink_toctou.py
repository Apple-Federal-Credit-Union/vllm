# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File-URL confinement must open the file that passed the check.

``MediaConnector._load_file_url`` resolves a ``file:`` path and requires the
allowed directory to be a parent, then ``load_file`` opens the original path.
These tests retarget that path in the window between the check and the read.
They fail while the read still follows the new symlink.
"""

import asyncio
import io
import os
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
import soundfile
from PIL import Image

from vllm.multimodal.media.connector import MediaConnector

pytestmark = pytest.mark.cpu_test

_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_BENIGN_AMPLITUDE = 0.25
_SECRET_AMPLITUDE = -0.75


def _path_of(file) -> Path | None:
    try:
        raw = os.fspath(file)
    except TypeError:
        return None
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _write_solid_video(path: Path, bgr: tuple[int, int, int]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (8, 8),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :] = bgr
    try:
        for _ in range(4):
            writer.write(frame)
    finally:
        writer.release()
    if path.stat().st_size == 0:
        raise RuntimeError(f"failed to write video fixture {path}")


class _MediaTree:
    """Allowed directory plus a file outside it, linked from inside."""

    def __init__(self, root: Path, modality: str) -> None:
        self.modality = modality
        self.allowed = root / "allowed"
        self.outside = root / "outside"
        self.allowed.mkdir()
        self.outside.mkdir()
        if modality == "image":
            self.benign = self.allowed / "benign.png"
            self.secret = self.outside / "secret.png"
            Image.new("RGB", (2, 2), _GREEN).save(self.benign)
            Image.new("RGB", (2, 2), _RED).save(self.secret)
        elif modality == "audio":
            self.benign = self.allowed / "benign.wav"
            self.secret = self.outside / "secret.wav"
            benign_wave = np.full(32, _BENIGN_AMPLITUDE, dtype=np.float32)
            secret_wave = np.full(32, _SECRET_AMPLITUDE, dtype=np.float32)
            soundfile.write(self.benign, benign_wave, 16000)
            soundfile.write(self.secret, secret_wave, 16000)
            self.benign_wave, self.benign_sr = soundfile.read(
                self.benign, dtype="float32"
            )
            self.secret_wave, _secret_sr = soundfile.read(self.secret, dtype="float32")
        elif modality == "video":
            self.benign = self.allowed / "benign.avi"
            self.secret = self.outside / "secret.avi"
            _write_solid_video(self.benign, (0, 255, 0))
            _write_solid_video(self.secret, (0, 0, 255))
        else:
            raise AssertionError(modality)
        self.link = self.allowed / f"link{self.benign.suffix}"
        self.link.symlink_to(self.benign)
        self.benign_bytes = self.benign.read_bytes()
        self.secret_bytes = self.secret.read_bytes()
        self.outside_names = sorted(path.name for path in self.outside.iterdir())

    def connector(self) -> MediaConnector:
        media_io_kwargs = None
        if self.modality == "audio":
            media_io_kwargs = {"audio": {"audio_backend": "soundfile"}}
        return MediaConnector(
            allowed_local_media_path=str(self.allowed),
            media_io_kwargs=media_io_kwargs,
        )

    def assert_outside_untouched(self) -> None:
        assert self.secret.read_bytes() == self.secret_bytes
        names = sorted(path.name for path in self.outside.iterdir())
        assert names == self.outside_names

    def assert_checked_payload(self, result) -> None:
        if self.modality == "audio":
            waveform, sample_rate = result
            got = np.asarray(waveform, dtype=np.float32)
            assert sample_rate == self.benign_sr
            assert np.allclose(got, self.benign_wave)
            assert not np.allclose(got, self.secret_wave)
            return
        assert result.original_bytes == self.benign_bytes
        assert result.original_bytes != self.secret_bytes
        if self.modality == "image":
            assert result.getpixel((0, 0)) == _GREEN


def _fetch(connector: MediaConnector, modality: str, url: str, use_async: bool):
    if modality == "image":
        if use_async:
            return asyncio.run(connector.fetch_image_async(url))
        return connector.fetch_image(url)
    if modality == "audio":
        if use_async:
            return asyncio.run(connector.fetch_audio_async(url))
        return connector.fetch_audio(url)
    if modality == "video":
        if use_async:
            return asyncio.run(connector.fetch_video_async(url))
        return connector.fetch_video(url)
    raise AssertionError(modality)


def _install_open_hooks(monkeypatch, on_open):
    real_io_open = io.open
    real_os_open = os.open
    real_sf_open = soundfile.SoundFile._open
    real_av_open = av.open

    def io_open(file, mode="r", *args, **kwargs):
        on_open(file)
        return real_io_open(file, mode, *args, **kwargs)

    def os_open(path, flags, *args, **kwargs):
        on_open(path)
        return real_os_open(path, flags, *args, **kwargs)

    def sf_open(self, file, mode_int, closefd):
        on_open(file)
        return real_sf_open(self, file, mode_int, closefd)

    def av_open(file, *args, **kwargs):
        on_open(file)
        return real_av_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", io_open)
    monkeypatch.setattr(os, "open", os_open)
    monkeypatch.setattr(soundfile.SoundFile, "_open", sf_open)
    monkeypatch.setattr(av, "open", av_open)


def _arm_retarget_after_check(monkeypatch, media: _MediaTree) -> dict[str, bool]:
    """Point the symlink outside after the check's resolve, before the read."""
    state = {"done": False}
    real_resolve = Path.resolve

    def retarget() -> None:
        if state["done"]:
            return
        state["done"] = True
        media.link.unlink()
        media.link.symlink_to(media.secret)

    def resolve(self, *args, **kwargs):
        resolved = real_resolve(self, *args, **kwargs)
        if self == media.link:
            retarget()
        return resolved

    def on_open(file) -> None:
        path = _path_of(file)
        if path == media.link:
            retarget()

    monkeypatch.setattr(Path, "resolve", resolve)
    _install_open_hooks(monkeypatch, on_open)
    return state


def _arm_swap_checked_file(monkeypatch, media: _MediaTree) -> dict[str, bool]:
    """Replace the checked file with an outside symlink at the open itself."""
    state = {"done": False}

    def on_open(file) -> None:
        if state["done"]:
            return
        path = _path_of(file)
        if path is None or path.parent != media.allowed:
            return
        if path.name not in {media.benign.name, media.link.name}:
            return
        state["done"] = True
        saved = media.benign.with_name(media.benign.name + ".authorized")
        os.rename(media.benign, saved)
        os.symlink(media.secret, media.benign)

    _install_open_hooks(monkeypatch, on_open)
    return state


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_file_url_returns_inode_checked_before_symlink_retarget(
    tmp_path, monkeypatch, modality: str, use_async: bool
):
    """Opening the pre-check path, or resolving it again, must not win."""
    media = _MediaTree(tmp_path, modality)
    connector = media.connector()
    state = _arm_retarget_after_check(monkeypatch, media)

    result = _fetch(connector, modality, media.link.as_uri(), use_async)

    assert state["done"]
    media.assert_outside_untouched()
    media.assert_checked_payload(result)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_file_url_open_follows_the_checked_directory_entry(
    tmp_path, monkeypatch, modality: str, use_async: bool
):
    """A fresh lookup that still passes must not read a swapped directory entry."""
    media = _MediaTree(tmp_path, modality)
    connector = media.connector()
    state = _arm_swap_checked_file(monkeypatch, media)

    try:
        result = _fetch(connector, modality, media.link.as_uri(), use_async)
    except Exception:
        media.assert_outside_untouched()
        assert state["done"]
        return

    media.assert_outside_untouched()
    assert state["done"]
    media.assert_checked_payload(result)
