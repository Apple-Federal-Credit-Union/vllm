# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ovis2.5 startup profiling must reject a vision grid that cannot grow.

``vit_config.patch_size`` and ``temporal_patch_size`` come from the
checkpoint ``config.json``. Profiling counts dummy video frames until the
token count exceeds the context length. A grid that stays at zero, and a
temporal size that stops the token count from increasing, never crosses
that threshold.

These tests fail while that search can run without a bound.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parent / "ovis2_5_profile_worker.py"
REPO_ROOT = Path(__file__).resolve().parents[4]
PROFILE_TIMEOUT_S = 25

# A 1792-pixel side is what the profiler uses for every frame.
_ZERO_GRID_PATCH = 1793


def _run_profile(tmp_path: Path, spec: dict) -> tuple[dict, dict | None]:
    payload = dict(spec)
    payload["model_dir"] = str(tmp_path / "checkpoint")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, str(WORKER), json.dumps(payload)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=PROFILE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        loaded = _loaded_event(stdout)
        pytest.fail(
            "video profiling did not return within "
            f"{PROFILE_TIMEOUT_S}s after loading patch_size="
            f"{loaded.get('patch_size')!r} temporal_patch_size="
            f"{loaded.get('temporal_patch_size')!r}"
        )

    loaded = _loaded_event(completed.stdout)
    done = _done_event(completed.stdout)
    assert done is not None, completed.stderr[-2000:]
    return loaded, done


def _events(stdout: str) -> list[dict]:
    events = []
    for line in stdout.splitlines():
        if line.startswith("{"):
            events.append(json.loads(line))
    return events


def _loaded_event(stdout: str) -> dict:
    for event in _events(stdout):
        if event.get("event") == "loaded":
            return event
    return {}


def _done_event(stdout: str) -> dict | None:
    done = None
    for event in _events(stdout):
        if event.get("event") == "done":
            done = event
    return done


def _assert_call_rejected(name: str, outcome: dict) -> None:
    assert outcome["status"] == "error", (name, outcome)
    assert outcome["is_value_error"], (name, outcome)
    assert outcome["exc"] != "ZeroDivisionError"
    message = outcome["message"]
    assert "Failed to apply" not in message
    assert "PIL.Image" not in message
    assert any(
        token in message
        for token in ("patch_size", "temporal_patch_size", "positive", "grid")
    ), (name, message)


def _assert_rejected(loaded: dict, done: dict, spec: dict) -> None:
    assert loaded["patch_size"] == spec["patch_size"]
    assert loaded["temporal_patch_size"] == spec["temporal_patch_size"]
    calls = done["calls"]
    assert set(calls) == {
        "max_video_frames",
        "frames_per_video",
        "image_only_frames",
        "max_video_tokens",
        "dummy_image_only",
        "dummy_video",
        "startup_budget",
    }
    for name, outcome in calls.items():
        _assert_call_rejected(name, outcome)


@pytest.mark.parametrize(
    ("architecture", "patch_size", "temporal_patch_size", "max_model_len"),
    [
        ("Ovis2_5", 4096, 1, 32),
        ("Ovis2_5", _ZERO_GRID_PATCH, 1, 32),
        ("Ovis2_5", 1792.5, 1, 32),
        ("Ovis2_5", 0, 1, 32),
        ("Ovis2_5", -1, 1, 32),
        ("Ovis2_5", 16, 0, 32),
        ("Ovis2_5", 16, -1, 200_000),
        ("Ovis2_6_MoeForCausalLM", 4096, 1, 32),
    ],
)
def test_invalid_vision_grid_rejected_during_startup_profiling(
    tmp_path: Path,
    architecture: str,
    patch_size: int | float,
    temporal_patch_size: int,
    max_model_len: int,
) -> None:
    """Checkpoint vision sizes that cannot add tokens must be refused.

    The refusal has to happen in the frame search itself. Building dummy
    media and then failing in the image processor still keeps startup from
    finishing, and it does not bound the search.
    """
    spec = {
        "architecture": architecture,
        "patch_size": patch_size,
        "temporal_patch_size": temporal_patch_size,
        "max_model_len": max_model_len,
        "mode": "profile",
    }
    loaded, done = _run_profile(tmp_path, spec)
    assert done is not None
    _assert_rejected(loaded, done, spec)


def test_valid_vision_grid_profiles_within_context(tmp_path: Path) -> None:
    """A normal patch size still finishes profiling and stays in context."""
    spec = {
        "architecture": "Ovis2_5",
        "patch_size": 16,
        "temporal_patch_size": 1,
        "max_model_len": 32,
        "mode": "profile",
    }
    loaded, done = _run_profile(tmp_path, spec)
    assert done is not None
    assert loaded["patch_size"] == 16
    assert loaded["architecture"] == "Ovis2_5"
    calls = done["calls"]
    for name, outcome in calls.items():
        assert outcome["status"] == "ok", (name, outcome)
    limit = spec["max_model_len"]
    assert 0 <= calls["max_video_frames"]["value"] <= limit
    assert 1 <= calls["frames_per_video"]["value"] <= limit
    assert 1 <= calls["image_only_frames"]["value"] <= limit
    assert calls["max_video_tokens"]["value"] >= 1


def test_non_increasing_token_count_stops_frame_search(tmp_path: Path) -> None:
    """The frame search stops when each new frame adds no tokens.

    Validation of ``patch_size`` alone does not cover this. The search reads
    the token count on every iteration, including after a caller replaces
    that count with a value that stays at zero.
    """
    spec = {
        "architecture": "Ovis2_5",
        "patch_size": 16,
        "temporal_patch_size": 1,
        "max_model_len": 32,
        "mode": "non_increasing",
    }
    loaded, done = _run_profile(tmp_path, spec)
    assert done is not None
    assert loaded["patch_size"] == 16
    limit = spec["max_model_len"]
    frame_calls = {
        "max_video_frames",
        "frames_per_video",
        "image_only_frames",
        "dummy_image_only",
        "dummy_video",
        "startup_budget",
    }
    for name, outcome in done["calls"].items():
        if outcome["status"] == "ok":
            if name in frame_calls:
                assert outcome["value"] <= limit, (name, outcome)
            continue
        assert outcome["is_value_error"], (name, outcome)
        assert "Failed to apply" not in outcome["message"]
