# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunASR must not open a checkpoint-chosen CMVN path outside the model dir.

The preprocessor config is publisher-controlled data. ``get_processor`` loads
it with ``trust_remote_code=False``. These tests fail while ``load_cmvn``
still opens that path with an unbounded ``readlines``.
"""

import ctypes
import json
import os
import resource
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm.transformers_utils.processor import get_processor
from vllm.transformers_utils.processors.funasr import FunASRFeatureExtractor

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALID_CMVN = (
    "<AddShift>\n<LearnRateCoef> 1 1 0.1 ]\n<Rescale>\n<LearnRateCoef> 1 1 1.0 ]\n"
)
_READ_BUDGET = 1_048_576
_IN_OPEN = 0x00000020
_IN_NONBLOCK = 0x00000800
_SETUP_ERRORS = {"ImportError", "ModuleNotFoundError"}


class _Inotify:
    """Record IN_OPEN events on specific inodes. Observes the real open."""

    def __init__(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_init1.restype = ctypes.c_int
        libc.inotify_add_watch.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        libc.inotify_add_watch.restype = ctypes.c_int
        self._libc = libc
        self._fd = libc.inotify_init1(_IN_NONBLOCK)
        if self._fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, "inotify_init1")
        self._paths: dict[int, str] = {}

    def watch(self, path: str) -> None:
        wd = self._libc.inotify_add_watch(self._fd, os.fsencode(path), _IN_OPEN)
        if wd < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"inotify_add_watch {path}")
        self._paths[wd] = path

    def opened(self) -> list[str]:
        opened: list[str] = []
        while True:
            try:
                data = os.read(self._fd, 4096)
            except BlockingIOError:
                break
            offset = 0
            while offset + 16 <= len(data):
                wd, mask, _cookie, name_len = struct.unpack_from("iIII", data, offset)
                offset += 16 + name_len
                if mask & _IN_OPEN and wd in self._paths:
                    opened.append(self._paths[wd])
        return opened

    def close(self) -> None:
        os.close(self._fd)


def _rchar() -> int:
    with open("/proc/self/io", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("rchar:"):
                return int(line.split()[1])
    raise RuntimeError("rchar missing from /proc/self/io")


def _write_repo(model_dir: Path, cmvn_file: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel({"<|AUDIO|>": 0, "<unk>": 1}, unk_token="<unk>"))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    fast.save_pretrained(model_dir)
    config = {
        "feature_extractor_type": "FunASRFeatureExtractor",
        "processor_class": "FunASRProcessor",
        "feature_size": 80,
        "sampling_rate": 16000,
        "hop_length": 160,
        "chunk_length": 30,
        "n_fft": 400,
        "padding_value": 0.0,
        "dither": 0.0,
        "frontend_conf": {
            "fs": 16000,
            "n_mels": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "lfr_m": 1,
            "lfr_n": 1,
            "cmvn_file": cmvn_file,
        },
    }
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )


def _cap_address_space() -> None:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    limit = (rss_kb + 256 * 1024) * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _install_toctou_hook(link_path: str, outside_path: str) -> None:
    """Retarget ``link_path`` in the ``open`` audit hook, before the syscall.

    ``os.path.realpath`` does not emit this event, so a check that resolves
    the link and then opens the original path reads the new target.
    """

    def _hook(event: str, args: tuple[object, ...]) -> None:
        if event != "open" or not args:
            return
        target = args[0]
        if isinstance(target, bytes):
            target = os.fsdecode(target)
        if target != link_path or not os.path.islink(link_path):
            return
        os.remove(link_path)
        os.symlink(outside_path, link_path)

    sys.addaudithook(_hook)


def _load(entry: str, model_dir: str):
    if entry == "processor":
        processor = get_processor(model_dir, trust_remote_code=False)
        return processor, processor.feature_extractor
    extractor = FunASRFeatureExtractor.from_pretrained(
        model_dir, trust_remote_code=False
    )
    return None, extractor


def _transcribe(entry: str, processor, extractor) -> None:
    audio = [np.zeros(1600, dtype=np.float32)]
    if entry == "processor":
        processor(audio=audio, text=["<|AUDIO|>"], sampling_rate=16000)
        return
    extractor(audio, sampling_rate=16000)


def _worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = Path(request["result"])
    payload: dict[str, object] = {
        "status": "crashed",
        "exc_type": None,
        "exc": None,
        "traceback": None,
        "opened": [],
        "rchar_delta": None,
        "reached_funasr": False,
        "cmvn_seen": None,
    }
    watcher = _Inotify()
    try:
        os.chdir(request["cwd"])
        for path in request["watch"]:
            watcher.watch(path)
        if request.get("toctou_link"):
            _install_toctou_hook(request["toctou_link"], request["toctou_outside"])
        try:
            processor, extractor = _load(request["entry"], request["model_dir"])
        except Exception as exc:
            payload["status"] = "load_error"
            payload["exc_type"] = type(exc).__name__
            payload["exc"] = str(exc)
            payload["traceback"] = traceback.format_exc()
            payload["reached_funasr"] = (
                "transformers_utils/processors/funasr.py" in payload["traceback"]
            )
            return
        payload["reached_funasr"] = True
        payload["cmvn_seen"] = extractor.frontend_conf.get("cmvn_file")
        Path(request["ready"]).write_text("ready", encoding="utf-8")
        if request.get("limit_memory"):
            _cap_address_space()
        before = _rchar()
        try:
            _transcribe(request["entry"], processor, extractor)
        except Exception as exc:
            payload["status"] = "error"
            payload["exc_type"] = type(exc).__name__
            payload["exc"] = str(exc)
            payload["traceback"] = traceback.format_exc()
        else:
            payload["status"] = "ok"
        payload["rchar_delta"] = _rchar() - before
    finally:
        payload["opened"] = watcher.opened()
        watcher.close()
        result_path.write_text(json.dumps(payload), encoding="utf-8")


def _run(request: dict[str, object], result_timeout: float) -> dict[str, object]:
    work = Path(tempfile.mkdtemp())
    ready = work / "ready"
    result = work / "result.json"
    request_path = work / "request.json"
    request["ready"] = str(ready)
    request["result"] = str(result)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT), env.get("PYTHONPATH", "")])
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd="/",
    )
    try:
        deadline = time.monotonic() + 90
        while (
            time.monotonic() < deadline and not ready.exists() and proc.poll() is None
        ):
            time.sleep(0.05)
        if ready.exists() and proc.poll() is None:
            deadline = time.monotonic() + result_timeout
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.05)
        if proc.poll() is None:
            proc.kill()
            stdout, stderr = proc.communicate()
            return {
                "status": "blocked",
                "exc_type": None,
                "exc": "cmvn read did not return",
                "traceback": None,
                "opened": [],
                "rchar_delta": None,
                "reached_funasr": True,
                "cmvn_seen": request.get("cmvn_file"),
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
            }
        stdout, stderr = proc.communicate()
        text_out = stdout.decode(errors="replace")
        text_err = stderr.decode(errors="replace")
        if not result.exists():
            return {
                "status": "crashed",
                "exc_type": None,
                "exc": f"worker exited {proc.returncode}",
                "traceback": None,
                "opened": [],
                "rchar_delta": None,
                "reached_funasr": False,
                "cmvn_seen": None,
                "stdout": text_out,
                "stderr": text_err,
            }
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["stdout"] = text_out
        payload["stderr"] = text_err
        return payload
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def _assert_refused(payload: dict[str, object], forbidden: list[str]) -> None:
    assert payload["reached_funasr"] is True, payload
    assert payload["exc_type"] not in _SETUP_ERRORS, payload
    assert payload["exc_type"] != "MemoryError", payload
    opened = set(payload["opened"])
    assert opened.isdisjoint(forbidden), payload
    assert payload["status"] not in {"ok", "blocked", "crashed"}, payload


def _request(
    tmp: Path,
    *,
    entry: str,
    cmvn_file: str,
    cwd: Path,
    watch: list[str],
    toctou_link: str | None = None,
    toctou_outside: str | None = None,
    limit_memory: bool = False,
) -> dict[str, object]:
    return {
        "entry": entry,
        "model_dir": str(tmp / "model"),
        "cmvn_file": cmvn_file,
        "cwd": str(cwd),
        "watch": watch,
        "toctou_link": toctou_link,
        "toctou_outside": toctou_outside,
        "limit_memory": limit_memory,
    }


def test_processor_refuses_absolute_cmvn_path(tmp_path: Path) -> None:
    """An absolute cmvn_file outside the repo must not be opened."""
    model = tmp_path / "model"
    outside = tmp_path / "outside.mvn"
    outside.write_text(_VALID_CMVN, encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_repo(model, str(outside))
    payload = _run(
        _request(
            tmp_path,
            entry="processor",
            cmvn_file=str(outside),
            cwd=cwd,
            watch=[str(outside)],
        ),
        result_timeout=30,
    )
    _assert_refused(payload, [str(outside)])


def test_feature_extractor_refuses_absolute_cmvn_path(tmp_path: Path) -> None:
    """The feature extractor opens the same path without the processor."""
    model = tmp_path / "model"
    outside = tmp_path / "outside.mvn"
    outside.write_text(_VALID_CMVN, encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_repo(model, str(outside))
    payload = _run(
        _request(
            tmp_path,
            entry="extractor",
            cmvn_file=str(outside),
            cwd=cwd,
            watch=[str(outside)],
        ),
        result_timeout=30,
    )
    _assert_refused(payload, [str(outside)])


def test_processor_refuses_parent_segment_cmvn_path(tmp_path: Path) -> None:
    """A relative parent segment escapes the model directory."""
    model = tmp_path / "model"
    outside = tmp_path / "outside.mvn"
    outside.write_text(_VALID_CMVN, encoding="utf-8")
    _write_repo(model, "../outside.mvn")
    payload = _run(
        _request(
            tmp_path,
            entry="processor",
            cmvn_file="../outside.mvn",
            cwd=model,
            watch=[str(outside)],
        ),
        result_timeout=30,
    )
    _assert_refused(payload, [str(outside)])


def test_processor_refuses_prefix_bypass_cmvn_path(tmp_path: Path) -> None:
    """A path under the model dir string still escapes through '..'."""
    model = tmp_path / "model"
    outside = tmp_path / "outside.mvn"
    outside.write_text(_VALID_CMVN, encoding="utf-8")
    cmvn_file = str(model) + "/../outside.mvn"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_repo(model, cmvn_file)
    payload = _run(
        _request(
            tmp_path,
            entry="processor",
            cmvn_file=cmvn_file,
            cwd=cwd,
            watch=[str(outside)],
        ),
        result_timeout=30,
    )
    _assert_refused(payload, [str(outside)])


def test_processor_refuses_symlink_cmvn_file(tmp_path: Path) -> None:
    """A symlink whose path is inside the model dir must not be followed."""
    model = tmp_path / "model"
    outside = tmp_path / "outside.mvn"
    outside.write_text(_VALID_CMVN, encoding="utf-8")
    model.mkdir()
    link = model / "linked.mvn"
    link.symlink_to(outside)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_repo(model, str(link))
    payload = _run(
        _request(
            tmp_path,
            entry="processor",
            cmvn_file=str(link),
            cwd=cwd,
            watch=[str(outside)],
        ),
        result_timeout=30,
    )
    _assert_refused(payload, [str(outside)])


def test_processor_refuses_symlink_directory_component(tmp_path: Path) -> None:
    """A symlink in a parent directory component must not be followed."""
    model = tmp_path / "model"
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    outside = outside_dir / "cmvn.mvn"
    outside.write_text(_VALID_CMVN, encoding="utf-8")
    model.mkdir()
    (model / "sub").symlink_to(outside_dir)
    cmvn_file = str(model / "sub" / "cmvn.mvn")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_repo(model, cmvn_file)
    payload = _run(
        _request(
            tmp_path,
            entry="processor",
            cmvn_file=cmvn_file,
            cwd=cwd,
            watch=[str(outside)],
        ),
        result_timeout=30,
    )
    _assert_refused(payload, [str(outside)])


def test_processor_does_not_reopen_swapped_cmvn_symlink(tmp_path: Path) -> None:
    """Opening the checked path must not observe a symlink retarget.

    The link starts at a file inside the model dir. On the ``open`` audit
    event, which runs before the syscall, the test points the link at a file
    outside the model dir. A check that resolves the link and then opens the
    original path reads the outside file.
    """
    model = tmp_path / "model"
    inside = model / "inside.mvn"
    outside = tmp_path / "outside.mvn"
    model.mkdir()
    inside.write_text(_VALID_CMVN, encoding="utf-8")
    outside.write_text(_VALID_CMVN, encoding="utf-8")
    link = model / "cmvn.mvn"
    link.symlink_to(inside)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_repo(model, str(link))
    payload = _run(
        _request(
            tmp_path,
            entry="processor",
            cmvn_file=str(link),
            cwd=cwd,
            watch=[str(outside)],
            toctou_link=str(link),
            toctou_outside=str(outside),
        ),
        result_timeout=30,
    )
    assert payload["status"] != "blocked", payload
    assert payload["status"] != "crashed", payload
    assert payload["reached_funasr"] is True, payload
    assert payload["exc_type"] not in _SETUP_ERRORS, payload
    assert str(outside) not in set(payload["opened"]), payload


def test_processor_dev_zero_cmvn_does_not_block(tmp_path: Path) -> None:
    """``/dev/zero`` never yields a newline and must not be read."""
    model = tmp_path / "model"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_repo(model, "/dev/zero")
    payload = _run(
        _request(
            tmp_path,
            entry="processor",
            cmvn_file="/dev/zero",
            cwd=cwd,
            watch=[],
            limit_memory=True,
        ),
        result_timeout=3,
    )
    _assert_refused(payload, [])


def test_processor_caps_bytes_read_from_model_cmvn(tmp_path: Path) -> None:
    """An in-repo CMVN file is still an unbounded ``readlines`` today."""
    model = tmp_path / "model"
    model.mkdir()
    huge = model / "huge.mvn"
    huge.write_bytes(b"0" * (8 * 1024 * 1024))
    _write_repo(model, "huge.mvn")
    payload = _run(
        _request(
            tmp_path,
            entry="processor",
            cmvn_file="huge.mvn",
            cwd=model,
            watch=[],
        ),
        result_timeout=30,
    )
    assert payload["status"] not in {"ok", "blocked", "crashed"}, payload
    assert payload["reached_funasr"] is True, payload
    assert payload["exc_type"] not in _SETUP_ERRORS, payload
    assert payload["exc_type"] != "MemoryError", payload
    delta = payload["rchar_delta"]
    assert isinstance(delta, int)
    assert delta < _READ_BUDGET, payload


def test_processor_reads_cmvn_file_inside_model_dir(tmp_path: Path) -> None:
    """A regular CMVN file stored in the model directory remains readable."""
    model = tmp_path / "model"
    inside = model / "am.mvn"
    model.mkdir()
    inside.write_text(_VALID_CMVN, encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _write_repo(model, str(inside))
    payload = _run(
        _request(
            tmp_path,
            entry="processor",
            cmvn_file=str(inside),
            cwd=cwd,
            watch=[str(inside)],
        ),
        result_timeout=30,
    )
    assert payload["status"] == "ok", payload
    assert str(inside) in set(payload["opened"]), payload


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]))
