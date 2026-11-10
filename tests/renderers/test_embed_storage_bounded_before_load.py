# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client embedding checkpoints must be rejected before storage is allocated.

`torch.load` of a zip checkpoint allocates each storage record from the
uncompressed size in the zip header, then `safe_to_dense` compares that
tensor to `VLLM_MAX_EMBED_DECODE_BYTES`. A deflated storage record keeps the
caller bytes far below the allocation. These tests fail while that order is
unchanged: the child records a `posix_memalign` of the declared storage.
"""

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pybase64
import torch

STORAGE_BYTES = 32 * 1024 * 1024
EMBED_LIMIT_BYTES = 4096
CHILD_FLAG = "VLLM_EMBED_STORAGE_LIMIT_CHILD"

# Interposes the allocator torch 2.14 uses for CPU storage (posix_memalign in
# c10/core/impl/alloc_cpu.cpp). Armed only after import so library load does
# not count. A hit of STORAGE_BYTES means torch.load materialized the record.
_ALLOC_WATCH_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
int posix_memalign(void **memptr, size_t alignment, size_t size) {
    static int (*real_posix_memalign)(void **, size_t, size_t);
    if (!real_posix_memalign)
        real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    const char *min_env = getenv("ALLOC_WATCH_MIN");
    size_t min_size = min_env ? strtoull(min_env, NULL, 10) : 0;
    if (getenv("ALLOC_WATCH_ARMED") && min_size && size >= min_size)
        fprintf(stderr, "ALLOC posix_memalign %zu\n", size);
    return real_posix_memalign(memptr, alignment, size);
}
"""

# Local text config so prompt-embed loading does not fetch a Hub repo.
# The over-limit cases are rejected before hidden-size is read.
_OPT_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 1,
    "vocab_size": 50272,
    "word_embed_proj_dim": 768,
    "max_position_embeddings": 2048,
    "ffn_dim": 3072,
    "activation_function": "relu",
    "do_layer_norm_before": True,
    "bos_token_id": 2,
    "eos_token_id": 2,
    "pad_token_id": 1,
}


def _deflated_checkpoint(storage_bytes: int) -> bytes:
    """Torch zip whose `data/0` record is deflated zeros of `storage_bytes`."""
    tensor = torch.zeros(storage_bytes, dtype=torch.uint8)
    raw = io.BytesIO()
    torch.save(tensor, raw)
    del tensor
    source = zipfile.ZipFile(io.BytesIO(raw.getvalue()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for info in source.infolist():
            payload = source.read(info.filename)
            compress = (
                zipfile.ZIP_DEFLATED if info.file_size > 4096 else zipfile.ZIP_STORED
            )
            archive.writestr(info.filename, payload, compress_type=compress)
    return out.getvalue()


def _declared_storage_bytes(payload: bytes) -> int:
    archive = zipfile.ZipFile(io.BytesIO(payload))
    stored = [
        info.file_size
        for info in archive.infolist()
        if info.filename.endswith("data/0")
    ]
    assert len(stored) == 1
    return stored[0]


def _child() -> None:
    os.environ["VLLM_MAX_EMBED_DECODE_BYTES"] = str(EMBED_LIMIT_BYTES)

    from vllm.config import ModelConfig
    from vllm.multimodal.media.audio import AudioEmbeddingMediaIO
    from vllm.multimodal.media.connector import MediaConnector
    from vllm.multimodal.media.image import ImageEmbeddingMediaIO
    from vllm.multimodal.media.video import VideoEmbeddingMediaIO
    from vllm.renderers.embed_utils import safe_load_prompt_embeds

    root = Path(os.environ["VLLM_EMBED_STORAGE_LIMIT_DIR"])
    model_dir = root / "opt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_OPT_CONFIG), encoding="utf-8")
    model_config = ModelConfig(
        model=str(model_dir),
        skip_tokenizer_init=True,
        dtype="float32",
        enable_prompt_embeds=True,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
    )

    bomb = (root / "bomb.pt").read_bytes()
    small = (root / "small.pt").read_bytes()
    bomb_b64 = pybase64.b64encode(bomb).decode("utf-8")
    bomb_path = root / "bomb-file.pt"
    bomb_path.write_bytes(bomb)
    connector = MediaConnector()
    # Import is finished. Allocations from here are the loaders under test.
    os.environ["ALLOC_WATCH_ARMED"] = "1"

    def run(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            result = {"name": name, "exc": type(exc).__name__, "msg": str(exc)}
        else:
            result = {"name": name, "exc": None, "msg": ""}
        print("CASE " + json.dumps(result), flush=True)

    image = ImageEmbeddingMediaIO()
    audio = AudioEmbeddingMediaIO()
    video = VideoEmbeddingMediaIO()
    run("image-bytes-within-limit", lambda: image.load_bytes(small))
    run("image-bytes", lambda: image.load_bytes(bomb))
    run("image-base64", lambda: image.load_base64("", bomb_b64))
    run("image-file", lambda: image.load_file(bomb_path))
    run("audio-bytes", lambda: audio.load_bytes(bomb))
    run("audio-base64", lambda: audio.load_base64("", bomb_b64))
    run("audio-file", lambda: audio.load_file(bomb_path))
    run("video-bytes", lambda: video.load_bytes(bomb))
    run("video-base64", lambda: video.load_base64("", bomb_b64))
    run("video-file", lambda: video.load_file(bomb_path))
    run("connector-image", lambda: connector.fetch_image_embedding(bomb_b64))
    run("connector-audio", lambda: connector.fetch_audio_embedding(bomb_b64))
    run("connector-video", lambda: connector.fetch_video_embedding(bomb_b64))
    run(
        "prompt-embeds",
        lambda: safe_load_prompt_embeds(model_config, pybase64.b64encode(bomb)),
    )


def test_deflated_storage_is_smaller_than_the_tensor_it_declares(tmp_path: Path):
    """The request body is the compressed record, not the storage it names."""
    payload = _deflated_checkpoint(STORAGE_BYTES)
    assert _declared_storage_bytes(payload) == STORAGE_BYTES
    assert len(payload) * 100 < STORAGE_BYTES
    (tmp_path / "bomb.pt").write_bytes(payload)


def _alloc_watch(tmp_path: Path) -> Path:
    source = tmp_path / "alloc_watch.c"
    library = tmp_path / "alloc_watch.so"
    source.write_text(_ALLOC_WATCH_C, encoding="utf-8")
    compiled = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", str(library), str(source), "-ldl"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    return library


def test_over_limit_storage_is_rejected_before_it_is_allocated(tmp_path: Path):
    """Every client embed loader must refuse the checkpoint before allocating.

    The child loads a deflated checkpoint whose uncompressed storage is
    `STORAGE_BYTES`, above `VLLM_MAX_EMBED_DECODE_BYTES`. `torch.load` asks
    the CPU allocator for that storage before `safe_to_dense` runs. The
    watcher records that `posix_memalign`. Rejecting from the declared
    uncompressed size does not.
    """
    bomb = _deflated_checkpoint(STORAGE_BYTES)
    small = _deflated_checkpoint(128)
    assert _declared_storage_bytes(bomb) > EMBED_LIMIT_BYTES
    assert _declared_storage_bytes(small) <= EMBED_LIMIT_BYTES
    (tmp_path / "bomb.pt").write_bytes(bomb)
    (tmp_path / "small.pt").write_bytes(small)
    library = _alloc_watch(tmp_path)

    env = os.environ.copy()
    env[CHILD_FLAG] = "1"
    env["VLLM_EMBED_STORAGE_LIMIT_DIR"] = str(tmp_path)
    env["PYTHONUNBUFFERED"] = "1"
    env["LD_PRELOAD"] = str(library)
    env["ALLOC_WATCH_MIN"] = str(STORAGE_BYTES)
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [repo_root]
    )
    completed = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    cases = [
        json.loads(line.removeprefix("CASE "))
        for line in completed.stdout.splitlines()
        if line.startswith("CASE ")
    ]
    assert completed.returncode == 0, completed.stderr
    names = [case["name"] for case in cases]
    assert names == [
        "image-bytes-within-limit",
        "image-bytes",
        "image-base64",
        "image-file",
        "audio-bytes",
        "audio-base64",
        "audio-file",
        "video-bytes",
        "video-base64",
        "video-file",
        "connector-image",
        "connector-audio",
        "connector-video",
        "prompt-embeds",
    ]

    within = cases[0]
    assert within["exc"] is None, within
    assert "ALLOC posix_memalign" not in completed.stderr

    for case in cases[1:]:
        assert case["exc"] == "VLLMValidationError", case
        assert "VLLM_MAX_EMBED_DECODE_BYTES" in case["msg"], case
        assert "ALLOC posix_memalign" not in completed.stderr


if __name__ == "__main__" and os.environ.get(CHILD_FLAG) == "1":
    _child()
