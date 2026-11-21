# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Molmo2 crop-budget bounds.

These tests fail while a checkpoint can set ``max_crops`` high enough that
profiling or an image request spends unbounded CPU in the tiling search.
They pass once that budget is refused or capped before the search runs.

``vllm/model_executor/models/molmo.py`` repeats the same search. It is not
covered here: Molmo's processor is not in the shipped transformers package,
and loading it executes repository code.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Molmo2Config, PreTrainedTokenizerFast

from vllm.config import ModelConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.model_executor.models.molmo2 import Molmo2DummyInputsBuilder
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item

# Published Molmo2 checkpoints use 8. A few thousand already makes one tiling
# search take seconds, and profiling repeats that search for every layout.
_HOSTILE_MAX_CROPS = 20_000
_SEARCH_DEADLINE_S = 5
_SUBPROCESS_DEADLINE_S = 90

pytestmark = pytest.mark.skip_global_cleanup

_CHILD = textwrap.dedent(
    """
    import signal
    import sys

    from PIL import Image

    from vllm.config import ModelConfig
    from vllm.config.multimodal import MultiModalDummyOptions
    from vllm.model_executor.models.molmo2 import Molmo2DummyInputsBuilder
    from vllm.multimodal import MULTIMODAL_REGISTRY

    class SearchTooSlow(Exception):
        pass

    def _alarm(signum, frame):
        raise SearchTooSlow(f"tiling search exceeded {sys.argv[2]}s")

    def run(name, fn):
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, float(sys.argv[2]))
        try:
            fn()
        except ValueError:
            print(name, "refused", flush=True)
        else:
            print(name, "ok", flush=True)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    repo = sys.argv[1]
    model_config = ModelConfig(
        model=repo,
        trust_remote_code=False,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    assert model_config.trust_remote_code is False
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    info = processor.info
    image_processor = info.get_hf_processor().image_processor

    run("profile", info.get_image_size_with_most_features)
    run(
        "select",
        lambda: info.select_tiling(
            image_width=32,
            image_height=32,
            image_processor=image_processor,
        ),
    )
    run(
        "grid",
        lambda: info.get_patches_grid_size(
            image_width=32,
            image_height=32,
            image_processor=image_processor,
        ),
    )
    run(
        "tokens",
        lambda: info.get_num_image_tokens(
            image_height=32,
            image_width=32,
            processor=info.get_hf_processor(),
        ),
    )
    run(
        "dummy",
        lambda: Molmo2DummyInputsBuilder(info).get_dummy_mm_data(
            128,
            {"image": 1, "video": 0},
            MultiModalDummyOptions(),
        ),
    )

    image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    items = info.parse_mm_data({"image": image})
    run("request", lambda: processor("<|image|> hello", items))
    print("DONE", flush=True)
    """
)


def _write_checkpoint(root: Path, max_crops: int) -> None:
    """Write a local Molmo2 repo whose only hostile field is ``max_crops``."""
    config = Molmo2Config(architectures=["Molmo2ForConditionalGeneration"])
    # The shipped Molmo2Config omits ids the vLLM processor reads. The
    # published checkpoint carries them as ordinary config fields.
    config.image_col_id = 151939
    config.image_low_res_id = 151942
    config.use_frame_special_tokens = False
    config.save_pretrained(root)

    root.joinpath("preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "Molmo2ImageProcessor",
                "processor_class": "Molmo2Processor",
                "do_convert_rgb": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "max_crops": max_crops,
                "overlap_margins": [4, 4],
                "patch_size": 14,
                "pooling_size": [2, 2],
                "resample": 2,
                "size": {"height": 378, "width": 378},
            }
        )
    )
    root.joinpath("processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Molmo2Processor",
                "image_use_col_tokens": True,
                "use_single_crop_col_tokens": False,
                "use_single_crop_start_token": True,
                "video_use_col_tokens": False,
                "use_frame_special_tokens": False,
            }
        )
    )
    root.joinpath("video_preprocessor_config.json").write_text(
        json.dumps(
            {
                "video_processor_type": "Molmo2VideoProcessor",
                "processor_class": "Molmo2Processor",
                "do_convert_rgb": True,
                "do_normalize": True,
                "do_rescale": True,
                "do_resize": True,
                "do_sample_frames": True,
                "frame_sample_mode": "uniform_last_frame",
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "max_fps": 2.0,
                "num_frames": 2,
                "patch_size": 14,
                "pooling_size": [2, 2],
                "resample": 2,
                "rescale_factor": 0.00392156862745098,
                "sampling_fps": 2,
                "size": {"height": 378, "width": 378},
            }
        )
    )

    specials = [
        "<unk>",
        "<|endoftext|>",
        "<|image|>",
        "<|video|>",
        "<im_patch>",
        "<im_col>",
        "<im_start>",
        "<low_res_im_start>",
        "<frame_start>",
        "<im_end>",
        "<frame_end>",
        "<im_low>",
        "hello",
    ]
    vocab = {token: index for index, token in enumerate(specials)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|image|>",
                "<|video|>",
                "<im_patch>",
                "<im_col>",
                "<im_start>",
                "<low_res_im_start>",
                "<frame_start>",
                "<im_end>",
                "<frame_end>",
                "<im_low>",
            ]
        }
    )
    tokenizer.save_pretrained(root)


def _model_config(root: Path) -> ModelConfig:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return ModelConfig(
        model=str(root),
        trust_remote_code=False,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )


def test_published_crop_budget_profiles_and_serves_an_image(tmp_path: Path) -> None:
    """A normal Molmo2 preprocessor config still profiles and accepts an image."""
    repo = tmp_path / "molmo2"
    repo.mkdir()
    _write_checkpoint(repo, max_crops=8)

    model_config = _model_config(repo)
    assert model_config.trust_remote_code is False
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    image_processor = processor.info.get_hf_processor().image_processor
    assert image_processor.max_crops == 8

    per_item = get_mm_max_toks_per_item(processor, {"image": 1}, cache=None)
    assert per_item["image"] > 0

    image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    items = processor.info.parse_mm_data({"image": image})
    processed = processor("<|image|> hello", items)
    assert len(processed["prompt_token_ids"]) > 1
    assert processed["mm_placeholders"]["image"]


def test_checkpoint_max_crops_cannot_stall_profiling_or_requests(
    tmp_path: Path,
) -> None:
    """A published ``max_crops`` must not pin profiling or the next image."""
    repo = tmp_path / "molmo2"
    repo.mkdir()
    _write_checkpoint(repo, max_crops=_HOSTILE_MAX_CROPS)

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[4]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(repo),
            str(_SEARCH_DEADLINE_S),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_DEADLINE_S,
        cwd=repo_root,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "DONE" in completed.stdout
    for operation in ("profile", "select", "grid", "tokens", "dummy", "request"):
        assert operation in completed.stdout
