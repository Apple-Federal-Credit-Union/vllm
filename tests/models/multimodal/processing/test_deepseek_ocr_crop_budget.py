# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek OCR crop-budget bounds.

These tests fail while a checkpoint can set ``max_crops`` high enough that
startup profiling or a later image request stays in the tiling search.
They pass once that budget is refused or capped before the search runs.

``get_num_image_tokens`` / ``count_tiles`` on DeepSeek OCR and DeepSeek OCR 2
are not covered. Those helpers pass the built-in crop constant and do not
read the checkpoint budget. Unlimited OCR is not covered either: its
processor constructor already passes a fixed ``max_crops``.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item

# A few hundred crops already costs seconds. The startup search is cubic
# in this value, so an uncapped budget cannot finish inside the deadline.
_HOSTILE_MAX_CROPS = 20_000
_SEARCH_DEADLINE_S = 5
_SUBPROCESS_DEADLINE_S = 90

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

_CHILD = textwrap.dedent(
    """
    import signal
    import sys

    from PIL import Image

    import vllm.model_executor.models.deepseek_ocr  # noqa: F401
    import vllm.model_executor.models.deepseek_ocr2  # noqa: F401
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item

    class SearchTooSlow(Exception):
        pass

    def _alarm(signum, frame):
        raise SearchTooSlow(f"tiling search exceeded {sys.argv[2]}s")

    def _is_timeout(exc):
        seen = set()
        current = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, SearchTooSlow):
                return True
            current = current.__cause__ or current.__context__
        return False

    def run(name, fn):
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, float(sys.argv[2]))
        try:
            fn()
        except ValueError as exc:
            # The processor wrapper turns the alarm into ValueError. That is
            # the uncapped search, not a budget rejection.
            if _is_timeout(exc):
                raise
            print(name, "refused", flush=True)
        else:
            print(name, "ok", flush=True)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    repo = sys.argv[1]
    mm_kwargs = {}
    if len(sys.argv) > 3 and sys.argv[3]:
        mm_kwargs["max_crops"] = int(sys.argv[3])

    try:
        model_config = ModelConfig(
            model=repo,
            trust_remote_code=False,
            max_model_len=2048,
            dtype="float32",
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs=mm_kwargs or None,
        )
        assert model_config.trust_remote_code is False
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    except ValueError:
        print("load refused", flush=True)
        print("DONE", flush=True)
        sys.exit(0)

    run(
        "profile",
        lambda: get_mm_max_toks_per_item(processor, {"image": 1}, cache=None),
    )
    size = processor.info.get_image_size_with_most_features()
    image = Image.new("RGB", (size.width, size.height), color=(10, 20, 30))
    items = processor.info.parse_mm_data({"image": image})
    run("request", lambda: processor("<image>", items))
    print("DONE", flush=True)
    """
)

# Same shape as a published DeepSeek-OCR config.json, without auto_map.
# language_config is what DeepseekVLV2Config copies into text_config.
_MODEL_CONFIG = {
    "architectures": ["DeepseekOCRForCausalLM"],
    "candidate_resolutions": [[1024, 1024]],
    "global_view_pos": "head",
    "language_config": {
        "architectures": ["DeepseekV2ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 1,
        "first_k_dense_replace": 1,
        "hidden_size": 1280,
        "intermediate_size": 6848,
        "kv_lora_rank": None,
        "lm_head": True,
        "max_position_embeddings": 8192,
        "moe_intermediate_size": 896,
        "n_group": 1,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_attention_heads": 10,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 12,
        "num_key_value_heads": 10,
        "q_lora_rank": None,
        "qk_nope_head_dim": 0,
        "qk_rope_head_dim": 0,
        "rm_head": False,
        "topk_group": 1,
        "topk_method": "greedy",
        "torch_dtype": "bfloat16",
        "use_mla": False,
        "v_head_dim": 0,
        "vocab_size": 129280,
    },
    "model_type": "deepseek_vl_v2",
    "projector_config": {
        "input_dim": 2048,
        "model_type": "mlp_projector",
        "n_embed": 1280,
        "projector_type": "linear",
    },
    "tile_tag": "2D",
    "torch_dtype": "bfloat16",
    "vision_config": {
        "image_size": 1024,
        "mlp_ratio": 3.7362,
        "model_name": "deeplip_b_l",
        "model_type": "vision",
        "width": 1024,
    },
    "bos_token_id": 0,
    "eos_token_id": 1,
    "hidden_size": 1280,
    "num_attention_heads": 10,
    "num_hidden_layers": 12,
    "vocab_size": 129280,
}


def _write_checkpoint(root: Path, max_crops: int, architecture: str) -> None:
    """Write a local DeepSeek OCR repo whose hostile field is ``max_crops``."""
    config = dict(_MODEL_CONFIG)
    config["architectures"] = [architecture]
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("config.json").write_text(json.dumps(config))
    root.joinpath("processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "DeepseekOCRProcessor",
                "image_token": "<image>",
                "pad_token": "<｜▁pad▁｜>",
                "max_crops": max_crops,
            }
        )
    )

    specials = ["<unk>", "<s>", "</s>", "<image>", "<｜▁pad▁｜>"]
    vocab = {token: index for index, token in enumerate(specials)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<｜▁pad▁｜>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    tokenizer.save_pretrained(root)


def _run_budget(root: Path, override: int | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[4]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    argv = [
        sys.executable,
        "-c",
        _CHILD,
        str(root),
        str(_SEARCH_DEADLINE_S),
    ]
    if override is not None:
        argv.append(str(override))
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_DEADLINE_S,
        cwd=repo_root,
        env=env,
    )


def _assert_search_finishes(completed: subprocess.CompletedProcess[str]) -> None:
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "DONE" in completed.stdout
    assert "profile" in completed.stdout
    assert "request" in completed.stdout


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM"],
)
def test_checkpoint_max_crops_cannot_stall_profiling_or_requests(
    tmp_path: Path, architecture: str
) -> None:
    """processor_config.json max_crops must not pin profiling or an image."""
    repo = tmp_path / "deepseek-ocr"
    _write_checkpoint(repo, _HOSTILE_MAX_CROPS, architecture)
    completed = _run_budget(repo, override=None)
    _assert_search_finishes(completed)


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM"],
)
def test_mm_processor_kwargs_max_crops_cannot_stall(
    tmp_path: Path, architecture: str
) -> None:
    """A later max_crops override must not restart an uncapped tiling search."""
    repo = tmp_path / "deepseek-ocr"
    _write_checkpoint(repo, max_crops=6, architecture=architecture)
    completed = _run_budget(repo, override=_HOSTILE_MAX_CROPS)
    _assert_search_finishes(completed)


@pytest.mark.parametrize("max_crops", [0, -1])
def test_non_positive_max_crops_is_rejected(tmp_path: Path, max_crops: int) -> None:
    """A non-positive crop budget is refused before the layout search."""
    repo = tmp_path / "deepseek-ocr"
    _write_checkpoint(repo, max_crops, "DeepseekOCRForCausalLM")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with pytest.raises(ValueError):
        model_config = ModelConfig(
            model=str(repo),
            trust_remote_code=False,
            max_model_len=2048,
            dtype="float32",
            limit_mm_per_prompt={"image": 1},
        )
        assert model_config.trust_remote_code is False
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
        get_mm_max_toks_per_item(processor, {"image": 1}, cache=None)
