# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint tile limits must be rejected before InternVL enumeration.

These tests fail while the planner still builds candidate layouts with three
nested loops over an uncapped checkpoint integer. A fix has to refuse that
integer on the same processing-info methods startup profiling and image
requests call, before those loops run.

`NemotronH_Nano_VL_V2` is not covered. Its processor keeps `max_num_tiles` at
the built-in default unless the caller passes it, and that default does not
come from the checkpoint fields exercised here.

`QianfanOCRForConditionalGeneration` is not covered. Its processing info
requires the built-in `QianfanOCRConfig`, and that class has no
`min_dynamic_patch` or `max_dynamic_patch` field for a checkpoint to set.
"""

from __future__ import annotations

import json
import signal
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from vllm.config.model import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

_DEADLINE_S = 2.0
_UNBOUNDED = 800
_BOUNDED = 12

# Architectures whose processing info copies min/max_dynamic_patch off the
# loaded checkpoint config and into the shared InternVL planner.
_CHAT_ARCHITECTURES = (
    "InternVLChatModel",
    "H2OVLChatModel",
    "NVLM_D",
    "SkyworkR1VChatModel",
    "Eagle2_5_VLForConditionalGeneration",
)

_CHAT_CONFIG_PY = """
from transformers import PreTrainedConfig, Qwen2Config


class InternVisionConfig(PreTrainedConfig):
    model_type = "intern_vit"

    def __init__(self, image_size=448, patch_size=14, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size


class InternVLChatConfig(PreTrainedConfig):
    model_type = "internvl_chat"

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        downsample_ratio=0.5,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        use_msac=False,
        force_image_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = InternVisionConfig(**(vision_config or {}))
        llm = llm_config or {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "vocab_size": 8,
        }
        self.llm_config = Qwen2Config(**llm)
        self.text_config = self.llm_config
        self.downsample_ratio = downsample_ratio
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_msac = use_msac
        self.force_image_size = force_image_size
"""


def _write_tokenizer(root: Path) -> None:
    vocab = {
        "<unk>": 0,
        "<img>": 1,
        "</img>": 2,
        "<IMG_CONTEXT>": 3,
        "<image>": 4,
        "<video>": 5,
    }
    Tokenizer(WordLevel(vocab, unk_token="<unk>")).save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "model_max_length": 2048,
                "start_image_token": "<img>",
                "end_image_token": "</img>",
                "context_image_token": "<IMG_CONTEXT>",
                "video_token": "<video>",
            }
        )
    )


def _model_config(root: Path, *, trust_remote_code: bool) -> ModelConfig:
    return ModelConfig(
        model=str(root),
        tokenizer=str(root),
        tokenizer_mode="hf",
        trust_remote_code=trust_remote_code,
        dtype="float32",
        seed=0,
        max_model_len=512,
        enforce_eager=True,
    )


def _processing_info(root: Path, *, trust_remote_code: bool):
    config = _model_config(root, trust_remote_code=trust_remote_code)
    return MULTIMODAL_REGISTRY.get_processing_info(config)


def _assert_rejected_before_deadline(fn: Callable[[], object]) -> None:
    """Fail if `fn` returns or is still enumerating when the deadline hits."""

    def _on_alarm(_signum: int, _frame: object) -> None:
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, _DEADLINE_S)
    try:
        try:
            fn()
        except TimeoutError as exc:
            raise AssertionError(
                "checkpoint patch limit was still being enumerated "
                f"after {_DEADLINE_S}s"
            ) from exc
        except Exception:
            return
        raise AssertionError("checkpoint patch limit was accepted")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _write_hf_internvl(root: Path, max_patches: int) -> None:
    root.mkdir(parents=True)
    _write_tokenizer(root)
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "GotOcr2ImageProcessorFast",
                "crop_to_patches": True,
                "min_patches": 1,
                "max_patches": max_patches,
                "size": {"height": 448, "width": 448},
            }
        )
    )
    (root / "processor_config.json").write_text(
        json.dumps({"image_seq_length": 256, "processor_class": "InternVLProcessor"})
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["InternVLForConditionalGeneration"],
                "model_type": "internvl",
                "downsample_ratio": 0.5,
                "image_seq_length": 256,
                "image_token_id": 3,
                "projector_hidden_act": "gelu",
                "vision_feature_layer": -1,
                "vision_feature_select_strategy": "default",
                "torch_dtype": "float32",
                "vision_config": {
                    "model_type": "internvl_vision",
                    "image_size": [448, 448],
                    "patch_size": [14, 14],
                    "hidden_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "intermediate_size": 64,
                },
                "text_config": {
                    "model_type": "qwen2",
                    "hidden_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "intermediate_size": 64,
                    "vocab_size": 8,
                    "max_position_embeddings": 2048,
                },
            }
        )
    )


def _write_chat_checkpoint(
    root: Path,
    architecture: str,
    max_dynamic_patch: int,
    *,
    processor_max_input_tiles: int | None = None,
) -> None:
    root.mkdir(parents=True)
    _write_tokenizer(root)
    (root / "configuration_internvl_chat.py").write_text(_CHAT_CONFIG_PY)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "model_type": "internvl_chat",
                "auto_map": {
                    "AutoConfig": "configuration_internvl_chat.InternVLChatConfig"
                },
                "downsample_ratio": 0.5,
                "dynamic_image_size": True,
                "use_thumbnail": True,
                "min_dynamic_patch": 1,
                "max_dynamic_patch": max_dynamic_patch,
                "use_msac": False,
                "force_image_size": 448,
                "vision_config": {"image_size": 448, "patch_size": 14},
                "llm_config": {
                    "hidden_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "intermediate_size": 64,
                    "vocab_size": 8,
                    "max_position_embeddings": 2048,
                },
            }
        )
    )
    if processor_max_input_tiles is not None:
        (root / "processor_config.json").write_text(
            json.dumps(
                {
                    "min_input_tiles": 1,
                    "max_input_tiles": processor_max_input_tiles,
                    "dynamic_image_size": True,
                }
            )
        )


def _exercise_chat_info(info) -> None:
    processor = info.get_hf_processor()
    info.get_num_image_tokens(
        image_width=32,
        image_height=32,
        processor=processor,
    )
    info.get_image_size_with_most_features()
    image = Image.new("RGB", (32, 32))
    processor.image_processor(image)


def test_hf_internvl_rejects_uncapped_max_patches(tmp_path: Path) -> None:
    root = tmp_path / "internvl-hf"
    _write_hf_internvl(root, _UNBOUNDED)
    info = _processing_info(root, trust_remote_code=False)

    def _profile() -> None:
        info.get_image_size_with_most_features()

    _assert_rejected_before_deadline(_profile)


def test_hf_internvl_profiles_bounded_max_patches(tmp_path: Path) -> None:
    root = tmp_path / "internvl-hf"
    _write_hf_internvl(root, _BOUNDED)
    info = _processing_info(root, trust_remote_code=False)
    size = info.get_image_size_with_most_features()
    assert size.width > 0 and size.height > 0


@pytest.mark.parametrize("architecture", _CHAT_ARCHITECTURES)
def test_chat_family_rejects_uncapped_max_dynamic_patch(
    tmp_path: Path, architecture: str
) -> None:
    root = tmp_path / architecture
    _write_chat_checkpoint(root, architecture, _UNBOUNDED)
    info = _processing_info(root, trust_remote_code=True)
    _assert_rejected_before_deadline(lambda: _exercise_chat_info(info))


@pytest.mark.parametrize("architecture", _CHAT_ARCHITECTURES)
def test_chat_family_profiles_bounded_max_dynamic_patch(
    tmp_path: Path, architecture: str
) -> None:
    root = tmp_path / architecture
    _write_chat_checkpoint(root, architecture, _BOUNDED)
    info = _processing_info(root, trust_remote_code=True)
    size = info.get_image_size_with_most_features()
    assert size.width > 0 and size.height > 0


def test_embed_processor_config_rejects_uncapped_max_input_tiles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "embed"
    # config.json itself stays inside the historical default. The limit that
    # reaches the planner is processor_config.json.
    _write_chat_checkpoint(
        root,
        "LlamaNemotronVLModel",
        _BOUNDED,
        processor_max_input_tiles=_UNBOUNDED,
    )
    info = _processing_info(root, trust_remote_code=True)
    _assert_rejected_before_deadline(lambda: _exercise_chat_info(info))
