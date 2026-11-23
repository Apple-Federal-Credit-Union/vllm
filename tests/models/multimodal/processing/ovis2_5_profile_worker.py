# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load an Ovis2.5 checkpoint and run its startup video-profile search.

The checkpoint config classes are the published model files. They copy
``vit_config.patch_size`` and ``temporal_patch_size`` through with no
range check. This process then runs the same profiling calls the scheduler
makes while the server is still starting.
"""

import json
import shutil
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _write_checkpoint(root: Path, spec: dict) -> None:
    architecture = spec["architecture"]
    if architecture == "Ovis2_5":
        config_src = FIXTURES / "configuration_ovis2_5.py.txt"
        config_name = "configuration_ovis2_5.py"
        auto_config = "configuration_ovis2_5.Ovis2_5_Config"
        model_type = "ovis2_5"
        llm_model_type = "qwen3"
    elif architecture == "Ovis2_6_MoeForCausalLM":
        config_src = FIXTURES / "configuration_ovis2_6.py.txt"
        config_name = "configuration_ovis2_6.py"
        auto_config = "configuration_ovis2_6.Ovis2_6_Moe_Config"
        model_type = "ovis2_6_moe"
        llm_model_type = "qwen3_moe"
    else:
        raise ValueError(f"unsupported architecture {architecture}")

    root.mkdir(parents=True)
    shutil.copy(config_src, root / config_name)
    shutil.copy(
        FIXTURES / "ovis_preprocessor_config.json",
        root / "preprocessor_config.json",
    )

    vocab = {f"tok{i}": i for i in range(1, 64)}
    vocab["<unk>"] = 0
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        bos_token="<unk>",
        eos_token="<unk>",
    )
    tokenizer.save_pretrained(root)

    max_model_len = spec["max_model_len"]
    llm_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "max_position_embeddings": max_model_len,
        "model_type": llm_model_type,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
    }
    if llm_model_type == "qwen3":
        llm_config["head_dim"] = 16
    config = {
        "architectures": [architecture],
        "auto_map": {"AutoConfig": auto_config},
        "hidden_size": 64,
        "vocab_size": 128,
        "num_attention_heads": 4,
        "max_position_embeddings": max_model_len,
        "model_type": model_type,
        "visual_vocab_size": 128,
        "llm_config": llm_config,
        "vit_config": {
            "hidden_size": 64,
            "hidden_stride": 2,
            "image_size": 512,
            "intermediate_size": 128,
            "model_type": "siglip2_navit",
            "num_attention_heads": 4,
            "num_channels": 3,
            "num_hidden_layers": 2,
            "patch_size": spec["patch_size"],
            "temporal_patch_size": spec["temporal_patch_size"],
        },
    }
    (root / "config.json").write_text(json.dumps(config))


def _emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def _profile(model_dir: Path, spec: dict) -> None:
    from vllm.config import DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget

    max_model_len = spec["max_model_len"]
    model_config = ModelConfig(
        model=str(model_dir),
        trust_remote_code=True,
        dtype="float32",
        seed=0,
        max_model_len=max_model_len,
    )
    vit_config = model_config.hf_config.vit_config
    _emit(
        {
            "event": "loaded",
            "patch_size": vit_config.patch_size,
            "temporal_patch_size": vit_config.temporal_patch_size,
            "architecture": model_config.architecture,
        }
    )

    info = MULTIMODAL_REGISTRY.get_processing_info(model_config)
    if spec["mode"] == "non_increasing":

        def _constant_tokens(
            *,
            image_width: int,
            image_height: int,
            num_frames: int = 1,
        ) -> int:
            del image_width, image_height, num_frames
            return 0

        info.get_num_image_tokens = _constant_tokens  # type: ignore[method-assign]

    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    if spec["mode"] == "non_increasing":
        processor.info.get_num_image_tokens = info.get_num_image_tokens

    mm_options = model_config.get_multimodal_config().limit_per_prompt
    video_counts = {"image": 0, "video": 1}
    image_only_counts = {"image": 1, "video": 0}
    scheduler = SchedulerConfig(
        max_num_seqs=2,
        max_num_batched_tokens=max_model_len,
        max_model_len=max_model_len,
        is_encoder_decoder=False,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler,
        device_config=DeviceConfig(device="cpu"),
    )

    def _max_frames() -> int:
        return info._get_max_video_frames(max_model_len)

    def _per_video() -> int:
        return info.get_num_frames_with_most_features(max_model_len, video_counts)

    def _image_only() -> int:
        return info.get_num_frames_with_most_features(
            max_model_len, image_only_counts
        )

    def _video_tokens() -> int:
        return info.get_max_video_tokens(max_model_len, video_counts)

    def _dummy_image_only() -> int:
        processor.dummy_inputs.get_dummy_mm_data(
            max_model_len, image_only_counts, mm_options
        )
        return _image_only()

    def _dummy_video() -> int:
        processor.dummy_inputs.get_dummy_mm_data(
            max_model_len, video_counts, mm_options
        )
        return _per_video()

    def _startup_budget() -> int:
        MultiModalBudget(vllm_config, MULTIMODAL_REGISTRY)
        return _per_video()

    calls = {
        "max_video_frames": _max_frames,
        "frames_per_video": _per_video,
        "image_only_frames": _image_only,
        "max_video_tokens": _video_tokens,
        "dummy_image_only": _dummy_image_only,
        "dummy_video": _dummy_video,
        "startup_budget": _startup_budget,
    }
    outcomes: dict[str, dict] = {}
    for name, call in calls.items():
        try:
            outcomes[name] = {"status": "ok", "value": call()}
        except Exception as exc:
            outcomes[name] = {
                "status": "error",
                "exc": type(exc).__name__,
                "is_value_error": isinstance(exc, ValueError),
                "message": str(exc).splitlines()[0][:500],
            }
    _emit({"event": "done", "status": "ok", "calls": outcomes})


def main() -> None:
    spec = json.loads(sys.argv[1])
    model_dir = Path(spec["model_dir"])
    try:
        _write_checkpoint(model_dir, spec)
        _profile(model_dir, spec)
    except Exception as exc:
        message = str(exc).splitlines()[0][:500]
        _emit(
            {
                "event": "done",
                "status": "error",
                "exc": type(exc).__name__,
                "is_value_error": isinstance(exc, ValueError),
                "message": message,
            }
        )


if __name__ == "__main__":
    main()
