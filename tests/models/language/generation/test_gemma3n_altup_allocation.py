# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma3n must not size AltUp from an uncapped checkpoint integer.

``altup_num_inputs`` is read from the checkpoint ``config.json`` and, on the
text and multimodal language-model paths, sizes:

* ``Gemma3nAltUp.prediction_coefs`` as ``(altup_num_inputs**2, altup_num_inputs)``
* ``Gemma3nTextModel.altup_unembed_projections``
* ``Gemma3nSelfDecoder.altup_projections``
* the fast-prefill hidden-state buffer when that cache flag is on

Published Gemma3n configs use 4. A checkpoint may set a much larger integer.
Construction has to reject that value, or clamp every one of those sites to a
small count, before allocating the cubic prediction matrix.
"""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.load import LoadConfig
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.gemma3n import Gemma3nForCausalLM, Gemma3nTextModel
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.utils.torch_utils import set_default_torch_dtype

# Real Gemma3n checkpoints use 4. Sixteen is still a tiny AltUp width
# (16**3 parameters). Anything above this is the unbounded checkpoint field.
_MAX_ALTUP_INPUTS = 16
_MALICIOUS_ALTUP_INPUTS = 48
_MAX_ELEMENTS = 1_000_000

_DIST_READY = False


def _text_config(altup_num_inputs: int, *, num_hidden_layers: int) -> dict:
    return {
        "model_type": "gemma3n_text",
        "architectures": ["Gemma3nForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "sliding_window": 32,
        "vocab_size_per_layer_input": 64,
        "hidden_size_per_layer_input": 16,
        "altup_num_inputs": altup_num_inputs,
        "altup_active_idx": 0,
        "altup_coef_clip": 120.0,
        "altup_correct_scale": True,
        "num_kv_shared_layers": 0,
        "laurel_rank": 8,
        "tie_word_embeddings": True,
    }


def _write_config(directory: Path, payload: dict) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(directory)


def _shape_of(args: tuple, kwargs: dict) -> tuple[int, ...] | None:
    if args and isinstance(args[0], (tuple, list, torch.Size)):
        return tuple(int(dim) for dim in args[0])
    dims: list[int] = []
    for arg in args:
        if isinstance(arg, bool) or not isinstance(arg, int):
            break
        dims.append(arg)
    if dims:
        return tuple(dims)
    size = kwargs.get("size")
    if isinstance(size, (tuple, list, torch.Size)):
        return tuple(int(dim) for dim in size)
    if isinstance(size, int):
        return (size,)
    return None


class _AllocationTrace:
    """Records tensor shapes requested while a model is constructed."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []
        self.unembed_projections = 0
        self.embed_projections = 0
        self._empty = torch.empty
        self._zeros = torch.zeros
        self._column_init = ColumnParallelLinear.__init__

    def __enter__(self) -> "_AllocationTrace":
        trace = self

        def empty(*args, **kwargs):
            shape = _shape_of(args, kwargs)
            if shape is not None:
                trace.shapes.append(shape)
                if math.prod(shape) > _MAX_ELEMENTS:
                    raise RuntimeError(
                        f"weight allocation of shape {shape} exceeds the "
                        "fixed element budget"
                    )
            return trace._empty(*args, **kwargs)

        def zeros(*args, **kwargs):
            shape = _shape_of(args, kwargs)
            if shape is not None:
                trace.shapes.append(shape)
                if math.prod(shape) > _MAX_ELEMENTS:
                    raise RuntimeError(
                        f"buffer allocation of shape {shape} exceeds the "
                        "fixed element budget"
                    )
            return trace._zeros(*args, **kwargs)

        def column_init(layer, *args, **kwargs):
            prefix = kwargs.get("prefix", "")
            if "altup_unembed_projections" in prefix:
                trace.unembed_projections += 1
            elif "altup_projections" in prefix:
                trace.embed_projections += 1
            return trace._column_init(layer, *args, **kwargs)

        torch.empty = empty
        torch.zeros = zeros
        ColumnParallelLinear.__init__ = column_init
        return self

    def __exit__(self, *exc) -> None:
        torch.empty = self._empty
        torch.zeros = self._zeros
        ColumnParallelLinear.__init__ = self._column_init

    def cubic_shapes(self, altup_num_inputs: int) -> list[tuple[int, ...]]:
        banned = altup_num_inputs**3
        return [shape for shape in self.shapes if math.prod(shape) >= banned]


def _ensure_dist(vllm_config: VllmConfig) -> None:
    global _DIST_READY
    if _DIST_READY:
        return
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
    _DIST_READY = True


def _vllm_config(
    model_dir: str,
    *,
    fast_prefill: bool,
) -> tuple[ModelConfig, VllmConfig]:
    model_config = ModelConfig(
        model=model_dir,
        tokenizer_mode="skip",
        max_model_len=32,
        dtype="float32",
        enforce_eager=True,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(kv_sharing_fast_prefill=fast_prefill),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=128,
            max_num_seqs=4,
            max_model_len=32,
            is_encoder_decoder=False,
        ),
    )
    return model_config, vllm_config


def _create_causal_lm(
    model_dir: str,
    *,
    fast_prefill: bool,
) -> Gemma3nForCausalLM:
    model_config, vllm_config = _vllm_config(model_dir, fast_prefill=fast_prefill)
    _ensure_dist(vllm_config)
    loader = DefaultModelLoader(LoadConfig())
    with set_default_torch_dtype(model_config.dtype):
        with torch.device("cpu"):
            model = loader.create_model(
                vllm_config=vllm_config,
                model_config=model_config,
            )
    assert isinstance(model, Gemma3nForCausalLM)
    return model


def _create_conditional_language_model(model_dir: str) -> Gemma3nForCausalLM:
    """Build the language model the multimodal class hands ``text_config`` to."""
    model_config, vllm_config = _vllm_config(model_dir, fast_prefill=False)
    assert model_config.hf_config.text_config.altup_num_inputs
    _ensure_dist(vllm_config)
    with set_default_torch_dtype(model_config.dtype):
        with torch.device("cpu"):
            model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=model_config.hf_config.text_config,
                architectures=["Gemma3nForCausalLM"],
                prefix="language_model",
            )
    assert isinstance(model, Gemma3nForCausalLM)
    return model


def _assert_altup_bounded(model: Gemma3nForCausalLM, requested: int) -> None:
    text: Gemma3nTextModel = model.model
    assert text.layers, "decoder layers were not built"
    widths = set()
    for layer in text.layers:
        weight = layer.altup.prediction_coefs.weight
        width = weight.shape[1]
        widths.add(width)
        assert width <= _MAX_ALTUP_INPUTS, (
            f"prediction_coefs kept checkpoint altup_num_inputs={requested} "
            f"and allocated shape {tuple(weight.shape)}"
        )
        assert weight.shape[0] == width**2
        correction = layer.altup.correction_coefs.weight
        assert correction.shape == (width, width)
        router = layer.altup.modality_router.weight
        assert router.shape[0] == width
    assert len(widths) == 1
    width = widths.pop()
    assert len(text.altup_unembed_projections) <= _MAX_ALTUP_INPUTS - 1
    assert len(text.self_decoder.altup_projections) <= _MAX_ALTUP_INPUTS - 1
    assert len(text.altup_unembed_projections) == max(width - 1, 0)
    assert len(text.self_decoder.altup_projections) == max(width - 1, 0)
    hidden_states = getattr(text, "hidden_states", None)
    if hidden_states is not None:
        assert hidden_states.shape[-1] == width
        assert hidden_states.shape[-1] <= _MAX_ALTUP_INPUTS


def _assert_refusal_left_no_cubic_state(
    trace: _AllocationTrace,
    requested: int,
) -> None:
    assert trace.unembed_projections <= _MAX_ALTUP_INPUTS - 1, (
        "altup_unembed_projections were built from the uncapped count "
        f"({trace.unembed_projections}) before construction was refused"
    )
    assert trace.embed_projections <= _MAX_ALTUP_INPUTS - 1, (
        "altup_projections were built from the uncapped count "
        f"({trace.embed_projections}) before construction was refused"
    )
    cubic = trace.cubic_shapes(requested)
    assert not cubic, f"cubic AltUp weights were allocated: {cubic}"


@pytest.fixture(scope="module", autouse=True)
def _offline_hub():
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    yield
    if previous is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = previous


@pytest.mark.cpu_test
def test_published_altup_width_still_builds(tmp_path: Path) -> None:
    """A checkpoint that uses the published AltUp width still constructs."""
    model_dir = _write_config(tmp_path, _text_config(4, num_hidden_layers=2))
    with _AllocationTrace() as trace:
        model = _create_causal_lm(model_dir, fast_prefill=True)
    weight = model.model.layers[0].altup.prediction_coefs.weight
    assert tuple(weight.shape) == (16, 4)
    assert len(model.model.layers) == 2
    assert tuple(model.model.layers[1].altup.prediction_coefs.weight.shape) == (
        16,
        4,
    )
    assert len(model.model.altup_unembed_projections) == 3
    assert len(model.model.self_decoder.altup_projections) == 3
    assert model.model.hidden_states.shape[-1] == 4
    assert trace.unembed_projections == 3
    assert trace.embed_projections == 3


@pytest.mark.cpu_test
def test_uncapped_text_checkpoint_does_not_allocate_cubic_altup(
    tmp_path: Path,
) -> None:
    """A large text-config ``altup_num_inputs`` must not allocate N**3."""
    requested = _MALICIOUS_ALTUP_INPUTS
    model_dir = _write_config(
        tmp_path,
        _text_config(requested, num_hidden_layers=2),
    )
    model_config, _ = _vllm_config(model_dir, fast_prefill=True)
    assert model_config.hf_config.altup_num_inputs == requested

    with _AllocationTrace() as trace:
        try:
            model = _create_causal_lm(model_dir, fast_prefill=True)
        except ValueError as exc:
            assert "altup" in str(exc).lower()
            _assert_refusal_left_no_cubic_state(trace, requested)
            return
    _assert_altup_bounded(model, requested)
    _assert_refusal_left_no_cubic_state(trace, requested)


@pytest.mark.cpu_test
def test_uncapped_multimodal_text_config_does_not_allocate_cubic_altup(
    tmp_path: Path,
) -> None:
    """Nested ``text_config.altup_num_inputs`` takes the same language-model path.

    ``Gemma3nForConditionalGeneration`` passes ``config.text_config`` to
    ``init_vllm_registered_model``. The vision and audio towers do not size
    AltUp and are not built here.
    """
    requested = _MALICIOUS_ALTUP_INPUTS
    payload = {
        "model_type": "gemma3n",
        "architectures": ["Gemma3nForConditionalGeneration"],
        "text_config": _text_config(requested, num_hidden_layers=2),
        "vision_config": {
            "hidden_size": 32,
            "vocab_size": 8,
            "vocab_offset": 128,
        },
        "audio_config": {
            "hidden_size": 32,
            "vocab_size": 8,
            "vocab_offset": 136,
            "conf_num_hidden_layers": 1,
        },
    }
    model_dir = _write_config(tmp_path, payload)
    model_config, _ = _vllm_config(model_dir, fast_prefill=False)
    assert model_config.hf_config.text_config.altup_num_inputs == requested

    with _AllocationTrace() as trace:
        try:
            model = _create_conditional_language_model(model_dir)
        except ValueError as exc:
            assert "altup" in str(exc).lower()
            _assert_refusal_left_no_cubic_state(trace, requested)
            return
    _assert_altup_bounded(model, requested)
    _assert_refusal_left_no_cubic_state(trace, requested)
