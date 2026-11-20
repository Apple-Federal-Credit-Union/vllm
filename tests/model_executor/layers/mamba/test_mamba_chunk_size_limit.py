# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint mamba chunk size must be a small positive integer.

Mamba2-family configs carry ``chunk_size`` or ``mamba_chunk_size``. Startup
profiling runs each Mamba2 mixer with no attention metadata, and that path
sizes its dummy sequence to the checkpoint value. These tests are the spec
for rejecting values that are not positive integers at most 8192 before any
tensor is allocated from that value, including when the field is read again
between the check and the allocation.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba import mamba_mixer2 as mixer_module
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    MambaMixer2,
    mamba_mixer2,
)
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

# Inclusive upper bound. Real Mamba2-family checkpoints use 128 or 256.
MAX_MAMBA_CHUNK_SIZE = 8192
# Larger than any legitimate warmup, small enough that a missed check still
# reports the requested shape instead of exhausting this process.
_ALLOCATION_CEILING = 16384
HUGE_CHUNK_SIZE = 2**30

_LAYER_COUNTER = 0


def _mamba2_config(chunk_size: int) -> dict:
    return {
        "model_type": "mamba2",
        "architectures": ["Mamba2ForCausalLM"],
        "hidden_size": 64,
        "num_heads": 4,
        "head_dim": 32,
        "expand": 2,
        "n_groups": 1,
        "state_size": 16,
        "num_hidden_layers": 1,
        "vocab_size": 128,
        "conv_kernel": 4,
        "chunk_size": chunk_size,
    }


def _falcon_h1_config(mamba_chunk_size: int) -> dict:
    return {
        "model_type": "falcon_h1",
        "architectures": ["FalconH1ForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "mamba_d_ssm": 64,
        "mamba_n_heads": 4,
        "mamba_d_head": 16,
        "mamba_n_groups": 1,
        "mamba_d_state": 8,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "mamba_chunk_size": mamba_chunk_size,
    }


def _granite_config(mamba_chunk_size: int) -> dict:
    return {
        "model_type": "granitemoehybrid",
        "architectures": ["GraniteMoeHybridForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "layer_types": ["linear_attention"],
        "mamba_n_heads": 4,
        "mamba_d_head": 32,
        "mamba_expand": 2,
        "mamba_n_groups": 1,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_chunk_size": mamba_chunk_size,
    }


def _nemotron_h_config(chunk_size: int) -> dict:
    return {
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "layer_types": ["linear_attention"],
        "ssm_state_size": 16,
        "mamba_num_heads": 4,
        "mamba_head_dim": 32,
        "n_groups": 1,
        "chunk_size": chunk_size,
    }


def _nemotron_h_alias_config(mamba_chunk_size: int) -> dict:
    """Nemotron copies ``mamba_chunk_size`` onto ``chunk_size`` while parsing."""
    config = _nemotron_h_config(128)
    del config["chunk_size"]
    config["mamba_chunk_size"] = mamba_chunk_size
    return config


def _zamba2_config(chunk_size: int) -> dict:
    return {
        "model_type": "zamba2",
        "architectures": ["Zamba2ForCausalLM"],
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "layers_block_type": ["linear_attention"],
        "num_attention_heads": 4,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "mamba_d_state": 16,
        "mamba_expand": 2,
        "n_mamba_heads": 4,
        "chunk_size": chunk_size,
    }


_FAMILIES = {
    "mamba2": _mamba2_config,
    "falcon_h1": _falcon_h1_config,
    "granitemoehybrid": _granite_config,
    "nemotron_h": _nemotron_h_config,
    "nemotron_h_mamba_chunk_size": _nemotron_h_alias_config,
    "zamba2": _zamba2_config,
}


def _write_checkpoint(tmp_path, family: str, chunk_size: int):
    config = _FAMILIES[family](chunk_size)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def _load_model_config(model_dir) -> ModelConfig:
    return ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        max_model_len=128,
        enforce_eager=True,
    )


def _vllm_config(model_config: ModelConfig) -> VllmConfig:
    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=16, cache_dtype="auto"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=16,
            max_model_len=128,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        parallel_config=ParallelConfig(),
    )


def _leading_dim(args, kwargs) -> int | None:
    if args and isinstance(args[0], int):
        return args[0]
    if args and isinstance(args[0], (tuple, list)) and args[0]:
        first = args[0][0]
        return first if isinstance(first, int) else None
    size = kwargs.get("size")
    if isinstance(size, int):
        return size
    if isinstance(size, (tuple, list)) and size and isinstance(size[0], int):
        return size[0]
    return None


@contextmanager
def _guard_warmup_allocations() -> Iterator[list[int]]:
    """Record leading dims of mixer warmup allocations and refuse huge ones.

    The product computes the shape. Sizes at or below the ceiling are really
    allocated. Larger sizes are the checkpoint-controlled buffer this test
    exists to forbid, so they are refused after the requested dim is recorded.
    """
    requested: list[int] = []
    originals = {
        name: getattr(mixer_module.torch, name) for name in ("empty", "randn", "zeros")
    }
    # Warmup frees autotune scratch after the sized allocations. That call is
    # not what chooses the sequence length. On a host with no accelerator it
    # raises before the test can see the shapes the product requested.
    empty_cache = mixer_module.torch.accelerator.empty_cache
    mixer_module.torch.accelerator.empty_cache = lambda: None

    def wrap(fn):
        def inner(*args, **kwargs):
            leading = _leading_dim(args, kwargs)
            if leading is not None:
                requested.append(leading)
                if leading > _ALLOCATION_CEILING:
                    raise MemoryError(f"warmup requested leading dimension {leading}")
            return fn(*args, **kwargs)

        return inner

    for name, fn in originals.items():
        setattr(mixer_module.torch, name, wrap(fn))
    try:
        yield requested
    finally:
        for name, fn in originals.items():
            setattr(mixer_module.torch, name, fn)
        mixer_module.torch.accelerator.empty_cache = empty_cache


def _profile_mixer_forward(vllm_config: VllmConfig) -> None:
    """Run the custom op the profile forward runs with no attention metadata.

    ``GPUModelRunner.profile_run`` calls ``_dummy_run(is_profile=True)``, which
    forces eager mode and leaves ``attn_metadata`` unset. ``MambaMixer2`` then
    always enters ``_warmup_ssd_kernels``.
    """
    global _LAYER_COUNTER
    _LAYER_COUNTER += 1
    prefix = f"model.layers.{_LAYER_COUNTER}.mixer"
    with set_current_vllm_config(vllm_config):
        mixer = MambaMixer2(
            hidden_size=64,
            ssm_state_size=16,
            conv_kernel_size=4,
            intermediate_size=128,
            use_conv_bias=True,
            use_bias=False,
            n_groups=1,
            num_heads=4,
            head_dim=32,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            prefix=prefix,
        )
        last_dim = (
            mixer.tped_intermediate_size + mixer.tped_conv_size + mixer.tped_dt_size
        )
        projected = torch.zeros(1, last_dim)
        output = torch.empty(1, mixer.num_heads * mixer.head_dim)
        with set_forward_context(None, vllm_config):
            mamba_mixer2(projected, output, prefix)


def _uses_malicious_length(requested: list[int], malicious: int) -> bool:
    return any(
        leading == malicious or leading > MAX_MAMBA_CHUNK_SIZE for leading in requested
    )


class _LaterLookup:
    """First read returns a legal chunk size; every later read returns a huge one."""

    def __init__(self, first: int, later: int):
        self._reads = 0
        self._first = first
        self._later = later

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        self._reads += 1
        return self._first if self._reads == 1 else self._later

    def __set__(self, obj, value):
        raise AttributeError(value)


def _install_later_lookup(config, name: str, first: int, later: int) -> None:
    class _Proxy(config.__class__):
        pass

    setattr(_Proxy, name, _LaterLookup(first, later))
    config.__class__ = _Proxy


@pytest.fixture
def tp_world(default_vllm_config):
    import os
    import tempfile

    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    backend = current_platform.dist_backend or "gloo"
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend=backend,
    )
    initialize_model_parallel(1, 1)
    try:
        yield
    finally:
        os.unlink(temp_file)


@pytest.mark.parametrize("family", list(_FAMILIES))
@pytest.mark.parametrize("chunk_size", [0, -8, 8193, HUGE_CHUNK_SIZE])
def test_profile_warmup_rejects_checkpoint_chunk_size(
    tp_world, tmp_path, family, chunk_size
):
    """A crafted config must fail before warmup allocates that sequence length."""
    model_dir = _write_checkpoint(tmp_path, family, chunk_size)
    with (
        _guard_warmup_allocations() as requested,
        pytest.raises(ValueError, match="chunk"),
    ):
        model_config = _load_model_config(model_dir)
        _profile_mixer_forward(_vllm_config(model_config))
    assert not _uses_malicious_length(requested, chunk_size)


@pytest.mark.parametrize(
    "family,chunk_size",
    [("mamba2", 256), ("mamba2", MAX_MAMBA_CHUNK_SIZE), ("falcon_h1", 256)],
)
def test_profile_warmup_uses_accepted_chunk_size(
    tp_world, tmp_path, family, chunk_size
):
    """A legal checkpoint value is what the profile dummy sequence is sized to."""
    model_dir = _write_checkpoint(tmp_path, family, chunk_size)
    with _guard_warmup_allocations() as requested:
        model_config = _load_model_config(model_dir)
        _profile_mixer_forward(_vllm_config(model_config))
    assert chunk_size in requested
    assert not _uses_malicious_length(requested, HUGE_CHUNK_SIZE)


@pytest.mark.parametrize(
    ("family", "field_name"),
    [("mamba2", "chunk_size"), ("falcon_h1", "mamba_chunk_size")],
)
def test_warmup_does_not_use_a_chunk_size_reread(
    tp_world, tmp_path, family, field_name
):
    """Checking the field and then reading it again must not observe a new value.

    The first read is a legal chunk size. A second read of the same attribute
    returns a huge value. The profile warmup may use only the value that was
    validated, and must not allocate the later value.
    """
    model_dir = _write_checkpoint(tmp_path, family, 256)
    model_config = _load_model_config(model_dir)
    _install_later_lookup(model_config.hf_text_config, field_name, 256, HUGE_CHUNK_SIZE)
    with _guard_warmup_allocations() as requested:
        try:
            _profile_mixer_forward(_vllm_config(model_config))
        except ValueError as exc:
            assert "chunk" in str(exc).lower()
    assert HUGE_CHUNK_SIZE not in requested
    assert not any(leading > MAX_MAMBA_CHUNK_SIZE for leading in requested)


def _build_metadata(vllm_config: VllmConfig):
    spec = MambaSpec(
        block_size=16,
        shapes=((16, 64),),
        dtypes=(torch.float32,),
    )
    builder = Mamba2AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[8], query_lens=[8]),
        block_size=16,
        device=torch.device("cpu"),
    )
    common = common.replace(is_prefilling=torch.ones(1, dtype=torch.bool))
    return builder.build(common_prefix_len=0, common_attn_metadata=common)


@pytest.mark.parametrize("chunk_size", [0, -8, HUGE_CHUNK_SIZE])
def test_mamba2_metadata_rejects_checkpoint_chunk_size(tp_world, tmp_path, chunk_size):
    """Metadata construction divides by the same checkpoint chunk size."""
    model_dir = _write_checkpoint(tmp_path, "mamba2", chunk_size)
    with pytest.raises(ValueError, match="chunk"):
        model_config = _load_model_config(model_dir)
        _build_metadata(_vllm_config(model_config))


def test_mamba2_metadata_does_not_use_a_chunk_size_reread(tp_world, tmp_path):
    """The chunk size stored for metadata must be the validated read."""
    model_dir = _write_checkpoint(tmp_path, "mamba2", 256)
    model_config = _load_model_config(model_dir)
    _install_later_lookup(model_config.hf_text_config, "chunk_size", 256, 0)
    vllm_config = _vllm_config(model_config)
    try:
        metadata = _build_metadata(vllm_config)
    except ValueError as exc:
        assert "chunk" in str(exc).lower()
        return
    assert metadata.chunk_size == 256
    assert metadata.chunk_size > 0
