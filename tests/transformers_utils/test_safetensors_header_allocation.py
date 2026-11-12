# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safetensors header lengths must be rejected before the read allocates.

``parse_safetensors_file_metadata`` and the weight-cache header reader pass
the file's little-endian uint64 header length straight to ``file.read``.
CPython allocates a bytes object of that size before it knows how much of
the file is left. These tests are the spec for closing that:

* Raise ``ValueError`` whose message contains ``safetensors header``.
* Do that when the declared length is greater than the bytes still in the
  opened file, or greater than the safetensors limit of 100_000_000 bytes.
* Measure the file through the descriptor that will be read. A size looked
  up from the path must not be what authorizes the read.
* Leave the directory untouched: rejecting the header must not create,
  delete, or rewrite a file.
* Keep accepting a header whose length equals the bytes that follow it and
  is within that limit.

The tests fail while the unbounded read is still in place.
"""

import json
import os
import struct
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import PreTrainedConfig

# Platform detection and Hub lookups run at import time. Pin them before
# importing vLLM so configuration tests stay on the local checkpoint.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm.config import CacheConfig, DeviceConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.model_loader.weight_cache.protocol import (
    WeightCacheKey,
    hash_checkpoint,
)
from vllm.transformers_utils.config import (
    MistralConfigParser,
    checkpoint_has_lm_head,
    get_safetensors_params_metadata,
)
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.transformers_utils.utils import parse_safetensors_file_metadata

# safetensors 0.8.0 rejects anything above this (`MAX_HEADER_SIZE`).
SAFETENSORS_MAX_HEADER_LENGTH = 100_000_000
# Large enough that a buffered read of the declared length is obvious, and
# still under the cap so a cap-only check would let it through.
ALLOCATION_PROBE_LENGTH = 8 * 1024 * 1024
# Peak traced allocation allowed while a header is being rejected.
MAX_REJECT_PEAK_BYTES = 1_048_576
HEADER_ERROR = "safetensors header"

DirectoryCallback = Callable[[Path], None]


def _llama_config(*, tie: bool, include_dtype: bool, quant: dict | None) -> dict:
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "tie_word_embeddings": tie,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    if include_dtype:
        config["torch_dtype"] = "float16"
        config["dtype"] = "float16"
    if quant is not None:
        config["quantization_config"] = quant
    return config


def _write_safetensors(path: Path, declared_length: int, body: bytes) -> None:
    path.write_bytes(struct.pack("<Q", declared_length) + body)


def _write_sparse_safetensors(path: Path, declared_length: int) -> None:
    """A header claim that fits in ``st_size`` without storing those bytes."""
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", declared_length))
        handle.truncate(8 + declared_length)


def _checkpoint(
    directory: Path,
    declared_length: int,
    body: bytes = b"{}",
    *,
    tie: bool = True,
    include_dtype: bool = True,
    quant: dict | None = None,
    params: dict | None = None,
    sparse: bool = False,
    weights_name: str = "model.safetensors",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    config_path.write_text(
        json.dumps(_llama_config(tie=tie, include_dtype=include_dtype, quant=quant))
    )
    weights = directory / weights_name
    if sparse:
        _write_sparse_safetensors(weights, declared_length)
    else:
        _write_safetensors(weights, declared_length, body)
    if params is not None:
        (directory / "params.json").write_text(json.dumps(params))
    return directory


def _encoded_header(payload: dict) -> bytes:
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _directory_snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _assert_header_rejected(
    directory: Path,
    callback: DirectoryCallback,
    *,
    bound_allocation: bool = True,
) -> None:
    before = _directory_snapshot(directory)
    tracemalloc.start()
    with pytest.raises(ValueError, match=HEADER_ERROR):
        callback(directory)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Model and engine config construction allocates for unrelated reasons.
    # The readers those configs call are bounded by the direct tests above.
    if bound_allocation:
        assert peak < MAX_REJECT_PEAK_BYTES, peak
    assert _directory_snapshot(directory) == before


def _oversized_directory(tmp_path: Path, declared_length: int) -> Path:
    """Tiny file whose header claims ``declared_length`` and whose tail is JSON."""
    return _checkpoint(tmp_path / "model", declared_length, b"{}")


def _parse(directory: Path) -> None:
    parse_safetensors_file_metadata(directory / "model.safetensors")


def _metadata(directory: Path) -> None:
    get_safetensors_params_metadata(str(directory))


def _lm_head(directory: Path) -> None:
    checkpoint_has_lm_head.cache_clear()
    checkpoint_has_lm_head(str(directory))


def _hash(directory: Path) -> None:
    hash_checkpoint(str(directory))


def _awq(directory: Path) -> None:
    AutoAWQConfig(4, 128, True, False).maybe_update_config(str(directory))


def _gptq(directory: Path) -> None:
    AutoGPTQConfig(4, 128, False, True, False, {}, {}).maybe_update_config(
        str(directory)
    )


def _dtype(directory: Path) -> None:
    ModelArchConfigConvertorBase.get_torch_dtype(
        PreTrainedConfig(),
        str(directory),
        None,
        "hf",
    )


def _mistral(directory: Path) -> None:
    MistralConfigParser().parse(str(directory), trust_remote_code=False)


def _weight_cache_key(directory: Path) -> None:
    model_config = ModelConfig(
        model=str(directory),
        tokenizer=str(directory),
        dtype="float16",
        max_model_len=16,
    )
    WeightCacheKey.from_model_config(
        model_config,
        tp_size=1,
        tp_rank=0,
        cache_config=CacheConfig(),
    )


def _vllm_config(
    directory: Path,
    *,
    load_format: str,
    quantization: str | None = None,
) -> VllmConfig:
    model_config = ModelConfig(
        model=str(directory),
        tokenizer=str(directory),
        dtype="float16",
        max_model_len=16,
        quantization=quantization,
    )
    return VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format=load_format),
        device_config=DeviceConfig(device="cpu"),
    )


READER_CASES = [
    pytest.param(_parse, id="parse_safetensors_file_metadata"),
    pytest.param(_metadata, id="get_safetensors_params_metadata"),
    pytest.param(_lm_head, id="checkpoint_has_lm_head"),
    pytest.param(_hash, id="hash_checkpoint"),
    pytest.param(_awq, id="auto_awq_maybe_update_config"),
    pytest.param(_gptq, id="auto_gptq_maybe_update_config"),
    pytest.param(_dtype, id="get_torch_dtype"),
    pytest.param(_mistral, id="mistral_config_parser"),
]

_MISTRAL_PARAMS = {
    "dim": 32,
    "n_layers": 1,
    "n_heads": 4,
    "n_kv_heads": 4,
    "hidden_dim": 64,
    "vocab_size": 128,
    "norm_eps": 1e-5,
    "max_seq_len": 32,
}


@pytest.mark.parametrize("read_header", READER_CASES)
def test_header_longer_than_file_is_rejected(tmp_path: Path, read_header) -> None:
    """An 8 MiB claim in a file of a few bytes must not be read or returned."""
    params = _MISTRAL_PARAMS if read_header is _mistral else None
    directory = _checkpoint(
        tmp_path / "model",
        ALLOCATION_PROBE_LENGTH,
        b"{}",
        params=params,
    )
    _assert_header_rejected(directory, read_header)


@pytest.mark.parametrize(
    "read_header",
    [
        pytest.param(_parse, id="parse_safetensors_file_metadata"),
        pytest.param(_hash, id="hash_checkpoint"),
    ],
)
def test_header_above_safetensors_limit_is_rejected(
    tmp_path: Path, read_header
) -> None:
    """A sparse file can report a huge ``st_size`` without holding the bytes.

    The declared length fits in that size, so a check against ``stat`` alone
    still allocates it. The safetensors cap has to reject it first.
    """
    declared = SAFETENSORS_MAX_HEADER_LENGTH + 1
    directory = _checkpoint(tmp_path / "model", declared, sparse=True)
    _assert_header_rejected(directory, read_header)


def test_one_byte_past_the_file_is_rejected(tmp_path: Path) -> None:
    """The length is compared with the bytes left after the 8-byte prefix."""
    directory = _oversized_directory(tmp_path, len(b"{}") + 1)
    _assert_header_rejected(directory, _parse)
    _assert_header_rejected(directory, _metadata)


def test_length_that_does_not_fit_an_index_is_rejected(tmp_path: Path) -> None:
    directory = _oversized_directory(tmp_path, 2**64 - 1)
    _assert_header_rejected(directory, _parse)
    _assert_header_rejected(directory, _hash)


def test_multi_gigabyte_header_is_rejected_without_memory_error(
    tmp_path: Path,
) -> None:
    """The size that aborts startup today is a terabyte-sized ``read``."""
    directory = _oversized_directory(tmp_path, 2**40)
    _assert_header_rejected(directory, _parse)
    _assert_header_rejected(directory, _metadata)


def _lie_about_file_size(path: Path, reported_size: int):
    real_stat = os.stat
    identity = real_stat(path)

    def lying_stat(target, *args, **kwargs):
        result = real_stat(target, *args, **kwargs)
        if (result.st_dev, result.st_ino) != (identity.st_dev, identity.st_ino):
            return result
        values = list(result)
        values[6] = reported_size
        return os.stat_result(values)

    return patch("os.stat", lying_stat)


@pytest.mark.parametrize(
    "read_header",
    [
        pytest.param(_parse, id="parse_safetensors_file_metadata"),
        pytest.param(_metadata, id="get_safetensors_params_metadata"),
        pytest.param(_hash, id="hash_checkpoint"),
    ],
)
def test_path_stat_cannot_authorize_the_header(tmp_path: Path, read_header) -> None:
    """A second lookup of the path must not be the size that is enforced.

    ``os.stat`` reports a file large enough for the claim. The open file is
    only a few bytes. The read has to follow the open file.
    """
    directory = _oversized_directory(tmp_path, ALLOCATION_PROBE_LENGTH)
    weights = directory / "model.safetensors"
    reported = 8 + ALLOCATION_PROBE_LENGTH
    with _lie_about_file_size(weights, reported):
        _assert_header_rejected(directory, read_header)


def test_sibling_shard_does_not_hide_an_oversized_header(tmp_path: Path) -> None:
    directory = _checkpoint(tmp_path / "model", ALLOCATION_PROBE_LENGTH, b"{}")
    valid = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [1, 1],
            "data_offsets": [0, 2],
        }
    }
    (directory / "model-00002-of-00002.safetensors").write_bytes(_encoded_header(valid))
    _assert_header_rejected(directory, _metadata)


def test_exact_header_is_parsed(tmp_path: Path) -> None:
    payload = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [1, 1],
            "data_offsets": [0, 2],
        }
    }
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(_encoded_header(payload))
    assert parse_safetensors_file_metadata(directory / "model.safetensors") == payload
    metadata = get_safetensors_params_metadata(str(directory))
    assert metadata["model.embed_tokens.weight"]["dtype"] == "F16"
    checkpoint_has_lm_head.cache_clear()
    assert checkpoint_has_lm_head(str(directory)) is False
    assert hash_checkpoint(str(directory))


def test_vllm_config_rejects_oversized_header_when_embeddings_are_tied(
    tmp_path: Path,
) -> None:
    directory = _oversized_directory(tmp_path, ALLOCATION_PROBE_LENGTH)
    _assert_header_rejected(
        directory,
        lambda path: _vllm_config(path, load_format="safetensors"),
        bound_allocation=False,
    )


def test_vllm_config_rejects_terabyte_header_when_embeddings_are_tied(
    tmp_path: Path,
) -> None:
    directory = _oversized_directory(tmp_path, 2**40)
    _assert_header_rejected(
        directory,
        lambda path: _vllm_config(path, load_format="auto"),
        bound_allocation=False,
    )


def test_vllm_config_awq_rejects_oversized_header(tmp_path: Path) -> None:
    """AWQ rebuilds ``modules_to_not_convert`` while the config is built.

    Tied embeddings are off and the load format reads no weights, so the
    only header read is the quant config update.
    """
    quant = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
    }
    directory = _checkpoint(
        tmp_path / "model",
        ALLOCATION_PROBE_LENGTH,
        b"{}",
        tie=False,
        quant=quant,
    )
    _assert_header_rejected(
        directory,
        lambda path: _vllm_config(path, load_format="dummy", quantization="awq"),
        bound_allocation=False,
    )


def test_vllm_config_gptq_rejects_oversized_header(tmp_path: Path) -> None:
    quant = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
    }
    directory = _checkpoint(
        tmp_path / "model",
        ALLOCATION_PROBE_LENGTH,
        b"{}",
        tie=False,
        quant=quant,
    )
    _assert_header_rejected(
        directory,
        lambda path: _vllm_config(path, load_format="dummy"),
        bound_allocation=False,
    )


def test_model_config_auto_dtype_rejects_oversized_header(tmp_path: Path) -> None:
    """Missing ``dtype`` is filled from the safetensors header at startup."""
    directory = _checkpoint(
        tmp_path / "model",
        ALLOCATION_PROBE_LENGTH,
        b"{}",
        tie=False,
        include_dtype=False,
    )
    _assert_header_rejected(
        directory,
        lambda path: ModelConfig(
            model=str(path),
            tokenizer=str(path),
            dtype="auto",
            max_model_len=16,
        ),
        bound_allocation=False,
    )


def test_weight_cache_key_rejects_oversized_header(tmp_path: Path) -> None:
    directory = _oversized_directory(tmp_path, ALLOCATION_PROBE_LENGTH)
    _assert_header_rejected(directory, _weight_cache_key, bound_allocation=False)


def test_vllm_config_loads_a_checkpoint_whose_header_matches(
    tmp_path: Path,
) -> None:
    payload = {
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [1, 1],
            "data_offsets": [0, 2],
        },
        "lm_head.weight": {
            "dtype": "F16",
            "shape": [1, 1],
            "data_offsets": [2, 4],
        },
    }
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(_llama_config(tie=True, include_dtype=True, quant=None))
    )
    (directory / "model.safetensors").write_bytes(_encoded_header(payload))
    checkpoint_has_lm_head.cache_clear()
    config = _vllm_config(directory, load_format="safetensors")
    assert config.model_config.hf_config.tie_word_embeddings is False
    assert config.model_config.word_embeddings_untied_by_checkpoint is True
