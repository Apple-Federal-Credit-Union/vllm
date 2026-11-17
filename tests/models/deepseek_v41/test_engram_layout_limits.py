# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram layout bounds for checkpoint-controlled prime search.

``EngramLayout.from_config`` is how both DeepSeek V4.1 model constructors
turn ``config.json`` into hash-bucket primes
(``DeepseekV4Model.__init__`` in ``nvidia/model.py`` and ``amd/model.py``).
The primality test is only defined for integers below ``2**32``. These tests
require the layout builder to reject checkpoint fields outside the ranges
below before that search runs, and to keep building layouts for the
DeepSeek-V4.1-Flash checkpoint and for the top of each range.

Accepted ranges (inclusive, except an empty ``engram_layer_ids`` which skips
the layout):

- ``engram_vocab_size``: ``2`` .. ``2**31``
- ``engram_n_heads``: ``1`` .. ``64``
- ``engram_max_ngram_size``: ``2`` .. ``8``
- ``len(engram_layer_ids)``: ``0`` .. ``64``
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from vllm.models.deepseek_v41.common.engram import EngramLayout
from vllm.transformers_utils.config import get_config, get_hf_text_config

pytestmark = pytest.mark.skip_global_cleanup

_REPO_ROOT = Path(__file__).resolve().parents[3]

# DeepSeek-V4.1-Flash text_config Engram fields.
_FLASH_ENGRAM = {
    "engram_layer_ids": [1, 14],
    "engram_num_embeddings": [384006168, 384016682],
    "engram_max_ngram_size": 4,
    "engram_vocab_size": 16_000_000,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_pad_token_id": 2,
    "engram_compressed_vocab_size": 99092,
}

_MAX_VOCAB_SIZE = 2**31
_MAX_N_HEADS = 64
_MAX_NGRAM_SIZE = 8
_MAX_LAYER_COUNT = 64


def _engram_fields(**overrides: object) -> dict:
    layer_ids = overrides.get("engram_layer_ids", [1])
    fields: dict = {
        "engram_layer_ids": layer_ids,
        "engram_num_embeddings": [128] * len(layer_ids),
        "engram_max_ngram_size": 2,
        "engram_vocab_size": 1000,
        "engram_n_heads": 1,
        "engram_head_dim": 8,
        "engram_pad_token_id": 0,
        "engram_compressed_vocab_size": 16,
    }
    fields.update(overrides)
    if "engram_num_embeddings" not in overrides:
        fields["engram_num_embeddings"] = [128] * len(fields["engram_layer_ids"])
    return fields


def _config_json(engram: dict, vocab_digits: str | None = None) -> str:
    """Serialize a checkpoint config. ``vocab_digits`` is written as a JSON integer.

    ``json.dumps`` refuses integers wider than the interpreter digit limit, so
    the widest value ``config.json`` can still carry is spliced in as text.
    """
    body = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": 128,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            **engram,
        },
        "vision_config": {
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
        },
    }
    if vocab_digits is None:
        return json.dumps(body)
    body["text_config"]["engram_vocab_size"] = 0
    text = json.dumps(body)
    needle = '"engram_vocab_size": 0'
    if needle not in text:
        raise RuntimeError("engram_vocab_size placeholder missing from config.json")
    return text.replace(needle, '"engram_vocab_size": ' + vocab_digits, 1)


def _load_config(engram: dict, vocab_digits: str | None = None):
    """Load a DeepSeek V4.1 ``config.json`` the way ``ModelConfig`` does."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(
            _config_json(engram, vocab_digits),
            encoding="utf-8",
        )
        return get_config(tmp, trust_remote_code=False)


def _assert_rejected(engram: dict, field_name: str) -> None:
    config = _load_config(engram)
    with pytest.raises(ValueError, match=field_name):
        EngramLayout.from_config(config)


def test_flash_checkpoint_engram_layout_builds() -> None:
    """Published Flash sizes stay inside the range and still hash to primes."""
    config = _load_config(dict(_FLASH_ENGRAM))
    assert get_hf_text_config(config) is config
    assert config.engram_vocab_size == 16_000_000
    layout = EngramLayout.from_config(config)
    assert layout is not None
    assert len(layout.primes) == 2
    assert len(layout.primes[0]) == 3
    assert len(layout.primes[0][0]) == 8
    assert layout.primes[0][0][0] > 16_000_000


def test_empty_engram_layer_ids_skip_layout() -> None:
    config = _load_config(_engram_fields(engram_layer_ids=[]))
    assert EngramLayout.from_config(config) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"engram_vocab_size": _MAX_VOCAB_SIZE},
        {"engram_n_heads": _MAX_N_HEADS},
        {"engram_max_ngram_size": _MAX_NGRAM_SIZE},
        {"engram_layer_ids": list(range(_MAX_LAYER_COUNT))},
    ],
)
def test_upper_bound_engram_fields_still_build(overrides: dict) -> None:
    config = _load_config(_engram_fields(**overrides))
    layout = EngramLayout.from_config(config)
    assert layout is not None
    assert layout.primes


def test_vocab_at_or_above_2_32_is_rejected() -> None:
    _assert_rejected(
        _engram_fields(engram_vocab_size=2**32),
        "engram_vocab_size",
    )


def test_vocab_above_documented_max_is_rejected() -> None:
    _assert_rejected(
        _engram_fields(engram_vocab_size=_MAX_VOCAB_SIZE + 1),
        "engram_vocab_size",
    )


def test_too_many_heads_is_rejected() -> None:
    _assert_rejected(
        _engram_fields(engram_n_heads=_MAX_N_HEADS + 1),
        "engram_n_heads",
    )


def test_ngram_size_above_max_is_rejected() -> None:
    _assert_rejected(
        _engram_fields(engram_max_ngram_size=_MAX_NGRAM_SIZE + 1),
        "engram_max_ngram_size",
    )


def test_too_many_engram_layers_is_rejected() -> None:
    _assert_rejected(
        _engram_fields(engram_layer_ids=list(range(_MAX_LAYER_COUNT + 1))),
        "engram_layer_ids",
    )


def _max_json_int_digits() -> str:
    """Widest odd integer ``json.loads`` still accepts at the default digit limit.

    CPython's default ``int`` conversion limit is 4300 decimal digits. One
    more digit raises ``ValueError`` inside ``json.loads`` and never reaches
    the layout builder. This value is odd and is accepted by that parser.
    """
    return "1" + ("0" * 4298) + "1"


def _child_wide_vocab() -> int:
    """Return 0 when a max-width vocab is rejected before the prime search."""
    config = _load_config(
        _engram_fields(engram_vocab_size=0),
        vocab_digits=_max_json_int_digits(),
    )
    vocab = config.engram_vocab_size
    if not isinstance(vocab, int) or vocab.bit_length() < 14000:
        return 5
    try:
        layout = EngramLayout.from_config(config)
    except ValueError as exc:
        return 0 if "engram_vocab_size" in str(exc) else 3
    return 2 if layout is not None else 4


def test_wide_vocab_is_rejected_before_prime_search() -> None:
    """A 4300-digit ``engram_vocab_size`` must not enter modular exponentiation.

    That width is still a legal JSON integer. One primality test on it takes
    several seconds, the candidate is not prime, and the search steps forward.
    Rejection has to win the race against that work.
    """
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = (
        "from tests.models.deepseek_v41.test_engram_layout_limits import "
        "_child_wide_vocab; raise SystemExit(_child_wide_vocab())"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=_REPO_ROOT,
            env=env,
            timeout=20,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "EngramLayout.from_config did not reject a 4300-digit "
            f"engram_vocab_size within {exc.timeout}s"
        )
    assert completed.returncode == 0, completed.stderr


if __name__ == "__main__":
    raise SystemExit(_child_wide_vocab())
