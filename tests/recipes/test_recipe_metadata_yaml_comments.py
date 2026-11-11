# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup

_CONVERTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "recipes"
    / "recipe_json_to_vllm_config.py"
)
_CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "recipe_json_to_vllm_config",
    _CONVERTER_PATH,
)
assert _CONVERTER_SPEC is not None and _CONVERTER_SPEC.loader is not None
_CONVERTER = importlib.util.module_from_spec(_CONVERTER_SPEC)
_CONVERTER_SPEC.loader.exec_module(_CONVERTER)

INTENDED_MODEL = "org/intended-model"
INJECTED_MODEL = "attacker/custom-model"
INJECTED_SERVED_NAME = "injected-served-name"
_ARGV = ["vllm", "serve", INTENDED_MODEL, "--tensor-parallel-size", "1"]
_METADATA_FIELDS = ("hardware", "strategy", "variant", "deploy_type")
_LINE_BREAKS = [
    pytest.param("\n", id="lf"),
    pytest.param("\r", id="cr"),
    pytest.param("\r\n", id="crlf"),
    pytest.param("\u0085", id="nel"),
    pytest.param("\u2028", id="line-separator"),
    pytest.param("\u2029", id="paragraph-separator"),
]


def _serve_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser()
    parser.add_argument("serve")
    parser.add_argument("model_tag", nargs="?")
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--served-model-name")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _parse_printed_serve_config(config_path: Path) -> argparse.Namespace:
    """Parse `vllm serve --config <config_path>` the way the CLI loads YAML."""
    args = _serve_parser().parse_args(["serve", "--config", str(config_path)])
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    return args


def _assert_argv_translation(args: argparse.Namespace) -> None:
    assert args.model_tag is None
    assert args.trust_remote_code is False
    assert args.served_model_name is None
    assert args.dtype is None
    assert args.model == INTENDED_MODEL
    assert args.tensor_parallel_size == 1


def _injected_metadata(
    line_break: str,
    prefix: str,
    *,
    document_separator: bool,
) -> str:
    parts = [
        prefix,
        f"model: {INJECTED_MODEL}",
        "trust-remote-code: true",
        f"served-model-name: {INJECTED_SERVED_NAME}",
        "dtype: float32",
    ]
    if document_separator:
        parts.append("---")
    return line_break.join(parts)


def _run_converter(
    monkeypatch: pytest.MonkeyPatch,
    recipe_path: Path,
    config_path: Path,
    env_path: Path,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recipe_json_to_vllm_config.py",
            str(recipe_path),
            "--config-out",
            str(config_path),
            "--env-out",
            str(env_path),
        ],
    )
    return _CONVERTER.main()


def _assert_outputs_follow_argv(
    code: int,
    config_path: Path,
    env_path: Path,
) -> None:
    if code != 0:
        assert not config_path.exists()
        assert not env_path.exists()
        return
    _assert_argv_translation(_parse_printed_serve_config(config_path))


def _write_recipe(path: Path, recipe: dict) -> None:
    path.write_text(json.dumps(recipe), encoding="utf-8")


@pytest.mark.parametrize("field", _METADATA_FIELDS)
@pytest.mark.parametrize("line_break", _LINE_BREAKS)
def test_metadata_line_break_cannot_add_serve_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    line_break: str,
) -> None:
    """Recipe metadata line breaks must not become serve arguments."""
    prefix = "single_node" if field == "deploy_type" else "h100"
    recipe = {
        "argv": _ARGV,
        "hardware": "h100",
        "strategy": "single_node_tp",
        "variant": "default",
        "deploy_type": "single_node",
        field: _injected_metadata(line_break, prefix, document_separator=False),
    }
    recipe_path = tmp_path / "recipe.json"
    config_path = tmp_path / "config.yml"
    env_path = tmp_path / "env.sh"
    _write_recipe(recipe_path, recipe)
    code = _run_converter(monkeypatch, recipe_path, config_path, env_path)
    _assert_outputs_follow_argv(code, config_path, env_path)


@pytest.mark.parametrize("field", _METADATA_FIELDS)
@pytest.mark.parametrize("line_break", _LINE_BREAKS)
def test_metadata_document_separator_cannot_replace_serve_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    line_break: str,
) -> None:
    """A document separator in metadata must not replace the argv config."""
    prefix = "single_node" if field == "deploy_type" else "h100"
    recipe = {
        "argv": _ARGV,
        "hardware": "h100",
        "strategy": "single_node_tp",
        "variant": "default",
        "deploy_type": "single_node",
        field: _injected_metadata(line_break, prefix, document_separator=True),
    }
    recipe_path = tmp_path / "recipe.json"
    config_path = tmp_path / "config.yml"
    env_path = tmp_path / "env.sh"
    _write_recipe(recipe_path, recipe)
    code = _run_converter(monkeypatch, recipe_path, config_path, env_path)
    _assert_outputs_follow_argv(code, config_path, env_path)


def test_benign_recipe_serve_config_matches_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recipe without line breaks loads as the argv translation."""
    recipe = {
        "argv": _ARGV,
        "hardware": "h100",
        "strategy": "single_node_tp",
        "variant": "default",
        "deploy_type": "single_node",
    }
    recipe_path = tmp_path / "recipe.json"
    config_path = tmp_path / "config.yml"
    env_path = tmp_path / "env.sh"
    _write_recipe(recipe_path, recipe)
    code = _run_converter(monkeypatch, recipe_path, config_path, env_path)
    assert code == 0
    _assert_argv_translation(_parse_printed_serve_config(config_path))


def test_source_line_break_cannot_inject_config_or_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source path is written into comments and must not break them."""
    directory = tmp_path / (
        "dir\ntrust-remote-code: true\n"
        f"served-model-name: {INJECTED_SERVED_NAME}\n"
        "dtype: float32"
    )
    directory.mkdir()
    recipe_path = directory / "recipe.json"
    config_path = tmp_path / "config.yml"
    env_path = tmp_path / "env.sh"
    _write_recipe(
        recipe_path,
        {
            "argv": _ARGV,
            "hardware": "h100",
            "strategy": "single_node_tp",
            "variant": "default",
            "deploy_type": "single_node",
        },
    )
    code = _run_converter(monkeypatch, recipe_path, config_path, env_path)
    _assert_outputs_follow_argv(code, config_path, env_path)
    if code != 0:
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        assert (
            line == ""
            or line.startswith("#")
            or line.startswith("#!")
            or line.startswith("export ")
        )
