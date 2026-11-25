# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import json
import logging
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.common import startup
from vllm.entrypoints.common.cli_args import make_arg_parser
from vllm.entrypoints.common.startup import (
    jsonify_non_default_args,
    redact_sensitive_args,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


class TestRedactSensitiveArgs:
    API_KEY = "sk-test-secret-12345"

    def test_redact_replaces_sensitive_values_only(self):
        args = {
            "api_key": self.API_KEY,
            "hf_token": "hf_secret",
            "watermark_config": {"algorithm": "gumbel", "key": 42},
            "other": "visible",
        }
        redacted = redact_sensitive_args(args)
        assert redacted == {
            "api_key": "***",
            "hf_token": "***",
            "watermark_config": "***",
            "other": "visible",
        }
        # original dict must not be mutated
        assert args == {
            "api_key": self.API_KEY,
            "hf_token": "hf_secret",
            "watermark_config": {"algorithm": "gumbel", "key": 42},
            "other": "visible",
        }

    def test_no_sensitive_fields_returns_original(self):
        args = {"model_tag": "org/model", "other": "visible"}
        assert redact_sensitive_args(args) is args

    def test_api_key_not_in_log(self, monkeypatch, caplog):
        non_default = {
            "model_tag": "org/model",
            "default_chat_template_kwargs": {"enable_thinking": False},
            "api_key": self.API_KEY,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "qwen3_coder",
        }
        monkeypatch.setattr(startup, "get_non_default_args", lambda args: non_default)
        with caplog.at_level("INFO", logger="vllm.entrypoints.common.startup"):
            startup.log_non_default_args(args=Namespace())
        message = caplog.text
        assert self.API_KEY not in message
        assert "'api_key': '***'" in message
        # non-sensitive args are still logged
        assert "org/model" in message
        assert "qwen3_coder" in message

    def test_hf_token_not_in_log(self, monkeypatch, caplog):
        hf_token = "hf_TESTTOKENVALUE123"
        non_default = {
            "model_tag": "org/model",
            "hf_token": hf_token,
            "tool_call_parser": "qwen3_coder",
        }
        monkeypatch.setattr(startup, "get_non_default_args", lambda args: non_default)
        with caplog.at_level("INFO", logger="vllm.entrypoints.common.startup"):
            startup.log_non_default_args(args=Namespace())
        message = caplog.text
        assert hf_token not in message
        assert "'hf_token': '***'" in message
        assert "org/model" in message


# Obvious stand-ins for the object-store fields documented on the obj tier.
# They are not credentials.
_ACCESS_KEY = "placeholder-access-key"
_SECRET_KEY = "placeholder-secret-key"
_SESSION_TOKEN = "placeholder-session-token"
_MODEL = "org/placeholder-model"


def _obj_store_extra_config() -> dict:
    """KV connector extra config shaped like the documented obj tier."""
    return {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": 1024,
        "secondary_tiers": [
            {
                "type": "obj",
                "store_config": {
                    "bucket": "placeholder-bucket",
                    "endpoint_override": "placeholder.example",
                    "scheme": "https",
                    "access_key": _ACCESS_KEY,
                    "secret_key": _SECRET_KEY,
                    "session_token": _SESSION_TOKEN,
                },
            }
        ],
    }


def _kv_transfer_payload(shape: str) -> dict:
    """OffloadingConnector config, or the same tier nested under MultiConnector.

    MultiConnector stores each child in ``connectors``, and each child carries
    its own ``kv_connector_extra_config``. A redactor that only inspects the
    top-level extra-config dict still has to hide these nested values.
    """
    extra = _obj_store_extra_config()
    if shape == "direct":
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }
    if shape == "multi":
        return {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": extra,
                    }
                ]
            },
        }
    raise ValueError(shape)


_SHAPES = ["direct", "multi"]


def _parse_serve_namespace(argv: list[str]) -> Namespace:
    """Parse argv with the same parser ``vllm serve`` and the API server use.

    ``ServeSubcommand.cmd`` copies the positional model onto ``args.model``
    before ``setup_server`` logs non-default arguments.
    """
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(argv)
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    return args


def _logged_messages(logger_name: str, emit) -> list[str]:
    logger = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.INFO)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        emit()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    messages = []
    for record in records:
        assert record.levelno == logging.INFO
        messages.append(record.getMessage())
    return messages


def _assert_credential_values_absent(text: str) -> None:
    assert _ACCESS_KEY not in text
    assert _SECRET_KEY not in text
    assert _SESSION_TOKEN not in text


def _live_extra_config(value):
    if isinstance(value, dict):
        return value["kv_connector_extra_config"]
    return value.kv_connector_extra_config


def _assert_live_credentials_unchanged(before, after) -> None:
    """Redaction is for the log line. The config the server uses stays intact."""
    assert after == before
    rendered = json.dumps(after)
    assert _ACCESS_KEY in rendered
    assert _SECRET_KEY in rendered
    assert _SESSION_TOKEN in rendered


class TestKvOffloadCredentialsNotLogged:
    """Object-store credentials in kv_transfer_config must not reach startup logs.

    ``setup_server`` and ``LLM`` both call ``log_non_default_args``. The Rust
    frontend logs ``redact_sensitive_args(jsonify_non_default_args(...))``.
    ``ObjStoreConfig`` hides the same fields only after the tier is constructed,
    which is later than these logs.
    """

    @pytest.mark.parametrize("shape", _SHAPES)
    def test_cli_namespace_startup_log_omits_obj_store_credentials(self, shape: str):
        args = _parse_serve_namespace(
            [
                _MODEL,
                "--kv-transfer-config",
                json.dumps(_kv_transfer_payload(shape)),
            ]
        )
        before = copy.deepcopy(_live_extra_config(args.kv_transfer_config))
        messages = _logged_messages(
            "vllm.entrypoints.common.startup",
            lambda: startup.log_non_default_args(args),
        )
        assert messages
        text = "\n".join(messages)
        assert "non-default args:" in text
        assert _MODEL in text
        _assert_live_credentials_unchanged(
            before, _live_extra_config(args.kv_transfer_config)
        )
        _assert_credential_values_absent(text)

    @pytest.mark.parametrize("shape", _SHAPES)
    def test_yaml_config_startup_log_omits_obj_store_credentials(
        self, shape: str, tmp_path: Path
    ):
        config_path = tmp_path / "serve.yaml"
        config_path.write_text(
            yaml.safe_dump({"kv-transfer-config": _kv_transfer_payload(shape)}),
            encoding="utf-8",
        )
        # FlexibleArgumentParser.load_config_file turns this nested mapping
        # back into a --kv-transfer-config JSON argument.
        args = _parse_serve_namespace([_MODEL, "--config", str(config_path)])
        before = copy.deepcopy(_live_extra_config(args.kv_transfer_config))
        messages = _logged_messages(
            "vllm.entrypoints.common.startup",
            lambda: startup.log_non_default_args(args),
        )
        text = "\n".join(messages)
        assert "non-default args:" in text
        _assert_live_credentials_unchanged(
            before, _live_extra_config(args.kv_transfer_config)
        )
        _assert_credential_values_absent(text)

    @pytest.mark.parametrize("shape", _SHAPES)
    @pytest.mark.parametrize("as_dict", [False, True], ids=["config", "dict"])
    def test_engine_args_startup_log_omits_obj_store_credentials(
        self, shape: str, as_dict: bool
    ):
        # LLM.__init__ builds EngineArgs and logs it. Library callers pass the
        # parsed config object or the same mapping as a plain dict.
        parsed = _parse_serve_namespace(
            [
                _MODEL,
                "--kv-transfer-config",
                json.dumps(_kv_transfer_payload(shape)),
            ]
        )
        config = _kv_transfer_payload(shape) if as_dict else parsed.kv_transfer_config
        engine_args = EngineArgs(model=_MODEL, kv_transfer_config=config)
        before = copy.deepcopy(_live_extra_config(engine_args.kv_transfer_config))
        messages = _logged_messages(
            "vllm.entrypoints.common.startup",
            lambda: startup.log_non_default_args(engine_args),
        )
        text = "\n".join(messages)
        assert "non-default args:" in text
        assert _MODEL in text
        _assert_live_credentials_unchanged(
            before, _live_extra_config(engine_args.kv_transfer_config)
        )
        _assert_credential_values_absent(text)

    @pytest.mark.parametrize("shape", _SHAPES)
    def test_rust_frontend_launch_log_omits_obj_store_credentials(self, shape: str):
        # RustFrontendProcessManager logs this string before spawning the
        # frontend. The child still receives the unredacted args on its
        # command line; this test covers the logged copy.
        args = _parse_serve_namespace(
            [
                _MODEL,
                "--kv-transfer-config",
                json.dumps(_kv_transfer_payload(shape)),
            ]
        )
        before = copy.deepcopy(_live_extra_config(args.kv_transfer_config))
        args_dict = jsonify_non_default_args(
            args,
            exclude={
                "api_server_count",
                "data_parallel_rank",
                "data_parallel_external_lb",
                "data_parallel_hybrid_lb",
            },
        )
        model_tag = getattr(args, "model_tag", None) or getattr(args, "model", None)
        if model_tag is not None:
            args_dict["model_tag"] = model_tag
        redacted_json = json.dumps(redact_sensitive_args(args_dict), sort_keys=True)
        assert _MODEL in redacted_json
        _assert_live_credentials_unchanged(
            before, _live_extra_config(args.kv_transfer_config)
        )
        _assert_credential_values_absent(redacted_json)
