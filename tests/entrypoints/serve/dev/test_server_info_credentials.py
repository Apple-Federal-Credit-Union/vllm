# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Credential exposure on the development ``/server_info`` route.

``/server_info`` is mounted only when ``VLLM_SERVER_DEV_MODE`` is enabled.
When an API key is configured, the route must enforce that key the same way
``/v1`` does. Both the text and JSON config forms must omit credential
fields that remain set on the live ``VllmConfig`` (``hf_token``, and the
watermark key).
"""

import json
from argparse import Namespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vllm.config import ModelConfig, VllmConfig, WatermarkConfig
from vllm.entrypoints.launchers.app import build_app

pytestmark = pytest.mark.skip_global_cleanup

# Obvious placeholders. These are not real credentials.
HF_TOKEN_PLACEHOLDER = "placeholder-hf-token"
API_KEY_PLACEHOLDER = "placeholder-api-key"
WATERMARK_KEY_PLACEHOLDER = 10**12 + 7

_OPT_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_hidden_layers": 1,
    "vocab_size": 32,
    "max_position_embeddings": 16,
    "ffn_dim": 16,
    "word_embed_proj_dim": 8,
}


def _server_args(api_key: list[str] | None) -> Namespace:
    return Namespace(
        disable_fastapi_docs=True,
        enable_offline_docs=False,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=api_key,
        enable_request_id_headers=False,
        enable_fault_tolerance=False,
        middleware=[],
        log_error_stack=False,
    )


def _live_config(tmp_path: Path) -> tuple[VllmConfig, str]:
    """Build a real config whose in-memory ``hf_token`` is the placeholder."""
    model_dir = tmp_path / "opt-local"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_OPT_CONFIG))
    model_path = str(model_dir)
    model_config = ModelConfig(
        model=model_path,
        tokenizer=model_path,
        skip_tokenizer_init=True,
        enforce_eager=True,
        hf_token=HF_TOKEN_PLACEHOLDER,
        max_model_len=8,
        trust_remote_code=False,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        watermark_config=WatermarkConfig(key=WATERMARK_KEY_PLACEHOLDER),
    )
    assert vllm_config.model_config.hf_token == HF_TOKEN_PLACEHOLDER
    assert vllm_config.watermark_config is not None
    assert vllm_config.watermark_config.key == WATERMARK_KEY_PLACEHOLDER
    return vllm_config, model_path


def _client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    dev_mode: bool,
    api_key: list[str] | None,
    env_api_key: str | None = None,
) -> tuple[TestClient, str]:
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1" if dev_mode else "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    if env_api_key is None:
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("VLLM_API_KEY", env_api_key)
    vllm_config, model_path = _live_config(tmp_path)
    app = build_app(_server_args(api_key), supported_tasks=())
    app.state.vllm_config = vllm_config
    app.state.log_stats = False
    return TestClient(app), model_path


def _assert_credentials_absent(body: str) -> None:
    assert HF_TOKEN_PLACEHOLDER not in body
    assert str(WATERMARK_KEY_PLACEHOLDER) not in body
    assert API_KEY_PLACEHOLDER not in body


@pytest.mark.parametrize("config_format", ["text", "json"])
@pytest.mark.parametrize(
    "api_key_source",
    ["args", "env"],
)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-placeholder"},
    ],
)
def test_server_info_requires_configured_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_format: str,
    api_key_source: str,
    headers: dict[str, str],
):
    """A configured API key is checked before ``/server_info`` is served.

    The same app rejects an unauthenticated ``/v1`` request. ``/server_info``
    must do the same for both config formats, and the refusal body must not
    contain the live config.
    """
    if api_key_source == "args":
        api_key: list[str] | None = [API_KEY_PLACEHOLDER]
        env_api_key = None
    else:
        api_key = None
        env_api_key = API_KEY_PLACEHOLDER

    client, model_path = _client(
        monkeypatch,
        tmp_path,
        dev_mode=True,
        api_key=api_key,
        env_api_key=env_api_key,
    )
    with client:
        guarded = client.get("/v1/models", headers=headers)
        response = client.get(
            "/server_info",
            params={"config_format": config_format},
            headers=headers,
        )

    assert guarded.status_code == 401
    assert response.status_code == 401
    _assert_credentials_absent(response.text)
    assert model_path not in response.text


@pytest.mark.parametrize("config_format", ["text", "json"])
def test_authorized_server_info_omits_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_format: str,
):
    """A valid API key may read server info, without credential fields."""
    client, model_path = _client(
        monkeypatch,
        tmp_path,
        dev_mode=True,
        api_key=[API_KEY_PLACEHOLDER],
    )
    with client:
        response = client.get(
            "/server_info",
            params={"config_format": config_format},
            headers={"Authorization": f"Bearer {API_KEY_PLACEHOLDER}"},
        )

    assert response.status_code == 200
    assert model_path in response.text
    _assert_credentials_absent(response.text)


@pytest.mark.parametrize("config_format", ["text", "json"])
def test_server_info_omits_credentials_when_api_key_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_format: str,
):
    """With no API key configured, the route stays open and still redacts."""
    client, model_path = _client(
        monkeypatch,
        tmp_path,
        dev_mode=True,
        api_key=None,
    )
    with client:
        response = client.get(
            "/server_info",
            params={"config_format": config_format},
        )

    assert response.status_code == 200
    assert model_path in response.text
    _assert_credentials_absent(response.text)


@pytest.mark.parametrize("config_format", ["text", "json"])
def test_server_info_absent_when_dev_mode_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_format: str,
):
    """Development routes, including ``/server_info``, are not mounted."""
    client, model_path = _client(
        monkeypatch,
        tmp_path,
        dev_mode=False,
        api_key=None,
    )
    with client:
        response = client.get(
            "/server_info",
            params={"config_format": config_format},
        )

    assert response.status_code == 404
    _assert_credentials_absent(response.text)
    assert model_path not in response.text
