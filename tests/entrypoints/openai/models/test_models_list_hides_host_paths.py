# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GET /v1/models may list public model ids and must not list host paths.

`root` on a base-model card is the public id. LoRA cards may use that same
public id, or omit `root`, and neither card may carry the weights directory
or an adapter directory. Those paths are the server's `--model` value and
the adapter path, which can name a home directory or a cache layout.
"""

import argparse
import asyncio
import json
from argparse import Namespace
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.common.cli_args import LoRAParserAction
from vllm.entrypoints.openai.models.api_router import attach_router
from vllm.entrypoints.openai.models.protocol import BaseModelPath, LoRAModulePath
from vllm.entrypoints.openai.models.serving import (
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from vllm.entrypoints.serve.lora.api_router import attach_router as attach_lora_router
from vllm.entrypoints.serve.middleware.register import init_entrypoints_middleware
from vllm.entrypoints.serve.utils.api_utils import process_lora_modules
from vllm.lora.request import LoRARequest

# Obvious placeholder. Not a credential used by any deployment.
_API_KEY = "placeholder-api-key"
_PUBLIC_MODEL = "public-model"
_SECOND_ALIAS = "second-alias"


class _ListingModelConfig:
    max_model_len = 2048


class _ListingEngine:
    """Engine stand-in for the models list.

    `OpenAIServingModels` stores the `LoRARequest` it built from the adapter
    path after `add_lora` returns. `AsyncLLM.add_lora` forwards that object
    into the engine, and the worker resolves a local copy of the path without
    writing it back onto the request the API process keeps.
    """

    def __init__(self) -> None:
        self.model_config = _ListingModelConfig()
        self.renderer = None
        self.input_processor = None

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        del lora_request
        return True


def _host_dir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "home" / "deploy-user" / ".cache" / name
    path.mkdir(parents=True)
    return path


def _parse_lora_modules(items: list[str]) -> list[LoRAModulePath]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-modules", nargs="+", action=LoRAParserAction)
    parsed = parser.parse_args(["--lora-modules", *items])
    return parsed.lora_modules


async def _serving_models(
    model_path: str,
    served_names: list[str],
    lora_modules: list[LoRAModulePath] | None = None,
    default_mm_loras: dict[str, str] | None = None,
) -> OpenAIServingModels:
    # Same construction as `init_app_state`: every served name shares
    # `args.model`, and static adapters include default multimodal LoRAs.
    base_model_paths = [
        BaseModelPath(name=name, model_path=model_path) for name in served_names
    ]
    serving = OpenAIServingModels(
        engine_client=_ListingEngine(),
        base_model_paths=base_model_paths,
        lora_modules=process_lora_modules(lora_modules, default_mm_loras),
    )
    await serving.init_static_loras()
    return serving


def _registry(model_path: str, served_names: list[str]) -> OpenAIModelRegistry:
    # Same construction as `init_render_app_state`.
    return OpenAIModelRegistry(
        model_config=_ListingModelConfig(),
        base_model_paths=[
            BaseModelPath(name=name, model_path=model_path) for name in served_names
        ],
    )


def _client(handler: OpenAIServingModels | OpenAIModelRegistry, api_key: str | None):
    app = FastAPI()
    attach_router(app)
    if api_key is not None:
        attach_lora_router(app)
    app.state.openai_serving_models = handler
    init_entrypoints_middleware(
        Namespace(
            allowed_origins=[],
            allow_credentials=False,
            allowed_methods=["*"],
            allowed_headers=["*"],
            api_key=[api_key] if api_key is not None else None,
            enable_request_id_headers=False,
            middleware=[],
        ),
        app,
        ("generate",),
    )
    return TestClient(app)


def _cards(response) -> dict[str, dict]:
    body = response.json()
    assert body["object"] == "list"
    return {card["id"]: card for card in body["data"]}


def _assert_hidden(response, secrets: list[str]) -> None:
    for secret in secrets:
        assert secret not in response.text


def _assert_base_cards(cards: dict[str, dict], public_ids: list[str]) -> None:
    for public_id in public_ids:
        card = cards[public_id]
        assert card["root"] == public_id
        assert card["parent"] is None


def _assert_adapter_card(card: dict, public_id: str, parent: str) -> None:
    assert card["id"] == public_id
    assert card.get("root") in (None, public_id)
    assert card["parent"] == parent


def test_get_v1_models_omits_aliased_base_model_filesystem_path(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    weights = _host_dir(tmp_path, "weights-not-published")
    serving = asyncio.run(_serving_models(str(weights), [_PUBLIC_MODEL, _SECOND_ALIAS]))
    with _client(serving, api_key=None) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    cards = _cards(response)
    _assert_base_cards(cards, [_PUBLIC_MODEL, _SECOND_ALIAS])
    _assert_hidden(response, [str(weights), "deploy-user"])


def test_get_v1_models_omits_static_lora_filesystem_paths(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    weights = _host_dir(tmp_path, "base-weights")
    absolute = _host_dir(tmp_path, "absolute-adapter")
    relative = "deploy-user-relative/hidden-adapter"
    tilde = "~/deploy-user/hidden-lora-weights"
    modules = _parse_lora_modules(
        [
            f"absolute-adapter={absolute}",
            json.dumps(
                {
                    "name": "relative-adapter",
                    "path": relative,
                    "base_model_name": _PUBLIC_MODEL,
                }
            ),
            json.dumps({"name": "tilde-adapter", "path": tilde}),
        ]
    )
    serving = asyncio.run(
        _serving_models(str(weights), [_PUBLIC_MODEL], lora_modules=modules)
    )
    with _client(serving, api_key=None) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    cards = _cards(response)
    _assert_base_cards(cards, [_PUBLIC_MODEL])
    _assert_adapter_card(cards["absolute-adapter"], "absolute-adapter", _PUBLIC_MODEL)
    _assert_adapter_card(cards["relative-adapter"], "relative-adapter", _PUBLIC_MODEL)
    _assert_adapter_card(cards["tilde-adapter"], "tilde-adapter", _PUBLIC_MODEL)
    _assert_hidden(response, [str(weights), str(absolute), relative, tilde])


def test_get_v1_models_omits_default_multimodal_lora_filesystem_path(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    weights = _host_dir(tmp_path, "mm-base-weights")
    audio = _host_dir(tmp_path, "audio-adapter")
    serving = asyncio.run(
        _serving_models(
            str(weights),
            [_PUBLIC_MODEL],
            default_mm_loras={"audio": str(audio)},
        )
    )
    with _client(serving, api_key=None) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    cards = _cards(response)
    _assert_base_cards(cards, [_PUBLIC_MODEL])
    _assert_adapter_card(cards["audio"], "audio", _PUBLIC_MODEL)
    _assert_hidden(response, [str(weights), str(audio)])


def test_get_v1_models_omits_runtime_loaded_lora_filesystem_path(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "1")
    weights = _host_dir(tmp_path, "runtime-base-weights")
    adapter = _host_dir(tmp_path, "runtime-adapter-weights")
    serving = asyncio.run(_serving_models(str(weights), [_PUBLIC_MODEL]))
    with _client(serving, api_key=_API_KEY) as client:
        denied = client.get("/v1/models")
        loaded = client.post(
            "/v1/load_lora_adapter",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={"lora_name": "runtime-adapter", "lora_path": str(adapter)},
        )
        listed = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {_API_KEY}"},
        )

    assert denied.status_code == 401
    _assert_hidden(denied, [str(weights), str(adapter)])
    assert loaded.status_code == 200
    assert listed.status_code == 200
    cards = _cards(listed)
    _assert_base_cards(cards, [_PUBLIC_MODEL])
    _assert_adapter_card(cards["runtime-adapter"], "runtime-adapter", _PUBLIC_MODEL)
    _assert_hidden(listed, [str(weights), str(adapter)])


def test_render_models_list_omits_base_model_filesystem_path(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    weights = _host_dir(tmp_path, "render-weights")
    registry = _registry(str(weights), [_PUBLIC_MODEL])
    with _client(registry, api_key=None) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    cards = _cards(response)
    _assert_base_cards(cards, [_PUBLIC_MODEL])
    _assert_hidden(response, [str(weights), "deploy-user"])
