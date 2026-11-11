# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client multimodal blobs on POST /inference/v1/generate must not unpickle.

The handler decodes ``features.kwargs_data`` and ``features.mm_metadata`` with
the worker msgpack decoder. These tests post a msgpack extension through that
route and fail if decoding the body runs the payload. They fail while that
decoder still unpickles extension types.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cloudpickle
import pybase64
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from msgspec import msgpack

from vllm.entrypoints.scale_out.token_in_token_out.api_router import router
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.envs import disable_envs_cache
from vllm.v1.serial_utils import CUSTOM_TYPE_CLOUDPICKLE, CUSTOM_TYPE_PICKLE

_EXTENSION_CODES = {
    "pickle": CUSTOM_TYPE_PICKLE,
    "cloudpickle": CUSTOM_TYPE_CLOUDPICKLE,
}


def record_client_payload_execution(marker_path: str) -> int:
    """Write a marker file. Invoked only if the request body is unpickled."""
    Path(marker_path).write_text("executed", encoding="utf-8")
    return 0


class _ExecuteOnReduce:
    """Pickle reducible whose reconstruction writes ``marker_path``."""

    def __init__(self, marker_path: str) -> None:
        self.marker_path = marker_path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (record_client_payload_execution, (self.marker_path,))


class _ModelConfig:
    generation_config = "auto"
    is_encoder_decoder = False
    max_model_len = 128

    def get_diff_sampling_param(self) -> dict[str, object]:
        return {}


class _Renderer:
    tokenizer = None


class _SchedulerConfig:
    max_num_seqs = 16


class _VllmConfig:
    scheduler_config = _SchedulerConfig()
    kv_transfer_config = None


class _Engine:
    """Attributes ``ServingTokens`` reads before it decodes multimodal blobs."""

    errored = False
    dead_error = RuntimeError("engine dead")
    model_config = _ModelConfig()
    renderer = _Renderer()
    input_processor = None
    vllm_config = _VllmConfig()

    def check_admission(self, n: int) -> None:
        del n


class _Models:
    lora_requests: dict[str, object] = {}
    base_model_paths: list[object] = []

    def model_name(self, lora_request: object = None) -> str:
        del lora_request
        return "mm-payload"


class _OnlineRenderer:
    class renderer:
        mm_processor_cache = None


def _extension_b64(marker: Path, extension: str, *, nested: bool) -> str:
    code = _EXTENSION_CODES[extension]
    obj = _ExecuteOnReduce(str(marker))
    payload = (
        cloudpickle.dumps(obj)
        if extension == "cloudpickle"
        else pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    )
    ext = msgpack.Ext(code, payload)
    document = msgpack.encode({"pixel_values": ext} if nested else ext)
    return pybase64.b64encode(document).decode("ascii")


def _generate_body(blob: str, field: str) -> dict[str, object]:
    features: dict[str, object] = {
        "mm_hashes": {"image": ["item-hash"]},
        "mm_placeholders": {"image": [{"offset": 0, "length": 1}]},
    }
    body: dict[str, object] = {
        "token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 1},
        "features": features,
    }
    if field == "kwargs_data":
        features["kwargs_data"] = {"image": [blob]}
    elif field == "mm_metadata":
        # Metadata without tensor data is only accepted alongside EC transfer.
        features["kwargs_data"] = {"image": [None]}
        features["mm_metadata"] = {"image": [blob]}
        body["ec_transfer_params"] = {"source": "prefill"}
    else:
        raise AssertionError(field)
    return body


@pytest.fixture
def generate_client() -> TestClient:
    serving = ServingTokens(
        _Engine(),
        _Models(),
        _OnlineRenderer(),
        online_derenderer=object(),
        request_logger=None,
        force_no_detokenize=False,
    )
    app = FastAPI()
    app.state.serving_tokens = serving
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


def _post_generate(
    client: TestClient,
    body: dict[str, object],
    *,
    allow_insecure_serialization: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "VLLM_ALLOW_INSECURE_SERIALIZATION",
        "1" if allow_insecure_serialization else "0",
    )
    disable_envs_cache()
    return client.post("/inference/v1/generate", json=body)


@pytest.mark.parametrize("field", ["kwargs_data", "mm_metadata"])
@pytest.mark.parametrize("layout", ["root", "nested"])
@pytest.mark.parametrize("extension", ["pickle", "cloudpickle"])
def test_generate_does_not_unpickle_client_multimodal_blob(
    generate_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    layout: str,
    extension: str,
) -> None:
    """Insecure serialization must not unpickle generate-request blobs.

    The marker is written by the extension payload itself, before the decoder
    can reject the resulting object. A refusal that still unpickles fails.
    """
    marker = tmp_path / "executed"
    blob = _extension_b64(marker, extension, nested=layout == "nested")
    response = _post_generate(
        generate_client,
        _generate_body(blob, field),
        allow_insecure_serialization=True,
        monkeypatch=monkeypatch,
    )

    assert not marker.exists()
    assert response.status_code >= 400


def test_generate_rejects_pickle_extension_when_insecure_serialization_disabled(
    generate_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default configuration must not run a pickle extension either."""
    marker = tmp_path / "executed"
    blob = _extension_b64(marker, "pickle", nested=False)
    response = _post_generate(
        generate_client,
        _generate_body(blob, "kwargs_data"),
        allow_insecure_serialization=False,
        monkeypatch=monkeypatch,
    )

    assert not marker.exists()
    assert response.status_code >= 400
