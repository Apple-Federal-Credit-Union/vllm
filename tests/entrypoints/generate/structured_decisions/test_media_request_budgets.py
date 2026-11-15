# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Budgets for media on POST /v1/systemone.

These tests are the spec for the missing per-request media limits. They fail
while a structured-decisions read fetches every URL on a connector that has no
request download budget and no decoded-media budget.

Each of ``images``, ``audio`` and ``videos`` may contain at most
``MAX_MEDIA_URLS`` URLs. A longer list is a 400 from ``create_decision``, and
nothing is fetched or decoded. Media that is fetched for one request shares
one ``MediaRequestDownloadBudget`` and one ``MediaDecodedBudget``, both created
for that request and both charged from every image, audio and video load
(including a video soundtrack and a ``video/jpeg`` data URL). Two items that
are each under the limit are rejected when their sum exceeds it. The next
request starts from an empty budget.

``LiquidStrategy.read`` and ``DiffusionGemmaCanvasStrategy.read`` /
``read_joint`` call ``NextTokenStrategy._image_parts`` and do not fetch audio
or video, so the next-token image tests cover them. ``POST /v1/decisions``
rejects images before a read strategy runs.
"""

import base64
import io
import os
import tempfile
import threading
import wave
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import av
import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.structured_decisions.api_router import (
    register_structured_decisions_api_router,
)
from vllm.entrypoints.generate.structured_decisions.protocol import (
    StructuredDecisionRequest,
)
from vllm.entrypoints.generate.structured_decisions.serving import (
    ServingStructuredDecisions,
)
from vllm.entrypoints.openai.decisions.strategies import (
    ClefFlashStrategy,
    D3Strategy,
    JointSchemaStrategy,
    NextTokenStrategy,
    ReadContext,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.exception_handling.register import (
    init_exception_handler,
)
from vllm.multimodal.media.connector import MediaConnector
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers import get_tokenizer

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

# Each media list on one /v1/systemone request.
MAX_MEDIA_URLS = 8
_DOWNLOAD_LIMIT_MB = 1
# Just above one 10x10 image (400 bytes) and below two.
_IMAGE_DECODE_LIMIT = 600
# Just above one 4-frame 16x16 video (3072 bytes) and below two.
_VIDEO_DECODE_LIMIT = 4000
# Just above one 1-second 8 kHz mono float32 clip (32000 bytes) and below two.
_AUDIO_DECODE_LIMIT = 40_000
_LOADED = "LOADED"
_DOWNLOAD_TOKEN = "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB"
_DECODE_TOKEN = "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"

_STRATEGIES = {
    "next": NextTokenStrategy,
    "joint": JointSchemaStrategy,
    "clef": ClefFlashStrategy,
    "d3": D3Strategy,
}
# NextTokenStrategy advertises only images. Audio and video are rejected as
# unsupported before the list-length check, so they are not cases for it.
_ACCEPTED = {
    "next": ("images",),
    "joint": ("images", "audio", "videos"),
    "clef": ("images", "videos"),
    "d3": ("images", "videos"),
}
_COUNT_CASES = [
    (kind, field) for kind, fields in _ACCEPTED.items() for field in fields
]
_IMAGE_KINDS = [kind for kind, fields in _ACCEPTED.items() if "images" in fields]
_VIDEO_KINDS = [kind for kind, fields in _ACCEPTED.items() if "videos" in fields]
_D3_DECISION = {
    "format_version": 1,
    "prompt": "d3",
    "codes": ["A", "B", "C"],
    "token_ids": [0, 1, 2],
    "temperature": 1.0,
}


def _png_bytes(width: int = 10, height: int = 10) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(width: int = 10, height: int = 10) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (4, 5, 6)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _wav_bytes(seconds: float, rate: int = 8000) -> bytes:
    frames = int(seconds * rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x01" * frames)
    return buffer.getvalue()


def _data_url(media_type: str, payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _avi_bytes() -> bytes:
    """Four 16x16 frames. Each strategy's video backend decodes 3072 bytes."""
    fd, name = tempfile.mkstemp(suffix=".avi")
    os.close(fd)
    writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"MJPG"), 2, (16, 16))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not write an MJPG AVI")
    rng = np.random.default_rng(0)
    for _ in range(4):
        frame = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    with open(name, "rb") as handle:
        data = handle.read()
    os.remove(name)
    return data


def _muxed_avi() -> bytes:
    """Picture plus PCM, between half a mebibyte and one mebibyte.

    Downloading the file once fits in ``_DOWNLOAD_LIMIT_MB``. Downloading it
    twice, as JointSchemaStrategy does for a video and its soundtrack, does not.
    """
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="avi")
    video_stream = container.add_stream("mjpeg", rate=2)
    video_stream.width = 16
    video_stream.height = 16
    video_stream.pix_fmt = "yuvj420p"
    audio_stream = container.add_stream("pcm_s16le", rate=16000)
    audio_stream.layout = "mono"
    for index in range(4):
        frame = av.VideoFrame(16, 16, "rgb24")
        frame.planes[0].update(bytes([40]) * 16 * 16 * 3)
        frame.pts = index
        for packet in video_stream.encode(frame):
            container.mux(packet)
    samples = np.zeros((1, 16000 * 20), np.int16)
    audio = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    audio.sample_rate = 16000
    audio.pts = 0
    for packet in audio_stream.encode(audio):
        container.mux(packet)
    for packet in video_stream.encode(None):
        container.mux(packet)
    for packet in audio_stream.encode(None):
        container.mux(packet)
    container.close()
    return buffer.getvalue()


class _MediaServer:
    def __init__(self, bodies: dict[str, bytes]):
        self.bodies = bodies
        self.hits: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.hits.append(self.path)
                body = owner.bodies[self.path]
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def url(self, path: str) -> str:
        port = self._httpd.server_address[1]
        return f"http://127.0.0.1:{port}{path}"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@contextmanager
def _serve(bodies: dict[str, bytes]):
    server = _MediaServer(bodies)
    try:
        yield server
    finally:
        server.close()


def _decision(**media: Any) -> StructuredDecisionRequest:
    return StructuredDecisionRequest(
        state="The card was charged twice.",
        questions={
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "payments", "shipping": "deliveries"},
            }
        },
        **media,
    )


def _json_body(**media: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "state": "The card was charged twice.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "payments", "shipping": "deliveries"},
            }
        },
    }
    body.update(media)
    return body


def _make_serving(engine, strategy) -> ServingStructuredDecisions:
    name = engine.model_config.model
    models = OpenAIServingModels(engine, [BaseModelPath(name=name, model_path=name)])
    return ServingStructuredDecisions(models, strategy)


def _stop_after_load(strategy, monkeypatch) -> None:
    """Fail once media has been decoded, before the renderer or the engine."""

    async def stop_prompts(read_request, contents, instructions, image_parts):
        raise RuntimeError(f"{_LOADED} image={len(image_parts)}")

    async def stop_engine(prompt, arrival_time):
        data = prompt.get("multi_modal_data") or {}
        counts = {key: len(value) for key, value in data.items()}
        raise RuntimeError(f"{_LOADED} {counts}")

    monkeypatch.setattr(strategy, "_prompts", stop_prompts, raising=False)
    renderer = strategy.context.online_renderer.renderer
    monkeypatch.setattr(renderer, "process_for_engine_async", stop_engine)


def _post(handler: ServingStructuredDecisions, payload: dict[str, Any]):
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.state.serving_structured_decisions = handler
    init_exception_handler(app)
    register_structured_decisions_api_router(app)
    with TestClient(app) as client:
        return client.post("/v1/systemone", json=payload)


async def _run(handler, **media):
    try:
        return await handler.create_decision(_decision(**media))
    except Exception as exc:
        return exc


def _assert_loaded(outcome) -> None:
    assert isinstance(outcome, RuntimeError), outcome
    assert str(outcome).startswith(_LOADED)


def _assert_budget(outcome, token: str) -> None:
    if isinstance(outcome, ErrorResponse):
        text = outcome.error.message
    elif isinstance(outcome, Exception):
        text = str(outcome)
    else:
        text = repr(outcome)
    assert _LOADED not in text, text
    assert token in text, text


def _assert_count(outcome, field: str) -> None:
    assert isinstance(outcome, ErrorResponse), outcome
    assert outcome.error.code == 400
    text = outcome.error.message
    assert str(MAX_MEDIA_URLS) in text
    assert field.rstrip("s") in text


def _http_urls(field: str, count: int, body: bytes):
    bodies = {f"/{field}/{index}": body for index in range(count)}
    return bodies


def _renderer(model_config, tokenizer):
    renderer = HfRenderer(
        SimpleNamespace(
            model_config=model_config,
            parallel_config=SimpleNamespace(_api_process_rank=0),
        ),
        tokenizer,
    )
    online = OnlineRenderer(
        model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    return renderer, online


@pytest.fixture(scope="module")
def qwen_stack():
    model_config = ModelConfig(model="Qwen/Qwen3-0.6B", dtype="float32")
    renderer, online_renderer = _renderer(
        model_config, get_tokenizer("Qwen/Qwen3-0.6B")
    )
    # D3 inserts <|vision_end|> with the checkpoint's vision chat template.
    # Qwen3-0.6B's text template has no such token, so the media read never
    # starts. Qwen2-VL's template does, which is what this path needs.
    vision_renderer, vision_online = _renderer(
        model_config, get_tokenizer("Qwen/Qwen2-VL-2B-Instruct")
    )
    engine = MagicMock(spec=EngineClient)
    engine.errored = False
    engine.model_config = model_config
    engine.renderer = renderer
    engine.input_processor = MagicMock()
    yield engine, online_renderer, vision_online
    renderer.shutdown()
    vision_renderer.shutdown()


@pytest.fixture
def build_handler(qwen_stack, monkeypatch):
    engine, online_renderer, vision_online = qwen_stack
    model_config = engine.model_config
    monkeypatch.setattr(
        ModelConfig,
        "supports_multimodal_inputs",
        property(lambda self: True),
    )

    def get_multimodal_config(self):
        if self.multimodal_config is not None:
            return self.multimodal_config
        return MultiModalConfig()

    monkeypatch.setattr(ModelConfig, "get_multimodal_config", get_multimodal_config)

    def build(kind: str):
        if kind == "d3":
            model_config.hf_config.decision_config = dict(_D3_DECISION)
            context_renderer = vision_online
        else:
            context_renderer = online_renderer
        strategy = _STRATEGIES[kind](
            ReadContext(engine, context_renderer, None, "auto", {})
        )
        return _make_serving(engine, strategy), strategy

    return build


@pytest.fixture(scope="module")
def tiny_png():
    return _png_bytes()


@pytest.fixture(scope="module")
def padded_png(tiny_png):
    return tiny_png + b"\0" * (600 * 1024)


@pytest.fixture(scope="module")
def tiny_wav():
    return _wav_bytes(1.0)


@pytest.fixture(scope="module")
def wide_wav():
    # 16000 Hz * 20 s * 2 bytes sits between 0.5 MiB and 1 MiB.
    return _wav_bytes(20.0, rate=16000)


@pytest.fixture(scope="module")
def tiny_avi():
    return _avi_bytes()


@pytest.fixture(scope="module")
def wide_avi():
    return _muxed_avi()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,field", _COUNT_CASES)
async def test_too_many_http_urls_are_rejected_before_fetch(
    build_handler, monkeypatch, kind, field
):
    handler, strategy = build_handler(kind)
    _stop_after_load(strategy, monkeypatch)
    bodies = _http_urls(field, MAX_MEDIA_URLS + 1, b"x")
    with _serve(bodies) as server:
        urls = [server.url(path) for path in bodies]
        outcome = await _run(handler, **{field: urls})
        assert server.hits == [], outcome
    _assert_count(outcome, field)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,field", _COUNT_CASES)
async def test_too_many_data_urls_are_rejected_before_decode(
    build_handler, monkeypatch, kind, field
):
    handler, strategy = build_handler(kind)
    _stop_after_load(strategy, monkeypatch)
    loaded: list[str] = []
    real_load = MediaConnector.load_from_url_async

    async def spy(self, url, media_io, **kwargs):
        loaded.append(url)
        return await real_load(self, url, media_io, **kwargs)

    monkeypatch.setattr(MediaConnector, "load_from_url_async", spy)
    media_type = {
        "images": "image/png",
        "audio": "audio/wav",
        "videos": "video/mp4",
    }
    urls = [
        _data_url(media_type[field], b"not-media") for _ in range(MAX_MEDIA_URLS + 1)
    ]
    outcome = await _run(handler, **{field: urls})
    assert loaded == [], outcome
    _assert_count(outcome, field)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,field", _COUNT_CASES)
async def test_max_media_urls_are_fetched(
    build_handler, monkeypatch, kind, field, tiny_png, tiny_wav, tiny_avi
):
    handler, strategy = build_handler(kind)
    _stop_after_load(strategy, monkeypatch)
    body = {"images": tiny_png, "audio": tiny_wav, "videos": tiny_avi}[field]
    bodies = _http_urls(field, MAX_MEDIA_URLS, body)
    with _serve(bodies) as server:
        urls = [server.url(path) for path in bodies]
        outcome = await _run(handler, **{field: urls})
        assert len(server.hits) >= MAX_MEDIA_URLS
    _assert_loaded(outcome)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _IMAGE_KINDS)
async def test_image_downloads_share_a_request_budget(
    build_handler, monkeypatch, kind, padded_png
):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", _DOWNLOAD_LIMIT_MB
    )
    handler, strategy = build_handler(kind)
    _stop_after_load(strategy, monkeypatch)
    bodies = _http_urls("images", 2, padded_png)
    with _serve(bodies) as server:
        urls = [server.url(path) for path in bodies]
        outcome = await _run(handler, images=urls)
        assert server.hits
    _assert_budget(outcome, _DOWNLOAD_TOKEN)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _IMAGE_KINDS)
async def test_image_decodes_share_a_request_budget(build_handler, monkeypatch, kind):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", _IMAGE_DECODE_LIMIT
    )
    handler, strategy = build_handler(kind)
    _stop_after_load(strategy, monkeypatch)
    urls = [_data_url("image/png", _png_bytes()) for _ in range(2)]
    outcome = await _run(handler, images=urls)
    _assert_budget(outcome, _DECODE_TOKEN)


@pytest.mark.asyncio
async def test_image_budget_is_per_request(build_handler, monkeypatch, padded_png):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", _DOWNLOAD_LIMIT_MB
    )
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", _IMAGE_DECODE_LIMIT
    )
    handler, strategy = build_handler("next")
    _stop_after_load(strategy, monkeypatch)
    bodies = _http_urls("images", 1, padded_png)
    with _serve(bodies) as server:
        url = server.url(next(iter(bodies)))
        first = await _run(handler, images=[url])
        second = await _run(handler, images=[url])
        assert len(server.hits) >= 2
    _assert_loaded(first)
    _assert_loaded(second)

    image = _data_url("image/png", _png_bytes())
    _assert_loaded(await _run(handler, images=[image]))
    _assert_loaded(await _run(handler, images=[image]))
    _assert_budget(await _run(handler, images=[image, image]), _DECODE_TOKEN)


@pytest.mark.asyncio
async def test_audio_downloads_share_a_request_budget(
    build_handler, monkeypatch, wide_wav
):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", _DOWNLOAD_LIMIT_MB
    )
    handler, strategy = build_handler("joint")
    _stop_after_load(strategy, monkeypatch)
    bodies = _http_urls("audio", 2, wide_wav)
    with _serve(bodies) as server:
        urls = [server.url(path) for path in bodies]
        outcome = await _run(handler, audio=urls)
        assert server.hits
    _assert_budget(outcome, _DOWNLOAD_TOKEN)


@pytest.mark.asyncio
async def test_audio_decodes_share_a_request_budget(build_handler, monkeypatch):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", _AUDIO_DECODE_LIMIT
    )
    handler, strategy = build_handler("joint")
    _stop_after_load(strategy, monkeypatch)
    urls = [_data_url("audio/wav", _wav_bytes(1.0)) for _ in range(2)]
    outcome = await _run(handler, audio=urls)
    _assert_budget(outcome, _DECODE_TOKEN)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _VIDEO_KINDS)
async def test_video_downloads_share_a_request_budget(
    build_handler, monkeypatch, kind, wide_avi
):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", _DOWNLOAD_LIMIT_MB
    )
    handler, strategy = build_handler(kind)
    _stop_after_load(strategy, monkeypatch)
    bodies = _http_urls("videos", 2, wide_avi)
    with _serve(bodies) as server:
        urls = [server.url(path) for path in bodies]
        outcome = await _run(handler, videos=urls)
        assert server.hits
    _assert_budget(outcome, _DOWNLOAD_TOKEN)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _VIDEO_KINDS)
async def test_video_decodes_share_a_request_budget(
    build_handler, monkeypatch, kind, tiny_avi
):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", _VIDEO_DECODE_LIMIT
    )
    handler, strategy = build_handler(kind)
    _stop_after_load(strategy, monkeypatch)
    bodies = _http_urls("videos", 2, tiny_avi)
    with _serve(bodies) as server:
        urls = [server.url(path) for path in bodies]
        outcome = await _run(handler, videos=urls)
        assert server.hits
    _assert_budget(outcome, _DECODE_TOKEN)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _VIDEO_KINDS)
async def test_video_jpeg_data_urls_share_a_decode_budget(
    build_handler, monkeypatch, kind
):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", _IMAGE_DECODE_LIMIT
    )
    handler, strategy = build_handler(kind)
    _stop_after_load(strategy, monkeypatch)
    frame = base64.b64encode(_jpeg_bytes()).decode("ascii")
    urls = [f"data:video/jpeg;base64,{frame}" for _ in range(2)]
    outcome = await _run(handler, videos=urls)
    _assert_budget(outcome, _DECODE_TOKEN)


@pytest.mark.asyncio
async def test_video_soundtrack_shares_the_download_budget(
    build_handler, monkeypatch, wide_avi
):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", _DOWNLOAD_LIMIT_MB
    )
    joint, joint_strategy = build_handler("joint")
    clef, clef_strategy = build_handler("clef")
    _stop_after_load(joint_strategy, monkeypatch)
    _stop_after_load(clef_strategy, monkeypatch)
    bodies = {"/clip.avi": wide_avi}
    with _serve(bodies) as server:
        url = server.url("/clip.avi")
        # Clef-Flash does not fetch a soundtrack, so one download of this
        # file stays under the request cap and must still be accepted.
        clef_outcome = await _run(clef, videos=[url])
        assert len(server.hits) == 1
    _assert_loaded(clef_outcome)
    with _serve(bodies) as server:
        url = server.url("/clip.avi")
        joint_outcome = await _run(joint, videos=[url])
        assert len(server.hits) >= 2
    # The picture and the soundtrack together exceed the request download cap.
    _assert_budget(joint_outcome, _DOWNLOAD_TOKEN)


@pytest.mark.asyncio
async def test_route_rejects_too_many_images_before_fetch(build_handler, monkeypatch):
    handler, strategy = build_handler("next")
    _stop_after_load(strategy, monkeypatch)
    bodies = _http_urls("images", MAX_MEDIA_URLS + 1, b"x")
    with _serve(bodies) as server:
        urls = [server.url(path) for path in bodies]
        response = _post(handler, _json_body(images=urls))
        assert server.hits == [], response.text
    assert response.status_code == 400
    assert str(MAX_MEDIA_URLS) in response.text
    assert "image" in response.text


@pytest.mark.asyncio
async def test_route_rejects_image_downloads_over_the_request_budget(
    build_handler, monkeypatch, padded_png
):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", _DOWNLOAD_LIMIT_MB
    )
    handler, strategy = build_handler("next")
    _stop_after_load(strategy, monkeypatch)
    bodies = _http_urls("images", 2, padded_png)
    with _serve(bodies) as server:
        urls = [server.url(path) for path in bodies]
        response = _post(handler, _json_body(images=urls))
        assert server.hits
    assert response.status_code in (400, 422)
    assert _DOWNLOAD_TOKEN in response.text
    assert _LOADED not in response.text


@pytest.mark.asyncio
async def test_route_rejects_image_decodes_over_the_request_budget(
    build_handler, monkeypatch
):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", _IMAGE_DECODE_LIMIT
    )
    handler, strategy = build_handler("next")
    _stop_after_load(strategy, monkeypatch)
    urls = [_data_url("image/png", _png_bytes()) for _ in range(2)]
    response = _post(handler, _json_body(images=urls))
    assert response.status_code in (400, 422)
    assert _DECODE_TOKEN in response.text
    assert _LOADED not in response.text
