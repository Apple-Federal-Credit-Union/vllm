# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request media budgets must cover a whole HTTP request.

``VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB`` and
``VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES`` are documented as aggregate caps for
one request. These tests send one request whose conversations or documents
are each under the cap and whose sum is over it. They fail while each item
gets a fresh budget.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import pytest
from PIL import Image

import vllm.connections as connections
import vllm.envs as envs
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    PoolerConfig,
)
from vllm.entrypoints.openai.chat_completion.batch_serving import (
    OpenAIServingChatBatch,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.base.serving import PoolingServing
from vllm.entrypoints.pooling.embed.io_processor import EmbedIOProcessor
from vllm.entrypoints.pooling.embed.protocol import (
    CohereEmbedRequest,
    EmbeddingBatchChatRequest,
)
from vllm.entrypoints.pooling.scoring.io_processor import (
    BiEncoderIOProcessor,
    CrossEncoderIOProcessor,
)
from vllm.entrypoints.pooling.scoring.protocol import (
    RerankRequest,
    ScoreQueriesDocumentsRequest,
)
from vllm.entrypoints.pooling.typing import PoolingServeContext
from vllm.multimodal.media.image import ImageMediaIO
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.chat_utils import ChatTemplateConfig
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers import get_tokenizer
from vllm.utils.mem_constants import MiB_bytes

_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
_DOWNLOAD_LIMIT_MB = 1
_DOWNLOAD_LIMIT_BYTES = _DOWNLOAD_LIMIT_MB * MiB_bytes
# Each remote image is under the request cap. Two of them are over it, past
# the bytes one in-flight read can pull after the cap is reached.
_IMAGE_BYTES = 800_000
_DOWNLOAD_SLACK = 4 * connections._RESPONSE_READ_CHUNK_SIZE
_DECODE_LIMIT = 100
_DECODE_EDGE = 4

pytestmark = pytest.mark.skip_global_cleanup


class _FrontendConfig:
    """Real config objects the renderer reads while it is loading media."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.parallel_config = ParallelConfig()
        self.observability_config = ObservabilityConfig()
        self.cache_config = CacheConfig(enable_prefix_caching=False)
        self.scheduler_config = None


class _Engine:
    def __init__(self, model_config: ModelConfig, renderer, vllm_config) -> None:
        self.model_config = model_config
        self.renderer = renderer
        self.vllm_config = vllm_config
        self.input_processor = None
        self.errored = False

    @property
    def dead_error(self) -> BaseException:
        return RuntimeError("engine is dead")

    def check_admission(self, n: int = 1, request_id: str | None = None) -> None:
        return None

    async def generate(self, *_args, **_kwargs):
        """The engine is downstream of media loading. Yield one finished output."""
        yield RequestOutput(
            request_id="media-budget",
            prompt=None,
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="ok",
                    token_ids=[2],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            finished=True,
        )


def _png(color: tuple[int, int, int], size: int) -> bytes:
    image = Image.new("RGB", (size, size), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _padded_png(color: tuple[int, int, int], nbytes: int) -> bytes:
    raw = _png(color, 8)
    if len(raw) >= nbytes:
        raise RuntimeError(f"png is {len(raw)} bytes, target is {nbytes}")
    return raw + b"\x00" * (nbytes - len(raw))


def _data_url(color: tuple[int, int, int], size: int = _DECODE_EDGE) -> str:
    import base64

    encoded = base64.b64encode(_png(color, size)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_message(url: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]


def _score_image(url: str) -> dict:
    return {"content": [{"type": "image_url", "image_url": {"url": url}}]}


class _ImageServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(handler) -> None:  # noqa: N802
                body = bodies.get(handler.path)
                if body is None:
                    handler.send_error(404)
                    return
                handler.send_response(200)
                handler.send_header("Content-Type", "image/png")
                # No Content-Length: the size is known only by reading.
                handler.end_headers()
                handler.wfile.write(body)

            def log_message(self, format: str, *args) -> None:
                return

        super().__init__(("127.0.0.1", 0), _Handler)

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


@pytest.fixture(scope="module")
def image_server():
    bodies = {
        "/a.png": _padded_png((255, 0, 0), _IMAGE_BYTES),
        "/b.png": _padded_png((0, 0, 255), _IMAGE_BYTES),
    }
    server = _ImageServer(bodies)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


@pytest.fixture(scope="module")
def media_stack():
    model_config = ModelConfig(
        _MODEL_ID,
        runner="generate",
        dtype="float32",
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=2048,
        generation_config="vllm",
        limit_mm_per_prompt={"image": 4},
        mm_processor_cache_gb=0,
    )
    tokenizer = get_tokenizer(model_config.tokenizer, trust_remote_code=True)
    vllm_config = _FrontendConfig(model_config)
    renderer = HfRenderer(vllm_config, tokenizer)
    online = OnlineRenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    engine = _Engine(model_config, renderer, vllm_config)
    models = OpenAIServingModels(
        engine,
        [BaseModelPath(name=_MODEL_ID, model_path=_MODEL_ID)],
    )
    chat = OpenAIServingChatBatch(
        engine,
        models,
        "assistant",
        online_renderer=online,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    model_config.pooler_config = PoolerConfig(task="embed")
    template = ChatTemplateConfig()
    embed_io = EmbedIOProcessor(vllm_config, renderer, template)
    bi_io = BiEncoderIOProcessor(vllm_config, renderer, template)
    cross_io = CrossEncoderIOProcessor(vllm_config, renderer, template)
    yield {
        "chat": chat,
        "embed_io": embed_io,
        "bi_io": bi_io,
        "cross_io": cross_io,
    }
    renderer.shutdown()


@pytest.fixture(autouse=True)
def _reset_http_session():
    """Each test opens its own aiohttp session on its own event loop."""
    connection = connections.global_http_connection
    connection._async_client = None
    yield
    connection._async_client = None


@pytest.fixture
def media_meters(monkeypatch):
    downloaded = [0]
    decoded: list[int] = []
    lock = threading.Lock()
    original_append = connections._append_chunk
    original_load = ImageMediaIO.load_bytes

    def append_chunk(
        chunks,
        chunk,
        received_bytes,
        max_bytes,
        media_download_limit,
        request_budget=None,
    ):
        if chunk:
            with lock:
                downloaded[0] += len(chunk)
        return original_append(
            chunks,
            chunk,
            received_bytes,
            max_bytes,
            media_download_limit,
            request_budget,
        )

    def load_bytes(self, data: bytes):
        loaded = original_load(self, data)
        width, height = loaded.media.size
        with lock:
            decoded.append(width * height * 4)
        return loaded

    monkeypatch.setattr(connections, "_append_chunk", append_chunk)
    monkeypatch.setattr(ImageMediaIO, "load_bytes", load_bytes)
    monkeypatch.setattr(envs, "VLLM_MEDIA_CACHE", "")
    monkeypatch.setattr(envs, "VLLM_MEDIA_FETCH_MAX_RETRIES", 1)
    return downloaded, decoded


def _limit_download(monkeypatch) -> None:
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", _DOWNLOAD_LIMIT_MB
    )
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 10_000_000)


def _limit_decode(monkeypatch) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", _DECODE_LIMIT)
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", 256)


def _error_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


def _assert_budget(
    caught: BaseException | None,
    token: str,
    measured: int,
    limit: int,
) -> None:
    assert caught is not None, (
        f"{token} was not enforced; measured {measured} bytes, limit {limit}"
    )
    text = _error_text(caught)
    assert token in text, (
        f"expected rejection from {token}, measured {measured} bytes "
        f"(limit {limit}), got:\n{text}"
    )
    assert measured <= limit, (
        f"{token} still accepted {measured} bytes; limit is {limit}"
    )


async def _run(call) -> BaseException | None:
    try:
        await call()
    except Exception as exc:
        return exc
    return None


async def _preprocess(io, request) -> None:
    """Run ``PoolingServing._preprocessing`` for one pooling request."""
    ctx = PoolingServeContext(
        request=request,
        model_name=_MODEL_ID,
        request_id="media-budget",
        pooling_params=io.create_pooling_params(request),
        lora_request=None,
        priorities=getattr(request, "priority", 0),
        prompt_extras=None,
    )
    await PoolingServing._preprocessing(None, io, ctx)


def _http_urls(server: _ImageServer) -> tuple[str, str]:
    return f"{server.base_url}/a.png", f"{server.base_url}/b.png"


@pytest.mark.asyncio
async def test_one_conversation_download_budget_rejects_two_images(
    image_server, media_stack, media_meters, monkeypatch
):
    """A single conversation already shares one download budget."""
    downloaded, _decoded = media_meters
    _limit_download(monkeypatch)
    first, second = _http_urls(image_server)
    request = BatchChatCompletionRequest(
        model=_MODEL_ID,
        messages=[
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": first}},
                        {"type": "image_url", "image_url": {"url": second}},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ]
        ],
    )
    caught = await _run(lambda: media_stack["chat"].render_batch_chat_request(request))
    _assert_budget(
        caught,
        "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB",
        downloaded[0],
        _DOWNLOAD_LIMIT_BYTES + _DOWNLOAD_SLACK,
    )


@pytest.mark.asyncio
async def test_one_conversation_decode_budget_rejects_two_images(
    media_stack, media_meters, monkeypatch
):
    """A single conversation already shares one decode budget."""
    _downloaded, decoded = media_meters
    _limit_decode(monkeypatch)
    request = BatchChatCompletionRequest(
        model=_MODEL_ID,
        messages=[
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url((255, 0, 0))},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url((0, 255, 0))},
                        },
                        {"type": "text", "text": "describe"},
                    ],
                }
            ]
        ],
    )
    caught = await _run(lambda: media_stack["chat"].render_batch_chat_request(request))
    _assert_budget(
        caught,
        "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES",
        sum(decoded),
        _DECODE_LIMIT,
    )


@pytest.mark.asyncio
async def test_batch_chat_download_budget_is_per_request(
    image_server, media_stack, media_meters, monkeypatch
):
    downloaded, _decoded = media_meters
    _limit_download(monkeypatch)
    first, second = _http_urls(image_server)
    request = BatchChatCompletionRequest(
        model=_MODEL_ID,
        messages=[_image_message(first), _image_message(second)],
    )
    caught = await _run(
        lambda: media_stack["chat"].create_batch_chat_completion(request)
    )
    _assert_budget(
        caught,
        "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB",
        downloaded[0],
        _DOWNLOAD_LIMIT_BYTES + _DOWNLOAD_SLACK,
    )


@pytest.mark.asyncio
async def test_batch_chat_decode_budget_is_per_request(
    media_stack, media_meters, monkeypatch
):
    _downloaded, decoded = media_meters
    _limit_decode(monkeypatch)
    request = BatchChatCompletionRequest(
        model=_MODEL_ID,
        messages=[
            _image_message(_data_url((255, 0, 0))),
            _image_message(_data_url((0, 0, 255))),
        ],
    )
    caught = await _run(
        lambda: media_stack["chat"].create_batch_chat_completion(request)
    )
    _assert_budget(
        caught,
        "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES",
        sum(decoded),
        _DECODE_LIMIT,
    )


async def _assert_pooling(
    io,
    request,
    meters,
    monkeypatch,
    kind: str,
) -> None:
    downloaded, decoded = meters
    if kind == "download":
        _limit_download(monkeypatch)
        token = "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB"
        limit = _DOWNLOAD_LIMIT_BYTES + _DOWNLOAD_SLACK
    else:
        _limit_decode(monkeypatch)
        token = "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"
        limit = _DECODE_LIMIT
    caught = await _run(lambda: _preprocess(io, request))
    measured = downloaded[0] if kind == "download" else sum(decoded)
    _assert_budget(caught, token, measured, limit)


def _embed_messages(first: str, second: str) -> EmbeddingBatchChatRequest:
    return EmbeddingBatchChatRequest(
        model=_MODEL_ID,
        messages=[_image_message(first), _image_message(second)],
    )


@pytest.mark.asyncio
async def test_embedding_batch_download_budget_is_per_request(
    image_server, media_stack, media_meters, monkeypatch
):
    first, second = _http_urls(image_server)
    await _assert_pooling(
        media_stack["embed_io"],
        _embed_messages(first, second),
        media_meters,
        monkeypatch,
        "download",
    )


@pytest.mark.asyncio
async def test_embedding_batch_decode_budget_is_per_request(
    media_stack, media_meters, monkeypatch
):
    await _assert_pooling(
        media_stack["embed_io"],
        _embed_messages(_data_url((255, 0, 0)), _data_url((0, 0, 255))),
        media_meters,
        monkeypatch,
        "decode",
    )


@pytest.mark.asyncio
async def test_cohere_embed_images_download_budget_is_per_request(
    image_server, media_stack, media_meters, monkeypatch
):
    first, second = _http_urls(image_server)
    await _assert_pooling(
        media_stack["embed_io"],
        CohereEmbedRequest(model=_MODEL_ID, images=[first, second]),
        media_meters,
        monkeypatch,
        "download",
    )


@pytest.mark.asyncio
async def test_bi_encoder_rerank_download_budget_is_per_request(
    image_server, media_stack, media_meters, monkeypatch
):
    first, second = _http_urls(image_server)
    await _assert_pooling(
        media_stack["bi_io"],
        RerankRequest(
            model=_MODEL_ID,
            query="query",
            documents=[_score_image(first), _score_image(second)],
        ),
        media_meters,
        monkeypatch,
        "download",
    )


@pytest.mark.asyncio
async def test_bi_encoder_rerank_decode_budget_is_per_request(
    media_stack, media_meters, monkeypatch
):
    await _assert_pooling(
        media_stack["bi_io"],
        RerankRequest(
            model=_MODEL_ID,
            query="query",
            documents=[
                _score_image(_data_url((255, 0, 0))),
                _score_image(_data_url((0, 0, 255))),
            ],
        ),
        media_meters,
        monkeypatch,
        "decode",
    )


@pytest.mark.asyncio
async def test_cross_encoder_score_download_budget_is_per_request(
    image_server, media_stack, media_meters, monkeypatch
):
    first, second = _http_urls(image_server)
    await _assert_pooling(
        media_stack["cross_io"],
        ScoreQueriesDocumentsRequest(
            model=_MODEL_ID,
            queries="query",
            documents=[_score_image(first), _score_image(second)],
        ),
        media_meters,
        monkeypatch,
        "download",
    )


@pytest.mark.asyncio
async def test_cross_encoder_score_decode_budget_is_per_request(
    media_stack, media_meters, monkeypatch
):
    await _assert_pooling(
        media_stack["cross_io"],
        ScoreQueriesDocumentsRequest(
            model=_MODEL_ID,
            queries="query",
            documents=[
                _score_image(_data_url((255, 0, 0))),
                _score_image(_data_url((0, 0, 255))),
            ],
        ),
        media_meters,
        monkeypatch,
        "decode",
    )
