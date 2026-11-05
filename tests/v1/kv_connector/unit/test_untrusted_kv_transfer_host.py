# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordinary inference requests must not steer KV transfer at a chosen host.

These tests fail while a chat, completion, responses, Anthropic, Cohere, or
batch chat body can place ``kv_transfer_params`` on the engine request and a
Mooncake, NIXL, or MoRI-IO connector then dials that address. They pass once
that field is rejected for ordinary clients, or the connector refuses the
host before any socket or HTTP request is opened.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import select
import socket
import threading
import time

os.environ.setdefault("VLLM_MOONCAKE_CONNECTOR_TIMEOUT", "1")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgspec
import pytest
import torch
import zmq
import zmq.asyncio
from cohere.types import UserChatMessageV2

from vllm.config import CacheConfig, DeviceConfig, KVTransferConfig, VllmConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeXferResponse,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.entrypoints.anthropic.protocol import (
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.cohere.protocol import CohereChatV2Request
from vllm.entrypoints.cohere.serving import CohereServingChatV2
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.exceptions import VLLMValidationError
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.skip_global_cleanup

_PROMPT = [11, 22, 33, 44]
_ENGINE = "remote-engine"


class _QuietHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        server = self.server
        assert isinstance(server, _BootstrapServer)
        server.http_paths.append(self.path)
        body = json.dumps(server.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


class _BootstrapServer(ThreadingHTTPServer):
    def __init__(self, payload: dict, http_paths: list[str]) -> None:
        super().__init__(("127.0.0.1", 0), _QuietHandler)
        self.payload = payload
        self.http_paths = http_paths


class _Listener:
    """Records HTTP GETs and TCP accepts on loopback."""

    def __init__(self) -> None:
        self.http_paths: list[str] = []
        self.tcp_peers: list[tuple[str, int]] = []
        self._stop = threading.Event()
        self._tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp.bind(("127.0.0.1", 0))
        self._tcp.listen(16)
        self._tcp.setblocking(False)
        self.tcp_port = self._tcp.getsockname()[1]
        payload = {
            "0": {
                "engine_id": _ENGINE,
                "worker_addr": {"0": {"0": f"tcp://127.0.0.1:{self.tcp_port}"}},
            }
        }
        self._http = _BootstrapServer(payload, self.http_paths)
        self.http_port = self._http.server_address[1]
        self._threads = [
            threading.Thread(target=self._http.serve_forever, daemon=True),
            threading.Thread(target=self._accept, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _accept(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([self._tcp], [], [], 0.05)
            if not readable:
                continue
            try:
                conn, addr = self._tcp.accept()
            except BlockingIOError:
                continue
            self.tcp_peers.append(addr)
            conn.close()

    @property
    def bootstrap_url(self) -> str:
        return f"http://127.0.0.1:{self.http_port}"

    def wait(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.http_paths or self.tcp_peers:
                return
            time.sleep(0.01)

    def close(self) -> None:
        self._stop.set()
        self._http.shutdown()
        self._http.server_close()
        self._tcp.close()

    def assert_unused(self) -> None:
        assert not self.http_paths and not self.tcp_peers, (
            "kv_transfer_params opened a server-side connection before "
            f"refusal: http={self.http_paths} tcp={self.tcp_peers}"
        )


@pytest.fixture
def listener():
    opened = _Listener()
    try:
        yield opened
    finally:
        opened.close()


def _kv_cache_config() -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )


def _vllm_config(
    connector: str,
    role: str,
    extra: dict | None = None,
) -> VllmConfig:
    return VllmConfig(
        cache_config=CacheConfig(block_size=16),
        device_config=DeviceConfig("cpu"),
        scheduler_config=SchedulerConfig.default_factory(
            max_model_len=128,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=True,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector=connector,
            kv_role=role,
            engine_id="local-engine",
            kv_connector_extra_config=extra or {},
        ),
    )


def _blocks() -> KVCacheBlocks:
    return KVCacheBlocks(blocks=((KVCacheBlock(block_id=7),),))


def _scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _engine_request(external_id: str, sampling, prompt: list[int]) -> Request:
    # InputProcessor.process_inputs deep-copies sampling params and does not
    # otherwise rewrite kv_transfer_params. assign_request_id then suffixes
    # the id the serving layer already prefixed.
    core = EngineCoreRequest(
        request_id=external_id,
        prompt_token_ids=list(prompt),
        mm_features=None,
        sampling_params=sampling.clone(),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    InputProcessor.assign_request_id(core)
    return Request.from_engine_core_request(core, block_hasher=None)


def _chat_request(params: dict, request_id: str) -> Request:
    chat = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        kv_transfer_params=params,
        request_id=request_id,
    )
    external = f"chatcmpl-{BaseServing._base_request_id(None, chat.request_id)}"
    sampling = chat.to_sampling_params(8, {})
    return _engine_request(external, sampling, _PROMPT)


def _completion_request(params: dict, request_id: str) -> Request:
    completion = CompletionRequest(
        prompt="hi",
        max_tokens=8,
        kv_transfer_params=params,
        request_id=request_id,
    )
    external = f"cmpl-{BaseServing._base_request_id(None, completion.request_id)}"
    sampling = completion.to_sampling_params(8, {})
    return _engine_request(external, sampling, _PROMPT)


def _responses_request(params: dict, request_id: str) -> Request:
    responses = ResponsesRequest(
        input="hi",
        kv_transfer_params=params,
        request_id=request_id,
    )
    sampling = responses.to_sampling_params(8, {})
    return _engine_request(responses.request_id, sampling, _PROMPT)


def _anthropic_request(params: dict) -> Request:
    anthropic = AnthropicMessagesRequest(
        model="test",
        messages=[AnthropicMessage(role="user", content="hi")],
        max_tokens=8,
        kv_transfer_params=params,
    )
    chat = AnthropicServingMessages._build_base_request(
        anthropic, [{"role": "user", "content": "hi"}]
    )
    external = f"chatcmpl-{BaseServing._base_request_id(None, chat.request_id)}"
    return _engine_request(external, chat.to_sampling_params(8, {}), _PROMPT)


def _cohere_request(params: dict) -> Request:
    cohere = CohereChatV2Request(
        model="test",
        messages=[UserChatMessageV2(content="hi")],
        kv_transfer_params=params,
    )
    chat = CohereServingChatV2._build_base_chat_completion(
        cohere, [{"role": "user", "content": "hi"}]
    )
    external = f"chatcmpl-{BaseServing._base_request_id(None, chat.request_id)}"
    return _engine_request(external, chat.to_sampling_params(8, {}), _PROMPT)


def _batch_request(params: dict) -> Request:
    batch = BatchChatCompletionRequest(
        messages=[[{"role": "user", "content": "hi"}]],
        kv_transfer_params=params,
    )
    chat = batch.to_chat_completion_request(batch.messages[0])
    external = f"chatcmpl-{BaseServing._base_request_id(None, chat.request_id)}"
    return _engine_request(external, chat.to_sampling_params(8, {}), _PROMPT)


def _mooncake_params(listener: _Listener) -> dict:
    return {
        "do_remote_prefill": True,
        "remote_engine_id": _ENGINE,
        "remote_bootstrap_addr": listener.bootstrap_url,
        "transfer_id": "xfer-1",
    }


def _nixl_params(listener: _Listener) -> dict:
    return {
        "do_remote_prefill": True,
        "remote_engine_id": _ENGINE,
        "remote_request_id": "prefill-1",
        "remote_host": "127.0.0.1",
        "remote_port": listener.tcp_port,
        "remote_block_ids": [1, 2],
        "tp_size": 1,
    }


def _run_quietly(action, listener: _Listener) -> None:
    caught: Exception | None = None
    try:
        action()
    except VLLMValidationError:
        caught = None
    except Exception as exc:
        caught = exc
    listener.wait(0.5)
    listener.assert_unused()
    if caught is not None:
        raise caught


def _schedule_mooncake(request: Request):
    scheduler = MooncakeConnectorScheduler(
        _vllm_config("MooncakeConnector", "kv_consumer"),
        "local-engine",
        _kv_cache_config(),
    )
    scheduler.update_state_after_alloc(request, _blocks(), len(_PROMPT))
    return scheduler.build_connector_meta(_scheduler_output())


def _drive_mooncake(meta) -> None:
    if not meta.reqs_to_recv:
        return
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker._remote_agents = {}
    worker._tp_size = {}
    worker._pending_bootstrap_queries = {}
    worker.pp_size = 1
    worker.pp_rank = 0
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.hostname = "127.0.0.1"
    worker.rpc_port = 9
    worker._kv_layout_id = 0
    worker._acked_layout_peers = set()
    worker.kv_caches_base_addr = []
    worker.block_len_per_layer = []
    worker.kv_block_len_per_layer = []
    worker.registered_layer_names = []
    worker.registered_layer_indices = []
    worker.registered_group_indices = []
    worker.region_shared_groups = []
    worker.region_row_offsets = []
    worker.finished_recving_reqs = set()
    worker._invalid_block_ids = queue.Queue()
    worker._failed_recv_reqs = queue.Queue()
    worker.xfer_stats = MooncakeKVConnectorStats()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.transfer_topo = TransferTopology(
        tp_rank=0,
        tp_size=1,
        block_size=16,
        engine_id="local-engine",
        is_mla=False,
        is_mamba=False,
        total_num_kv_heads=1,
        attn_backends=[],
    )
    worker._encoder = msgspec.msgpack.Encoder()
    worker._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
    worker.async_zmq_ctx = zmq.asyncio.Context()

    async def drive() -> None:
        try:
            for engine_id, pulls in meta.reqs_to_recv.items():
                await worker.handle_new_engine_id(engine_id, pulls)
            await asyncio.sleep(0.2)
        finally:
            current = asyncio.current_task()
            for task in asyncio.all_tasks():
                if task is not current:
                    task.cancel()

    try:
        asyncio.run(drive())
    finally:
        worker.async_zmq_ctx.term()


def _schedule_nixl(request: Request):
    scheduler = NixlPullConnectorScheduler(
        _vllm_config("NixlConnector", "kv_consumer"),
        "local-engine",
        _kv_cache_config(),
    )
    scheduler.on_new_request(request)
    scheduler.update_state_after_alloc(request, _blocks(), len(_PROMPT))
    return scheduler.build_connector_meta(_scheduler_output())


def _dial_nixl(host: str, port: int, tp_size: int, engine_id: str) -> None:
    worker = NixlBaseConnectorWorker.__new__(NixlBaseConnectorWorker)
    worker._is_csa_linear = False
    worker.use_host_buffer = True
    worker.transfer_topo = TransferTopology(
        tp_rank=0,
        tp_size=1,
        block_size=16,
        engine_id="local-engine",
        is_mla=False,
        is_mamba=False,
        total_num_kv_heads=1,
        attn_backends=[],
    )
    try:
        worker._nixl_handshake(host, int(port), int(tp_size), engine_id, 1, 1)
    except Exception:
        return


def _drive_nixl(meta) -> None:
    targets: list[tuple[str, int, int, str]] = []
    for req_meta in meta.reqs_to_recv.values():
        remote = req_meta.remote
        if remote is None:
            continue
        targets.append(
            (remote.host, int(remote.port), req_meta.tp_size, remote.engine_id)
        )
    for engine_id, heartbeat in meta.heartbeat_by_engine.items():
        targets.append(
            (
                heartbeat.host,
                int(heartbeat.port),
                heartbeat.tp_size,
                engine_id,
            )
        )
    threads = [
        threading.Thread(
            target=_dial_nixl, args=target, daemon=True
        )
        for target in targets
    ]
    for thread in threads:
        thread.start()


def _moriio_config(read_mode: str) -> VllmConfig:
    return _vllm_config(
        "MoRIIOConnector",
        "kv_consumer",
        {
            "handshake_port": 50000,
            "notify_port": 50001,
            "host_ip": "127.0.0.1",
            "read_mode": read_mode,
        },
    )


def _moriio_scheduler(read_mode: str) -> MoRIIOConnectorScheduler:
    return MoRIIOConnectorScheduler(
        _moriio_config(read_mode),
        "127.0.0.1:50000",
        _kv_cache_config(),
    )


def _peer_request_id(listener: _Listener) -> str:
    peer = f"host:127.0.0.1,handshake:{listener.tcp_port},notify:{listener.tcp_port}"
    other = "host:203.0.113.9,handshake:2,notify:3"
    uuid = "0123456789abcdef0123456789abcdef"
    return f"___prefill_addr_{peer}___decode_addr_{other}_{uuid}"


def _drive_moriio_handshake(meta) -> None:
    if not meta.reqs_to_recv:
        return
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.world_size = 1
    worker.tp_rank = 0

    def dial(req_meta) -> None:
        try:
            worker._moriio_handshake(
                req_meta.remote_host,
                int(req_meta.remote_handshake_port),
                int(req_meta.tp_size) or 1,
                req_meta.remote_engine_id,
                0,
                0,
            )
        except Exception:
            return

    for req_meta in meta.reqs_to_recv.values():
        threading.Thread(target=dial, args=(req_meta,), daemon=True).start()


def test_chat_completion_mooncake_does_not_query_client_bootstrap(listener):
    """POST /v1/chat/completions must not GET a client bootstrap URL."""

    def action() -> None:
        request = _chat_request(_mooncake_params(listener), "chat-1")
        _drive_mooncake(_schedule_mooncake(request))

    _run_quietly(action, listener)


def test_completion_nixl_does_not_handshake_client_host(listener):
    """POST /v1/completions must not NIXL-handshake a client host."""

    def action() -> None:
        request = _completion_request(_nixl_params(listener), "cmpl-1")
        _drive_nixl(_schedule_nixl(request))

    _run_quietly(action, listener)


def test_responses_mooncake_does_not_query_client_bootstrap(listener):
    """POST /v1/responses copies the same object into the engine request."""

    def action() -> None:
        request = _responses_request(_mooncake_params(listener), "resp-1")
        _drive_mooncake(_schedule_mooncake(request))

    _run_quietly(action, listener)


def test_anthropic_messages_mooncake_does_not_query_client_bootstrap(listener):
    """Anthropic messages forwards kv_transfer_params into chat sampling."""

    def action() -> None:
        request = _anthropic_request(_mooncake_params(listener))
        _drive_mooncake(_schedule_mooncake(request))

    _run_quietly(action, listener)


def test_cohere_chat_mooncake_does_not_query_client_bootstrap(listener):
    """Cohere chat v2 forwards kv_transfer_params into chat sampling."""

    def action() -> None:
        request = _cohere_request(_mooncake_params(listener))
        _drive_mooncake(_schedule_mooncake(request))

    _run_quietly(action, listener)


def test_batch_chat_mooncake_does_not_query_client_bootstrap(listener):
    """Batch chat keeps extra kv_transfer_params on each child request."""

    def action() -> None:
        request = _batch_request(_mooncake_params(listener))
        _drive_mooncake(_schedule_mooncake(request))

    _run_quietly(action, listener)


def test_moriio_notify_does_not_dial_client_remote_host(listener):
    """MoRI-IO WRITE decode must not ZMQ-notify kv_transfer_params.remote_host."""

    def action() -> None:
        params = {
            "do_remote_prefill": True,
            "remote_engine_id": _ENGINE,
            "remote_block_ids": [1],
            "remote_host": "127.0.0.1",
            "remote_handshake_port": listener.tcp_port,
            "remote_notify_port": listener.tcp_port,
            "transfer_id": "xfer-1",
            "tp_size": 1,
        }
        request = _chat_request(params, "plain-request")
        scheduler = _moriio_scheduler("false")
        try:
            scheduler.update_state_after_alloc(request, _blocks(), len(_PROMPT))
        finally:
            scheduler.shutdown()

    _run_quietly(action, listener)


def test_moriio_notify_does_not_dial_request_id_peer(listener):
    """MoRI-IO must not parse a peer address out of the client request id."""

    def action() -> None:
        params = {
            "do_remote_prefill": True,
            "remote_engine_id": _ENGINE,
            "remote_block_ids": [1],
            "transfer_id": "xfer-1",
            "tp_size": 1,
        }
        request = _completion_request(params, _peer_request_id(listener))
        scheduler = _moriio_scheduler("false")
        try:
            scheduler.update_state_after_alloc(request, _blocks(), len(_PROMPT))
        finally:
            scheduler.shutdown()

    _run_quietly(action, listener)


def test_moriio_read_does_not_handshake_request_id_peer(listener):
    """MoRI-IO READ handshake uses the address parsed in add_new_req."""

    def action() -> None:
        params = {
            "do_remote_prefill": True,
            "remote_engine_id": _ENGINE,
            "remote_block_ids": [[1, 2]],
            "transfer_id": "xfer-1",
            "tp_size": 1,
        }
        request = _chat_request(params, _peer_request_id(listener))
        scheduler = _moriio_scheduler("true")
        try:
            scheduler.update_state_after_alloc(request, _blocks(), len(_PROMPT))
            meta = scheduler.build_connector_meta(_scheduler_output())
        finally:
            scheduler.shutdown()
        _drive_moriio_handshake(meta)

    _run_quietly(action, listener)


def test_moriio_read_does_not_handshake_client_remote_host(listener):
    """MoRI-IO READ falls back to kv_transfer_params.remote_host."""

    def action() -> None:
        params = {
            "do_remote_prefill": True,
            "remote_engine_id": _ENGINE,
            "remote_block_ids": [[1, 2]],
            "remote_host": "127.0.0.1",
            "remote_handshake_port": listener.tcp_port,
            "remote_notify_port": listener.tcp_port,
            "transfer_id": "xfer-1",
            "tp_size": 1,
        }
        request = _responses_request(params, "plain-read")
        scheduler = _moriio_scheduler("true")
        try:
            scheduler.update_state_after_alloc(request, _blocks(), len(_PROMPT))
            meta = scheduler.build_connector_meta(_scheduler_output())
        finally:
            scheduler.shutdown()
        _drive_moriio_handshake(meta)

    _run_quietly(action, listener)
