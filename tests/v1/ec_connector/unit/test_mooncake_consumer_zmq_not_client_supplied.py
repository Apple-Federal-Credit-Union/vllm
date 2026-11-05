# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client-supplied Mooncake encoder-cache addresses must not be dialed.

These tests fail while a caller can put ``consumer_zmq`` on a chat,
completion, responses, or Anthropic request and have the producer open a
ZMQ connection to it. They pass once that address is ignored and no
reservation, completion, or tensor write is sent to a peer the caller
named, including a peer discovered by rewriting the host of that address
with ports from a handshake reply.

``mooncake-transfer-engine==0.3.14`` is installed, but its ``engine.so``
needs ``libcudart.so.12``, which this environment does not have. The
recording engine below is only the Python entry point named
``batch_transfer_sync_write`` in that package's ``engine.pyi``. It records
the arguments and returns success so the producer continues into the
completion RPC. It does not open a socket. The real body
(``TransferEnginePy::batchTransferSyncWrite`` in Mooncake v0.3.14) opens
the session named by ``target_hostname`` and submits a WRITE of the local
buffers at the peer offsets.
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import pytest
import torch
import zmq

_TRANSFER_WRITES: list[dict[str, Any]] = []


class RecordingTransferEngine:
    """Record ``batch_transfer_sync_write`` without opening a socket."""

    def initialize(
        self,
        local_hostname: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ) -> int:
        return 0

    def get_rpc_port(self) -> int:
        return 1

    def register_memory(
        self,
        buffer_addr: int,
        capacity: int,
        location: str = "*",
        remote_accessible: bool = True,
    ) -> int:
        return 0

    def unregister_memory(self, buffer_addr: int) -> int:
        return 0

    def batch_register_memory(
        self,
        buffer_addresses: list[int],
        capacities: list[int],
        location: str = "*",
    ) -> int:
        return 0

    def batch_unregister_memory(self, buffer_addresses: list[int]) -> int:
        return 0

    def batch_transfer_sync_write(
        self,
        target_hostname: str,
        buffers: list[int],
        peer_buffer_addresses: list[int],
        lengths: list[int],
        transport_hint: str = "",
        nic_hint: str = "",
    ) -> int:
        _TRANSFER_WRITES.append(
            {
                "target_hostname": target_hostname,
                "buffers": [int(value) for value in buffers],
                "peer_buffer_addresses": [
                    int(value) for value in peer_buffer_addresses
                ],
                "lengths": [int(value) for value in lengths],
            }
        )
        return 0


def _install_recording_engine() -> None:
    engine = types.ModuleType("mooncake.engine")
    engine.TransferEngine = RecordingTransferEngine  # type: ignore[attr-defined]
    package = sys.modules.get("mooncake")
    if package is None:
        package = types.ModuleType("mooncake")
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules["mooncake"] = package
    sys.modules["mooncake.engine"] = engine


_install_recording_engine()

from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.mooncake import (
    transfer as mooncake_transfer,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
)
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
# TEST-NET address (RFC 5737). The data-plane session is only a string the
# peer handshake hands back; nothing in this file connects to it.
_ATTACKER_SESSION = "192.0.2.10:40000"
_ATTACKER_PTR = 4096
_EMBEDS = 2
_WIDTH = 4

assert mooncake_transfer.TransferEngine is RecordingTransferEngine


class _SwitchingAddress(dict):
    """Return one consumer address on the first read and another after that."""

    def __init__(self, first: str, second: str, items: list[dict[str, str]]) -> None:
        super().__init__(consumer_zmq=first, ec_items=items)
        self._first = first
        self._second = second
        self.reads = 0

    def _current(self) -> str:
        self.reads += 1
        return self._first if self.reads == 1 else self._second

    def get(self, key: str, default: Any = None) -> Any:
        if key == "consumer_zmq":
            return self._current()
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        if key == "consumer_zmq":
            return self._current()
        return super().__getitem__(key)


def _sampling_params(api: str, ec_transfer_params: dict[str, Any]) -> SamplingParams:
    if api == "chat":
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4,
            ec_transfer_params=ec_transfer_params,
        )
        return request.to_sampling_params(max_tokens=4, default_sampling_params={})
    if api == "completion":
        request = CompletionRequest(
            model="test-model",
            prompt="hi",
            max_tokens=4,
            ec_transfer_params=ec_transfer_params,
        )
        return request.to_sampling_params(max_tokens=4, default_sampling_params={})
    if api == "responses":
        request = ResponsesRequest(
            model="test-model",
            input="hi",
            max_output_tokens=4,
            ec_transfer_params=ec_transfer_params,
        )
        return request.to_sampling_params(4, {})
    if api == "anthropic":
        anthropic = AnthropicMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4,
            ec_transfer_params=ec_transfer_params,
        )
        chat = AnthropicServingMessages._build_base_request(
            anthropic, [{"role": "user", "content": "hi"}]
        )
        return chat.to_sampling_params(max_tokens=4, default_sampling_params={})
    raise AssertionError(api)


def _engine_request(
    sampling_params: SamplingParams, count: int, idents: list[str]
) -> Request:
    return Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=ident,
                mm_position=PlaceholderRange(offset=0, length=_EMBEDS),
            )
            for ident in idents
        ],
    )


def _producer_config():
    config = create_ec_vllm_config(
        ec_role="ec_producer",
        dtype=torch.float16,
        encoder_output_widths={"image": _WIDTH},
    )
    config.ec_transfer_config.ec_connector = "ECMooncakeConnector"
    config.ec_transfer_config.ec_buffer_device = "cpu"
    config.ec_transfer_config.ec_buffer_size = 65536
    config.ec_transfer_config.ec_connector_extra_config = {
        "control_timeout_s": 1,
        "mooncake_protocol": "tcp",
    }
    return config


class _Peer:
    """REP peer that records every control request and answers the handshake."""

    def __init__(self) -> None:
        self.transcript: list[tuple[str, dict[str, Any]]] = []
        self._stop = threading.Event()
        self._context = zmq.Context()
        self._sockets: list[zmq.Socket] = []
        self._threads: list[threading.Thread] = []
        self.base = self._bind()
        self.shard = self._bind()
        self.base_port = int(self.base.rsplit(":", 1)[1])
        self.shard_port = int(self.shard.rsplit(":", 1)[1])

    def _bind(self) -> str:
        sock = self._context.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 100)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind("tcp://127.0.0.1:*")
        endpoint = sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        self._sockets.append(sock)
        thread = threading.Thread(
            target=self._serve, args=(sock, endpoint), daemon=True
        )
        thread.start()
        self._threads.append(thread)
        return endpoint

    def _serve(self, sock: zmq.Socket, endpoint: str) -> None:
        while not self._stop.is_set():
            try:
                message = sock.recv_json()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                return
            if not isinstance(message, dict):
                message = {"_non_object": repr(message)}
            self.transcript.append((endpoint, message))
            sock.send_json(self._reply(message))

    def _reply(self, message: dict[str, Any]) -> dict[str, Any]:
        op = message.get("op")
        if op == "peers":
            return {
                "ok": True,
                "result": {"ports": [self.base_port, self.shard_port]},
            }
        reservation = {
            "reservation_id": "reserved",
            "dst_session": _ATTACKER_SESSION,
            "dst_ptr": _ATTACKER_PTR,
            "nbytes": message.get("nbytes"),
            "write": True,
            "ready": False,
            "cached": False,
        }
        if op == "reserve":
            return {"ok": True, "result": reservation}
        if op == "reserve_batch":
            items = message.get("items") or []
            return {
                "ok": True,
                "result": {
                    "items": [
                        {
                            "ok": True,
                            "result": {
                                **reservation,
                                "nbytes": item.get("nbytes"),
                            },
                        }
                        for item in items
                    ]
                },
            }
        if op == "complete":
            return {"ok": True, "result": {"completed": True, "became_ready": False}}
        if op == "complete_batch":
            items = message.get("items") or []
            return {
                "ok": True,
                "result": {
                    "items": [
                        {"completed": True, "became_ready": False} for _ in items
                    ]
                },
            }
        if op == "cancel":
            return {"ok": True, "result": {"cancelled": True}}
        if op == "event_port":
            return {"ok": True, "result": self.shard_port}
        return {"ok": False, "error": "unexpected"}

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1)
        for sock in self._sockets:
            sock.close(linger=0)
        self._context.term()


def _wait_for_attack(peers: list[_Peer]) -> None:
    """Wait until a dial shows up, or long enough that one would have."""
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if _TRANSFER_WRITES or any(peer.transcript for peer in peers):
            break
        time.sleep(0.02)
    # Let a peers reply fan out into reserve, write, and complete.
    time.sleep(0.5)


def _run_producer(
    request: Request,
    *,
    trigger: str,
    idents: list[str],
    peers: list[_Peer],
) -> None:
    config = _producer_config()
    scheduler = ECMooncakeConnector(config, ECConnectorRole.SCHEDULER)
    worker = None
    try:
        worker = ECMooncakeConnector(config, ECConnectorRole.WORKER)
        if trigger == "schedule":
            scheduler.ensure_cache_available(request, 0)
        elif trigger == "alloc":
            for index in range(len(idents)):
                scheduler.update_state_after_alloc(request, index)
        else:
            raise AssertionError(trigger)
        metadata = scheduler.build_connector_meta(SchedulerOutput.make_empty())
        worker.bind_connector_metadata(metadata)
        worker.start_save_caches(
            encoder_cache={
                ident: torch.ones((_EMBEDS, _WIDTH), dtype=torch.float16)
                for ident in idents
            }
        )
        for ident in idents:
            worker.save_caches(
                {ident: torch.ones((_EMBEDS, _WIDTH), dtype=torch.float16)},
                ident,
            )
        worker.build_connector_worker_meta()
        _wait_for_attack(peers)
    finally:
        scheduler.shutdown()
        if worker is not None:
            worker.shutdown()


def _assert_unused(peers: list[_Peer]) -> None:
    transcript = [(peer.base, peer.shard, list(peer.transcript)) for peer in peers]
    assert not any(peer.transcript for peer in peers) and not _TRANSFER_WRITES, (
        "a client-supplied consumer address was dialed or written: "
        f"zmq={transcript} writes={_TRANSFER_WRITES}"
    )


@pytest.fixture(autouse=True)
def _clear_writes():
    _TRANSFER_WRITES.clear()
    yield
    _TRANSFER_WRITES.clear()


pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("api", ["chat", "completion", "responses", "anthropic"])
@pytest.mark.parametrize("trigger", ["schedule", "alloc"])
@pytest.mark.parametrize("count", [1, 2])
def test_client_consumer_zmq_is_not_dialed(api: str, trigger: str, count: int):
    """A request field must not become a ZMQ peer or a tensor-write target.

    ``count == 2`` exercises the batched reservation RPC. Both the scheduler
    admission path and the post-allocation path prepare the same push.
    """
    peer = _Peer()
    try:
        idents = [f"mm-{index}" for index in range(count)]
        params = {
            "consumer_zmq": peer.base,
            "ec_items": [
                {"mm_hash": ident, "transfer_id": f"xfer-{index}"}
                for index, ident in enumerate(idents)
            ],
        }
        sampling = _sampling_params(api, params)
        request = _engine_request(sampling, count, idents)
        assert request.ec_transfer_params is not None
        assert request.ec_transfer_params.get("consumer_zmq") == peer.base
        _run_producer(request, trigger=trigger, idents=idents, peers=[peer])
        _assert_unused([peer])
    finally:
        peer.close()


def test_push_without_transfer_id_does_not_dial_client_address():
    """A missing ec_items id still must not dial consumer_zmq."""
    peer = _Peer()
    try:
        params = {"consumer_zmq": peer.base}
        sampling = _sampling_params("chat", params)
        request = _engine_request(sampling, 1, ["mm-0"])
        _run_producer(request, trigger="schedule", idents=["mm-0"], peers=[peer])
        _assert_unused([peer])
    finally:
        peer.close()


def test_reread_of_consumer_zmq_is_not_dialed():
    """Looking the address up again must not dial a later value.

    The first read and every later read are both caller-controlled. Neither
    may be connected, and swapping the field after the scheduler has already
    looked at the request must not redirect the dial.
    """
    first = _Peer()
    second = _Peer()
    try:
        items = [{"mm_hash": "mm-0", "transfer_id": "xfer-0"}]
        switching = _SwitchingAddress(first.base, second.base, items)
        sampling = SamplingParams(
            max_tokens=4,
            extra_args={"ec_transfer_params": switching},
        )
        request = _engine_request(sampling, 1, ["mm-0"])
        request.ec_transfer_params = switching
        config = _producer_config()
        scheduler = ECMooncakeConnector(config, ECConnectorRole.SCHEDULER)
        worker = ECMooncakeConnector(config, ECConnectorRole.WORKER)
        try:
            scheduler.ensure_cache_available(request, 0)
            switching["consumer_zmq"] = second.base
            metadata = scheduler.build_connector_meta(SchedulerOutput.make_empty())
            worker.bind_connector_metadata(metadata)
            tensor = torch.ones((_EMBEDS, _WIDTH), dtype=torch.float16)
            worker.start_save_caches(encoder_cache={"mm-0": tensor})
            worker.save_caches({"mm-0": tensor}, "mm-0")
            worker.build_connector_worker_meta()
            _wait_for_attack([first, second])
        finally:
            scheduler.shutdown()
            worker.shutdown()
        _assert_unused([first, second])
    finally:
        first.close()
        second.close()
