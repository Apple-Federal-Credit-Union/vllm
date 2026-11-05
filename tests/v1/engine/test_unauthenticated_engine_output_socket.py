# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-node engine output sockets must ignore unauthenticated peers.

Data-parallel serving binds the frontend output socket as a TCP ZMQ PULL on
the data-parallel master address. These tests connect to that socket the way
an arbitrary peer can and require that:

* a frame that is not a valid engine output does not stop the output reader
  or the API server
* a forged but valid engine output is not delivered to the API
* a frame sent the way an engine sends outputs is still delivered after those
  unauthenticated frames

The tests fail while any TCP peer can push a frame that the frontend decodes
as an engine result.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import msgspec.msgpack
import pytest
import zmq

import vllm.v1.engine.core_client as core_client
from vllm.config import ParallelConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.entrypoints.launchers.launcher import terminate_if_errored
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
    UtilityOutput,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient, SyncMPClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.utils import (
    CoreEngineLaunch,
    bind_engine_zmq_listeners,
)
from vllm.v1.serial_utils import MsgpackEncoder

_MASTER_IP = "127.0.0.1"
_LEGITIMATE_REQUEST = "engine-finished-request"
_FORGED_REQUEST = "forged-finished-request"
_DELIVERY_WINDOW_S = 1.5


def _multi_node_config() -> VllmConfig:
    """Internal data-parallel config whose engines are not all local."""
    return VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=2,
            data_parallel_size_local=1,
            data_parallel_master_ip=_MASTER_IP,
        ),
        device_config=DeviceConfig(device="cpu"),
    )


def _finished_output(request_id: str, token_id: int) -> EngineCoreOutputs:
    return EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id=request_id,
                new_token_ids=[token_id],
                finish_reason=FinishReason.STOP,
            )
        ],
        finished_requests={request_id},
    )


def _encode(outputs: EngineCoreOutputs) -> list[bytes]:
    return [bytes(frame) for frame in MsgpackEncoder().encode(outputs)]


def _push_raw(address: str, payload: bytes | list[bytes]) -> None:
    """Push from a peer that only has a plain ZMQ socket, not an engine."""
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 2000)
    socket.connect(address)
    socket.poll(200, zmq.POLLOUT)
    if isinstance(payload, (bytes, bytearray)):
        socket.send(payload)
    else:
        socket.send_multipart(payload)
    socket.close()
    context.term()


def _push_engine_output(address: str, outputs: EngineCoreOutputs) -> None:
    """Send the way EngineCoreProc.process_output_sockets sends one output."""
    context = zmq.Context()
    socket = make_zmq_socket(context, address, zmq.PUSH, bind=False, linger=4000)
    socket.send_multipart(MsgpackEncoder().encode(outputs))
    socket.close()
    context.term()


def _send_engine_ready(address: str, engine_count: int, hold: threading.Event) -> None:
    """Complete the frontend ready handshake for ``engine_count`` engines."""
    context = zmq.Context()
    sockets = []
    for rank in range(engine_count):
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, rank.to_bytes(2, "little"))
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(address)
        sockets.append(socket)
    # The frontend binds, then waits. One empty payload is a valid ready
    # response; repeating it makes the handshake reject the identity.
    time.sleep(0.3)
    for socket in sockets:
        socket.send(b"")
    hold.wait(10)
    for socket in sockets:
        socket.close(linger=0)
    context.term()


def _open_async_client() -> tuple[DPLBAsyncMPClient, threading.Event]:
    """API client for multi-node internal DP, using the supervisor listeners."""
    config = _multi_node_config()
    assert not config.parallel_config.local_engines_only
    listeners = bind_engine_zmq_listeners(config)
    output_address = listeners.outputs[0].address
    input_address = listeners.inputs[0].address
    assert output_address.startswith(f"tcp://{_MASTER_IP}:")
    hold = threading.Event()
    threading.Thread(
        target=_send_engine_ready,
        args=(input_address, 2, hold),
        daemon=True,
    ).start()
    client = DPLBAsyncMPClient(
        config,
        object,
        False,
        client_addresses={
            "input_address": input_address,
            "output_address": output_address,
            "input_listener": listeners.inputs[0].socket,
            "output_listener": listeners.outputs[0].socket,
        },
    )
    assert client.output_address.startswith(f"tcp://{_MASTER_IP}:")
    return client, hold


class _OutputWaiter:
    def __init__(self, client: SyncMPClient) -> None:
        self._client = client
        self.output: EngineCoreOutputs | None = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.output = self._client.get_output()
        except BaseException as exc:
            self.error = exc

    def finished(self) -> bool:
        return not self._thread.is_alive()

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)


@contextmanager
def _engines_not_started(vllm_config, executor_class, log_stats, addresses):
    """Skip model-process launch. The output socket thread still runs."""
    yield CoreEngineLaunch(
        engine_manager=None,
        coordinator=None,
        addresses=addresses,
        tensor_queue=None,
    )


def _open_sync_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SyncMPClient, threading.Event]:
    endpoints: dict[int, str] = {}
    real_make_zmq_socket = core_client.make_zmq_socket

    def spy_make_zmq_socket(ctx, path, socket_type, *args, **kwargs):
        socket = real_make_zmq_socket(ctx, path, socket_type, *args, **kwargs)
        endpoints[socket_type] = socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        return socket

    monkeypatch.setattr(core_client, "launch_core_engines", _engines_not_started)
    monkeypatch.setattr(core_client, "make_zmq_socket", spy_make_zmq_socket)
    hold = threading.Event()

    def handshake_when_bound() -> None:
        deadline = time.monotonic() + 10
        while zmq.ROUTER not in endpoints and time.monotonic() < deadline:
            time.sleep(0.02)
        if zmq.ROUTER in endpoints:
            _send_engine_ready(endpoints[zmq.ROUTER], 2, hold)

    threading.Thread(target=handshake_when_bound, daemon=True).start()
    client = SyncMPClient(_multi_node_config(), object, False)
    assert client.output_address.startswith(f"tcp://{_MASTER_IP}:")
    return client, hold


def _fatal_payloads() -> list[tuple[str, bytes | list[bytes]]]:
    return [
        ("malformed", b"\xff\x00not-an-engine-output"),
        ("engine_dead_sentinel", EngineCoreProc.ENGINE_CORE_DEAD),
        (
            "unknown_utility",
            _encode(
                EngineCoreOutputs(
                    utility_output=UtilityOutput(
                        call_id=999,
                        failure_message="unauthenticated",
                    )
                )
            ),
        ),
    ]


async def _async_output_survives(
    client: DPLBAsyncMPClient, payload: bytes | list[bytes]
) -> None:
    received: list[EngineCoreOutputs | BaseException] = []

    async def read_output() -> None:
        try:
            received.append(await client.get_output_async())
        except BaseException as exc:
            received.append(exc)

    reader = asyncio.create_task(read_output())
    await asyncio.sleep(0.2)
    output_task = client.resources.output_queue_task
    assert output_task is not None
    _push_raw(client.output_address, payload)
    deadline = time.monotonic() + _DELIVERY_WINDOW_S
    while time.monotonic() < deadline:
        if output_task.done() or received or client.resources.engine_dead:
            break
        await asyncio.sleep(0.05)
    assert not client.resources.engine_dead
    assert not output_task.done()
    assert not received
    _push_engine_output(client.output_address, _finished_output(_LEGITIMATE_REQUEST, 4))
    await asyncio.wait_for(reader, timeout=_DELIVERY_WINDOW_S)
    assert len(received) == 1
    result = received[0]
    assert isinstance(result, EngineCoreOutputs)
    assert result.outputs[0].request_id == _LEGITIMATE_REQUEST
    assert result.outputs[0].new_token_ids == [4]
    assert result.outputs[0].finish_reason == FinishReason.STOP
    assert not output_task.done()
    assert not client.resources.engine_dead


@pytest.mark.parametrize(
    ("name", "payload"),
    _fatal_payloads(),
    ids=[item[0] for item in _fatal_payloads()],
)
def test_async_api_output_socket_ignores_unauthenticated_frame(name, payload):
    """A peer that is not an engine cannot stop the async output reader."""
    del name
    client, hold = _open_async_client()
    try:
        asyncio.run(_async_output_survives(client, payload))
    finally:
        hold.set()
        client.shutdown()


def test_async_api_output_socket_drops_forged_finished_output():
    """A forged finished output must not enter the result stream."""
    client, hold = _open_async_client()

    async def scenario() -> None:
        received: list[EngineCoreOutputs | BaseException] = []

        async def read_output() -> None:
            try:
                received.append(await client.get_output_async())
            except BaseException as exc:
                received.append(exc)

        reader = asyncio.create_task(read_output())
        await asyncio.sleep(0.2)
        _push_raw(client.output_address, _encode(_finished_output(_FORGED_REQUEST, 9)))
        deadline = time.monotonic() + _DELIVERY_WINDOW_S
        while time.monotonic() < deadline and not received:
            await asyncio.sleep(0.05)
        assert not received, received
        _push_engine_output(
            client.output_address, _finished_output(_LEGITIMATE_REQUEST, 4)
        )
        await asyncio.wait_for(reader, timeout=_DELIVERY_WINDOW_S)
        result = received[0]
        assert isinstance(result, EngineCoreOutputs)
        assert result.outputs[0].request_id == _LEGITIMATE_REQUEST
        assert _FORGED_REQUEST not in (result.finished_requests or set())

    try:
        asyncio.run(scenario())
    finally:
        hold.set()
        client.shutdown()


def test_malformed_output_frame_does_not_stop_api_server():
    """The API output handler must keep serving and must not set should_exit."""
    client, hold = _open_async_client()
    engine = object.__new__(AsyncLLM)
    engine.output_handler = None
    engine.engine_core = client
    engine.output_processor = OutputProcessor(None, log_stats=False)
    engine.log_stats = False
    engine.logger_manager = None
    engine.renderer = SimpleNamespace(
        mm_processor_cache=None, stat_mm_cache=lambda: None
    )
    engine.shutdown = lambda *args, **kwargs: None
    server = SimpleNamespace(should_exit=False)

    async def scenario() -> None:
        engine._run_output_handler()
        await asyncio.sleep(0.2)
        _push_raw(client.output_address, b"\xff\x00not-an-engine-output")
        deadline = time.monotonic() + _DELIVERY_WINDOW_S
        while time.monotonic() < deadline:
            terminate_if_errored(server, engine)
            if server.should_exit or not engine.is_running:
                break
            await asyncio.sleep(0.05)
        terminate_if_errored(server, engine)
        assert engine.is_running
        assert not engine.errored
        assert not server.should_exit
        assert not client.resources.engine_dead
        _push_engine_output(
            client.output_address, _finished_output(_LEGITIMATE_REQUEST, 4)
        )
        await asyncio.sleep(0.5)
        terminate_if_errored(server, engine)
        assert engine.is_running
        assert not server.should_exit
        handler = engine.output_handler
        assert handler is not None
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler

    try:
        asyncio.run(scenario())
    finally:
        hold.set()
        client.shutdown()


def _sync_output_survives(client: SyncMPClient, payload: bytes | list[bytes]) -> None:
    waiter = _OutputWaiter(client)
    time.sleep(0.2)
    _push_raw(client.output_address, payload)
    deadline = time.monotonic() + _DELIVERY_WINDOW_S
    while time.monotonic() < deadline:
        if waiter.finished() or client.resources.engine_dead:
            break
        time.sleep(0.05)
    assert waiter.error is None
    assert waiter.output is None
    assert client.output_queue_thread.is_alive()
    assert not client.resources.engine_dead
    _push_engine_output(client.output_address, _finished_output(_LEGITIMATE_REQUEST, 4))
    waiter.join(_DELIVERY_WINDOW_S)
    assert waiter.error is None
    assert waiter.output is not None
    assert waiter.output.outputs[0].request_id == _LEGITIMATE_REQUEST
    assert client.output_queue_thread.is_alive()
    assert not client.resources.engine_dead


@pytest.mark.parametrize(
    ("name", "payload"),
    _fatal_payloads(),
    ids=[item[0] for item in _fatal_payloads()],
)
def test_sync_output_socket_ignores_unauthenticated_frame(
    monkeypatch: pytest.MonkeyPatch, name, payload
):
    """Offline SyncMPClient uses the same unauthenticated PULL reader."""
    del name
    client, hold = _open_sync_client(monkeypatch)
    try:
        _sync_output_survives(client, payload)
    finally:
        hold.set()
        client.shutdown()


def test_sync_output_socket_drops_forged_finished_output(
    monkeypatch: pytest.MonkeyPatch,
):
    client, hold = _open_sync_client(monkeypatch)
    try:
        waiter = _OutputWaiter(client)
        time.sleep(0.2)
        _push_raw(client.output_address, _encode(_finished_output(_FORGED_REQUEST, 9)))
        deadline = time.monotonic() + _DELIVERY_WINDOW_S
        while time.monotonic() < deadline and not waiter.finished():
            time.sleep(0.05)
        assert waiter.output is None
        assert waiter.error is None
        _push_engine_output(
            client.output_address, _finished_output(_LEGITIMATE_REQUEST, 4)
        )
        waiter.join(_DELIVERY_WINDOW_S)
        assert waiter.error is None
        assert waiter.output is not None
        assert waiter.output.outputs[0].request_id == _LEGITIMATE_REQUEST
    finally:
        hold.set()
        client.shutdown()


def _start_coordinator() -> tuple[DPCoordinator, str, zmq.Socket, zmq.Context]:
    config = _multi_node_config().parallel_config
    coordinator = DPCoordinator(config, enable_wave_coordination=True)
    front = coordinator.get_stats_publish_address()
    back_in, back_out = coordinator.get_engine_socket_addresses()
    assert back_out.startswith(f"tcp://{_MASTER_IP}:")
    context = zmq.Context()
    subscribers = []
    try:
        for _ in range(config.data_parallel_size):
            socket = make_zmq_socket(context, back_in, zmq.XSUB, bind=False, linger=0)
            socket.send(b"\x01")
            subscribers.append(socket)
        for socket in subscribers:
            assert socket.recv() == b"READY"
        front_sub = context.socket(zmq.SUB)
        front_sub.setsockopt(zmq.LINGER, 0)
        front_sub.setsockopt(zmq.RCVTIMEO, 200)
        front_sub.connect(front)
        front_sub.setsockopt(zmq.SUBSCRIBE, b"")
        time.sleep(0.3)
        while True:
            try:
                front_sub.recv()
            except zmq.Again:
                break
        front_sub.setsockopt(zmq.RCVTIMEO, int(_DELIVERY_WINDOW_S * 1000))
        return coordinator, back_out, front_sub, context
    except BaseException:
        coordinator.shutdown(timeout=2)
        context.term()
        raise


def _recv_wave(front_sub: zmq.Socket) -> tuple | None:
    try:
        message = front_sub.recv()
    except zmq.Again:
        return None
    decoded = msgspec.msgpack.decode(message)
    assert isinstance(decoded, (list, tuple)) and len(decoded) == 3
    return decoded


def test_dp_coordinator_ignores_unauthenticated_output_frames():
    """The coordinator PULL on the master IP shares the output-socket bug."""
    coordinator, back_out, front_sub, context = _start_coordinator()
    try:
        _push_raw(back_out, b"\xff\x00not-an-engine-output")
        time.sleep(0.4)
        assert coordinator.proc.is_alive()
        _push_raw(back_out, _encode(EngineCoreOutputs(engine_index=0, wave_complete=7)))
        forged = _recv_wave(front_sub)
        assert forged is None or forged[1] == 0
        _push_engine_output(
            back_out, EngineCoreOutputs(engine_index=0, wave_complete=1)
        )
        published = _recv_wave(front_sub)
        assert published is not None
        assert published[0] is None
        assert published[1] == 2
        assert published[2] is False
        assert coordinator.proc.is_alive()
    finally:
        coordinator.shutdown(timeout=2)
        front_sub.close(linger=0)
        context.term()
