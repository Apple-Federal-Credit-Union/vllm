# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unauthorized subscribers must not read the multi-node worker broadcast.

These tests fail while ``MessageQueue`` publishes pickled collective RPCs on
an unauthenticated TCP XPUB. They pass once an extra subscriber, attached
after the expected remote readers have finished the handshake, cannot recover
method names, arguments, prompt token ids, or prompt embeddings.
"""

from __future__ import annotations

import pickle
import queue
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import zmq

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.utils.network_utils import get_ip
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc

PROMPT_TOKEN_IDS = [101, 202, 303, 404]
PROMPT_EMBED_VALUES = (-2.25, 4.5)
REQ_ID = "prompt-req-9f3a"
CACHED_REQ_ID = "cached-req-9f3a"


def _scheduler_output(req_id: str, token_ids: list[int]) -> SchedulerOutput:
    embeds = torch.tensor(
        [[PROMPT_EMBED_VALUES[0], PROMPT_EMBED_VALUES[1]]],
        dtype=torch.float32,
    )
    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=list(token_ids),
                mm_features=[],
                sampling_params=None,
                pooling_params=None,
                block_ids=([1, 2],),
                num_computed_tokens=0,
                lora_request=None,
                prompt_embeds=embeds,
            )
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[CACHED_REQ_ID],
            resumed_req_ids=set(),
            new_token_ids=[[7, 8, 9]],
            all_token_ids={CACHED_REQ_ID: [11, 22, 33]},
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_output_tokens=[1],
        ),
        num_scheduled_tokens={req_id: len(token_ids)},
        total_num_scheduled_tokens=len(token_ids),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _loads_frames(frames: list[bytes]) -> object | None:
    try:
        return pickle.loads(frames[0], buffers=frames[1:])
    except Exception:
        return None


def _prompt_embed_leak(obj: object) -> str | None:
    """Return a description when ``obj`` is an RPC carrying the test embeds."""
    if not isinstance(obj, tuple) or len(obj) < 2:
        return None
    args = obj[1]
    if not isinstance(args, tuple) or not args:
        return None
    sched = args[0]
    if not isinstance(sched, SchedulerOutput) or not sched.scheduled_new_reqs:
        return None
    embeds = sched.scheduled_new_reqs[0].prompt_embeds
    if embeds is None:
        return None
    expected = torch.tensor(PROMPT_EMBED_VALUES, dtype=torch.float32)
    if torch.allclose(embeds.detach().reshape(-1).cpu(), expected):
        return f"prompt embeds {embeds.detach().reshape(-1).cpu().tolist()}"
    return None


def _visible_markers(frames_list: list[list[bytes]], markers: tuple[bytes, ...]) -> str:
    """Return why an outsider could read ``markers``, or an empty string."""
    reasons = []
    for frames in frames_list:
        blob = b"".join(frames)
        for marker in markers:
            if marker in blob:
                reasons.append(f"raw frame contains {marker.decode()}")
        loaded = _loads_frames(frames)
        if loaded is None:
            continue
        rendered = repr(loaded)
        for marker in markers:
            text = marker.decode()
            if text in rendered:
                reasons.append(f"unpickled object contains {text}")
        embed_leak = _prompt_embed_leak(loaded)
        if embed_leak is not None:
            reasons.append(embed_leak)
    return "; ".join(reasons)


@dataclass
class _BroadcastSession:
    """Leader queue plus one subscriber that was not part of the handshake."""

    writer: MessageQueue
    attacker: zmq.Socket
    subscribed: bool
    inbox: queue.Queue
    errors: list[BaseException]
    n_messages: int

    def take(self) -> object:
        try:
            return self.inbox.get(timeout=15)
        except queue.Empty as exc:
            if self.errors:
                raise self.errors[0] from exc
            raise

    def drain_attacker(self) -> list[list[bytes]]:
        timeout_ms = 2000 if self.subscribed else 200
        frames_list: list[list[bytes]] = []
        while self.attacker.poll(timeout_ms if not frames_list else 100):
            frames_list.append(self.attacker.recv_multipart())
        return frames_list


class _LeaderBroadcast:
    def __init__(
        self,
        connect_ip: str | None,
        n_local: int,
        n_remote: int,
        n_messages: int,
    ) -> None:
        self.connect_ip = connect_ip
        self.n_local = n_local
        self.n_remote = n_remote
        self.n_messages = n_messages
        self.writer: MessageQueue | None = None
        self.attacker: zmq.Socket | None = None
        self.context: zmq.Context | None = None
        self.threads: list[threading.Thread] = []

    def __enter__(self) -> _BroadcastSession:
        writer = MessageQueue(
            n_reader=self.n_local + self.n_remote,
            n_local_reader=self.n_local,
            connect_ip=self.connect_ip,
        )
        self.writer = writer
        handle = writer.export_handle()
        assert handle.remote_subscribe_addr is not None
        inbox: queue.Queue = queue.Queue()
        errors: list[BaseException] = []

        def run_reader(rank: int, collect: bool) -> None:
            try:
                reader = MessageQueue.create_from_handle(handle, rank)
                reader.wait_until_ready()
                if collect:
                    for _ in range(self.n_messages):
                        inbox.put(reader.dequeue(timeout=15))
            except Exception as exc:
                errors.append(exc)

        for rank in range(self.n_local):
            self.threads.append(threading.Thread(target=run_reader, args=(rank, False)))
        for offset in range(self.n_remote):
            collect = offset == 0
            rank = self.n_local + offset
            self.threads.append(
                threading.Thread(target=run_reader, args=(rank, collect))
            )
        for thread in self.threads:
            thread.start()
        writer.wait_until_ready()

        # The handshake has already consumed the expected readers' subscriptions.
        assert writer.remote_socket.poll(0) == 0
        context = zmq.Context()
        attacker = context.socket(zmq.SUB)
        attacker.setsockopt(zmq.LINGER, 0)
        attacker.setsockopt_string(zmq.SUBSCRIBE, "")
        if handle.remote_addr_ipv6:
            attacker.setsockopt(zmq.IPV6, 1)
        attacker.connect(handle.remote_subscribe_addr)
        self.context = context
        self.attacker = attacker
        subscribed = bool(writer.remote_socket.poll(5000))
        return _BroadcastSession(
            writer=writer,
            attacker=attacker,
            subscribed=subscribed,
            inbox=inbox,
            errors=errors,
            n_messages=self.n_messages,
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        for thread in self.threads:
            thread.join(timeout=20)
        if self.attacker is not None:
            self.attacker.close(linger=0)
        if self.context is not None:
            self.context.term()
        if self.writer is not None:
            self.writer.shutdown()


def _assert_hidden(session: _BroadcastSession, markers: tuple[bytes, ...]) -> None:
    leaked = _visible_markers(session.drain_attacker(), markers)
    assert not leaked, (
        "subscriber that joined after the remote-reader handshake read the "
        f"worker broadcast ({leaked})"
    )


def _executor(writer: MessageQueue) -> MultiprocExecutor:
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.rpc_broadcast_mq = writer
    executor.is_failed = False
    executor.futures_queue = deque()
    executor.output_rank = 0
    executor.kv_output_aggregator = None
    executor.ec_output_aggregator = None
    # collective_rpc indexes this before waiting for a worker reply.
    executor.response_mqs = [MessageQueue(n_reader=1, n_local_reader=1)]
    return executor


def _response_pair() -> tuple[MessageQueue, MessageQueue]:
    writer = MessageQueue(n_reader=1, n_local_reader=1)
    reader = MessageQueue.create_from_handle(writer.export_handle(), 0)
    thread = threading.Thread(target=writer.wait_until_ready)
    thread.start()
    reader.wait_until_ready()
    thread.join()
    return writer, reader


def test_extra_subscriber_cannot_read_execute_model_scheduler_output():
    """Decode steps must not publish scheduler outputs to a late TCP subscriber.

    The leader queue matches a data-parallel leader whose world size is larger
    than the local world size: one local reader and one remote reader. The
    outsider connects only after ``wait_until_ready`` has accepted that remote
    reader.
    """
    token_ids = (PROMPT_TOKEN_IDS, [8, 16, 32])
    markers = tuple(
        marker.encode()
        for marker in (
            "execute_model",
            REQ_ID,
            f"{REQ_ID}-step-2",
            CACHED_REQ_ID,
            "101, 202, 303, 404",
            "-2.25",
        )
    )
    with _LeaderBroadcast(
        connect_ip=None, n_local=1, n_remote=1, n_messages=2
    ) as session:
        address = session.writer.export_handle().remote_subscribe_addr
        assert address is not None
        assert get_ip() in address
        executor = _executor(session.writer)
        for step, ids in enumerate(token_ids, start=1):
            req_id = REQ_ID if step == 1 else f"{REQ_ID}-step-2"
            executor.execute_model(_scheduler_output(req_id, list(ids)), non_block=True)
        first = session.take()
        second = session.take()
        assert first[0] == "execute_model"
        assert second[0] == "execute_model"
        assert first[1][0].scheduled_new_reqs[0].prompt_token_ids == list(token_ids[0])
        assert second[1][0].scheduled_new_reqs[0].prompt_token_ids == list(token_ids[1])
        embeds = first[1][0].scheduled_new_reqs[0].prompt_embeds
        assert embeds is not None
        torch.testing.assert_close(
            embeds.reshape(-1),
            torch.tensor(PROMPT_EMBED_VALUES, dtype=torch.float32),
        )
        assert not session.errors
        _assert_hidden(session, markers)


def test_extra_subscriber_cannot_read_ipv6_execute_model():
    """The IPv6 remote bind is the same unauthenticated XPUB."""
    with _LeaderBroadcast(
        connect_ip="::1", n_local=1, n_remote=1, n_messages=1
    ) as session:
        handle = session.writer.export_handle()
        assert handle.remote_addr_ipv6
        assert handle.remote_subscribe_addr is not None
        assert handle.remote_subscribe_addr.startswith("tcp://[::1]:")
        executor = _executor(session.writer)
        executor.execute_model(
            _scheduler_output(REQ_ID, PROMPT_TOKEN_IDS), non_block=True
        )
        received = session.take()
        assert received[0] == "execute_model"
        got_ids = received[1][0].scheduled_new_reqs[0].prompt_token_ids
        assert got_ids == PROMPT_TOKEN_IDS
        assert not session.errors
        _assert_hidden(
            session,
            (b"execute_model", REQ_ID.encode(), b"101, 202, 303, 404"),
        )


def test_remote_only_broadcast_object_hides_payload_from_extra_subscriber():
    """A queue whose only reader is remote still must not feed a second subscriber.

    ``MessageQueue.broadcast_object`` is what ``GroupCoordinator`` uses when a
    message-queue broadcaster is enabled, and a single remote reader is the
    shape used for a cross-node response queue.
    """
    marker = "broadcast-object-secret"
    with _LeaderBroadcast(
        connect_ip=get_ip(), n_local=0, n_remote=1, n_messages=1
    ) as session:
        payload = {
            "req_id": marker,
            "prompt_token_ids": PROMPT_TOKEN_IDS,
        }
        assert session.writer.broadcast_object(payload) == payload
        assert session.take() == payload
        assert not session.errors
        _assert_hidden(
            session,
            (marker.encode(), b"101, 202, 303, 404"),
        )


def _call_sample_tokens(executor: MultiprocExecutor, marker: str) -> None:
    grammar = GrammarOutput(
        structured_output_request_ids=[marker],
        grammar_bitmask=np.asarray([[9, 8, 7, 6]], dtype=np.int32),
    )
    executor.sample_tokens(grammar, non_block=True)


def _call_execute_dummy_batch(executor: MultiprocExecutor, marker: str) -> None:
    del marker
    executor.execute_dummy_batch()


def _call_take_draft_token_ids(executor: MultiprocExecutor, marker: str) -> None:
    del marker
    executor.take_draft_token_ids()


def _call_check_health(executor: MultiprocExecutor, marker: str) -> None:
    del marker
    executor.check_health()


def _call_collective_rpc(executor: MultiprocExecutor, marker: str) -> None:
    executor.collective_rpc(
        "apply_scheduler_metadata",
        args=(marker,),
        kwargs={"note": marker},
        non_block=True,
    )


def _call_pickled_callable(executor: MultiprocExecutor, marker: str) -> None:
    def _remote(worker: object, value: str) -> str:
        del worker
        return value

    executor.collective_rpc(_remote, args=(marker,), non_block=True)


RpcCall = Callable[[MultiprocExecutor, str], None]


@pytest.mark.parametrize(
    ("caller", "needs_response", "markers"),
    [
        (_call_sample_tokens, False, (b"sample_tokens", b"grammar-secret")),
        (_call_execute_dummy_batch, True, (b"execute_dummy_batch",)),
        (_call_take_draft_token_ids, True, (b"take_draft_token_ids",)),
        (_call_check_health, True, (b"check_health",)),
        (
            _call_collective_rpc,
            False,
            (b"apply_scheduler_metadata", b"rpc-arg-secret"),
        ),
        (_call_pickled_callable, False, (b"callable-arg-secret",)),
    ],
)
def test_extra_subscriber_cannot_read_collective_rpc(
    caller: RpcCall,
    needs_response: bool,
    markers: tuple[bytes, ...],
):
    """Every collective RPC shares the remote broadcast socket."""
    marker = markers[-1].decode()
    with _LeaderBroadcast(
        connect_ip=None, n_local=1, n_remote=1, n_messages=1
    ) as session:
        executor = _executor(session.writer)
        if needs_response:
            response_writer, response_reader = _response_pair()
            executor.response_mqs = [response_reader]
            response_writer.enqueue((WorkerProc.ResponseStatus.SUCCESS, None))
        else:
            executor.response_mqs = [MessageQueue(n_reader=1, n_local_reader=1)]
        caller(executor, marker)
        received = session.take()
        rendered = repr(received)
        assert marker in rendered
        if isinstance(received[0], str):
            assert received[0] == markers[0].decode()
        else:
            assert isinstance(received[0], bytes)
        assert not session.errors
        _assert_hidden(session, markers)
