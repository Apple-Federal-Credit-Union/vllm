# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security contract for the Mooncake bootstrap registry.

The prefiller listens with MooncakeBootstrapServer and workers register
their ZMQ addresses there. Decoders treat GET /query as the list of
endpoints to pull KV from.

These tests are expected to fail until that registry is closed:

* The listening socket must be the management host returned by
  get_mooncake_bootstrap_addr (127.0.0.1 when that host is loopback),
  not 0.0.0.0.
* POST /register and GET /query must reject any request that does not
  send Authorization: Bearer <VLLM_MOONCAKE_BOOTSTRAP_TOKEN>. A missing
  variable fails closed. A rejected register must not record a worker.
  The same token repeats an identical registration and still rejects a
  different address or engine id for an occupied rank.
* register_worker_with_bootstrap and the decoder query inside
  start_load_kv must send that credential. The decoder may dial only
  the worker address that an authenticated registration stored.

The example proxy get_prefiller_info also GETs {bootstrap}/query and is
not invoked here. It uses the same query operation these tests cover.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from dataclasses import dataclass

import httpx
import msgspec
import psutil
import pytest
import torch
import zmq

from tests.v1.kv_connector.unit.test_mooncake_connector import (
    FakeMooncakeWrapper,
    _make_test_kv_cache_config,
)
from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_connector as mc,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    MooncakeConnectorWorker,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    get_mooncake_bootstrap_addr,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_test

# Obvious placeholder. Not a deployed credential.
_TOKEN_ENV = "VLLM_MOONCAKE_BOOTSTRAP_TOKEN"
_PLACEHOLDER_TOKEN = "PLACEHOLDER_BOOTSTRAP_TOKEN"
_ENGINE_ID = "prefiller-engine"
_UNAUTHORIZED = {401, 403}


class _PpGroup:
    rank_in_group = 0


@dataclass
class _Peer:
    addr: str
    received: list[bytes]
    stop: threading.Event
    thread: threading.Thread
    socket: zmq.Socket


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _non_loopback_ipv4() -> list[str]:
    found: list[str] = []
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            ip = address.address
            if ip.startswith("127."):
                continue
            found.append(ip)
    return found


def _can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _assert_rejected(response: httpx.Response, secret_bits: tuple[str, ...]) -> None:
    assert response.status_code in _UNAUTHORIZED, response.status_code
    for bit in secret_bits:
        assert bit not in response.text


@contextlib.contextmanager
def _prefiller(monkeypatch, *, token: str | None, local_engines_only: bool):
    """Start the real prefiller bootstrap path with external pieces stubbed.

    TransferEngine, the accelerator, and process-group ranks are not part of
    the registry request path. This environment has none of them, so they are
    replaced only far enough for MooncakeConnectorWorker.__init__ to reach
    MooncakeBootstrapServer.
    """
    port = _free_tcp_port()
    monkeypatch.setenv("VLLM_MOONCAKE_BOOTSTRAP_PORT", str(port))
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("VLLM_MOONCAKE_CONNECTOR_TIMEOUT", "2")
    if token is None:
        monkeypatch.delenv(_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(_TOKEN_ENV, token)

    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"num_workers": 1},
    )
    if local_engines_only:
        vllm_config.parallel_config.data_parallel_external_lb = True
        vllm_config.parallel_config.data_parallel_rank_local = 0

    def _skip_block_sync(self) -> None:
        return None

    with (
        set_current_vllm_config(vllm_config),
        monkeypatch.context() as patch_ctx,
    ):
        patch_ctx.setattr(mc, "TransferEngine", FakeMooncakeWrapper)
        patch_ctx.setattr(torch.accelerator, "current_device_index", lambda: 0)
        patch_ctx.setattr(current_platform, "set_device", lambda device: None)
        patch_ctx.setattr(mc, "get_tensor_model_parallel_rank", lambda: 0)
        patch_ctx.setattr(mc, "get_tensor_model_parallel_world_size", lambda: 1)
        patch_ctx.setattr(mc, "get_pp_group", lambda: _PpGroup())
        patch_ctx.setattr(mc, "get_current_attn_backends", lambda *args, **kwargs: [])
        patch_ctx.setattr(
            MooncakeConnectorWorker,
            "_sync_block_size_with_kernel",
            _skip_block_sync,
        )
        producer = MooncakeConnectorWorker(
            vllm_config, _ENGINE_ID, _make_test_kv_cache_config()
        )
        try:
            yield producer, port
        finally:
            producer.shutdown()
            # __del__ calls shutdown again after these patches are gone.
            producer.shutdown = lambda: None


def _serve_peer() -> _Peer:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    received: list[bytes] = []
    stop = threading.Event()
    encoder = msgspec.msgpack.Encoder()

    def run() -> None:
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not stop.is_set():
            if sock not in dict(poller.poll(100)):
                continue
            parts = sock.recv_multipart()
            received.append(parts[-1])
            reply = encoder.encode(
                MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    ok_reqs=["d-req"],
                )
            )
            sock.send_multipart((parts[0], reply))

    thread = threading.Thread(target=run, name=f"bootstrap-peer-{port}", daemon=True)
    thread.start()
    return _Peer(
        addr=f"tcp://127.0.0.1:{port}",
        received=received,
        stop=stop,
        thread=thread,
        socket=sock,
    )


def _stop_peer(peer: _Peer) -> None:
    peer.stop.set()
    peer.thread.join(timeout=2)
    peer.socket.close(linger=0)


@pytest.mark.parametrize("local_engines_only", [False, True])
def test_prefiller_bootstrap_binds_management_host(monkeypatch, local_engines_only):
    """Loopback management hosts must not listen on every interface."""
    with _prefiller(
        monkeypatch, token=_PLACEHOLDER_TOKEN, local_engines_only=local_engines_only
    ) as (producer, port):
        intended_host, intended_port = get_mooncake_bootstrap_addr(producer.vllm_config)
        assert intended_host == "127.0.0.1"
        assert intended_port == port
        bound = producer.bootstrap_server._socket.getsockname()
        assert _can_connect("127.0.0.1", port)
        exposed = [ip for ip in _non_loopback_ipv4() if _can_connect(ip, port)]
        assert bound[0] == intended_host and not exposed, (
            bound[0],
            intended_host,
            exposed,
        )
        assert bound[1] == port


@pytest.mark.asyncio
async def test_register_and_query_reject_missing_or_wrong_token(monkeypatch):
    """Unauthenticated clients must not read or extend the worker map."""
    with _prefiller(
        monkeypatch, token=_PLACEHOLDER_TOKEN, local_engines_only=False
    ) as (producer, port):
        base = f"http://127.0.0.1:{port}"
        legit = {
            "engine_id": _ENGINE_ID,
            "dp_rank": producer.dp_rank,
            "tp_rank": producer.tp_rank,
            "pp_rank": producer.pp_rank,
            "addr": "tcp://198.51.100.10:5999",
        }
        other_rank = dict(
            legit,
            tp_rank=producer.tp_rank + 1,
            addr="tcp://198.51.100.11:5999",
        )
        hidden = (legit["addr"], legit["engine_id"], other_rank["addr"])

        async with httpx.AsyncClient(timeout=2.0) as client:
            empty = await client.get(f"{base}/query")
            _assert_rejected(empty, ())
            wrong = await client.get(
                f"{base}/query", headers=_bearer(_PLACEHOLDER_TOKEN + "-no")
            )
            _assert_rejected(wrong, ())
            basic = await client.get(
                f"{base}/query", headers={"Authorization": "Basic placeholder"}
            )
            _assert_rejected(basic, ())

            rejected = await client.post(f"{base}/register", json=legit)
            _assert_rejected(rejected, hidden)
            rejected_wrong = await client.post(
                f"{base}/register",
                json=legit,
                headers=_bearer(_PLACEHOLDER_TOKEN + "-no"),
            )
            _assert_rejected(rejected_wrong, hidden)
            assert producer.bootstrap_server.workers == {}

            ok = await client.post(
                f"{base}/register", json=legit, headers=_bearer(_PLACEHOLDER_TOKEN)
            )
            assert ok.status_code == 200

            leaked = await client.get(f"{base}/query")
            _assert_rejected(leaked, hidden)
            occupied = await client.post(f"{base}/register", json=other_rank)
            _assert_rejected(occupied, hidden)

            authed = await client.get(
                f"{base}/query", headers=_bearer(_PLACEHOLDER_TOKEN)
            )
            assert authed.status_code == 200
            body = authed.json()
            assert body[str(producer.dp_rank)]["engine_id"] == _ENGINE_ID
            assert (
                body[str(producer.dp_rank)]["worker_addr"][str(producer.tp_rank)][
                    str(producer.pp_rank)
                ]
                == legit["addr"]
            )
            assert other_rank["addr"] not in authed.text

            again = await client.post(
                f"{base}/register", json=legit, headers=_bearer(_PLACEHOLDER_TOKEN)
            )
            assert again.status_code == 200
            conflict = dict(legit, addr="tcp://198.51.100.12:5999")
            replaced = await client.post(
                f"{base}/register",
                json=conflict,
                headers=_bearer(_PLACEHOLDER_TOKEN),
            )
            assert replaced.status_code == 400
            mismatch = dict(legit, engine_id="other-engine", tp_rank=3)
            mismatched = await client.post(
                f"{base}/register",
                json=mismatch,
                headers=_bearer(_PLACEHOLDER_TOKEN),
            )
            assert mismatched.status_code == 400
            still = await client.get(
                f"{base}/query", headers=_bearer(_PLACEHOLDER_TOKEN)
            )
            assert (
                still.json()[str(producer.dp_rank)]["worker_addr"][
                    str(producer.tp_rank)
                ][str(producer.pp_rank)]
                == legit["addr"]
            )
            assert conflict["addr"] not in still.text
            assert "other-engine" not in still.text


@pytest.mark.asyncio
async def test_bootstrap_fails_closed_without_configured_token(monkeypatch):
    """No shared token configured must not leave the registry open."""
    with _prefiller(monkeypatch, token=None, local_engines_only=False) as (
        producer,
        port,
    ):
        payload = {
            "engine_id": _ENGINE_ID,
            "dp_rank": 0,
            "tp_rank": 0,
            "pp_rank": 0,
            "addr": "tcp://198.51.100.10:5999",
        }
        async with httpx.AsyncClient(timeout=2.0) as client:
            base = f"http://127.0.0.1:{port}"
            registered = await client.post(f"{base}/register", json=payload)
            queried = await client.get(f"{base}/query")
        _assert_rejected(registered, (payload["addr"], payload["engine_id"]))
        _assert_rejected(queried, (payload["addr"], payload["engine_id"]))
        assert producer.bootstrap_server.workers == {}


@pytest.mark.asyncio
async def test_unauthenticated_register_cannot_redirect_decoder_pull(monkeypatch):
    """A first unauthenticated registration must not become the KV peer."""
    attacker = _serve_peer()
    legitimate = _serve_peer()
    try:
        with _prefiller(
            monkeypatch, token=_PLACEHOLDER_TOKEN, local_engines_only=False
        ) as (producer, port):
            decoder_config = create_vllm_config(
                kv_connector="MooncakeConnector",
                kv_role="kv_consumer",
                kv_connector_extra_config={"num_workers": 1},
            )
            decoder = MooncakeConnectorWorker(
                decoder_config, "decoder-engine", _make_test_kv_cache_config()
            )
            try:
                base = f"http://127.0.0.1:{port}"
                forged = {
                    "engine_id": _ENGINE_ID,
                    "dp_rank": producer.dp_rank,
                    "tp_rank": producer.tp_rank,
                    "pp_rank": producer.pp_rank,
                    "addr": attacker.addr,
                }
                async with httpx.AsyncClient(timeout=2.0) as client:
                    forged_response = await client.post(f"{base}/register", json=forged)

                producer.side_channel_port = int(legitimate.addr.rsplit(":", 1)[1])
                assert producer.hostname == "127.0.0.1"
                legitimate_registered = False
                legitimate_status: int | None = None
                try:
                    await MooncakeConnectorWorker.register_worker_with_bootstrap(
                        producer
                    )
                    legitimate_registered = True
                except mc.httpx.HTTPStatusError as exc:
                    legitimate_status = exc.response.status_code

                metadata = MooncakeConnectorMetadata()
                metadata.add_new_req(
                    request_id="d-req",
                    local_block_ids=[[1]],
                    kv_transfer_params={
                        "transfer_id": "xfer-1",
                        "remote_engine_id": _ENGINE_ID,
                        "remote_bootstrap_addr": base,
                    },
                )
                decoder.start_load_kv(metadata)
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    if attacker.received or legitimate.received:
                        await asyncio.sleep(0.2)
                        break
                    await asyncio.sleep(0.05)

                stored = (
                    decoder._remote_agents.get(_ENGINE_ID, {})
                    .get(producer.tp_rank, {})
                    .get(producer.pp_rank)
                )
                problems: list[str] = []
                if forged_response.status_code not in _UNAUTHORIZED:
                    problems.append(
                        "unauthenticated /register returned "
                        f"{forged_response.status_code}"
                    )
                if not legitimate_registered:
                    problems.append(
                        "authenticated worker registration failed "
                        f"({legitimate_status})"
                    )
                if stored != legitimate.addr:
                    problems.append(f"decoder stored {stored!r}")
                if attacker.received:
                    problems.append(
                        "decoder pulled KV from the unauthenticated address"
                    )
                if not legitimate.received:
                    problems.append(
                        "decoder did not pull KV from the authenticated worker"
                    )
                assert not problems, "; ".join(problems)
            finally:
                decoder.shutdown()
                decoder.shutdown = lambda: None
    finally:
        _stop_peer(attacker)
        _stop_peer(legitimate)
