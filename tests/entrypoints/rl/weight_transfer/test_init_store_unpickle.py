# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Security spec for online weight-transfer initialization.

These tests fail while the server still accepts an unauthenticated
``POST /init_weight_transfer_engine`` and the worker unpickles bytes from
the caller-supplied rendezvous. They are expected to pass once:

* the RL routes require the same API key as the guarded prefixes,
* ``WeightTransferConfig.master_address`` and ``master_port`` are the only
  rendezvous the worker joins, and
* the NCCL unique-id exchange copies a fixed-size buffer and rejects any
  other length before the bytes are interpreted. The same rule applies to
  every init path that currently unpickles a client-chosen store, including
  the dense NCCL, sparse NCCL, and NCCL M2N engines.
"""

import os
import pickle
import socket
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi import FastAPI
from starlette.testclient import TestClient
from torch.distributed import TCPStore

from tests.wt_marker import RAN, note
from vllm.config import VllmConfig
from vllm.config.device import DeviceConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCL_UNIQUE_ID_BYTES,
    NCCLLibrary,
)
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.entrypoints.common.cli_args import FrontendArgs
from vllm.entrypoints.launchers.api_server.routers import register_api_routers
from vllm.entrypoints.serve.middleware.register import init_entrypoints_middleware
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.gpu_worker import Worker

# Obvious placeholder for the key configured on this test server. It is not a
# credential from any deployment.
PLACEHOLDER_API_KEY = "placeholder-api-key"

BACKENDS = ("nccl", "sparse_nccl", "nccl_m2n")

# Symbols NCCLLibrary binds before it returns. The real constructor does not
# read the TCPStore; it only has to succeed so PyNcclCommunicator reaches
# broadcast_obj. ncclCommInitRank returns an error so a later collective
# cannot block.
_STUB_NCCL_SOURCE = r"""
int ncclGetVersion(int *version) {
    if (version) *version = 21903;
    return 0;
}
const char *ncclGetErrorString(int result) { (void)result; return "stub-nccl"; }
const char *ncclGetLastError(void *comm) { (void)comm; return "stub-nccl"; }
int ncclGetUniqueId(void *unique_id) { (void)unique_id; return 0; }
int ncclCommInitRank(void *comm, int nranks, void *comm_id, int rank) {
    (void)comm; (void)nranks; (void)comm_id; (void)rank; return 1;
}
int ncclAllReduce(const void *s, void *r, unsigned long c, int dt, int op,
                  void *comm, void *stream) {
    (void)s;(void)r;(void)c;(void)dt;(void)op;(void)comm;(void)stream; return 1;
}
int ncclReduce(const void *s, void *r, unsigned long c, int dt, int op, int root,
               void *comm, void *stream) {
    (void)s;(void)r;(void)c;(void)dt;(void)op;(void)root;(void)comm;(void)stream;
    return 1;
}
int ncclAllGather(const void *s, void *r, unsigned long c, int dt, void *comm,
                  void *stream) {
    (void)s;(void)r;(void)c;(void)dt;(void)comm;(void)stream; return 1;
}
int ncclReduceScatter(const void *s, void *r, unsigned long c, int dt, int op,
                      void *comm, void *stream) {
    (void)s;(void)r;(void)c;(void)dt;(void)op;(void)comm;(void)stream; return 1;
}
int ncclSend(const void *s, unsigned long c, int dt, int dest, void *comm,
             void *stream) {
    (void)s;(void)c;(void)dt;(void)dest;(void)comm;(void)stream; return 1;
}
int ncclRecv(void *r, unsigned long c, int dt, int src, void *comm, void *stream) {
    (void)r;(void)c;(void)dt;(void)src;(void)comm;(void)stream; return 1;
}
int ncclBroadcast(const void *s, void *r, unsigned long c, int dt, int root,
                  void *comm, void *stream) {
    (void)s;(void)r;(void)c;(void)dt;(void)root;(void)comm;(void)stream; return 1;
}
int ncclCommDestroy(void *comm) { (void)comm; return 0; }
int ncclCommAbort(void *comm) { (void)comm; return 0; }
int ncclCommSuspend(void *comm, int flags) { (void)comm; (void)flags; return 0; }
int ncclCommResume(void *comm) { (void)comm; return 0; }
int ncclGroupStart(void) { return 0; }
int ncclGroupEnd(void) { return 0; }
int ncclCommWindowRegister(void *comm, void *buff, unsigned long size, void *win,
                           int flags) {
    (void)comm;(void)buff;(void)size;(void)win;(void)flags; return 0;
}
int ncclCommWindowDeregister(void *comm, void *win) {
    (void)comm;(void)win; return 0;
}
int ncclCommQueryProperties(void *comm, void *props) {
    (void)comm;(void)props; return 0;
}
"""


def _gadget(token: str, exact_len: int | None = None) -> bytes:
    """Pickle a reducer that records ``token`` when it is loaded."""

    def build(pad_len: int) -> bytes:
        pad = "A" * pad_len

        class _Gadget:
            def __reduce__(self):
                return (note, (token, pad))

        return pickle.dumps(_Gadget(), protocol=4)

    if exact_len is None:
        return build(0)
    previous = build(0)
    if len(previous) == exact_len:
        return previous
    for pad_len in range(1, exact_len + 1):
        blob = build(pad_len)
        if len(blob) == exact_len:
            return blob
        if len(blob) > exact_len > len(previous):
            break
        previous = blob
    raise AssertionError(
        f"could not build a {exact_len}-byte reducer (nearest {len(previous)})"
    )


class _StoreServer:
    """Rank-0 TCPStore serving one value at ``broadcast_from/0/0``."""

    def __init__(self, payload: bytes):
        self.payload = payload
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._wait_until_listening()

    @property
    def host(self) -> str:
        return "127.0.0.1"

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _serve(self) -> None:
        from datetime import timedelta

        store = TCPStore(
            self.host,
            self.port,
            world_size=2,
            is_master=True,
            timeout=timedelta(seconds=8),
            wait_for_workers=False,
            use_libuv=False,
        )
        store.set("broadcast_from/0/0", self.payload)
        self._stop.wait()
        del store

    def _wait_until_listening(self) -> None:
        deadline = threading.Event()

        def _arm() -> None:
            deadline.set()

        timer = threading.Timer(5, _arm)
        timer.start()
        while not deadline.is_set():
            try:
                with socket.create_connection((self.host, self.port), 0.2):
                    timer.cancel()
                    return
            except OSError:
                continue
        raise RuntimeError(f"TCPStore did not listen on {self.host}:{self.port}")


def _compile_stub(directory: Path) -> None:
    source = directory / "stub_nccl.c"
    library = directory / "libstubnccl.so"
    source.write_text(_STUB_NCCL_SOURCE)
    subprocess.check_call(
        ["gcc", "-shared", "-fPIC", "-o", str(library), str(source)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    os.environ["VLLM_NCCL_SO_PATH"] = str(library)
    NCCLLibrary.path_to_library_cache.clear()
    NCCLLibrary.path_to_dict_mapping.clear()
    NCCLLibrary()


def _init_body(backend: str, host: str, port: int) -> dict:
    if backend == "nccl_m2n":
        return {
            "schema_version": 2,
            "master_address": host,
            "master_port": port,
            "rank_offset": 1,
            "world_size": 2,
            "dst_mesh_dims": [1, 1],
            "source_digest": "unverified",
            "params": [],
        }
    return {
        "master_address": host,
        "master_port": port,
        "rank_offset": 1,
        "world_size": 2,
    }


class _InProcessEngine:
    """Same init call the API route makes, executed in this process.

    ``AsyncLLM.init_weight_transfer_engine`` forwards ``request.init_info``
    unchanged. The uniproc executor then ``run_method``s that dict into
    ``Worker.init_weight_transfer_engine``.
    """

    def __init__(self, worker: Worker):
        self._worker = worker

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        del timeout
        return run_method(self._worker, method, args, kwargs or {})

    async def init_weight_transfer_engine(self, request):
        return await AsyncLLM.init_weight_transfer_engine(self, request)


def _worker(backend: str, vllm_config: VllmConfig, pinned: tuple[str, int]):
    config = WeightTransferConfig(backend=backend)
    # The fixed server reads these instead of the request body. Setting them
    # on the instance keeps the test valid before the fields exist on the
    # dataclass; pydantic forbids unknown constructor arguments today.
    object.__setattr__(config, "master_address", pinned[0])
    object.__setattr__(config, "master_port", pinned[1])
    engine = WeightTransferEngineFactory.create_engine(
        config,
        vllm_config,
        torch.device("cpu"),
        torch.nn.Linear(1, 1),
    )
    worker = Worker.__new__(Worker)
    worker.weight_transfer_engine = engine
    return _InProcessEngine(worker)


def _post(client: TestClient, body: dict, headers: dict[str, str] | None):
    return client.post(
        "/init_weight_transfer_engine",
        json={"init_info": body},
        headers=headers,
    )


@pytest.fixture(scope="module")
def nccl_library(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Load a NCCL shared library so communicator init is not skipped."""
    directory = tmp_path_factory.mktemp("nccl-stub")
    try:
        _compile_stub(directory)
    except (OSError, subprocess.CalledProcessError):
        os.environ.pop("VLLM_NCCL_SO_PATH", None)
        NCCLLibrary.path_to_library_cache.clear()
        NCCLLibrary.path_to_dict_mapping.clear()
        try:
            NCCLLibrary()
        except Exception as exc:
            pytest.fail(
                "NCCLLibrary did not load, so PyNcclCommunicator returns "
                f"before reading the store: {exc}"
            )


@pytest.fixture(scope="module")
def vllm_config() -> VllmConfig:
    return VllmConfig(device_config=DeviceConfig(device="cpu"))


@pytest.fixture(scope="module")
def api_app(nccl_library: None):
    del nccl_library
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    app = FastAPI()
    args = FrontendArgs()
    args.api_key = [PLACEHOLDER_API_KEY]
    app.state.args = args
    register_api_routers(args, app, ())
    init_entrypoints_middleware(args, app, ())
    app.state.engine_client = None
    return app


@pytest.fixture
def client(api_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(api_app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_marker():
    RAN.clear()
    yield
    RAN.clear()


@pytest.fixture(autouse=True)
def _cpu_device_index(monkeypatch: pytest.MonkeyPatch):
    """Let init proceed where this process has no accelerator.

    ``torch.accelerator.current_device_index`` raises here before the worker
    joins the store. On a GPU worker it returns the selected device index,
    and that integer is consumed only after the unique id is read. Return 0,
    the index of the selected device, when the runtime has no accelerator.
    """
    real = torch.accelerator.current_device_index

    def current_device_index() -> int:
        try:
            return real()
        except RuntimeError as exc:
            if "none is available" not in str(exc):
                raise
            return 0

    monkeypatch.setattr(torch.accelerator, "current_device_index", current_device_index)


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_api_key_does_not_run_store_payload(
    backend: str,
    client: TestClient,
    api_app: FastAPI,
    vllm_config: VllmConfig,
):
    """An API key is configured, and the RL init route still requires it."""
    token = f"missing-key-{backend}"
    payload = _gadget(token, NCCL_UNIQUE_ID_BYTES)
    attacker = _StoreServer(payload)
    pinned = _StoreServer(b"\x00" * NCCL_UNIQUE_ID_BYTES)
    try:
        api_app.state.engine_client = _worker(
            backend, vllm_config, (pinned.host, pinned.port)
        )
        response = _post(
            client,
            _init_body(backend, attacker.host, attacker.port),
            None,
        )
    finally:
        attacker.close()
        pinned.close()
    assert token not in RAN
    assert response.status_code == 401


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrong_api_key_does_not_run_store_payload(
    backend: str,
    client: TestClient,
    api_app: FastAPI,
    vllm_config: VllmConfig,
):
    """A bearer token other than the configured key is rejected first."""
    token = f"wrong-key-{backend}"
    payload = _gadget(token, NCCL_UNIQUE_ID_BYTES)
    attacker = _StoreServer(payload)
    pinned = _StoreServer(b"\x00" * NCCL_UNIQUE_ID_BYTES)
    try:
        api_app.state.engine_client = _worker(
            backend, vllm_config, (pinned.host, pinned.port)
        )
        response = _post(
            client,
            _init_body(backend, attacker.host, attacker.port),
            {"Authorization": "Bearer placeholder-not-the-configured-key"},
        )
    finally:
        attacker.close()
        pinned.close()
    assert token not in RAN
    assert response.status_code == 401


@pytest.mark.parametrize("backend", BACKENDS)
def test_client_rendezvous_is_not_joined(
    backend: str,
    client: TestClient,
    api_app: FastAPI,
    vllm_config: VllmConfig,
):
    """The body cannot steer the worker at a store the config did not pin."""
    token = f"client-rendezvous-{backend}"
    payload = _gadget(token, NCCL_UNIQUE_ID_BYTES)
    attacker = _StoreServer(payload)
    pinned = _StoreServer(b"\x00" * NCCL_UNIQUE_ID_BYTES)
    try:
        api_app.state.engine_client = _worker(
            backend, vllm_config, (pinned.host, pinned.port)
        )
        _post(
            client,
            _init_body(backend, attacker.host, attacker.port),
            {"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"},
        )
    finally:
        attacker.close()
        pinned.close()
    assert token not in RAN


@pytest.mark.parametrize("backend", BACKENDS)
def test_configured_store_payload_is_not_interpreted(
    backend: str,
    client: TestClient,
    api_app: FastAPI,
    vllm_config: VllmConfig,
):
    """Even the pinned store is copied, not unpickled.

    The payload is exactly ``NCCL_UNIQUE_ID_BYTES`` long, so a length check
    that still calls ``pickle.loads`` on the accepted buffer does not pass.
    """
    token = f"configured-store-{backend}"
    payload = _gadget(token, NCCL_UNIQUE_ID_BYTES)
    pinned = _StoreServer(payload)
    try:
        api_app.state.engine_client = _worker(
            backend, vllm_config, (pinned.host, pinned.port)
        )
        _post(
            client,
            _init_body(backend, pinned.host, pinned.port),
            {"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"},
        )
    finally:
        pinned.close()
    assert token not in RAN


def test_second_store_read_is_not_unpickled(
    client: TestClient,
    api_app: FastAPI,
    vllm_config: VllmConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    """A length check must not fetch the same key again and interpret it.

    Both buffers are ``NCCL_UNIQUE_ID_BYTES`` long. The first read is one
    reducer and every later read is the reducer stored in the TCPStore.
    Interpreting either one records a token. Copying the first buffer without
    unpickling records neither.
    """
    first_token = "second-read-first"
    stored_token = "second-read-stored"
    stored = _gadget(stored_token, NCCL_UNIQUE_ID_BYTES)
    first = _gadget(first_token, NCCL_UNIQUE_ID_BYTES)
    server = _StoreServer(stored)
    reads: list[str] = []
    real_get = TCPStore.get

    def get(self, key: str) -> bytes:
        value = real_get(self, key)
        if key != "broadcast_from/0/0":
            return value
        reads.append(key)
        if len(reads) == 1:
            return first
        return value

    monkeypatch.setattr(TCPStore, "get", get)
    try:
        api_app.state.engine_client = _worker(
            "nccl", vllm_config, (server.host, server.port)
        )
        _post(
            client,
            _init_body("nccl", server.host, server.port),
            {"Authorization": f"Bearer {PLACEHOLDER_API_KEY}"},
        )
    finally:
        server.close()
    assert first_token not in RAN
    assert stored_token not in RAN
