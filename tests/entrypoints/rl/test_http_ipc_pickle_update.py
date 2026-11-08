# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP IPC weight updates must not execute a pickled request body.

``/update_weights`` and ``/collective_rpc`` are mounted in server dev mode and
are outside the API-key prefixes. With the IPC backend and
``VLLM_ALLOW_INSECURE_SERIALIZATION`` enabled, the worker rebuilds
``IPCWeightTransferUpdateInfo`` from that body and ``pickle.loads`` the
``ipc_handles_pickled`` field.

These tests drive the FastAPI routes, the AsyncLLM weight-update call, the
utility-RPC msgpack frames, the worker-queue pickle round-trip, and
``Worker.update_weights``. They fail while that field is unpickled.
"""

import io
import os
import pickle
from argparse import Namespace

import pybase64
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from vllm.config import DeviceConfig, VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.entrypoints.launchers.app import build_app
from vllm.envs import disable_envs_cache
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.worker.gpu_worker import Worker

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skip_global_cleanup,
]

_EXECUTIONS: list[str] = []
# Placeholder credential. The suite only checks that this header is not what
# authorizes execution of the update body.
_API_KEY = "placeholder-api-key"


def _note_pickled_ipc_body_executed() -> None:
    _EXECUTIONS.append("executed")


class _PickleExecutionMarker:
    """Pickle payload whose reduction calls a function in this module."""

    def __reduce__(self):
        return (_note_pickled_ipc_body_executed, ())


def _pickled_update() -> dict:
    encoded = pybase64.b64encode(
        pickle.dumps(
            _PickleExecutionMarker(),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    ).decode("ascii")
    return {
        "names": ["w"],
        "dtype_names": ["float32"],
        "shapes": [[1]],
        "ipc_handles_pickled": encoded,
    }


def _message_queue_roundtrip(obj):
    """Replay ``MessageQueue.enqueue`` for an object with no tensors.

    The worker RPC queue pickles ``(method, args, kwargs, output_rank)`` with
    ``pickle.Pickler`` at the highest protocol. Tensor reducers do not apply
    to this payload, so the main pickle buffer is the whole message.
    """
    buffer = io.BytesIO()
    pickler = pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dump(obj)
    return pickle.loads(buffer.getvalue())


class _WorkerExecutor:
    """In-process stand-in for the executor object ``EngineCore`` calls.

    ``collective_rpc`` applies the multiprocess worker-queue pickle, then the
    in-process ``UniProcExecutor.collective_rpc`` implementation, which calls
    ``run_method`` on the real worker.
    """

    def __init__(self, worker: Worker) -> None:
        self.driver_worker = worker

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
        single_value=False,
    ):
        if kwargs is None:
            kwargs = {}
        method, args, kwargs, _output_rank = _message_queue_roundtrip(
            (method, args, kwargs, None)
        )
        return UniProcExecutor.collective_rpc(
            self,
            method,
            timeout,
            args,
            kwargs,
            non_block=non_block,
            single_value=single_value,
        )


def _deliver_collective_rpc(worker, method, timeout, args, kwargs):
    """Send one utility RPC the way the async engine client and core do.

    The client msgpack-encodes ``(client_index, call_id, method, args)`` and
    the core's generic decoder reconstructs that tuple before
    ``_convert_msgspec_args`` and ``EngineCore.collective_rpc``.
    """
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()
    type_frame = EngineCoreRequestType.UTILITY.value
    data_frames = encoder.encode(
        (0, 1, "collective_rpc", (method, timeout, args, kwargs or {}))
    )
    assert EngineCoreRequestType(bytes(type_frame)) is EngineCoreRequestType.UTILITY
    _client_idx, _call_id, method_name, rpc_args = decoder.decode(data_frames)
    core = object.__new__(EngineCore)
    core.model_executor = _WorkerExecutor(worker)
    method_fn = getattr(core, method_name)
    converted = EngineCoreProc._convert_msgspec_args(method_fn, rpc_args)
    return method_fn(*converted)


class _HttpIpcEngine:
    """Engine client used by the RL routes, bound to one real worker."""

    def __init__(self, worker: Worker) -> None:
        self._worker = worker
        self.vllm_config = worker.vllm_config

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return _deliver_collective_rpc(self._worker, method, timeout, args, kwargs)

    async def update_weights(self, request):
        await AsyncLLM.update_weights(self, request)

    async def start_weight_update(self):
        await AsyncLLM.start_weight_update(self)


def _make_worker() -> Worker:
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    engine = WeightTransferEngineFactory.create_engine(
        WeightTransferConfig(backend="ipc"),
        config,
        torch.device("cpu"),
        nn.Linear(1, 1),
    )
    worker = object.__new__(Worker)
    worker.vllm_config = config
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker.rank = 1
    return worker


def _build_args() -> Namespace:
    return Namespace(
        disable_fastapi_docs=True,
        enable_offline_docs=False,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=[_API_KEY],
        enable_request_id_headers=False,
        enable_fault_tolerance=False,
        middleware=[],
        log_error_stack=False,
    )


@pytest.fixture
def ipc_server():
    """Dev-mode API server with IPC weight transfer and an open update session.

    ``Worker.update_weights`` rejects the body until ``start_weight_update``
    has opened a session. A configured RL server is already in that session
    while it accepts ``/update_weights``, so the fixture opens it through the
    real worker method before any HTTP request.
    """
    disable_envs_cache()
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    _EXECUTIONS.clear()
    worker = _make_worker()
    Worker.start_weight_update(worker)
    assert worker._weight_update_active
    app = build_app(_build_args(), supported_tasks=())
    app.state.engine_client = _HttpIpcEngine(worker)
    app.state.log_stats = False
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, worker
    _EXECUTIONS.clear()


def _headers(authenticated: bool) -> dict[str, str]:
    if not authenticated:
        return {}
    return {"Authorization": f"Bearer {_API_KEY}"}


def _assert_pickle_did_not_run(response) -> None:
    assert _EXECUTIONS == [], (
        "pickled ipc_handles_pickled executed in the worker; "
        f"status={response.status_code} body={response.text}"
    )


@pytest.mark.parametrize("authenticated", [False, True])
def test_update_weights_does_not_unpickle_http_body(ipc_server, authenticated):
    """A dict ``update_info`` posted to ``/update_weights`` must not run."""
    client, _worker = ipc_server
    response = client.post(
        "/update_weights",
        json={"update_info": _pickled_update()},
        headers=_headers(authenticated),
    )
    _assert_pickle_did_not_run(response)


@pytest.mark.parametrize("authenticated", [False, True])
def test_update_weights_does_not_unpickle_rank_local_list(ipc_server, authenticated):
    """Every rank-local ``update_info`` element must not run.

    The worker selects one element after the request is accepted. The pickled
    field is on every element, including index 0 and the caller's rank, so
    checking a different index from the one that is applied still fails.
    """
    client, worker = ipc_server
    parallel = worker.vllm_config.parallel_config
    selected = parallel.data_parallel_index * parallel.world_size + worker.rank
    assert selected != 0
    updates = [_pickled_update() for _ in range(selected + 1)]
    response = client.post(
        "/update_weights",
        json={"update_info": updates},
        headers=_headers(authenticated),
    )
    _assert_pickle_did_not_run(response)


@pytest.mark.parametrize("authenticated", [False, True])
def test_collective_rpc_update_weights_does_not_unpickle(ipc_server, authenticated):
    """``/collective_rpc`` method ``update_weights`` must not run the field."""
    client, _worker = ipc_server
    response = client.post(
        "/collective_rpc",
        json={
            "method": "update_weights",
            "kwargs": {"update_info": _pickled_update()},
        },
        headers=_headers(authenticated),
    )
    _assert_pickle_did_not_run(response)


def test_api_key_guards_v1_models(ipc_server):
    """The same server still rejects a guarded route missing the placeholder key."""
    client, _worker = ipc_server
    response = client.get("/v1/models")
    assert response.status_code == 401
    assert _EXECUTIONS == []
