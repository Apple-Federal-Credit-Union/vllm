# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hidden-state saves must stay inside the configured storage directory.

ExampleHiddenStatesConnector joins the engine request id onto
shared_storage_path and creates that path. The id keeps a caller-supplied
X-Request-Id or body request_id. These tests fail while that join can create
or overwrite files outside the storage directory.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import h11
import httptools
import pytest
import torch
from fastapi import Request
from safetensors.torch import load_file

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.request import Request as EngineRequest

pytestmark = pytest.mark.cpu_test

# First segment is glued to the serving prefix (`chatcmpl-` / `cmpl-`), so the
# `..` components have to be their own path segments to leave the storage dir.
_TRAVERSAL = "x/../../escaped/leaf"


class _CpuCopyDone:
    """Stand-in for the CUDA event _write_tensors waits on.

    The tensors passed in are already on CPU. synchronize() does not choose
    the output path; save_file does.
    """

    def synchronize(self) -> None:
        return None


class _HeaderSink:
    def __init__(self) -> None:
        self.headers: list[tuple[bytes, bytes]] = []

    def on_message_begin(self) -> None:
        return None

    def on_url(self, url: bytes) -> None:
        return None

    def on_header(self, name: bytes, value: bytes) -> None:
        self.headers.append((name, value))

    def on_headers_complete(self) -> None:
        return None

    def on_body(self, body: bytes) -> None:
        return None

    def on_message_complete(self) -> None:
        return None

    def on_chunk_header(self) -> None:
        return None

    def on_chunk_complete(self) -> None:
        return None

    def on_status(self, status: bytes) -> None:
        return None


def _http_request(header_value: str | None) -> bytes:
    lines = [
        b"POST /v1/chat/completions HTTP/1.1",
        b"Host: localhost",
        b"Content-Length: 2",
    ]
    if header_value is not None:
        lines.append(b"X-Request-Id: " + header_value.encode("ascii"))
    lines.extend([b"", b"{}"])
    return b"\r\n".join(lines)


def _headers_from_httptools(raw: bytes) -> list[tuple[bytes, bytes]]:
    """Parse with httptools, then lowercase names the way uvicorn does.

    HttpToolsProtocol.on_header lowercases the name before the ASGI scope is
    built. Starlette looks headers up with a lowercase key.
    """
    sink = _HeaderSink()
    parser = httptools.HttpRequestParser(sink)
    parser.feed_data(raw)
    return [(name.lower(), value) for name, value in sink.headers]


def _headers_from_h11(raw: bytes) -> list[tuple[bytes, bytes]]:
    connection = h11.Connection(h11.SERVER)
    connection.receive_data(raw)
    event = connection.next_event()
    assert isinstance(event, h11.Request)
    return list(event.headers)


def _raw_request(headers: list[tuple[bytes, bytes]]) -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    return Request(scope)


def _engine_request_id(
    external_request_id: str,
    sampling_params: SamplingParams | None = None,
) -> str:
    """Apply the same suffix the engine appends in assign_request_id."""
    request = EngineCoreRequest(
        request_id=external_request_id,
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    InputProcessor.assign_request_id(request)
    return request.request_id


def _storage(tmp_path: Path) -> Path:
    storage = tmp_path / "jail" / "storage"
    storage.mkdir(parents=True)
    return storage


def _connector(storage: Path) -> ExampleHiddenStatesConnector:
    """Real constructor. Only the config fields __init__ reads are filled in."""
    vllm_config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"shared_storage_path": str(storage)},
        ),
        cache_config=SimpleNamespace(block_size=16),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=[])
            )
        ),
    )
    return ExampleHiddenStatesConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.WORKER,
        kv_cache_config=None,  # type: ignore[arg-type]
    )


def _scheduler_output(
    req_id: str,
    sampling_params: SamplingParams | None = None,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=[11, 22],
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=([],),
            num_computed_tokens=0,
            lora_request=None,
        )
    ]
    return output


def _snapshot(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    found: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        found.add(path.resolve())
    return found


def _close_locks(conn: ExampleHiddenStatesConnector) -> None:
    for fd in list(conn._lock_fds.values()):
        os.close(fd)
    conn._lock_fds.clear()


def _save_like_worker(
    conn: ExampleHiddenStatesConnector,
    req_id: str,
    sampling_params: SamplingParams | None,
) -> None:
    """Run the scheduler join and both worker writes that use its filename.

    _submit_async_write is not called: it builds a CUDA stream and a pinned
    tensor before os.makedirs, and this host has no pinned-memory allocator.
    The directory and lock are created in wait_for_save. The safetensors file
    is created in _write_tensors, which _submit_async_write calls with the
    same filename.
    """
    output = _scheduler_output(req_id, sampling_params)
    try:
        meta = conn.build_connector_meta(output)
        conn.bind_connector_metadata(meta)
        conn.wait_for_save()
        request = EngineRequest(
            request_id=req_id,
            prompt_token_ids=[11, 22],
            sampling_params=sampling_params or SamplingParams(),
            pooling_params=None,
        )
        _delay_free, transfer = conn.request_finished(request, [0])
        assert transfer is not None
        filename = transfer["hidden_states_path"]
        pending = conn._pending_saves[req_id]
        assert pending.filename == filename
        ExampleHiddenStatesConnector._write_tensors(
            {
                "hidden_states": torch.zeros(2, 4),
                "token_ids": pending.token_ids,
            },
            _CpuCopyDone(),
            pending.filename,
            None,
        )
    finally:
        _close_locks(conn)


def _assert_contained(storage: Path, before: set[Path]) -> None:
    """Fail if the connector created anything whose real path is outside."""
    storage_root = storage.resolve()
    leaked = []
    for path in sorted(_snapshot(storage.parent) - before):
        if path != storage_root and storage_root not in path.parents:
            leaked.append(path)
    assert not leaked, (
        "hidden-state save created paths outside "
        f"{storage_root}: {leaked}"
    )


def _exercise(
    storage: Path,
    req_id: str,
    sampling_params: SamplingParams | None = None,
) -> None:
    before = _snapshot(storage.parent)
    conn = _connector(storage)
    try:
        try:
            _save_like_worker(conn, req_id, sampling_params)
        except (OSError, ValueError, RuntimeError):
            # A fix may refuse the path. A refusal that already created or
            # replaced a file outside storage is still a failure.
            pass
    finally:
        conn._executor.shutdown(wait=True, cancel_futures=True)
    _assert_contained(storage, before)


def test_chat_header_parent_segments_stay_in_storage(tmp_path: Path) -> None:
    """X-Request-Id with its own '..' segments must not leave storage.

    The header is parsed by httptools, which uvicorn selects when that
    package is installed, then read by BaseServing._base_request_id. Chat
    completions prefix chatcmpl- and the engine appends its random suffix.
    """
    raw = _http_request(_TRAVERSAL)
    headers = _headers_from_httptools(raw)
    request = _raw_request(headers)
    base = BaseServing._base_request_id(request, "body-is-not-used")
    assert base == _TRAVERSAL
    req_id = _engine_request_id(f"chatcmpl-{base}")
    storage = _storage(tmp_path)
    _exercise(storage, req_id)


def test_h11_chat_header_parent_segments_stay_in_storage(tmp_path: Path) -> None:
    """Same chat id when the request is parsed by h11 instead of httptools."""
    raw = _http_request(_TRAVERSAL)
    headers = _headers_from_h11(raw)
    request = _raw_request(headers)
    base = BaseServing._base_request_id(request, "body-is-not-used")
    assert base == _TRAVERSAL
    req_id = _engine_request_id(f"chatcmpl-{base}")
    _exercise(_storage(tmp_path), req_id)


def test_chat_body_request_id_parent_segments_stay_in_storage(
    tmp_path: Path,
) -> None:
    """Body request_id is used when X-Request-Id is absent."""
    body = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="test",
        request_id=_TRAVERSAL,
    )
    request = _raw_request(_headers_from_httptools(_http_request(None)))
    base = BaseServing._base_request_id(request, body.request_id)
    assert base == _TRAVERSAL
    req_id = _engine_request_id(f"chatcmpl-{base}")
    _exercise(_storage(tmp_path), req_id)


def test_completion_header_parent_segments_stay_in_storage(tmp_path: Path) -> None:
    """Completions prefix cmpl- and then join the same way as chat."""
    body = CompletionRequest(model="test", prompt="hi", request_id="unused-body")
    request = _raw_request(_headers_from_httptools(_http_request(_TRAVERSAL)))
    base = BaseServing._base_request_id(request, body.request_id)
    assert base == _TRAVERSAL
    req_id = _engine_request_id(f"cmpl-{base}")
    _exercise(_storage(tmp_path), req_id)


def test_parallel_sampling_child_id_stays_in_storage(tmp_path: Path) -> None:
    """n>1 children are scheduled as '{index}_{parent id}' with the traversal."""
    external = f"chatcmpl-{_TRAVERSAL}"
    sampling_params = SamplingParams(n=2)
    parent = EngineCoreRequest(
        request_id=external,
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    InputProcessor.assign_request_id(parent)
    child_id, _child_params = ParentRequest(parent).get_child_info(0)
    assert child_id.startswith("0_")
    assert _TRAVERSAL in child_id
    _exercise(_storage(tmp_path), child_id)


def test_symlink_component_in_request_id_stays_in_storage(tmp_path: Path) -> None:
    """A separator in the id must not follow a symlink out of storage.

    The symlink is already pointed outside before the save, so a check that
    only looks at the un-resolved string (storage / 'chatcmpl-link' / ...)
    and then opens that string still writes outside.
    """
    storage = _storage(tmp_path)
    outside = storage.parent / "escaped"
    outside.mkdir()
    link = storage / "chatcmpl-link"
    link.symlink_to(outside, target_is_directory=True)
    req_id = _engine_request_id("chatcmpl-link/leaf")
    _exercise(storage, req_id)


def test_absolute_engine_request_id_stays_in_storage(tmp_path: Path) -> None:
    """os.path.join drops the storage dir when the request id is absolute.

    Chat and completion ids are prefixed, so they are not absolute. An engine
    request id that is absolute still reaches the same join.
    """
    storage = _storage(tmp_path)
    outside = storage.parent / "escaped"
    outside.mkdir()
    absolute = str(outside / "abs")
    assert os.path.isabs(absolute)
    req_id = _engine_request_id(absolute)
    assert os.path.isabs(req_id)
    _exercise(storage, req_id)


def test_disabled_custom_save_path_does_not_allow_request_id_escape(
    tmp_path: Path,
) -> None:
    """allow_custom_save_path defaults off and must not be the only check.

    A hidden_states_path on the request is ignored in that mode. The request
    id is still joined onto the storage directory.
    """
    storage = _storage(tmp_path)
    outside = storage.parent / "escaped"
    custom = outside / "custom.safetensors"
    sampling_params = SamplingParams(
        extra_args={"kv_transfer_params": {"hidden_states_path": str(custom)}}
    )
    req_id = _engine_request_id(f"chatcmpl-{_TRAVERSAL}", sampling_params)
    _exercise(storage, req_id, sampling_params)
    assert not custom.exists()


def test_saved_tensors_match_what_was_written_when_path_is_safe(
    tmp_path: Path,
) -> None:
    """A request id with no path characters still saves inside storage.

    Guards a fix that refuses every save. The file has to contain the token
    ids handed to _write_tensors.
    """
    storage = _storage(tmp_path)
    req_id = _engine_request_id("chatcmpl-plain")
    before = _snapshot(storage.parent)
    conn = _connector(storage)
    try:
        _save_like_worker(conn, req_id, None)
    finally:
        conn._executor.shutdown(wait=True, cancel_futures=True)
    created = _snapshot(storage.parent) - before
    files = [path for path in created if path.is_file()]
    assert files, "safe request id did not write a hidden-state file"
    storage_root = storage.resolve()
    for path in files:
        assert storage_root == path or storage_root in path.parents
    data_files = [path for path in files if not path.name.endswith(".lock")]
    assert data_files
    tensors = load_file(data_files[0])
    assert tensors["token_ids"].tolist() == [11, 22]
