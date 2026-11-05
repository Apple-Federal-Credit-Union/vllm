# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authorization and cache-binding tests for the HF3FS metadata server.

The metadata server is a separate process. Its HTTP routes must refuse
callers that do not present ``VLLM_HF3FS_METADATA_TOKEN`` (placeholder values
only in this file). ``Hf3fsGlobalMetadataClient`` is the client the connector
uses and must send that credential.

``confirm_write`` may record a page only when that same rank allocated that
page for that same key. A later pair in the same request must not replace an
authorized page with a different index, and ``pages_to_release`` must not free
a page still bound to a key.

Block hashes must mix ``kv_connector_extra_config['hf3fs_block_hash_salt']``.
The unsalted ``md5(f"{previous}_{token_ids}")`` digest must not answer
existence checks. The scheduler must not treat a key aimed at another
request's page as a hit, and a load of that key must not read that page.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests
import torch
import uvicorn

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    HF3FSKVConnector,
    Hf3fsClient,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_metadata_server import (
    Hf3fsGlobalMetadataClient,
    Hf3fsMetadataServer,
    run_metadata_server,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request

TOKEN_ENV = "VLLM_HF3FS_METADATA_TOKEN"
# Obvious placeholders. These are not credentials for any deployment.
METADATA_TOKEN = "PLACEHOLDER_METADATA_TOKEN"
WRONG_TOKEN = "PLACEHOLDER_WRONG"
SALT_A = "PLACEHOLDER_BLOCK_HASH_SALT_A"
SALT_B = "PLACEHOLDER_BLOCK_HASH_SALT_B"
pytestmark = pytest.mark.skip_global_cleanup

BLOCK_SIZE = 16
BYTES_PER_PAGE = 64
NUM_PAGES = 4
VICTIM_KEY = "victim-key"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

# Minimal OPT config so the connector can be constructed without a download.
_OPT_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "vocab_size": 50272,
    "max_position_embeddings": 2048,
    "word_embed_proj_dim": 768,
    "ffn_dim": 3072,
    "activation_function": "relu",
    "do_layer_norm_before": True,
    "bos_token_id": 2,
    "eos_token_id": 2,
    "pad_token_id": 1,
    "torch_dtype": "float16",
}


class _QuietServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        return


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _json_body(response: requests.Response) -> dict:
    if not response.content:
        return {}
    try:
        body = response.json()
    except json.JSONDecodeError:
        return {}
    return body if isinstance(body, dict) else {}


def _pages(results) -> dict[str, int]:
    return dict(results)


class MetadataServerHandle:
    def __init__(self, port: int):
        os.environ[TOKEN_ENV] = METADATA_TOKEN
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.metadata = Hf3fsMetadataServer()
        config = uvicorn.Config(
            self.metadata.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
        )
        self.server = _QuietServer(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 10
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                requests.post(
                    f"{self.base}/keys/batch_exists",
                    json={"keys": []},
                    timeout=0.2,
                )
                break
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(0.05)
        else:
            raise RuntimeError(f"metadata server did not start: {last_error}")
        self.client = Hf3fsGlobalMetadataClient(self.base)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def post(
        self,
        path: str,
        payload: dict | None = None,
        headers: dict | None = None,
    ) -> requests.Response:
        return requests.post(
            f"{self.base}{path}",
            json={} if payload is None else payload,
            headers=headers or {},
            timeout=5,
        )


@pytest.fixture
def metadata_server() -> Iterator[MetadataServerHandle]:
    handle = MetadataServerHandle(_free_port())
    try:
        yield handle
    finally:
        handle.stop()


def _seed_victim(client: Hf3fsGlobalMetadataClient) -> int:
    """Initialize one rank and confirm one key on the page it was allocated."""
    client.initialize(0, NUM_PAGES, role="worker")
    allocated = _pages(client.allocate_pages_for_keys(0, [(VICTIM_KEY, "")]))
    page = allocated[VICTIM_KEY]
    assert page >= 0
    client.confirm_write_for_keys(0, [(VICTIM_KEY, page)])
    return page


def _assert_rejected(response: requests.Response) -> None:
    assert response.status_code in (401, 403), response.status_code


def _public_block_hashes(token_ids: list[int], block_size: int) -> list[str]:
    """Hash a prompt the way an outsider can, with no deployment salt."""
    hashes: list[str] = []
    previous = ""
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        if len(block) < block_size:
            break
        digest = hashlib.md5(f"{previous}_{block}".encode()).hexdigest()
        hashes.append(digest)
        previous = digest
    return hashes


def test_allocated_page_confirm_still_round_trips(
    metadata_server: MetadataServerHandle,
) -> None:
    """A confirmation of the page just allocated for that key stays readable."""
    page = _seed_victim(metadata_server.client)
    client = metadata_server.client
    assert client.batch_key_exists([VICTIM_KEY]) == [True]
    assert client.get_key_locations(0, [VICTIM_KEY]) == [page]
    assert client.batch_key_exists(["missing-key"]) == [False]


def test_initialize_without_credential_creates_no_rank(
    metadata_server: MetadataServerHandle,
) -> None:
    response = metadata_server.post(
        "/rank/3/initialize",
        {"role": "worker", "num_pages": NUM_PAGES},
    )
    try:
        metadata_server.client.allocate_pages_for_keys(3, [("other", "")])
        rank_created = True
    except RuntimeError:
        rank_created = False
    assert not rank_created
    _assert_rejected(response)


def test_allocate_without_credential_consumes_no_page(
    metadata_server: MetadataServerHandle,
) -> None:
    _seed_victim(metadata_server.client)
    response = metadata_server.post(
        "/keys/batch_allocate",
        {"rank": 0, "keys": [["stolen", ""]]},
    )
    # One page is bound to the victim, so three allocations still fit
    # unless the unauthenticated allocate consumed one.
    extra = _pages(
        metadata_server.client.allocate_pages_for_keys(
            0, [("e0", ""), ("e1", ""), ("e2", "")]
        )
    )
    assert all(page >= 0 for page in extra.values()), extra
    assert metadata_server.client.batch_key_exists(["stolen"]) == [False]
    _assert_rejected(response)


def test_exists_without_credential_is_not_an_oracle(
    metadata_server: MetadataServerHandle,
) -> None:
    _seed_victim(metadata_server.client)
    for headers in ({}, {"Authorization": f"Bearer {WRONG_TOKEN}"}):
        response = metadata_server.post(
            "/keys/batch_exists",
            {"keys": [VICTIM_KEY, "absent-key"]},
            headers=headers,
        )
        assert "exists" not in _json_body(response)
        _assert_rejected(response)


def test_locations_without_credential_hide_page_indexes(
    metadata_server: MetadataServerHandle,
) -> None:
    page = _seed_victim(metadata_server.client)
    response = metadata_server.post(
        "/keys/get_locations",
        {"rank": 0, "keys": [VICTIM_KEY]},
        headers={"Authorization": f"Bearer {WRONG_TOKEN}"},
    )
    body = _json_body(response)
    assert "locations" not in body
    assert str(page) not in response.text
    _assert_rejected(response)


def test_confirm_without_credential_does_not_bind_or_release(
    metadata_server: MetadataServerHandle,
) -> None:
    page = _seed_victim(metadata_server.client)
    response = metadata_server.post(
        "/keys/confirm_write",
        {
            "rank": 0,
            "confirmations": [["attacker-key", page]],
            "pages_to_release": [page],
        },
    )
    client = metadata_server.client
    assert client.batch_key_exists(["attacker-key"]) == [False]
    assert client.get_key_locations(0, [VICTIM_KEY]) == [page]
    _assert_rejected(response)
    # Releasing the victim page would return it to the pool and make four
    # new allocations succeed. Three free pages remain, so four do not.
    extra = _pages(
        client.allocate_pages_for_keys(0, [(f"n{i}", "") for i in range(4)])
    )
    assert any(allocated < 0 for allocated in extra.values()), extra


def test_clear_without_credential_keeps_cached_keys(
    metadata_server: MetadataServerHandle,
) -> None:
    _seed_victim(metadata_server.client)
    for headers in ({}, {"Authorization": f"Bearer {WRONG_TOKEN}"}):
        response = metadata_server.post("/clear", headers=headers)
        assert metadata_server.client.batch_key_exists([VICTIM_KEY]) == [True]
        _assert_rejected(response)


def test_confirm_rejects_page_not_allocated_to_that_key(
    metadata_server: MetadataServerHandle,
) -> None:
    """An authorized caller cannot aim a new key at another key's page."""
    client = metadata_server.client
    client.initialize(0, NUM_PAGES, role="worker")
    allocated = _pages(
        client.allocate_pages_for_keys(0, [(VICTIM_KEY, ""), ("attacker-key", "")])
    )
    victim_page = allocated[VICTIM_KEY]
    attacker_page = allocated["attacker-key"]
    assert victim_page != attacker_page
    client.confirm_write_for_keys(0, [(VICTIM_KEY, victim_page)])

    try:
        client.confirm_write_for_keys(0, [("attacker-key", victim_page)])
    except RuntimeError:
        pass

    assert client.batch_key_exists(["attacker-key"]) == [False]
    assert client.get_key_locations(0, [VICTIM_KEY]) == [victim_page]
    attacker_loc = client.get_key_locations(0, ["attacker-key"])[0]
    assert attacker_loc != victim_page


def test_confirm_batch_cannot_overwrite_with_a_later_page(
    metadata_server: MetadataServerHandle,
) -> None:
    """A matching pair must not authorize a different page later in the body."""
    client = metadata_server.client
    client.initialize(0, NUM_PAGES, role="worker")
    allocated = _pages(
        client.allocate_pages_for_keys(0, [(VICTIM_KEY, ""), ("attacker-key", "")])
    )
    victim_page = allocated[VICTIM_KEY]
    attacker_page = allocated["attacker-key"]
    client.confirm_write_for_keys(0, [(VICTIM_KEY, victim_page)])

    try:
        client.confirm_write_for_keys(
            0,
            [
                ("attacker-key", attacker_page),
                ("attacker-key", victim_page),
            ],
        )
    except RuntimeError:
        pass

    assert client.get_key_locations(0, [VICTIM_KEY]) == [victim_page]
    assert client.get_key_locations(0, ["attacker-key"])[0] != victim_page
    assert client.batch_key_exists([VICTIM_KEY]) == [True]


def test_confirm_cannot_release_a_page_bound_to_another_key(
    metadata_server: MetadataServerHandle,
) -> None:
    page = _seed_victim(metadata_server.client)
    client = metadata_server.client
    try:
        client.confirm_write_for_keys(0, [], pages_to_release=[page])
    except RuntimeError:
        pass
    assert client.get_key_locations(0, [VICTIM_KEY]) == [page]
    assert client.batch_key_exists([VICTIM_KEY]) == [True]
    extra = _pages(client.allocate_pages_for_keys(0, [(f"n{i}", "") for i in range(4)]))
    assert any(allocated < 0 for allocated in extra.values()), extra


def test_confirm_cannot_move_a_confirmed_key_onto_another_page(
    metadata_server: MetadataServerHandle,
) -> None:
    client = metadata_server.client
    client.initialize(0, NUM_PAGES, role="worker")
    allocated = _pages(
        client.allocate_pages_for_keys(0, [(VICTIM_KEY, ""), ("other-key", "")])
    )
    victim_page = allocated[VICTIM_KEY]
    other_page = allocated["other-key"]
    client.confirm_write_for_keys(0, [(VICTIM_KEY, victim_page)])
    try:
        client.confirm_write_for_keys(0, [(VICTIM_KEY, other_page)])
    except RuntimeError:
        pass
    assert client.get_key_locations(0, [VICTIM_KEY]) == [victim_page]


def test_run_helpers_do_not_default_to_all_interfaces() -> None:
    for fn in (run_metadata_server, Hf3fsMetadataServer.run):
        host = inspect.signature(fn).parameters["host"].default
        assert host in LOOPBACK_HOSTS, fn.__qualname__


def _listening_ips(port: int) -> list[str]:
    found: list[str] = []
    port_hex = f"{port:04X}"
    for proc_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        text = Path(proc_path).read_text()
        for line in text.splitlines()[1:]:
            local, _remote, state = line.split()[1:4]
            ip_hex, listed_port = local.split(":")
            if listed_port.upper() == port_hex and state == "0A":
                found.append(ip_hex)
    return found


def test_cli_without_host_listens_on_loopback_only() -> None:
    """The process started with default arguments must not listen on every NIC."""
    os.environ[TOKEN_ENV] = METADATA_TOKEN
    port = _free_port()
    module = (
        "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_metadata_server"
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", module, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.time() + 60
        listeners: list[str] = []
        while time.time() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read().decode() if proc.stdout else ""
                raise RuntimeError(f"metadata server exited early:\n{output}")
            listeners = _listening_ips(port)
            if listeners:
                break
            time.sleep(0.1)
        assert listeners, "metadata server did not listen"
        for ip_hex in listeners:
            assert set(ip_hex) != {"0"}, ip_hex
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def opt_model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("opt-config")
    (path / "config.json").write_text(json.dumps(_OPT_CONFIG))
    return str(path)


def _vllm_config(model_dir: str, salt: str) -> VllmConfig:
    model_config = ModelConfig(
        model=model_dir,
        tokenizer=model_dir,
        trust_remote_code=False,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    return VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=64,
            max_model_len=2048,
            enable_chunked_prefill=True,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="HF3FSKVConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"hf3fs_block_hash_salt": salt},
        ),
        device_config=DeviceConfig("cpu"),
    )


def _open_connector(config: VllmConfig) -> HF3FSKVConnector:
    previous = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)
    }
    connector = HF3FSKVConnector(
        vllm_config=config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=None,
    )
    connector._saved_signals = previous  # type: ignore[attr-defined]
    return connector


def _close_connector(connector: HF3FSKVConnector) -> None:
    previous = getattr(connector, "_saved_signals", {})
    connector.close()
    for sig, handler in previous.items():
        signal.signal(sig, handler)


@pytest.fixture
def connector_server(
    opt_model_dir: str,
) -> Iterator[tuple[MetadataServerHandle, HF3FSKVConnector]]:
    """Server on the port the connector client uses, plus one scheduler."""
    os.environ[TOKEN_ENV] = METADATA_TOKEN
    handle = MetadataServerHandle(18000)
    connector = _open_connector(_vllm_config(opt_model_dir, SALT_A))
    try:
        yield handle, connector
    finally:
        _close_connector(connector)
        handle.stop()


def _request(request_id: str, token_ids: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=1, temperature=0),
        pooling_params=None,
    )


def test_block_hash_mixes_configured_salt_and_hides_guesses(
    opt_model_dir: str,
    connector_server: tuple[MetadataServerHandle, HF3FSKVConnector],
) -> None:
    handle, connector = connector_server
    other = _open_connector(_vllm_config(opt_model_dir, SALT_B))
    try:
        token_ids = list(range(BLOCK_SIZE * 2))
        hashed = connector._generate_block_hashes(token_ids, 0)
        other_hashed = other._generate_block_hashes(token_ids, 0)
        public = _public_block_hashes(token_ids, BLOCK_SIZE)
        assert hashed != other_hashed
        assert hashed != public

        client = handle.client
        client.initialize(0, NUM_PAGES, role="worker")
        keys = [(digest, "") for digest in hashed]
        allocated = _pages(client.allocate_pages_for_keys(0, keys))
        client.confirm_write_for_keys(
            0, [(digest, allocated[digest]) for digest in hashed]
        )
        assert client.batch_key_exists(hashed) == [True, True]
        assert client.batch_key_exists(public) == [False, False]
    finally:
        _close_connector(other)


def _write_victim_page(path: Path, page: int, pattern: torch.Tensor) -> Hf3fsClient:
    storage = Hf3fsClient(
        path=str(path),
        size=BYTES_PER_PAGE * NUM_PAGES,
        bytes_per_page=BYTES_PER_PAGE,
        entries=4,
    )
    payload = pattern.contiguous().numpy().tobytes()
    assert len(payload) == BYTES_PER_PAGE
    with open(path, "r+b") as backing:
        backing.seek(page * BYTES_PER_PAGE)
        backing.write(payload)
    return storage


def test_scheduler_misses_key_aimed_at_another_users_page(
    connector_server: tuple[MetadataServerHandle, HF3FSKVConnector],
    tmp_path: Path,
) -> None:
    """Confirm-write of the caller's prompt hash must not load another page.

    The scheduler hashes the prompt and calls ``batch_key_exists``. The worker
    then calls ``get_key_locations`` and ``Hf3fsClient.batch_read`` at
    ``page_index * bytes_per_page`` (see ``_handle_load_task``). Scatter copies
    that buffer into the request KV. This environment has no CUDA, so the
    assertion is on that read buffer, which is the tensor scatter would store.
    """
    handle, connector = connector_server
    victim_block = list(range(100, 100 + BLOCK_SIZE))
    attacker_block = list(range(200, 200 + BLOCK_SIZE))
    victim_hash = connector._generate_block_hashes(victim_block, 0)[0]
    attacker_hash = connector._generate_block_hashes(attacker_block, 0)[0]
    public_attacker = _public_block_hashes(attacker_block, BLOCK_SIZE)[0]

    client = handle.client
    client.initialize(0, NUM_PAGES, role="worker")
    allocated = _pages(client.allocate_pages_for_keys(0, [(victim_hash, "")]))
    victim_page = allocated[victim_hash]
    client.confirm_write_for_keys(0, [(victim_hash, victim_page)])

    pattern = torch.full((BYTES_PER_PAGE // 4,), 7.0, dtype=torch.float32)
    storage = _write_victim_page(tmp_path / "kv", victim_page, pattern)
    try:
        # The attacker picks the hash the scheduler will compute for a prompt
        # they control, and also the unsalted digest of that prompt, and aims
        # both at the occupied page. No allocate was done for either hash.
        response = handle.post(
            "/keys/confirm_write",
            {
                "rank": 0,
                "confirmations": [
                    [public_attacker, victim_page],
                    [attacker_hash, victim_page],
                ],
                "pages_to_release": [],
            },
        )

        victim_prompt = victim_block + [1]
        attacker_prompt = attacker_block + [2]
        victim_hits, victim_load = connector.get_num_new_matched_tokens(
            _request("victim", victim_prompt), 0
        )
        attacker_hits, attacker_load = connector.get_num_new_matched_tokens(
            _request("attacker", attacker_prompt), 0
        )
        location = client.get_key_locations(0, [attacker_hash])[0]
        buffer = torch.zeros(BYTES_PER_PAGE // 4, dtype=torch.float32)
        if isinstance(location, int) and location >= 0:
            # Same offset ``_handle_load_task`` passes to ``batch_read``.
            read = storage.batch_read([location * BYTES_PER_PAGE], [buffer])
            assert read == [BYTES_PER_PAGE]
        copied = torch.equal(buffer, pattern)
        assert (
            victim_hits == BLOCK_SIZE
            and victim_load is True
            and attacker_hits == 0
            and attacker_load is False
            and location != victim_page
            and not copied
            and client.get_key_locations(0, [victim_hash]) == [victim_page]
        ), (
            f"victim_hits={victim_hits} attacker_hits={attacker_hits} "
            f"location={location} victim_page={victim_page} copied={copied}"
        )
        _assert_rejected(response)
    finally:
        storage.close()
