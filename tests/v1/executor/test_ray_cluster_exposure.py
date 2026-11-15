# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ray startup must not accept unauthenticated remote tasks.

These tests exercise the serving paths that call ``ray.init`` and the
worker environment copy in ``RayExecutorV2``. They fail while a process
that can route to the GCS port can run a task without a Ray auth token,
while that port is bound to every interface, or while driver environment
variables outside the worker allowlist are copied into workers.
"""

import os
import socket
import struct
import subprocess
import sys
import textwrap
from collections.abc import Callable
from urllib.parse import urlsplit

import pytest
import ray

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.v1.engine.utils import CoreEngineActorManager, EngineZmqAddresses
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.executor.ray_env_utils import get_driver_env_vars
from vllm.v1.executor.ray_executor_v2 import RayExecutorV2, RayWorkerProc
from vllm.v1.executor.ray_utils import (
    WORKER_SPECIFIC_ENV_VARS,
    initialize_ray_cluster,
)
from vllm.v1.executor.uniproc_executor import UniProcExecutor

# Names that are not on the worker allowlist. Values are placeholders.
_NON_ALLOWLISTED = {
    "AWS_SECRET_ACCESS_KEY": "placeholder",
    "AWS_ACCESS_KEY_ID": "placeholder",
    "OPENAI_API_KEY": "placeholder",
    "MYLIB_FOO": "bar",
}
_WILDCARD_HOSTS = {"0.0.0.0", "::", "*"}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

_CLIENT = textwrap.dedent(
    """
    import os
    import sys

    os.environ.pop("RAY_AUTH_MODE", None)
    os.environ.pop("RAY_AUTH_TOKEN", None)
    os.environ.pop("RAY_AUTH_TOKEN_PATH", None)

    import ray

    address = sys.argv[1]
    try:
        ray.init(address=address, include_dashboard=False, logging_level=40)

        @ray.remote
        def probe():
            return "executed"

        print(ray.get(probe.remote(), timeout=20), flush=True)
    except Exception as exc:
        print(f"rejected:{type(exc).__name__}", flush=True)
    finally:
        if ray.is_initialized():
            ray.shutdown()
    """
)


def _ipv4_host(ip_hex: str) -> str:
    return socket.inet_ntoa(struct.pack("<I", int(ip_hex, 16)))


def _ipv6_host(ip_hex: str) -> str:
    raw = bytes.fromhex(ip_hex)
    ordered = b"".join(raw[index : index + 4][::-1] for index in range(0, 16, 4))
    return socket.inet_ntop(socket.AF_INET6, ordered)


def _listening_sockets() -> dict[str, list[tuple[str, int]]]:
    """Map socket inode to listening (host, port) pairs."""
    listeners: dict[str, list[tuple[str, int]]] = {}
    for path, parse_host in (
        ("/proc/net/tcp", _ipv4_host),
        ("/proc/net/tcp6", _ipv6_host),
    ):
        with open(path, encoding="ascii") as handle:
            next(handle)
            for line in handle:
                parts = line.split()
                if parts[3] != "0A":  # TCP_LISTEN
                    continue
                local, inode = parts[1], parts[9]
                ip_hex, port_hex = local.split(":")
                listeners.setdefault(inode, []).append(
                    (parse_host(ip_hex), int(port_hex, 16))
                )
    return listeners


def _pids_with_basename(basename: str) -> set[int]:
    found: set[int] = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            command = os.readlink(f"/proc/{entry}/exe")
        except OSError:
            continue
        if os.path.basename(command) == basename:
            found.add(int(entry))
    return found


def _listen_addresses_for(basename: str) -> list[tuple[str, int]]:
    listeners = _listening_sockets()
    addresses: list[tuple[str, int]] = []
    for pid in _pids_with_basename(basename):
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if not target.startswith("socket:["):
                continue
            inode = target[len("socket:[") : -1]
            addresses.extend(listeners.get(inode, []))
    return addresses


def _unauthenticated_task(address: str) -> str:
    """Return ``executed`` when a token-less client runs a remote function."""
    env = os.environ.copy()
    for name in ("RAY_AUTH_MODE", "RAY_AUTH_TOKEN", "RAY_AUTH_TOKEN_PATH"):
        env.pop(name, None)
    completed = subprocess.run(
        [sys.executable, "-c", _CLIENT, address],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    output = completed.stdout.strip().splitlines()
    return output[-1] if output else f"rejected:exit-{completed.returncode}"


def _assert_started_cluster_is_closed() -> None:
    """Require loopback-or-authenticated GCS and no wildcard bind.

    A cluster that was never started satisfies the check: there is no
    port for another host to connect to.
    """
    if not ray.is_initialized():
        assert not _listen_addresses_for("gcs_server")
        return

    info = ray._private.worker._global_node.address_info
    gcs_listeners = _listen_addresses_for("gcs_server")
    assert gcs_listeners, "Ray reported a cluster but gcs_server is not listening"
    gcs_address = info["gcs_address"]
    remote_result = _unauthenticated_task(gcs_address)
    _host, port_text = gcs_address.rsplit(":", 1)
    loopback = f"127.0.0.1:{port_text}"
    loop_result = (
        remote_result if loopback == gcs_address else _unauthenticated_task(loopback)
    )
    assert remote_result != "executed"
    assert loop_result != "executed"

    wildcard = [item for item in gcs_listeners if item[0] in _WILDCARD_HOSTS]
    assert not wildcard, f"GCS is listening on a wildcard address: {wildcard}"

    webui = info.get("webui_url") or ""
    if webui:
        host = urlsplit(webui if "://" in webui else f"http://{webui}").hostname
        assert host in _LOOPBACK_HOSTS, webui


def _run_startup(start: Callable[[], None]) -> None:
    if ray.is_initialized():
        ray.shutdown()
    try:
        start()
    except (AssertionError, ValueError):
        # Placement and the CPU platform check run after ray.init. A
        # startup that never initializes Ray has not opened a port.
        if not ray.is_initialized():
            raise
    try:
        _assert_started_cluster_is_closed()
    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_explicit_ray_executor_rejects_unauthenticated_tasks():
    """``--distributed-executor-backend ray`` starts a closed cluster."""
    parallel = ParallelConfig(distributed_executor_backend="ray")
    assert parallel.distributed_executor_backend == "ray"
    assert Executor.get_class(VllmConfig(parallel_config=parallel)) is RayExecutorV2

    def start() -> None:
        initialize_ray_cluster(parallel)

    _run_startup(start)


def test_data_parallel_ray_backend_selects_ray_and_closes_cluster():
    """``--data-parallel-backend ray`` selects Ray and starts it closed."""
    parallel = ParallelConfig(data_parallel_size=2, data_parallel_backend="ray")
    assert parallel.distributed_executor_backend == "ray"
    assert Executor.get_class(VllmConfig(parallel_config=parallel)) is RayExecutorV2

    config = VllmConfig(parallel_config=parallel)

    class _ModelConfig:
        is_moe = False

    # ``is_moe`` only chooses the actor class. It is not an input to ray.init.
    config.model_config = _ModelConfig()
    addresses = EngineZmqAddresses(
        inputs=["tcp://127.0.0.1:9"],
        outputs=["tcp://127.0.0.1:10"],
    )

    def start() -> None:
        CoreEngineActorManager(
            vllm_config=config,
            addresses=addresses,
            executor_class=UniProcExecutor,
            log_stats=False,
        )

    _run_startup(start)


def test_rocm_and_xpu_startup_branch_rejects_unauthenticated_tasks(monkeypatch):
    """The ROCm/XPU ``ray.init`` branch uses the same closed cluster."""
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)

    def start() -> None:
        initialize_ray_cluster(ParallelConfig(distributed_executor_backend="ray"))

    _run_startup(start)


def test_driver_env_forwarded_to_workers_stays_on_the_allowlist(monkeypatch):
    """Worker env is the allowlist, and ``initialize_worker`` applies that."""
    for name, value in _NON_ALLOWLISTED.items():
        monkeypatch.setenv(name, value)

    copied = get_driver_env_vars(WORKER_SPECIFIC_ENV_VARS)
    allowed = get_env_vars_to_copy(exclude_vars=WORKER_SPECIFIC_ENV_VARS)

    for name in _NON_ALLOWLISTED:
        monkeypatch.delenv(name, raising=False)

    def _stop_before_model_load(self, *args, **kwargs):
        raise RuntimeError("stop before model load")

    monkeypatch.setattr(WorkerProc, "__init__", _stop_before_model_load)
    worker = RayWorkerProc(
        vllm_config=VllmConfig(
            parallel_config=ParallelConfig(distributed_executor_backend="ray")
        ),
        rank=0,
        input_shm_handle=Handle(),
        is_driver_worker=False,
    )
    with pytest.raises(RuntimeError, match="stop before model load"):
        worker.initialize_worker(
            local_rank=0,
            env_vars={},
            distributed_init_method="tcp://127.0.0.1:1",
            driver_env_vars=copied,
        )

    leaked = [name for name in _NON_ALLOWLISTED if name in os.environ]
    assert not leaked, f"initialize_worker copied {leaked}"

    extra = sorted(set(copied) - allowed)
    copied_secrets = [name for name in _NON_ALLOWLISTED if name in extra]
    assert not extra, (
        f"{len(extra)} non-allowlisted names were copied, including {copied_secrets}"
    )
