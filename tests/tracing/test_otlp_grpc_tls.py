# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OTLP gRPC trace export must use TLS unless the operator opts out.

``init_tracer`` (API server / ``LLMEngine`` / ``AsyncLLM``) and
``maybe_init_worker_tracer`` (engine core, Ray engine actor, and worker
processes) both install the exporter from ``get_span_exporter``. These tests
drive those two entry points, record one ``llm_request`` span through
``instrument_manual`` (the helper ``OutputProcessor.do_tracing`` uses), and
read the bytes the exporter writes to the collector socket.

TLS is required when the OpenTelemetry SDK would open a secure channel for
the same endpoint and environment: an ``https`` URL, a non-HTTP URL
(including ``grpc://`` and a bare ``host:port``), or
``OTEL_EXPORTER_OTLP_TRACES_INSECURE=false`` /
``OTEL_EXPORTER_OTLP_INSECURE=false``. Plaintext is required only for an
``http`` URL with neither variable requesting TLS, and when a variable is
explicitly ``true``.

The ``http/protobuf`` branch of ``get_span_exporter`` is not exercised. It
does not pass ``insecure=True``; that exporter lets ``requests`` follow the
URL scheme and the certificate verify flag.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from opentelemetry import trace

from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    init_tracer,
    instrument_manual,
    maybe_init_worker_tracer,
)

# Obvious placeholder. Not a real collector credential.
_HEADER_NAME = "collector-auth"
_HEADER_VALUE = "placeholder-credential"
_REQUEST_ID = "req-otlp-tls-check"
_HTTP2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
_EMPTY_SETTINGS = b"\x00\x00\x00\x04\x00\x00\x00\x00\x00"
_SETTINGS_ACK = b"\x00\x00\x00\x04\x01\x00\x00\x00\x00"

_GRPC = "grpc://127.0.0.1:{port}"
_BARE = "127.0.0.1:{port}"
_HTTP = "http://127.0.0.1:{port}"
_HTTPS = "https://127.0.0.1:{port}"


def _markers(data: bytes) -> tuple[bool, bool]:
    return _REQUEST_ID.encode() in data, _HEADER_VALUE.encode() in data


def _classify(data: bytes) -> str:
    if data.startswith(_HTTP2_PREFACE):
        return "plaintext"
    if len(data) >= 3 and data[0] == 0x16 and data[1] == 0x03:
        return "tls"
    return "unknown"


def _read_connection(conn: socket.socket) -> bytes:
    """Read the exporter's first flight, answering HTTP/2 settings if needed.

    gRPC writes the request id and metadata only after the server's settings
    frame. A TLS client instead starts with a ClientHello, which is returned
    as-is.
    """
    conn.settimeout(2)
    data = bytearray()
    while len(data) < len(_HTTP2_PREFACE):
        chunk = conn.recv(len(_HTTP2_PREFACE) - len(data))
        if not chunk:
            return bytes(data)
        data.extend(chunk)
    if not data.startswith(_HTTP2_PREFACE):
        while len(data) < 512:
            try:
                chunk = conn.recv(512 - len(data))
            except TimeoutError:
                break
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    conn.sendall(_EMPTY_SETTINGS)
    conn.sendall(_SETTINGS_ACK)
    while len(data) < 16000 and not all(_markers(data)):
        try:
            chunk = conn.recv(4096)
        except TimeoutError:
            break
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def _export_one_span(entry: str, endpoint: str) -> None:
    if entry == "engine":
        init_tracer("vllm.llm_engine", endpoint)
    elif entry == "worker":
        # Child processes inherit the endpoint the parent stored.
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = endpoint
        tracer = maybe_init_worker_tracer(
            "vllm.worker",
            "worker",
            "Worker_0",
        )
        if tracer is None:
            raise RuntimeError("worker tracer was not initialized")
    else:
        raise RuntimeError(f"unknown entry {entry}")

    instrument_manual(
        span_name="llm_request",
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        attributes={
            SpanAttributes.GEN_AI_REQUEST_ID: _REQUEST_ID,
            SpanAttributes.GEN_AI_REQUEST_TEMPERATURE: 0.2,
            SpanAttributes.GEN_AI_REQUEST_TOP_P: 0.9,
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: 3,
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: 4,
        },
        kind=SpanKind.SERVER,
    )
    trace.get_tracer_provider().force_flush(timeout_millis=5000)


def _capture(entry: str, template: str, extra_env: dict[str, str]) -> dict:
    for key in list(os.environ):
        if key.startswith("OTEL_"):
            del os.environ[key]
    os.environ["OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"] = "1"
    os.environ["OTEL_BSP_EXPORT_TIMEOUT"] = "3000"
    os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = f"{_HEADER_NAME}={_HEADER_VALUE}"
    os.environ.update(extra_env)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    endpoint = template.format(port=port)
    captured: dict[str, bytes | str] = {}

    def _serve() -> None:
        server.settimeout(15)
        try:
            conn, _ = server.accept()
        except OSError as exc:
            captured["error"] = str(exc)
            return
        try:
            captured["data"] = _read_connection(conn)
        finally:
            conn.close()

    thread = threading.Thread(target=_serve)
    thread.start()
    try:
        _export_one_span(entry, endpoint)
    finally:
        thread.join(timeout=20)
        server.close()

    data = captured.get("data", b"")
    if not isinstance(data, bytes):
        data = b""
    request_id_visible, header_visible = _markers(data)
    return {
        "transport": _classify(data),
        "request_id": int(request_id_visible),
        "header": int(header_visible),
        "nbytes": len(data),
        "error": str(captured.get("error", "")),
    }


def _run_case(entry: str, template: str, env: dict[str, str]) -> dict[str, str]:
    payload = json.dumps({"entry": entry, "template": template, "env": env})
    repo_root = str(Path(__file__).resolve().parents[2])
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, child_env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), payload],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env=child_env,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and len(lines) == 1, (
        f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(lines[0][len("RESULT ") :])


def _assert_transport(
    entry: str,
    template: str,
    env: dict[str, str],
    transport: str,
    *,
    cleartext_metadata: bool,
) -> None:
    result = _run_case(entry, template, env)
    assert result["transport"] == transport, result
    if cleartext_metadata:
        assert result["request_id"] == 1, result
        assert result["header"] == 1, result
    else:
        assert result["request_id"] == 0, result
        assert result["header"] == 0, result


def test_engine_grpc_endpoint_uses_tls_by_default():
    """A grpc:// collector, the documented endpoint form, is TLS by default."""
    _assert_transport("engine", _GRPC, {}, "tls", cleartext_metadata=False)


def test_engine_bare_endpoint_uses_tls_by_default():
    """host:port with no scheme is TLS by default."""
    _assert_transport("engine", _BARE, {}, "tls", cleartext_metadata=False)


def test_engine_https_endpoint_uses_tls():
    """An https:// collector is TLS even though the URL names a remote port."""
    _assert_transport("engine", _HTTPS, {}, "tls", cleartext_metadata=False)


def test_engine_grpc_honors_traces_insecure_false():
    """OTEL_EXPORTER_OTLP_TRACES_INSECURE=false keeps the grpc channel on TLS."""
    _assert_transport(
        "engine",
        _GRPC,
        {"OTEL_EXPORTER_OTLP_TRACES_INSECURE": "false"},
        "tls",
        cleartext_metadata=False,
    )


def test_engine_http_honors_traces_insecure_false():
    """An http URL still uses TLS when the operator sets the insecure flag false."""
    _assert_transport(
        "engine",
        _HTTP,
        {"OTEL_EXPORTER_OTLP_TRACES_INSECURE": "false"},
        "tls",
        cleartext_metadata=False,
    )


def test_engine_grpc_honors_generic_insecure_false():
    """OTEL_EXPORTER_OTLP_INSECURE=false keeps the grpc channel on TLS."""
    _assert_transport(
        "engine",
        _GRPC,
        {"OTEL_EXPORTER_OTLP_INSECURE": "false"},
        "tls",
        cleartext_metadata=False,
    )


def test_worker_grpc_endpoint_uses_tls_by_default():
    """A worker that inherits a grpc endpoint exports over TLS by default."""
    _assert_transport("worker", _GRPC, {}, "tls", cleartext_metadata=False)


def test_engine_http_without_tls_request_is_plaintext():
    """An http URL with no TLS request stays plaintext and still exports."""
    _assert_transport(
        "engine",
        _HTTP,
        {},
        "plaintext",
        cleartext_metadata=True,
    )


def test_engine_explicit_insecure_opt_in_is_plaintext():
    """OTEL_EXPORTER_OTLP_TRACES_INSECURE=true exports metadata in the clear."""
    _assert_transport(
        "engine",
        _GRPC,
        {"OTEL_EXPORTER_OTLP_TRACES_INSECURE": "true"},
        "plaintext",
        cleartext_metadata=True,
    )


def test_worker_explicit_insecure_opt_in_is_plaintext():
    """A worker honors an explicit insecure opt-in inherited in the environment."""
    _assert_transport(
        "worker",
        _GRPC,
        {"OTEL_EXPORTER_OTLP_TRACES_INSECURE": "true"},
        "plaintext",
        cleartext_metadata=True,
    )


def _main() -> None:
    try:
        payload = json.loads(sys.argv[1])
        result = _capture(payload["entry"], payload["template"], payload["env"])
    except Exception:
        traceback.print_exc()
        os._exit(1)
    print("RESULT " + json.dumps(result), flush=True)
    # Skip exporter atexit hooks; they retry the same failed export.
    os._exit(0)


if __name__ == "__main__":
    _main()
