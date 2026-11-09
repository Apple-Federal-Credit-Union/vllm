# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Responses MCP headers must not be sent to the operator tool server."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack, suppress

import mcp_types
import pytest
from openai.types.responses import ResponseFunctionToolCall
from openai_harmony import Author, Message, Role, TextContent

from vllm.entrypoints.mcp.tool_server import MCPToolServer
from vllm.entrypoints.openai.responses.context import (
    HarmonyContext,
    ParsableContext,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.openai.responses.utils import extract_tool_types
from vllm.envs import disable_envs_cache
from vllm.parser.harmony import HarmonyParser

CLIENT_SESSION = "placeholder-session"
CLIENT_AUTHORIZATION = "placeholder-authorization"
CLIENT_COOKIE = "placeholder-cookie"
SERVER_SESSION = "server-issued-session"
TOOL_OUTPUT = "operator-tool-output"

_CASES = [
    ("harmony", "python", "code_interpreter", "python", "python"),
    ("harmony", "browser", "web_search_preview", "browser.search", "search"),
    ("harmony", "container", "container", "container.exec", "exec"),
    ("parsable", "python", "code_interpreter", "code_interpreter", "python"),
    ("parsable", "browser", "web_search_preview", "web_search_preview", "search"),
    ("parsable", "container", "container", "container", "exec"),
]


def _enabled_builtin_tools(tool_server: MCPToolServer, request: ResponsesRequest):
    """Same selection as OpenAIServingResponses._create_responses."""
    requested = extract_tool_types(request.tools)
    enabled: list[str] = []
    if tool_server.has_tool("browser") and "web_search_preview" in requested:
        enabled.append("browser")
    if tool_server.has_tool("python") and "code_interpreter" in requested:
        enabled.append("python")
    if tool_server.has_tool("container") and "container" in requested:
        enabled.append("container")
    return enabled


def _header(headers: dict[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def _tool_output_text(result) -> str:
    item = result[0]
    output = getattr(item, "output", None)
    if isinstance(output, str):
        return output
    return item.content[0].text


class _RecordedHttpServer:
    """Minimal MCP SSE peer that records the headers httpx2 actually sends.

    The handshake and tool-result shapes are the models from mcp 2.3.0
    (mcp_types), which is the client vLLM opens in MCPToolServer.new_session.
    """

    def __init__(self, server_name: str):
        self.server_name = server_name
        self.requests: list[dict] = []
        self._server: asyncio.Server | None = None
        self._sse_writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()
        self._sse_ready = asyncio.Event()

    @property
    def hostport(self) -> str:
        assert self._server is not None
        port = self._server.sockets[0].getsockname()[1]
        return f"127.0.0.1:{port}"

    @property
    def url(self) -> str:
        return f"http://{self.hostport}/sse"

    async def __aenter__(self) -> _RecordedHttpServer:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *_args) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while not reader.at_eof():
                parsed = await self._read_request(reader)
                if parsed is None:
                    break
                method, path, headers, body = parsed
                self.requests.append(
                    {"method": method, "path": path, "headers": headers, "body": body}
                )
                if method == "GET" and path.split("?", 1)[0] == "/sse":
                    await self._serve_sse(reader, writer, headers)
                    return
                if method == "POST" and path.split("?", 1)[0] == "/messages":
                    await self._serve_message(writer, body)
                    continue
                await self._write_response(writer, 404, b"")
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _read_request(self, reader: asyncio.StreamReader):
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        except (TimeoutError, asyncio.IncompleteReadError):
            return None
        lines = head.decode("iso-8859-1").split("\r\n")
        request_line = lines[0].split(" ")
        if len(request_line) < 2:
            return None
        method, path = request_line[0], request_line[1]
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        body = b""
        while len(body) < length:
            chunk = await reader.read(length - len(body))
            if not chunk:
                break
            body += chunk
        return method, path, headers, body

    async def _serve_sse(self, reader, writer, _headers) -> None:
        async with self._write_lock:
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Cache-Control: no-store\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            endpoint = f"http://{self.hostport}/messages"
            writer.write(f"event: endpoint\ndata: {endpoint}\n\n".encode())
            await writer.drain()
        self._sse_writer = writer
        self._sse_ready.set()
        try:
            while await reader.read(1024):
                pass
        finally:
            if self._sse_writer is writer:
                self._sse_writer = None
                self._sse_ready.clear()

    async def _serve_message(self, writer, body: bytes) -> None:
        message = json.loads(body.decode()) if body else {}
        method = message.get("method")
        if message.get("id") is not None:
            await asyncio.wait_for(self._sse_ready.wait(), timeout=5)
            payload = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": self._result(method, message.get("params") or {}),
            }
            encoded = json.dumps(payload)
            async with self._write_lock:
                sse_writer = self._sse_writer
                assert sse_writer is not None
                sse_writer.write(f"event: message\ndata: {encoded}\n\n".encode())
                await sse_writer.drain()
        await self._write_response(writer, 202, b"")

    def _result(self, method: str, params: dict) -> dict:
        if method == "initialize":
            version = params.get("protocolVersion", "2025-11-25")
            result = mcp_types.InitializeResult(
                protocol_version=version,
                capabilities=mcp_types.ServerCapabilities(
                    tools=mcp_types.ToolsCapability()
                ),
                server_info=mcp_types.Implementation(
                    name=self.server_name, version="0"
                ),
                instructions="operator",
            )
        elif method == "tools/list":
            result = mcp_types.ListToolsResult(
                tools=[
                    mcp_types.Tool(
                        name=name,
                        description=name,
                        input_schema={"type": "object"},
                    )
                    for name in ("python", "search", "exec")
                ]
            )
        elif method == "tools/call":
            tool_name = params.get("name", "")
            result = mcp_types.CallToolResult(
                content=[
                    mcp_types.TextContent(
                        type="text", text=f"{TOOL_OUTPUT} {tool_name}"
                    )
                ],
                is_error=False,
            )
        else:
            raise AssertionError(f"unexpected MCP method {method}")
        return result.model_dump(by_alias=True, mode="json", exclude_none=True)

    async def _write_response(
        self, writer: asyncio.StreamWriter, status: int, body: bytes
    ) -> None:
        reason = {202: "Accepted", 404: "Not Found"}.get(status, "OK")
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n".encode()
        )
        writer.write(body)
        await writer.drain()


def _attacker_request(server_label: str, server_url: str) -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "use the tool",
            "request_id": SERVER_SESSION,
            "tools": [
                {
                    "type": "mcp",
                    "server_label": server_label,
                    "server_url": server_url,
                    "headers": {
                        "x-session-id": CLIENT_SESSION,
                        "Authorization": CLIENT_AUTHORIZATION,
                        "Cookie": CLIENT_COOKIE,
                    },
                }
            ],
        }
    )


def _context(
    kind: str, request: ResponsesRequest, available: list[str], call_name: str
):
    if kind == "harmony":
        if call_name == "python":
            content = "print(1)"
        elif call_name == "browser.search":
            content = json.dumps({"query": "placeholder"})
        else:
            content = json.dumps({"cmd": ["true"]})
        parser = HarmonyParser(tokenizer=object(), tools=request.tools)
        return HarmonyContext(
            [
                Message(
                    author=Author(role=Role.ASSISTANT),
                    content=[TextContent(text=content)],
                    recipient=call_name,
                    channel="commentary",
                )
            ],
            available,
            frozenset(),
            response_parser=parser,
            request=request,
        )
    if call_name == "code_interpreter":
        arguments = json.dumps({"code": "print(1)"})
    elif call_name == "web_search_preview":
        arguments = json.dumps({"query": "placeholder"})
    else:
        arguments = json.dumps({"cmd": ["true"]})
    return ParsableContext(
        response_messages=[
            ResponseFunctionToolCall(
                id="fc_placeholder",
                call_id="call_placeholder",
                name=call_name,
                arguments=arguments,
                type="function_call",
                status="completed",
            )
        ],
        tokenizer=object(),
        parser_cls=None,
        request=request,
        available_tools=available,
        chat_template=None,
        chat_template_content_format="string",
    )


def _session_requests(server: _RecordedHttpServer, baseline: int) -> list[dict]:
    return server.requests[baseline:]


def _assert_operator_session_is_not_caller_controlled(
    session_requests: list[dict], tool_name: str
) -> None:
    sse = [item for item in session_requests if item["path"].startswith("/sse")]
    calls = []
    for item in session_requests:
        if item["method"] != "POST":
            continue
        body = json.loads(item["body"].decode() or "{}")
        params = body.get("params") or {}
        if body.get("method") == "tools/call" and params.get("name") == tool_name:
            calls.append(item)
    assert sse, session_requests
    assert calls, session_requests
    for item in (sse[-1], calls[-1]):
        headers = item["headers"]
        assert _header(headers, "x-session-id") == SERVER_SESSION, headers
        assert _header(headers, "authorization") is None, headers
        assert _header(headers, "cookie") is None, headers
        assert CLIENT_SESSION not in headers.values(), headers
        assert CLIENT_AUTHORIZATION not in headers.values(), headers
        assert CLIENT_COOKIE not in headers.values(), headers


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("kind", "tool_name", "server_label", "call_name", "invoked_tool"),
    _CASES,
)
async def test_responses_mcp_headers_do_not_reach_operator_tool_server(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    tool_name: str,
    server_label: str,
    call_name: str,
    invoked_tool: str,
) -> None:
    disable_envs_cache()
    monkeypatch.setenv(
        "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS",
        "code_interpreter,web_search_preview,container",
    )
    async with (
        _RecordedHttpServer(tool_name) as operator,
        _RecordedHttpServer("decoy") as decoy,
    ):
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(operator.hostport)
        opened_during_startup = len(operator.requests)

        request = _attacker_request(server_label, decoy.url)
        tool = request.tools[0]
        assert tool.type == "mcp"
        assert tool.server_url == decoy.url
        assert tool.headers is not None
        assert tool.headers["x-session-id"] == CLIENT_SESSION
        assert tool.headers["Authorization"] == CLIENT_AUTHORIZATION
        assert tool.headers["Cookie"] == CLIENT_COOKIE

        available = _enabled_builtin_tools(tool_server, request)
        assert available == [tool_name]
        context = _context(kind, request, available, call_name)
        serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
        serving.tool_server = tool_server

        async with AsyncExitStack() as stack:
            await OpenAIServingResponses._initialize_tool_sessions(
                serving, request, context, stack
            )
            result = await context.call_tool()

        assert _tool_output_text(result) == f"{TOOL_OUTPUT} {invoked_tool}"
        assert decoy.requests == []
        _assert_operator_session_is_not_caller_controlled(
            _session_requests(operator, opened_during_startup), invoked_tool
        )


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
async def test_unlisted_mcp_label_does_not_open_operator_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disable_envs_cache()
    monkeypatch.delenv("VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS", raising=False)
    async with (
        _RecordedHttpServer("python") as operator,
        _RecordedHttpServer("decoy") as decoy,
    ):
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(operator.hostport)
        opened_during_startup = len(operator.requests)
        request = _attacker_request("code_interpreter", decoy.url)
        context = _context(
            "harmony",
            request,
            _enabled_builtin_tools(tool_server, request),
            "python",
        )
        serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
        serving.tool_server = tool_server
        async with AsyncExitStack() as stack:
            await OpenAIServingResponses._initialize_tool_sessions(
                serving, request, context, stack
            )

        assert context.available_tools == []
        assert _session_requests(operator, opened_during_startup) == []
        assert decoy.requests == []
