"""Real MCP (Model Context Protocol) client wrapper.

Uses the official `mcp` Python SDK to connect to one or more MCP servers over
stdio, discover their tools, and call them. The orchestrator consumes tools
from this registry without caring which server they live on.

Design notes:
- Each MCP server is launched as a subprocess via stdio (most common in the
  wild today). We keep the connection open for the lifetime of the registry.
- Tools are namespaced by server name to avoid collisions:
    `server_name__tool_name` → real tool on that server.
- We convert the MCP `inputSchema` (JSON Schema) directly into Anthropic's
  tool-use schema, since they're compatible.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agentmesh.utils.logging import get_logger
from agentmesh.utils.types import ToolCall, ToolResult

log = get_logger(__name__)

# Separator used when flattening (server, tool) → single identifier for the LLM.
_NS_SEP = "__"


@dataclass
class MCPServerSpec:
    """Declarative spec for one MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


@dataclass
class _ServerConn:
    spec: MCPServerSpec
    session: ClientSession
    tool_names: list[str]


class MCPToolRegistry:
    """Connects to many MCP servers and exposes a unified tool list."""

    def __init__(self, servers: list[MCPServerSpec]) -> None:
        self._servers = servers
        self._connections: dict[str, _ServerConn] = {}
        self._anthropic_tools: list[dict[str, Any]] = []
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    # --------------------------------------------------------------------
    # lifecycle
    # --------------------------------------------------------------------

    async def connect(self) -> None:
        """Start every server subprocess and list its tools."""
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        for spec in self._servers:
            try:
                await self._connect_one(spec)
            except Exception as e:  # pragma: no cover — defensive, logged
                log.error("mcp.connect_failed", server=spec.name, error=str(e))
                raise

        log.info(
            "mcp.registry_ready",
            servers=len(self._connections),
            total_tools=len(self._anthropic_tools),
        )

    async def _connect_one(self, spec: MCPServerSpec) -> None:
        assert self._stack is not None

        params = StdioServerParameters(
            command=spec.command,
            args=spec.args,
            env=spec.env,
        )
        transport = await self._stack.enter_async_context(stdio_client(params))
        read_stream, write_stream = transport
        session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        tools_resp = await session.list_tools()
        tool_names: list[str] = []
        for tool in tools_resp.tools:
            flat_name = f"{spec.name}{_NS_SEP}{tool.name}"
            tool_names.append(flat_name)
            self._anthropic_tools.append(
                {
                    "name": flat_name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema
                    or {"type": "object", "properties": {}},
                }
            )

        self._connections[spec.name] = _ServerConn(
            spec=spec, session=session, tool_names=tool_names
        )
        log.info("mcp.server_connected", server=spec.name, tools=len(tool_names))

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(None, None, None)
            self._stack = None
        self._connections.clear()
        self._anthropic_tools.clear()

    # --------------------------------------------------------------------
    # accessors
    # --------------------------------------------------------------------

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Schemas formatted for Anthropic's `tools` param."""
        return list(self._anthropic_tools)

    def list_tool_names(self) -> list[str]:
        return [t["name"] for t in self._anthropic_tools]

    # --------------------------------------------------------------------
    # dispatch
    # --------------------------------------------------------------------

    async def call(self, call: ToolCall) -> ToolResult:
        """Route a ToolCall to its server and return a normalised ToolResult."""
        server_name, _, tool_name = call.name.partition(_NS_SEP)
        if not tool_name or server_name not in self._connections:
            return ToolResult(
                call_id=call.id,
                is_error=True,
                content=f"Unknown tool: {call.name}",
            )

        conn = self._connections[server_name]
        async with self._lock:
            try:
                resp = await conn.session.call_tool(tool_name, call.arguments)
            except Exception as e:
                log.warning("mcp.tool_call_error", tool=call.name, error=str(e))
                return ToolResult(
                    call_id=call.id, is_error=True, content=f"Tool error: {e}"
                )

        # Normalise: MCP returns structured content blocks; we flatten to text.
        text_parts: list[str] = []
        for block in resp.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            else:
                # Images / resources — represent as a tag so the LLM knows
                # something came back, even if we can't pass it through raw.
                text_parts.append(f"[{btype or 'unknown'} content omitted]")

        return ToolResult(
            call_id=call.id,
            is_error=bool(getattr(resp, "isError", False)),
            content="\n".join(text_parts).strip() or "(empty result)",
        )
