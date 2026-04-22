"""End-to-end test for MCPToolRegistry against the real demo server.

Spawns the demo server as a subprocess and talks to it over stdio — no mocks.
This validates our MCP integration path (tool discovery, arg passing, content
normalisation) actually works. Skipped gracefully if the `mcp` SDK isn't
importable, so a partial install still lets the rest of the suite pass.
"""

from __future__ import annotations

import sys

import pytest

mcp_sdk = pytest.importorskip("mcp")

from agentmesh.mcp_client.client import MCPServerSpec, MCPToolRegistry  # noqa: E402
from agentmesh.utils.types import ToolCall  # noqa: E402


def _demo_spec() -> MCPServerSpec:
    return MCPServerSpec(
        name="demo",
        command=sys.executable,
        args=["-m", "agentmesh.mcp_client.demo_server"],
    )


@pytest.mark.asyncio
async def test_registry_discovers_demo_tools():
    reg = MCPToolRegistry([_demo_spec()])
    await reg.connect()
    try:
        names = reg.list_tool_names()
        # Flat names prefixed by server.
        assert "demo__calculator" in names
        assert "demo__search" in names
        assert "demo__kv_set" in names
        assert len(names) == 6

        schemas = reg.anthropic_tools()
        calc = next(s for s in schemas if s["name"] == "demo__calculator")
        assert "expression" in calc["input_schema"]["properties"]
    finally:
        await reg.close()


@pytest.mark.asyncio
async def test_calculator_tool_roundtrip():
    reg = MCPToolRegistry([_demo_spec()])
    await reg.connect()
    try:
        res = await reg.call(
            ToolCall(name="demo__calculator", arguments={"expression": "12*11"})
        )
        assert res.is_error is False
        assert "132" in res.content
    finally:
        await reg.close()


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_not_exception():
    reg = MCPToolRegistry([_demo_spec()])
    await reg.connect()
    try:
        res = await reg.call(ToolCall(name="demo__does_not_exist", arguments={}))
        # Graceful: either is_error is set, or the content explains the failure.
        # Either way, no exception should propagate.
        assert res.is_error or "unknown" in res.content.lower() or "not found" in res.content.lower()
    finally:
        await reg.close()


@pytest.mark.asyncio
async def test_kv_set_then_get():
    reg = MCPToolRegistry([_demo_spec()])
    await reg.connect()
    try:
        set_res = await reg.call(
            ToolCall(name="demo__kv_set", arguments={"key": "k", "value": "v"})
        )
        assert set_res.is_error is False

        get_res = await reg.call(
            ToolCall(name="demo__kv_get", arguments={"key": "k"})
        )
        assert get_res.is_error is False
        assert "v" in get_res.content
    finally:
        await reg.close()
