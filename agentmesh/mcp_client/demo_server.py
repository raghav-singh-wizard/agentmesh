"""A tiny in-process MCP server used by examples, tests, and benchmarks.

Exposes four toy tools:

- `calculator(expression)`  — safe eval of arithmetic
- `search(query, k)`        — canned lookups from a small corpus
- `text_stats(text)`         — word/char counts, reading time
- `key_value_set/get/list`   — scratch-pad storage

Run as a normal MCP stdio server:

    python -m agentmesh.mcp_client.demo_server
"""

from __future__ import annotations

import ast
import operator
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("agentmesh-demo")

# Persistent scratch store (process-lifetime, fine for demos / tests).
_KV: dict[str, str] = {}

# Canned corpus — lets us demo "search" deterministically.
_CORPUS = {
    "mcp": (
        "The Model Context Protocol (MCP) is an open standard introduced by "
        "Anthropic in late 2024 for connecting LLM applications to external "
        "data sources and tools."
    ),
    "anthropic": (
        "Anthropic is an AI safety company founded in 2021. It develops the "
        "Claude family of large language models."
    ),
    "react": (
        "ReAct is a prompting strategy that interleaves reasoning traces "
        "with tool-use actions, first proposed by Yao et al. (2022)."
    ),
    "langgraph": (
        "LangGraph is a library for building stateful, multi-actor LLM "
        "applications as graphs of nodes and edges."
    ),
    "redis": (
        "Redis is an in-memory data store often used for caches, queues, "
        "and short-lived session state."
    ),
    "chroma": (
        "Chroma is an open-source embedding database used for retrieval "
        "augmented generation (RAG) workflows."
    ),
}


# ---------------------------- safe calculator -------------------------------

_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")

    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))  # type: ignore[operator]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))  # type: ignore[operator]
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

    return _ev(tree)


# ---------------------------- handlers --------------------------------------


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression (e.g. '2 + 3 * 4').",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression. Supports + - * / // % ** and unary -.",
                    }
                },
                "required": ["expression"],
            },
        ),
        Tool(
            name="search",
            description="Search a small in-memory knowledge corpus by keyword.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="text_stats",
            description="Return word count, character count, and reading time for text.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        Tool(
            name="kv_set",
            description="Store a value under a key in a scratch KV store.",
            inputSchema={
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
        ),
        Tool(
            name="kv_get",
            description="Retrieve a value by key from the scratch KV store.",
            inputSchema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        ),
        Tool(
            name="kv_list",
            description="List all keys currently in the scratch KV store.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "calculator":
            result = _safe_eval(arguments["expression"])
            return [TextContent(type="text", text=str(result))]

        if name == "search":
            query = arguments["query"].lower()
            k = int(arguments.get("k", 3))
            hits: list[tuple[int, str, str]] = []
            for key, text in _CORPUS.items():
                score = 0
                if query in key.lower():
                    score += 2
                for word in query.split():
                    if word in text.lower():
                        score += 1
                if score > 0:
                    hits.append((score, key, text))
            hits.sort(key=lambda x: -x[0])
            hits = hits[:k]
            if not hits:
                return [TextContent(type="text", text="No results.")]
            lines = [f"- {key}: {text}" for _, key, text in hits]
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "text_stats":
            text = arguments["text"]
            words = len(text.split())
            chars = len(text)
            reading_s = round(words / 200 * 60, 1)  # 200 wpm
            return [
                TextContent(
                    type="text",
                    text=f"words={words} chars={chars} reading_time_s={reading_s}",
                )
            ]

        if name == "kv_set":
            _KV[arguments["key"]] = str(arguments["value"])
            return [TextContent(type="text", text="ok")]

        if name == "kv_get":
            val = _KV.get(arguments["key"])
            return [TextContent(type="text", text=val if val is not None else "null")]

        if name == "kv_list":
            keys = ", ".join(sorted(_KV)) or "(empty)"
            return [TextContent(type="text", text=keys)]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"ERROR: {e}")]


async def _main() -> None:
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        sys.exit(0)
