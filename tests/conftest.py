"""Shared pytest fixtures. No network, no real LLM, no real Redis."""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Avoid the config loader accidentally touching user env during tests.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("AGENTMESH_LOG_LEVEL", "WARNING")

from agentmesh.llm.anthropic_client import LLMResponse  # noqa: E402
from agentmesh.utils.types import ToolCall, ToolResult  # noqa: E402


# ---------------------------------------------------------------------------
# Stub LLM — scripted, deterministic, offline
# ---------------------------------------------------------------------------


class StubLLM:
    """A minimal stand-in for AnthropicLLM.

    Construct with a list of `LLMResponse` objects; each `call()` pops the
    next one. Used to drive the orchestrator and critic through specific
    control-flow paths without hitting a real API.
    """

    def __init__(self, scripted: list[LLMResponse]) -> None:
        self._queue = list(scripted)
        self.calls: list[dict[str, Any]] = []
        self.model = "stub"
        self.max_tokens = 1024
        self.temperature = 0.0

    async def call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        # Snapshot the messages list — orchestrator mutates it across turns,
        # so a shallow reference would make every recorded call look identical.
        import copy

        self.calls.append(
            {"system": system, "messages": copy.deepcopy(messages), "tools": tools}
        )
        if not self._queue:
            raise AssertionError("StubLLM exhausted — no more scripted responses")
        return self._queue.pop(0)

    async def aclose(self) -> None:
        return None


def make_text_response(text: str, tokens_in: int = 10, tokens_out: int = 20) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_uses=[],
        stop_reason="end_turn",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        raw=None,  # type: ignore[arg-type]
    )


def make_tool_use_response(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_id: str = "tu_1",
    text: str = "",
    tokens_in: int = 10,
    tokens_out: int = 20,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_uses=[{"id": tool_id, "name": tool_name, "input": tool_input}],
        stop_reason="tool_use",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        raw=None,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Stub tool registry
# ---------------------------------------------------------------------------


class StubTools:
    """Mimic MCPToolRegistry without real MCP subprocesses."""

    def __init__(self, tools: dict[str, dict[str, Any]] | None = None) -> None:
        # Map flat tool name → {description, input_schema, handler(args) -> str}
        self._tools = tools or {}
        self.calls: list[ToolCall] = []

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": meta.get("description", ""),
                "input_schema": meta.get("input_schema", {"type": "object", "properties": {}}),
            }
            for name, meta in self._tools.items()
        ]

    def list_tool_names(self) -> list[str]:
        return list(self._tools)

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def call(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        meta = self._tools.get(call.name)
        if meta is None:
            return ToolResult(call_id=call.id, is_error=True, content=f"unknown tool: {call.name}")
        handler = meta.get("handler")
        try:
            out = handler(call.arguments) if handler else "(ok)"
            return ToolResult(call_id=call.id, is_error=False, content=str(out))
        except Exception as e:
            return ToolResult(call_id=call.id, is_error=True, content=str(e))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_procedural_path(tmp_path: Path) -> str:
    return str(tmp_path / "procedural.json")


@pytest.fixture
def tmp_chroma_dir(tmp_path: Path) -> str:
    return str(tmp_path / "chroma")


class _DeterministicEmbedder:
    """Offline, deterministic embedder for tests.

    Chroma's default embedder downloads an ONNX model on first use, which
    breaks air-gapped CI. This produces a 64-dim vector from hashed token
    counts — good enough to separate unrelated sentences for test assertions,
    bad enough never to confuse with production.

    Implements both the legacy `__call__` interface and the modern
    `embed_documents` / `embed_query` interface so we work across chromadb
    versions.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def _embed_one(self, text: str) -> list[float]:
        import hashlib
        import re

        vec = [0.0] * self._dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big")
            vec[h % self._dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    # Legacy interface
    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [self._embed_one(t) for t in input]

    # Modern chromadb interface
    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [self._embed_one(t) for t in input]

    def embed_query(self, input: str | list[str]) -> list[list[float]]:  # noqa: A002
        if isinstance(input, str):
            return [self._embed_one(input)]
        return [self._embed_one(t) for t in input]

    def name(self) -> str:
        return "test-deterministic-hash"

    # Some chromadb versions check this attribute.
    is_legacy = False


@pytest.fixture
def test_embedder() -> _DeterministicEmbedder:
    return _DeterministicEmbedder()


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM([])
