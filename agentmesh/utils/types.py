"""Shared data types (pydantic models).

These are the wire-level objects passed between orchestrator, memory, critic,
and the API. Kept small and explicit on purpose — one change here cascades
through the system, so we prefer stable fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


StepKind = Literal["plan", "tool_call", "tool_result", "critique", "final", "error"]
CritiqueVerdict = Literal["accept", "retry", "abort"]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolCall(BaseModel):
    """A call the model wants to make into an MCP tool."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of running a ToolCall against the MCP server."""

    call_id: str
    is_error: bool = False
    content: str
    raw: dict[str, Any] | None = None


class Critique(BaseModel):
    """A critic's structured judgement of a tool result."""

    verdict: CritiqueVerdict
    reasoning: str
    suggested_fix: str | None = None


class Step(BaseModel):
    """One row in the execution trace."""

    kind: StepKind
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_utcnow_iso)


class AgentResult(BaseModel):
    """Terminal outcome of an Orchestrator.run() invocation."""

    task: str
    session_id: str
    success: bool
    final_answer: str
    steps: list[Step]
    duration_s: float
    tokens_in: int = 0
    tokens_out: int = 0

    # Was the critic disabled for this run (for ablation experiments)
    critic_enabled: bool = True

    @property
    def num_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool_call")

    @property
    def num_critic_rejections(self) -> int:
        return sum(
            1
            for s in self.steps
            if s.kind == "critique" and s.payload.get("verdict") != "accept"
        )


class MemoryRecord(BaseModel):
    """A generic record for all three memory tiers."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str | None = None
    kind: str                              # e.g. "episode" | "fact" | "procedure"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_utcnow_iso)
