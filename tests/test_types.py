"""Tests for the shared data types."""

from __future__ import annotations

from agentmesh.utils.types import AgentResult, Step, ToolCall


def test_tool_call_auto_id():
    c1 = ToolCall(name="x")
    c2 = ToolCall(name="x")
    assert c1.id != c2.id
    assert len(c1.id) == 12


def test_agent_result_derived_counts():
    steps = [
        Step(kind="plan", summary="p"),
        Step(kind="tool_call", summary="t1"),
        Step(kind="tool_result", summary="r1"),
        Step(kind="critique", summary="retry: x", payload={"verdict": "retry"}),
        Step(kind="tool_call", summary="t2"),
        Step(kind="tool_result", summary="r2"),
        Step(kind="critique", summary="accept", payload={"verdict": "accept"}),
        Step(kind="final", summary="done"),
    ]
    r = AgentResult(
        task="t",
        session_id="s",
        success=True,
        final_answer="done",
        steps=steps,
        duration_s=1.0,
    )
    assert r.num_tool_calls == 2
    assert r.num_critic_rejections == 1


def test_step_timestamp_is_iso():
    s = Step(kind="plan", summary="hi")
    assert "T" in s.timestamp
