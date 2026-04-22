"""Critic tests — parse correctness and verdict plumbing."""

from __future__ import annotations

import pytest

from agentmesh.critic.critic import Critic
from agentmesh.utils.types import ToolCall, ToolResult
from tests.conftest import StubLLM, make_text_response


@pytest.mark.asyncio
async def test_critic_accepts_valid_json():
    llm = StubLLM(
        [make_text_response('{"verdict": "accept", "reasoning": "looks good"}')]
    )
    crit = Critic(llm)  # type: ignore[arg-type]
    out = await crit.review(
        task="add 2 + 2",
        call=ToolCall(name="calc", arguments={"e": "2+2"}),
        result=ToolResult(call_id="x", content="4"),
    )
    assert out.verdict == "accept"
    assert "good" in out.reasoning


@pytest.mark.asyncio
async def test_critic_parses_retry_with_suggestion():
    payload = (
        '{"verdict": "retry", "reasoning": "tool returned empty string", '
        '"suggested_fix": "retry with a longer query"}'
    )
    llm = StubLLM([make_text_response(payload)])
    crit = Critic(llm)  # type: ignore[arg-type]
    out = await crit.review(
        task="search corpus",
        call=ToolCall(name="search", arguments={"q": "x"}),
        result=ToolResult(call_id="x", content=""),
    )
    assert out.verdict == "retry"
    assert out.suggested_fix == "retry with a longer query"


@pytest.mark.asyncio
async def test_critic_parses_abort():
    llm = StubLLM(
        [make_text_response('{"verdict": "abort", "reasoning": "task impossible"}')]
    )
    crit = Critic(llm)  # type: ignore[arg-type]
    out = await crit.review(
        task="t",
        call=ToolCall(name="t"),
        result=ToolResult(call_id="x", content=""),
    )
    assert out.verdict == "abort"


@pytest.mark.asyncio
async def test_critic_malformed_json_defaults_to_accept():
    llm = StubLLM([make_text_response("not json at all")])
    crit = Critic(llm)  # type: ignore[arg-type]
    out = await crit.review(
        task="t", call=ToolCall(name="t"), result=ToolResult(call_id="x", content="")
    )
    assert out.verdict == "accept"


@pytest.mark.asyncio
async def test_critic_unknown_verdict_defaults_to_accept():
    llm = StubLLM(
        [make_text_response('{"verdict": "maybe", "reasoning": "idk"}')]
    )
    crit = Critic(llm)  # type: ignore[arg-type]
    out = await crit.review(
        task="t", call=ToolCall(name="t"), result=ToolResult(call_id="x", content="")
    )
    assert out.verdict == "accept"


@pytest.mark.asyncio
async def test_critic_handles_llm_exception_gracefully():
    # An LLM that raises instead of returning.
    class BadLLM:
        async def call(self, system: str, messages: list, tools: list | None = None):
            raise RuntimeError("api down")

    crit = Critic(BadLLM())  # type: ignore[arg-type]
    out = await crit.review(
        task="t", call=ToolCall(name="t"), result=ToolResult(call_id="x", content="ok")
    )
    # Graceful fallback: never blocks the orchestrator.
    assert out.verdict == "accept"
    assert "critic unavailable" in out.reasoning
