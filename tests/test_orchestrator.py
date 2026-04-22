"""Orchestrator integration tests.

Uses stub LLM + stub tools + in-memory episodic + tmpdir semantic/procedural.
No network calls, no subprocesses — runs in <1s per test.
"""

from __future__ import annotations

import pytest

from agentmesh.critic.critic import Critic
from agentmesh.memory.base import MemoryHierarchy
from agentmesh.memory.episodic import InMemoryEpisodicMemory
from agentmesh.memory.procedural import ProceduralMemory
from agentmesh.memory.semantic import SemanticMemory
from agentmesh.orchestrator.core import Orchestrator
from tests.conftest import (
    StubLLM,
    StubTools,
    make_text_response,
    make_tool_use_response,
)


def _make_orch(
    tmp_procedural_path: str,
    tmp_chroma_dir: str,
    planner_llm: StubLLM,
    critic_llm: StubLLM,
    tools: StubTools,
    embedder,
    critic_enabled: bool = True,
    max_steps: int = 8,
) -> Orchestrator:
    memory = MemoryHierarchy(
        episodic=InMemoryEpisodicMemory(),
        semantic=SemanticMemory(
            persist_dir=tmp_chroma_dir, embedding_function=embedder
        ),
        procedural=ProceduralMemory(path=tmp_procedural_path),
    )
    orch = Orchestrator.__new__(Orchestrator)
    orch.llm = planner_llm  # type: ignore[assignment]
    orch._critic_llm = critic_llm  # type: ignore[assignment]
    orch.tools = tools  # type: ignore[assignment]
    orch.memory = memory
    orch.critic = Critic(critic_llm)  # type: ignore[arg-type]
    orch.max_steps = max_steps
    orch.critic_enabled = critic_enabled
    return orch


@pytest.mark.asyncio
async def test_orchestrator_direct_answer_no_tools(tmp_procedural_path, tmp_chroma_dir, test_embedder):
    """Planner replies with text immediately — one LLM call, no tool calls."""
    planner = StubLLM([make_text_response("The answer is 42.")])
    critic = StubLLM([])
    tools = StubTools({})

    orch = _make_orch(tmp_procedural_path, tmp_chroma_dir, planner, critic, tools, test_embedder)
    await orch.memory.initialize()

    res = await orch.run(task="what is the answer?", session_id="s1")

    assert res.success is True
    assert "42" in res.final_answer
    assert res.num_tool_calls == 0
    assert len(planner.calls) == 1
    assert len(critic.calls) == 0


@pytest.mark.asyncio
async def test_orchestrator_one_tool_then_answer(tmp_procedural_path, tmp_chroma_dir, test_embedder):
    """Planner uses one tool, reads the result, then answers."""
    planner = StubLLM(
        [
            make_tool_use_response("calculator", {"expression": "2+2"}, tool_id="tu1"),
            make_text_response("The result is 4."),
        ]
    )
    critic = StubLLM([make_text_response('{"verdict": "accept", "reasoning": "ok"}')])
    tools = StubTools(
        {"calculator": {"handler": lambda args: str(eval(args["expression"]))}}
    )

    orch = _make_orch(tmp_procedural_path, tmp_chroma_dir, planner, critic, tools, test_embedder)
    await orch.memory.initialize()

    res = await orch.run(task="compute 2+2", session_id="s1")

    assert res.success is True
    assert "4" in res.final_answer
    assert res.num_tool_calls == 1
    # planner is invoked twice: plan → tool_use, then plan → final
    assert len(planner.calls) == 2
    assert len(critic.calls) == 1  # one tool result reviewed


@pytest.mark.asyncio
async def test_orchestrator_critic_abort_stops_execution(
    tmp_procedural_path, tmp_chroma_dir, test_embedder
):
    """When critic returns 'abort', orchestrator halts without a final answer."""
    planner = StubLLM(
        [
            make_tool_use_response("bad_tool", {"x": 1}, tool_id="tu1"),
            # second response should never be requested because we abort
        ]
    )
    critic = StubLLM(
        [make_text_response('{"verdict": "abort", "reasoning": "unrecoverable"}')]
    )
    tools = StubTools({"bad_tool": {"handler": lambda _: "nonsense"}})

    orch = _make_orch(tmp_procedural_path, tmp_chroma_dir, planner, critic, tools, test_embedder)
    await orch.memory.initialize()

    res = await orch.run(task="try the bad tool", session_id="s1")

    assert res.success is False
    assert "aborted" in res.final_answer.lower()
    assert len(planner.calls) == 1


@pytest.mark.asyncio
async def test_orchestrator_critic_retry_annotates_result(
    tmp_procedural_path, tmp_chroma_dir, test_embedder
):
    """Critic 'retry' feeds suggestion into the tool_result; planner adjusts."""
    planner = StubLLM(
        [
            make_tool_use_response("search", {"q": ""}, tool_id="tu1"),
            make_tool_use_response("search", {"q": "mcp"}, tool_id="tu2"),
            make_text_response("Found: MCP is a protocol."),
        ]
    )
    critic = StubLLM(
        [
            make_text_response(
                '{"verdict": "retry", "reasoning": "empty result", '
                '"suggested_fix": "use a real query"}'
            ),
            make_text_response('{"verdict": "accept", "reasoning": "ok"}'),
        ]
    )
    tools = StubTools(
        {
            "search": {
                "handler": lambda args: "MCP is a protocol." if args.get("q") else ""
            }
        }
    )

    orch = _make_orch(tmp_procedural_path, tmp_chroma_dir, planner, critic, tools, test_embedder)
    await orch.memory.initialize()

    res = await orch.run(task="find info on MCP", session_id="s1")

    assert res.success is True
    assert "protocol" in res.final_answer.lower()
    assert res.num_tool_calls == 2
    # The second planner call (index 1) is the one that receives the
    # critic's retry annotation in the tool_result. The third call (index 2)
    # gets the accepted result and produces the final answer.
    second_call_msgs = planner.calls[1]["messages"]
    last_user = [m for m in second_call_msgs if m["role"] == "user"][-1]
    # content is a list of tool_result blocks with annotated text
    first_block = last_user["content"][0]
    assert first_block["type"] == "tool_result"
    assert "critic:" in first_block["content"]
    assert "suggestion" in first_block["content"]


@pytest.mark.asyncio
async def test_orchestrator_max_steps_terminates(tmp_procedural_path, tmp_chroma_dir, test_embedder):
    """Pathological tool-use loop must hit the step budget and stop."""
    # Planner always asks for a tool, never finishes.
    tool_responses = [
        make_tool_use_response("noop", {}, tool_id=f"tu{i}") for i in range(20)
    ]
    critic_responses = [
        make_text_response('{"verdict": "accept", "reasoning": "ok"}') for _ in range(20)
    ]

    planner = StubLLM(tool_responses)
    critic = StubLLM(critic_responses)
    tools = StubTools({"noop": {"handler": lambda _: "ok"}})

    orch = _make_orch(
        tmp_procedural_path, tmp_chroma_dir, planner, critic, tools, test_embedder, max_steps=3
    )
    await orch.memory.initialize()

    res = await orch.run(task="loop forever", session_id="s1")

    assert res.success is False
    assert "step budget" in res.final_answer.lower()
    assert res.num_tool_calls == 3


@pytest.mark.asyncio
async def test_orchestrator_disabled_critic_skips_review(
    tmp_procedural_path, tmp_chroma_dir, test_embedder
):
    """With critic disabled, the critic LLM is never invoked."""
    planner = StubLLM(
        [
            make_tool_use_response("calc", {"e": "1+1"}, tool_id="tu1"),
            make_text_response("2"),
        ]
    )
    critic = StubLLM([])  # will blow up if called — ensures it's not
    tools = StubTools({"calc": {"handler": lambda a: str(eval(a["e"]))}})

    orch = _make_orch(
        tmp_procedural_path,
        tmp_chroma_dir,
        planner,
        critic,
        tools,
        test_embedder,
        critic_enabled=False,
    )
    await orch.memory.initialize()

    res = await orch.run(task="add", session_id="s1")

    assert res.success is True
    assert "2" in res.final_answer
    assert len(critic.calls) == 0
    assert res.critic_enabled is False


@pytest.mark.asyncio
async def test_orchestrator_memory_recorded_on_success(
    tmp_procedural_path, tmp_chroma_dir, test_embedder
):
    """Successful run promotes an episodic trace + a semantic fact."""
    planner = StubLLM([make_text_response("The answer is blue.")])
    critic = StubLLM([])
    tools = StubTools({})

    orch = _make_orch(tmp_procedural_path, tmp_chroma_dir, planner, critic, tools, test_embedder)
    await orch.memory.initialize()

    await orch.run(task="what colour is the sky?", session_id="sky_test")

    # Episodic trace was populated.
    recent = await orch.memory.episodic.recent("sky_test", k=50)
    assert len(recent) >= 2  # at least plan + final

    # Semantic fact was learned.
    hits = await orch.memory.semantic.search("colour of the sky", k=3)
    assert any("blue" in h.content.lower() for h in hits)
