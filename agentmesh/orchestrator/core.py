"""The Orchestrator — the heart of AgentMesh.

Control flow per task:

    1. Retrieve memory context (episodic + semantic + procedural).
    2. Send (task, memory, tools) to the planner LLM.
    3. If planner returns a tool_use block → call it via MCPToolRegistry.
    4. Critic (second LLM) reviews the tool result:
          accept → append to history, loop.
          retry  → feed critic's suggestion back in instead of the result.
          abort  → end with failure.
    5. If planner returns a final text answer → extract, promote useful facts
       to semantic memory, done.
    6. Hard-capped at `max_steps` iterations so pathological loops terminate.

Everything is async so tool calls and critic review can overlap cleanly in
the future (we keep them sequential for now — simpler, and the latency cost
is dominated by the LLM round trips anyway).
"""

from __future__ import annotations

import time
from typing import Any

from agentmesh.config import Settings
from agentmesh.critic.critic import Critic
from agentmesh.llm.anthropic_client import AnthropicLLM, LLMResponse
from agentmesh.mcp_client.client import MCPServerSpec, MCPToolRegistry
from agentmesh.memory.base import MemoryHierarchy
from agentmesh.memory.episodic import EpisodicMemory
from agentmesh.memory.procedural import ProceduralMemory
from agentmesh.memory.semantic import SemanticMemory
from agentmesh.utils.logging import get_logger
from agentmesh.utils.types import AgentResult, Critique, Step, ToolCall, ToolResult

log = get_logger(__name__)


_PLANNER_SYSTEM = """You are AgentMesh, an autonomous task-completion agent.

You have access to a set of tools exposed over MCP (Model Context Protocol). \
To accomplish a user task, reason step by step and call tools as needed. When \
the task is complete, respond with a final text answer and stop calling tools.

GUIDELINES:
- Prefer calling tools over guessing. If a tool can answer the question, call it.
- Inspect the memory context below — it may contain facts, prior turns, or \
  known-good recipes for similar tasks. Use them.
- One tool call per step. Wait for the result before deciding the next step.
- If a tool returns an error, analyse why and try a different approach — do \
  not repeat the same call verbatim.
- When you have enough information, give a concise final answer. Do not pad."""


class Orchestrator:
    """Wires together LLM + MCP + memory + critic and runs the ReAct loop."""

    def __init__(
        self,
        llm: AnthropicLLM,
        critic_llm: AnthropicLLM,
        tools: MCPToolRegistry,
        memory: MemoryHierarchy,
        max_steps: int = 12,
        critic_enabled: bool = True,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.critic = Critic(critic_llm)
        self.max_steps = max_steps
        self.critic_enabled = critic_enabled
        self._critic_llm = critic_llm  # kept so we can close it

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        mcp_servers: list[MCPServerSpec] | None = None,
    ) -> Orchestrator:
        """Build a production-style orchestrator from env config.

        If `mcp_servers` is None, a single `demo` server (the built-in
        toy one) is registered — handy for examples/tests/benchmarks.
        """
        llm = AnthropicLLM(
            api_key=settings.anthropic_api_key,
            model=settings.planner_model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
        critic_llm = AnthropicLLM(
            api_key=settings.anthropic_api_key,
            model=settings.critic_model,
            max_tokens=512,
            temperature=0.0,
        )

        if mcp_servers is None:
            import sys

            mcp_servers = [
                MCPServerSpec(
                    name="demo",
                    command=sys.executable,
                    args=["-m", "agentmesh.mcp_client.demo_server"],
                )
            ]
        tools = MCPToolRegistry(mcp_servers)

        memory = MemoryHierarchy(
            episodic=EpisodicMemory(redis_url=settings.redis_url),
            semantic=SemanticMemory(persist_dir=settings.chroma_persist_dir),
            procedural=ProceduralMemory(path=settings.procedural_store_path),
        )
        return cls(
            llm=llm,
            critic_llm=critic_llm,
            tools=tools,
            memory=memory,
            max_steps=settings.max_steps,
            critic_enabled=settings.critic_enabled,
        )

    async def initialize(self) -> None:
        await self.tools.connect()
        await self.memory.initialize()

    async def close(self) -> None:
        await self.tools.close()
        await self.memory.close()
        await self.llm.aclose()
        await self._critic_llm.aclose()

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    async def run(self, task: str, session_id: str = "default") -> AgentResult:
        """Execute one task end-to-end and return the full trace."""

        t0 = time.perf_counter()
        steps: list[Step] = []
        messages: list[dict[str, Any]] = []
        tokens_in = tokens_out = 0

        # --- 1. memory retrieval & system prompt assembly ---
        mem_ctx = await self.memory.retrieve(task=task, session_id=session_id)
        system_prompt = _PLANNER_SYSTEM
        if not mem_ctx.is_empty:
            system_prompt += "\n\n" + mem_ctx.as_prompt_block()

        steps.append(
            Step(
                kind="plan",
                summary="Loaded memory context and initial task.",
                payload={
                    "episodic_count": len(mem_ctx.episodic),
                    "semantic_count": len(mem_ctx.semantic),
                    "procedural_count": len(mem_ctx.procedural),
                },
            )
        )

        messages.append({"role": "user", "content": task})
        tool_schemas = self.tools.anthropic_tools()

        # --- 2. ReAct loop ---
        success = False
        final_answer = ""
        aborted = False

        for _step_idx in range(self.max_steps):
            resp = await self.llm.call(
                system=system_prompt,
                messages=messages,
                tools=tool_schemas,
            )
            tokens_in += resp.tokens_in
            tokens_out += resp.tokens_out

            # No tool use → planner is finished.
            if not resp.tool_uses:
                final_answer = resp.text or "(no answer produced)"
                success = True
                steps.append(
                    Step(
                        kind="final",
                        summary=final_answer[:200],
                        payload={"stop_reason": resp.stop_reason},
                    )
                )
                break

            # Claude allows multiple tool_uses in one turn, but in practice
            # we handle them one at a time — simpler for the critic loop.
            assistant_blocks = self._assistant_blocks_from_response(resp)
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_results_blocks: list[dict[str, Any]] = []
            for tu in resp.tool_uses:
                call = ToolCall(id=tu["id"], name=tu["name"], arguments=tu["input"])
                steps.append(
                    Step(
                        kind="tool_call",
                        summary=f"{call.name}({_brief(call.arguments)})",
                        payload={"tool": call.name, "arguments": call.arguments},
                    )
                )

                result = await self.tools.call(call)
                steps.append(
                    Step(
                        kind="tool_result",
                        summary=f"{'ERROR' if result.is_error else 'ok'}: {_brief(result.content)}",
                        payload={"is_error": result.is_error},
                    )
                )

                # Critic review
                if self.critic_enabled:
                    crit = await self.critic.review(task=task, call=call, result=result)
                    steps.append(
                        Step(
                            kind="critique",
                            summary=f"{crit.verdict}: {crit.reasoning}",
                            payload={
                                "verdict": crit.verdict,
                                "reasoning": crit.reasoning,
                                "suggested_fix": crit.suggested_fix,
                            },
                        )
                    )
                    if crit.verdict == "abort":
                        steps.append(
                            Step(
                                kind="error",
                                summary="Critic aborted the task.",
                                payload={"reasoning": crit.reasoning},
                            )
                        )
                        final_answer = (
                            f"Task aborted by critic: {crit.reasoning}"
                        )
                        aborted = True
                        break
                    # "retry" → we still feed the result back, but annotate it
                    # with the critic's suggestion so the planner course-corrects.
                    effective_content = _merge_with_critique(result, crit)
                else:
                    effective_content = result.content

                tool_results_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": effective_content,
                        "is_error": result.is_error,
                    }
                )

            if aborted:
                break

            messages.append({"role": "user", "content": tool_results_blocks})

        else:
            # Loop exhausted without a final answer.
            steps.append(
                Step(
                    kind="error",
                    summary=f"Exceeded max_steps={self.max_steps}.",
                )
            )
            final_answer = "Task did not complete within the step budget."

        # --- 3. persist episodic trace & learn facts ---
        for s in steps:
            await self.memory.record_step(session_id=session_id, step=s)

        if success and final_answer:
            # Very small heuristic: promote the final answer as a fact so we
            # can retrieve it cross-session. Real systems would filter harder.
            await self.memory.learn_fact(
                fact=f"Q: {task}\nA: {final_answer}",
                metadata={"session_id": session_id},
            )

        duration = time.perf_counter() - t0
        return AgentResult(
            task=task,
            session_id=session_id,
            success=success and not aborted,
            final_answer=final_answer,
            steps=steps,
            duration_s=duration,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            critic_enabled=self.critic_enabled,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assistant_blocks_from_response(resp: LLMResponse) -> list[dict[str, Any]]:
        """Reconstruct assistant message blocks for the next turn."""
        blocks: list[dict[str, Any]] = []
        if resp.text:
            blocks.append({"type": "text", "text": resp.text})
        for tu in resp.tool_uses:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tu["id"],
                    "name": tu["name"],
                    "input": tu["input"],
                }
            )
        return blocks


def _brief(obj: Any, limit: int = 120) -> str:
    s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"


def _merge_with_critique(result: ToolResult, crit: Critique) -> str:
    """If the critic flags a retry, surface the suggestion to the planner."""
    if crit.verdict == "accept":
        return result.content
    note = f"[critic: {crit.verdict}] {crit.reasoning}"
    if crit.suggested_fix:
        note += f"\n[suggestion] {crit.suggested_fix}"
    return f"{result.content}\n\n{note}"
