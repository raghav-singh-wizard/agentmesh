"""Naive LangGraph baseline for head-to-head benchmarking.

We intentionally implement the *simple* LangGraph pattern — a ReAct-style
tool-calling loop with no critic, no persistent memory, no procedural recipes.
This is what a junior engineer would ship on day one, and it's what AgentMesh
needs to beat to justify its added complexity.

Tools are routed through the same MCPToolRegistry AgentMesh uses, so the
comparison is apples-to-apples at the tool layer.
"""

from __future__ import annotations

import time
from typing import Any

from typing_extensions import TypedDict

from agentmesh.config import get_settings
from agentmesh.mcp_client.client import MCPToolRegistry
from agentmesh.utils.logging import get_logger
from agentmesh.utils.types import AgentResult, Step, ToolCall

log = get_logger(__name__)


class _BaselineState(TypedDict):
    messages: list[Any]
    steps: list[Step]
    step_count: int
    tokens_in: int
    tokens_out: int


_SYSTEM = (
    "You are a helpful assistant with access to tools. Use them when needed. "
    "When you have the answer, respond directly without calling further tools."
)


class LangGraphBaseline:
    """Minimal ReAct agent built on LangGraph — no critic, no memory."""

    def __init__(self, tools: MCPToolRegistry, max_steps: int = 12) -> None:
        self._tools = tools
        self._max_steps = max_steps
        # langchain is heavy; import at construction time (after we know we
        # actually want to run the baseline) rather than at module load.
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "LangGraphBaseline requires `langgraph` and `langchain-anthropic`. "
                "Install with `pip install langgraph langchain-anthropic`."
            ) from e

        settings = get_settings()
        self._llm = ChatAnthropic(
            model=settings.planner_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
        self._graph = None  # built lazily after tools are connected

    def _build_graph(self):
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.tools import StructuredTool
        from langgraph.graph import END, StateGraph
        # Wrap each MCP tool as a LangChain StructuredTool.
        lc_tools: list[StructuredTool] = []
        for schema in self._tools.anthropic_tools():
            name = schema["name"]
            desc = schema["description"]
            input_schema = schema.get("input_schema", {"type": "object", "properties": {}})

            async def _runner(_name: str = name, **kwargs: Any) -> str:
                res = await self._tools.call(ToolCall(name=_name, arguments=kwargs))
                return res.content

            lc_tools.append(
                StructuredTool.from_function(
                    coroutine=_runner,
                    name=name,
                    description=desc,
                    args_schema=None,  # we validate via the MCP server
                )
            )

        llm_with_tools = self._llm.bind_tools(lc_tools)
        tools_by_name = {t.name: t for t in lc_tools}

        async def call_model(state: _BaselineState) -> _BaselineState:
            resp = await llm_with_tools.ainvoke(state["messages"])
            usage = getattr(resp, "usage_metadata", None) or {}
            state["tokens_in"] += int(usage.get("input_tokens", 0))
            state["tokens_out"] += int(usage.get("output_tokens", 0))
            state["messages"].append(resp)
            state["step_count"] += 1

            if getattr(resp, "tool_calls", None):
                for tc in resp.tool_calls:
                    state["steps"].append(
                        Step(
                            kind="tool_call",
                            summary=f"{tc['name']}(...)",
                            payload={"tool": tc["name"], "arguments": tc.get("args", {})},
                        )
                    )
            else:
                state["steps"].append(
                    Step(kind="final", summary=(resp.content or "")[:200])
                )
            return state

        async def call_tools(state: _BaselineState) -> _BaselineState:
            last = state["messages"][-1]
            if not isinstance(last, AIMessage) or not last.tool_calls:
                return state
            for tc in last.tool_calls:
                tool = tools_by_name.get(tc["name"])
                if tool is None:
                    content = f"Unknown tool: {tc['name']}"
                else:
                    try:
                        content = await tool.ainvoke(tc.get("args", {}))
                    except Exception as e:
                        content = f"ERROR: {e}"
                state["messages"].append(
                    ToolMessage(content=str(content), tool_call_id=tc["id"])
                )
                state["steps"].append(
                    Step(kind="tool_result", summary=str(content)[:120])
                )
            return state

        def should_continue(state: _BaselineState) -> str:
            if state["step_count"] >= self._max_steps:
                return END
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        g = StateGraph(_BaselineState)
        g.add_node("model", call_model)
        g.add_node("tools", call_tools)
        g.set_entry_point("model")
        g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
        g.add_edge("tools", "model")
        return g.compile()

    async def run(self, task: str, session_id: str = "default") -> AgentResult:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        if self._graph is None:
            self._graph = self._build_graph()

        t0 = time.perf_counter()
        init_state: _BaselineState = {
            "messages": [SystemMessage(content=_SYSTEM), HumanMessage(content=task)],
            "steps": [],
            "step_count": 0,
            "tokens_in": 0,
            "tokens_out": 0,
        }

        try:
            final_state = await self._graph.ainvoke(
                init_state, config={"recursion_limit": self._max_steps * 3}
            )
        except Exception as e:
            log.warning("baseline.failed", error=str(e))
            return AgentResult(
                task=task,
                session_id=session_id,
                success=False,
                final_answer=f"ERROR: {e}",
                steps=init_state["steps"],
                duration_s=time.perf_counter() - t0,
                critic_enabled=False,
            )

        # Extract final answer = last AIMessage content without tool_calls.
        final_answer = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        return AgentResult(
            task=task,
            session_id=session_id,
            success=bool(final_answer),
            final_answer=final_answer or "(no answer)",
            steps=final_state["steps"],
            duration_s=time.perf_counter() - t0,
            tokens_in=final_state["tokens_in"],
            tokens_out=final_state["tokens_out"],
            critic_enabled=False,
        )
