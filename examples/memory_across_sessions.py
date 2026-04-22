"""Demonstrates memory persistence across sessions.

Run this twice in a row. On the second run, AgentMesh should surface the fact
it learned the first time without having to call a tool again.

  python examples/memory_across_sessions.py
  python examples/memory_across_sessions.py     # second run uses semantic memory
"""

from __future__ import annotations

import asyncio

from agentmesh.config import get_settings
from agentmesh.orchestrator.core import Orchestrator
from agentmesh.utils.logging import setup_logging


async def main() -> None:
    setup_logging("INFO")
    orch = Orchestrator.from_settings(get_settings())
    await orch.initialize()
    try:
        # Run 1 — teach it a fact via a tool.
        r1 = await orch.run(
            task="Search the corpus for 'anthropic' and tell me what you find.",
            session_id="memory_demo_a",
        )
        print("\n--- Session A ---")
        print(r1.final_answer)
        print(f"tool_calls={r1.num_tool_calls}")

        # Run 2 — different session; semantic memory should carry over.
        r2 = await orch.run(
            task="What company develops the Claude family of models?",
            session_id="memory_demo_b",
        )
        print("\n--- Session B ---")
        print(r2.final_answer)
        print(f"tool_calls={r2.num_tool_calls}")
        print("\n(if tool_calls drops to 0 on the second run, semantic memory worked)")
    finally:
        await orch.close()


if __name__ == "__main__":
    asyncio.run(main())
