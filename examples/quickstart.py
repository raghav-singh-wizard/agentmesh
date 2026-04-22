"""Quickstart: run one task end-to-end against the demo MCP server.

Prerequisites:
  - ANTHROPIC_API_KEY set in env or .env
  - Redis running on localhost:6379 (see `make redis-up`)

Run:
  python examples/quickstart.py
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
        result = await orch.run(
            task="Compute 237 * 418, then search the corpus for MCP. "
                 "Return both the product and one sentence about MCP.",
            session_id="quickstart",
        )
        print("\n=== Final Answer ===")
        print(result.final_answer)
        print(f"\nsuccess={result.success}  tool_calls={result.num_tool_calls}  "
              f"duration={result.duration_s:.2f}s")
    finally:
        await orch.close()


if __name__ == "__main__":
    asyncio.run(main())
