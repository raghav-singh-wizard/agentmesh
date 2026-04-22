"""How to point AgentMesh at your own MCP server(s).

Any MCP server speaking stdio works. Here we spin up the bundled demo server
alongside a hypothetical external one — just to show the wiring. Replace the
second MCPServerSpec with whatever server you actually want (filesystem MCP,
GitHub MCP, Slack MCP, etc.).
"""

from __future__ import annotations

import asyncio
import sys

from agentmesh.config import get_settings
from agentmesh.mcp_client.client import MCPServerSpec
from agentmesh.orchestrator.core import Orchestrator
from agentmesh.utils.logging import setup_logging


async def main() -> None:
    setup_logging("INFO")

    servers = [
        # The built-in demo server (calculator, search, KV, text_stats).
        MCPServerSpec(
            name="demo",
            command=sys.executable,
            args=["-m", "agentmesh.mcp_client.demo_server"],
        ),
        # Example of adding another server. Uncomment and adapt:
        # MCPServerSpec(
        #     name="fs",
        #     command="npx",
        #     args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        # ),
    ]

    orch = Orchestrator.from_settings(get_settings(), mcp_servers=servers)
    await orch.initialize()
    try:
        print("Available tools (namespaced by server):")
        for t in orch.tools.list_tool_names():
            print(f"  - {t}")

        result = await orch.run(
            task="List all keys currently in the KV store.",
            session_id="custom_mcp_demo",
        )
        print(f"\nAnswer: {result.final_answer}")
    finally:
        await orch.close()


if __name__ == "__main__":
    asyncio.run(main())
