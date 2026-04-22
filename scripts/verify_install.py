#!/usr/bin/env python
"""Verify a fresh install: imports clean, config loads, demo MCP server boots.

Runs quickly (no LLM calls) so you can use it as a smoke test:
    python scripts/verify_install.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make this script work whether or not the package has been pip-installed.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def check_imports() -> bool:
    print("[1/4] import check ... ", end="", flush=True)
    try:
        import agentmesh  # noqa: F401
        from agentmesh import Orchestrator  # noqa: F401
        from agentmesh.mcp_client.client import MCPToolRegistry  # noqa: F401
        from agentmesh.memory.base import MemoryHierarchy  # noqa: F401
        from agentmesh.critic.critic import Critic  # noqa: F401
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def check_config() -> bool:
    print("[2/4] config loader ... ", end="", flush=True)
    try:
        from agentmesh.config import get_settings

        s = get_settings()
        assert s.max_steps > 0
        print(f"OK (planner={s.planner_model}, max_steps={s.max_steps})")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def check_benchmark_definitions() -> bool:
    print("[3/4] benchmark tasks ... ", end="", flush=True)
    try:
        from benchmarks.tasks import TASKS

        assert len(TASKS) == 50, f"expected 50 tasks, found {len(TASKS)}"
        ids = {t.id for t in TASKS}
        assert len(ids) == 50, "duplicate task ids"
        print(f"OK ({len(TASKS)} tasks)")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


async def check_demo_server() -> bool:
    print("[4/4] demo MCP server spawns ... ", end="", flush=True)
    try:
        from agentmesh.mcp_client.client import MCPServerSpec, MCPToolRegistry
        from agentmesh.utils.types import ToolCall

        reg = MCPToolRegistry(
            [
                MCPServerSpec(
                    name="demo",
                    command=sys.executable,
                    args=["-m", "agentmesh.mcp_client.demo_server"],
                )
            ]
        )
        await reg.connect()
        try:
            res = await reg.call(
                ToolCall(name="demo__calculator", arguments={"expression": "2+2"})
            )
            assert "4" in res.content
        finally:
            await reg.close()
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False


def main() -> int:
    ok = True
    ok &= check_imports()
    ok &= check_config()
    ok &= check_benchmark_definitions()
    ok &= asyncio.run(check_demo_server())
    print()
    if ok:
        print("All checks passed. Install looks healthy.")
        return 0
    print("One or more checks failed. See messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
