"""Command-line entry point: `python -m agentmesh` or `agentmesh`."""

from __future__ import annotations

import argparse
import asyncio
import sys

from agentmesh.config import get_settings
from agentmesh.orchestrator.core import Orchestrator
from agentmesh.utils.logging import setup_logging


async def _run_task(task: str, session_id: str, no_critic: bool) -> int:
    settings = get_settings()
    if no_critic:
        settings.critic_enabled = False

    orch = Orchestrator.from_settings(settings)
    try:
        await orch.initialize()
        result = await orch.run(task=task, session_id=session_id)
        print("\n=== Final Answer ===")
        print(result.final_answer)
        print(f"\n=== Trace ({len(result.steps)} steps, {result.duration_s:.2f}s) ===")
        for i, step in enumerate(result.steps, 1):
            print(f"[{i}] {step.kind}: {step.summary}")
        return 0 if result.success else 1
    finally:
        await orch.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentmesh", description="Run an AgentMesh task")
    parser.add_argument("task", help="Natural-language task description")
    parser.add_argument("--session", default="cli", help="Session id (memory scope)")
    parser.add_argument("--no-critic", action="store_true", help="Disable critic loop")
    parser.add_argument("--log-level", default=None, help="Override AGENTMESH_LOG_LEVEL")
    args = parser.parse_args()

    setup_logging(args.log_level)
    exit_code = asyncio.run(_run_task(args.task, args.session, args.no_critic))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
