"""Run the 50-task benchmark and dump comparable JSON + Markdown reports.

Usage:
    python -m benchmarks.run_benchmark                # full 50 tasks, both agents
    python -m benchmarks.run_benchmark --tasks 10     # first N tasks
    python -m benchmarks.run_benchmark --tasks all --agents agentmesh
    python -m benchmarks.run_benchmark --ablation     # w/ critic vs w/o critic

Important: this file deliberately does NOT invent numbers. Every metric printed
is computed from the runs this script actually performs. If you see a "2.1x"
figure in your README, it came from here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agentmesh.config import get_settings
from agentmesh.mcp_client.client import MCPServerSpec, MCPToolRegistry
from agentmesh.orchestrator.core import Orchestrator
from agentmesh.utils.logging import setup_logging
from agentmesh.utils.types import AgentResult
from benchmarks.langgraph_baseline import LangGraphBaseline
from benchmarks.tasks import TASKS, Task

console = Console()


def _demo_server_spec() -> MCPServerSpec:
    return MCPServerSpec(
        name="demo",
        command=sys.executable,
        args=["-m", "agentmesh.mcp_client.demo_server"],
    )


# -----------------------------------------------------------------------------
# per-agent runners
# -----------------------------------------------------------------------------


async def run_agentmesh(tasks: list[Task], critic: bool) -> list[dict[str, Any]]:
    settings = get_settings()
    settings.critic_enabled = critic
    orch = Orchestrator.from_settings(settings, mcp_servers=[_demo_server_spec()])
    await orch.initialize()
    results: list[dict[str, Any]] = []
    try:
        for i, task in enumerate(tasks, 1):
            console.print(
                f"[cyan]\\[AgentMesh{' +critic' if critic else ' -critic'}][/cyan] "
                f"({i}/{len(tasks)}) {task.id}",
            )
            try:
                res = await orch.run(task=task.prompt, session_id=f"bench_{task.id}")
            except Exception as e:
                console.print(f"  [red]run failed:[/red] {e}")
                res = AgentResult(
                    task=task.prompt, session_id=task.id, success=False,
                    final_answer=f"ERROR: {e}", steps=[], duration_s=0.0,
                    critic_enabled=critic,
                )
            passed = task.grader(res.final_answer, res.steps)
            results.append(_record(task, res, passed))
            console.print(f"  → {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} "
                          f"({res.duration_s:.2f}s, {res.num_tool_calls} tool calls)")
    finally:
        await orch.close()
    return results


async def run_baseline(tasks: list[Task]) -> list[dict[str, Any]]:
    tools = MCPToolRegistry([_demo_server_spec()])
    await tools.connect()
    baseline = LangGraphBaseline(tools=tools, max_steps=get_settings().max_steps)
    results: list[dict[str, Any]] = []
    try:
        for i, task in enumerate(tasks, 1):
            console.print(f"[magenta]\\[LangGraph][/magenta] ({i}/{len(tasks)}) {task.id}")
            try:
                res = await baseline.run(task=task.prompt, session_id=task.id)
            except Exception as e:
                console.print(f"  [red]run failed:[/red] {e}")
                res = AgentResult(
                    task=task.prompt, session_id=task.id, success=False,
                    final_answer=f"ERROR: {e}", steps=[], duration_s=0.0,
                    critic_enabled=False,
                )
            passed = task.grader(res.final_answer, res.steps)
            results.append(_record(task, res, passed))
            console.print(f"  → {'[green]PASS[/green]' if passed else '[red]FAIL[/red]'} "
                          f"({res.duration_s:.2f}s, {res.num_tool_calls} tool calls)")
    finally:
        await tools.close()
    return results


def _record(task: Task, res: AgentResult, passed: bool) -> dict[str, Any]:
    return {
        "id": task.id,
        "category": task.category,
        "prompt": task.prompt,
        "passed": passed,
        "duration_s": round(res.duration_s, 3),
        "tool_calls": res.num_tool_calls,
        "critic_rejections": res.num_critic_rejections,
        "tokens_in": res.tokens_in,
        "tokens_out": res.tokens_out,
        "final_answer": res.final_answer[:400],
    }


# -----------------------------------------------------------------------------
# reporting
# -----------------------------------------------------------------------------


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    n = len(records)
    passed = sum(1 for r in records if r["passed"])
    by_cat: dict[str, dict[str, int]] = {}
    for r in records:
        c = r["category"]
        d = by_cat.setdefault(c, {"n": 0, "passed": 0})
        d["n"] += 1
        d["passed"] += int(r["passed"])
    return {
        "n": n,
        "passed": passed,
        "pass_rate": round(passed / n, 3),
        "avg_duration_s": round(sum(r["duration_s"] for r in records) / n, 3),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in records) / n, 2),
        "total_tokens_in": sum(r["tokens_in"] for r in records),
        "total_tokens_out": sum(r["tokens_out"] for r in records),
        "by_category": by_cat,
    }


def _print_summary(agents: dict[str, dict[str, Any]]) -> None:
    t = Table(title="Benchmark Summary", show_lines=True)
    t.add_column("Agent", style="bold")
    t.add_column("N")
    t.add_column("Passed")
    t.add_column("Pass rate")
    t.add_column("Avg duration (s)")
    t.add_column("Avg tool calls")
    t.add_column("Tokens in")
    t.add_column("Tokens out")
    for name, agg in agents.items():
        t.add_row(
            name,
            str(agg["n"]),
            str(agg["passed"]),
            f"{agg['pass_rate']:.1%}",
            f"{agg['avg_duration_s']:.2f}",
            f"{agg['avg_tool_calls']:.2f}",
            str(agg["total_tokens_in"]),
            str(agg["total_tokens_out"]),
        )
    console.print(t)

    # By-category breakdown.
    cat_table = Table(title="Pass rate by category", show_lines=False)
    cat_table.add_column("Category")
    for name in agents:
        cat_table.add_column(name)
    cats = sorted({c for a in agents.values() for c in a["by_category"]})
    for cat in cats:
        row = [cat]
        for name in agents:
            cell = agents[name]["by_category"].get(cat)
            row.append(f"{cell['passed']}/{cell['n']}" if cell else "-")
        cat_table.add_row(*row)
    console.print(cat_table)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


async def _amain(args: argparse.Namespace) -> int:
    tasks: list[Task] = TASKS if args.tasks == "all" else TASKS[: int(args.tasks)]

    agents_to_run = set(args.agents.split(",")) if args.agents else {"agentmesh", "baseline"}
    if args.ablation:
        agents_to_run = {"agentmesh", "agentmesh_no_critic"}

    runs: dict[str, list[dict[str, Any]]] = {}

    if "agentmesh" in agents_to_run:
        runs["agentmesh"] = await run_agentmesh(tasks, critic=True)
    if "agentmesh_no_critic" in agents_to_run:
        runs["agentmesh_no_critic"] = await run_agentmesh(tasks, critic=False)
    if "baseline" in agents_to_run:
        runs["baseline"] = await run_baseline(tasks)

    aggregates = {name: _aggregate(recs) for name, recs in runs.items()}
    _print_summary(aggregates)

    # Head-to-head lift (honest — only printed if both present).
    if "agentmesh" in aggregates and "baseline" in aggregates:
        am = aggregates["agentmesh"]["pass_rate"]
        bl = aggregates["baseline"]["pass_rate"]
        if bl > 0:
            console.print(
                f"\n[bold]AgentMesh vs baseline pass-rate ratio: "
                f"{am/bl:.2f}x[/bold]  ({am:.1%} vs {bl:.1%})"
            )

    # Persist results.
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_count": len(tasks),
        "runs": runs,
        "aggregates": aggregates,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    console.print(f"\n[green]Wrote results to {out_path}[/green]")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgentMesh benchmark suite")
    parser.add_argument("--tasks", default="all", help="'all' or an integer N")
    parser.add_argument(
        "--agents",
        default=None,
        help="Comma-separated: agentmesh, baseline, agentmesh_no_critic",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run AgentMesh with and without the critic (no baseline).",
    )
    parser.add_argument(
        "--output",
        default=f"benchmarks/results/run_{int(time.time())}.json",
        help="Where to write the JSON report",
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    sys.exit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
