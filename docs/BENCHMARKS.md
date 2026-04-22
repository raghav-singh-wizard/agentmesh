# Benchmarks

This document describes the methodology behind the numbers in the README and
how to reproduce them. **All benchmark numbers you cite publicly should come
from a run on your own machine** — don't copy figures out of this doc without
verifying.

## What's being measured

Each agent is run against the same 50 tasks against the same MCP server
(the bundled demo server). We measure:

- **Pass rate** — did the final answer clear the task's grader?
- **Average steps to completion** — how many tool calls were made?
- **Average wall-clock duration** per task.
- **Token cost** — input + output tokens (proxy for $).

## The task suite (50 total)

Tasks are split into five categories, each probing a different failure mode:

| Category       | N  | What it tests                                          |
| -------------- | -- | ------------------------------------------------------ |
| `math`         | 10 | Single-tool correctness on a deterministic tool.       |
| `search`       | 15 | Tool output parsing; turning retrieval into answers.   |
| `composite`    | 10 | Multi-tool planning; combining results across calls.   |
| `stateful`     | 8  | State across tool calls (the KV store).                |
| `adversarial`  | 7  | Error recovery, empty results, tempting shortcuts.     |

Grading is deliberately lenient (case-insensitive substring matches, numeric
tolerance) so phrasing differences don't inflate failure counts. See
`benchmarks/tasks.py` for the exact graders.

## The comparison

We compare three configurations against the same tasks:

1. **AgentMesh (full)** — orchestrator + 3 memory tiers + critic.
2. **AgentMesh (no critic)** — same agent, critic loop disabled. This is the
   *ablation* that tells us how much the critic contributes on its own.
3. **LangGraph baseline** — a minimal ReAct agent built with LangGraph. No
   memory, no critic. Same tools via the same MCP registry. This is
   representative of what a first-pass agent usually looks like.

Same planner model, same temperature, same tool list — so differences come
from orchestration, not the underlying LLM.

## Running it yourself

**Prerequisites:** `ANTHROPIC_API_KEY` set, Redis running (`make redis-up`).

```bash
# Full benchmark — both agents, all 50 tasks
make bench

# Quick smoke — first 10 tasks only
make bench-quick

# Ablation — AgentMesh with and without the critic (no baseline)
python -m benchmarks.run_benchmark --ablation

# Single agent, specific task count
python -m benchmarks.run_benchmark --tasks 20 --agents agentmesh
```

Output:

- A Rich-formatted table in the terminal (pass rate per agent, per category).
- A JSON report in `benchmarks/results/run_<timestamp>.json` with every task,
  its answer, and the trace metadata.

## How to think about the numbers honestly

A few things the project brief called out, and a few things I want to flag
before you quote any of these numbers to an interviewer:

### Expect a gain in the 1.2×–1.8× range on pass rate, not 2.1×

Most of the "multi-agent gives 2× improvement" figures floating around come
from papers using tasks specifically chosen to expose baseline weaknesses
(long-horizon planning, error recovery). A general mixed suite like ours —
where 10/50 tasks are trivial arithmetic that any agent solves — has less
headroom. If you see a 2× gain on your run, be suspicious that your baseline
is broken. If you see a 1.3× gain with most of it concentrated in the
adversarial and composite categories, that's the realistic picture.

### The critic shows up asymmetrically

In our design, the critic mostly matters when the tool result is wrong or
empty. On clean tasks the critic just adds latency and cost. The ablation
run (`--ablation`) is where you see this clearly — expect `agentmesh` and
`agentmesh_no_critic` to be very close on math/search, with the gap opening
up on adversarial tasks.

### Memory has a cold-start problem

Semantic and procedural memory contribute nothing on the first run of the
benchmark — they haven't learned anything yet. Their value is cross-session
re-use, which this benchmark doesn't explicitly measure. To see the effect,
run the ablation twice back-to-back against the same data directory: the
second run will find facts from the first run and should need fewer tool
calls on related tasks.

### Token costs will be higher

Expect 2–3× the token count vs. the baseline:

- Memory context adds input tokens every turn.
- The critic call doubles per-step LLM usage.

That's the cost of the reliability gain. A production system would tune this
per deployment — maybe run the critic only on certain tool categories, or
cap episodic context more aggressively.

## What's in the JSON report

```json
{
  "timestamp": "2026-04-21T12:34:56+00:00",
  "task_count": 50,
  "runs": {
    "agentmesh": [
      {
        "id": "math_01",
        "category": "math",
        "prompt": "Compute 237 + 418.",
        "passed": true,
        "duration_s": 2.134,
        "tool_calls": 1,
        "critic_rejections": 0,
        "tokens_in": 1523,
        "tokens_out": 87,
        "final_answer": "The result is 655."
      },
      // ... 49 more
    ],
    "baseline": [ /* ... */ ]
  },
  "aggregates": {
    "agentmesh": {
      "n": 50,
      "passed": 39,
      "pass_rate": 0.78,
      "avg_duration_s": 3.21,
      "avg_tool_calls": 1.8,
      "total_tokens_in": 82341,
      "total_tokens_out": 5678,
      "by_category": {
        "math": {"n": 10, "passed": 10},
        "search": {"n": 15, "passed": 13},
        "composite": {"n": 10, "passed": 8},
        "stateful": {"n": 8, "passed": 6},
        "adversarial": {"n": 7, "passed": 2}
      }
    },
    // ...
  }
}
```

Use the JSON for follow-up analysis — per-category pass rates, per-task
latency distributions, token-cost breakdowns. The numbers in the terminal
table are rounded for display; the JSON has the raw values.

## Interpreting failures

When a task fails, read its trace from the JSON before concluding anything
about the agent. Common patterns:

- **Grader false negative** — the agent answered correctly but in a form the
  grader didn't match. Fix the grader, not the agent.
- **Planner shortcut** — the agent answered from its own knowledge without
  calling the required tool (the adversarial `adv_01` tests this).
- **Tool error loop** — the agent hit an error and re-tried the same call.
  This is what the critic's `retry` verdict is designed to catch.
- **Step budget exhausted** — the planner kept picking tools without
  converging. Usually a planning-prompt issue, sometimes a memory-context
  collision.

## Adding your own tasks

Tasks are just `Task(id, category, prompt, grader)` in `benchmarks/tasks.py`.
Add one, re-run, share the results. The only rule: if you add tasks the
baseline can't solve at all (e.g. ones that require a real filesystem or
network), add them to a separate file so the headline number stays
comparable.
