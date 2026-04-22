# Architecture

This document explains the design of AgentMesh — what the moving parts are,
why they exist, and how a single task flows through the system.

## Big picture

```
                        ┌──────────────────────────┐
User task ─────────────▶│      Orchestrator         │
                        │  (ReAct control loop)    │
                        └──────┬───────────┬───────┘
                               │           │
              ┌────────────────┘           └─────────────────┐
              ▼                                              ▼
    ┌────────────────────┐                       ┌────────────────────┐
    │   Memory Hierarchy │                       │  MCPToolRegistry   │
    │ ┌──────────────┐   │                       │  (one or more MCP  │
    │ │ Episodic     │ ◀─┼─── record_step()      │   servers over     │
    │ │ (Redis)      │   │                       │   stdio)           │
    │ └──────────────┘   │                       └────────┬───────────┘
    │ ┌──────────────┐   │                                │ tool call
    │ │ Semantic     │ ◀─┼─── learn_fact()                ▼
    │ │ (Chroma)     │   │                       ┌────────────────────┐
    │ └──────────────┘   │                       │  MCP Server(s)     │
    │ ┌──────────────┐   │                       │  - demo (bundled)  │
    │ │ Procedural   │ ◀─┼─── learn_procedure()  │  - filesystem      │
    │ │ (JSON)       │   │                       │  - github, etc.    │
    │ └──────────────┘   │                       └────────────────────┘
    └────────┬───────────┘
             │ retrieve()
             ▼                                   ┌────────────────────┐
    ┌────────────────────┐                       │   Anthropic API    │
    │   System prompt    │──── planner LLM ─────▶│   (planner model)  │
    │   (task + memory)  │                       └────────────────────┘
    └────────────────────┘

    After each tool result:
    ┌────────────────────┐   ┌────────────────────┐
    │   (task, call,     │──▶│   Critic LLM       │───▶ verdict ∈ {accept,
    │    result)         │   │   (second model)   │      retry, abort}
    └────────────────────┘   └────────────────────┘
```

## The one-task control flow

1. **Memory retrieval.** On entry to `Orchestrator.run()`, we query all three
   memory tiers for context relevant to the task. The results are stitched
   into the system prompt as `<episodic_memory>`, `<semantic_memory>`, and
   `<procedural_memory>` blocks. If a tier has nothing relevant, its block is
   omitted entirely — no empty sections, no wasted tokens.

2. **Planner call.** We send `(system_prompt, messages, tool_schemas)` to the
   planner model. Tool schemas come straight from the MCP registry — MCP's
   `inputSchema` is JSON Schema, which is what Anthropic's tool-use API
   expects, so no translation is needed.

3. **Three outcomes from the planner:**
   - **Text only** → final answer. Extract, promote to semantic memory,
     return.
   - **One or more `tool_use` blocks** → go to step 4.
   - **Neither** (shouldn't happen, but) → counted as a failed step.

4. **Tool execution.** For each tool_use, route it through `MCPToolRegistry`.
   The registry owns one subprocess per configured MCP server and routes
   calls by the `server__tool` namespace prefix.

5. **Critic review.** Before the tool result goes back to the planner, a
   second LLM reviews `(task, call, result)` and returns a structured
   verdict:
   - `accept` → proceed normally.
   - `retry` → still return the result to the planner, but annotated with the
     critic's suggested fix. The planner decides whether to try again.
   - `abort` → terminate with failure. Used sparingly — the critic is a
     helper, not a gate.

   Critic failures (timeout, malformed JSON, etc.) default to `accept` so the
   main loop never deadlocks on a critic problem.

6. **Loop.** Append results to the message history, go back to step 2.
   Hard-capped at `max_steps` iterations.

7. **Memory writeback.** When the loop exits, every step is appended to
   episodic memory, and successful final answers are promoted to semantic
   memory as `Q: <task>\nA: <answer>` records.

## Why three memory tiers

This is the phrase interviewers care about — and for good reason. A single
flat memory is fine for a demo; real agents that run for days need
differentiated storage because the tiers have **different access patterns**
and **different durability requirements**:

| Tier       | Lifetime    | Access pattern        | Backend    | Why not just use the others |
| ---------- | ----------- | --------------------- | ---------- | --------------------------- |
| Episodic   | Session     | Append / tail read    | Redis list | Vector search on a running trace is overkill; we just want "what just happened". |
| Semantic   | Forever     | Similarity search     | Chroma     | Can't do fuzzy retrieval on a Redis list; can't cheaply tail a vector store. |
| Procedural | Forever     | Pattern match by task | JSON file  | Procedures are *ordered* — similarity search flattens that structure. A small keyword index is the right shape. |

Episodic memory is **per-session** and bounded at 200 entries so long-running
sessions don't blow the context window. Semantic and procedural are
**cross-session** — facts and recipes learned for one user benefit the next.

This mirrors the distinction in cognitive architectures (Soar, ACT-R) where
memory isn't a single blob but a set of specialised stores. The pattern maps
cleanly to LLM agents: the planner needs *recent* context (episodic),
*durable* context (semantic), and *actionable templates* (procedural).

## Why a critic at all

The planner LLM is motivated to make progress. Given an empty or malformed
tool result, it will often just... keep going — write a plausible-sounding
final answer, move to the next step, paper over the failure. A dedicated
critic with a narrow job (judge one result) and a cold temperature (0.0)
catches cases the planner rationalises away.

The trade-off is **latency and cost** — one extra LLM call per tool use,
roughly 2x the budget. This is why `AGENTMESH_CRITIC_ENABLED` is a runtime
flag: you can turn it off for benchmarking (ablation) or for cost-sensitive
deployments.

Empirically (see `BENCHMARKS.md`), the critic's value shows up most on
adversarial tasks — where the naive path *looks* fine at each step but is
actually compounding errors. On clean tasks it's overhead.

## Why build on MCP specifically

The question the "senior vs junior" split really asks is: *does this project
couple to a specific tool implementation, or to a protocol?*

- **Junior approach:** hardcode a few tools (`search()`, `calculator()`)
  inside the agent codebase.
- **Mid approach:** load tools via a framework like LangChain — still coupled
  to that framework's tool abstraction.
- **Senior approach:** consume tools via an open protocol so any MCP server —
  filesystem, GitHub, Slack, a vendor's proprietary data API — drops in
  without code changes.

AgentMesh takes the last path. The `MCPToolRegistry.connect()` method
launches each configured server as a subprocess, calls `list_tools()` to
discover what's available, and flattens the result into the planner's tool
schema. Adding a new tool to the agent is a config change, not a code
change. This is the actual point of MCP, and it's what makes the agent
reusable across projects.

## What's intentionally *not* here

- **Multi-tenant auth.** Sessions are just string keys. A real product needs
  user IDs, isolation, and rate limiting — out of scope.
- **Distributed tool calls.** One orchestrator, one process, one Redis. No
  horizontal scale-out plumbing.
- **Fancy retrieval.** Semantic memory uses Chroma's default embedder; no
  re-ranking, no hybrid search. Would be a clear next step.
- **Tool selection policy.** The planner picks tools itself from the full
  list. For 100+ tools you'd want a retrieval step that narrows the candidate
  set before the planner sees it.
- **Streaming.** Responses are materialised fully before returning.

These are deliberate omissions — each is a real production concern, each
would double the code, and none of them are what the project is *about*.

## File layout quick reference

```
agentmesh/
├── orchestrator/core.py         ReAct loop, the conductor
├── mcp_client/client.py         MCPToolRegistry — the protocol glue
├── mcp_client/demo_server.py    Bundled toy MCP server for tests/examples
├── memory/base.py               MemoryHierarchy facade
├── memory/episodic.py           Redis-backed (+ in-memory for tests)
├── memory/semantic.py           Chroma-backed
├── memory/procedural.py         JSON-backed
├── critic/critic.py             Second-LLM verdict loop
├── llm/anthropic_client.py      Async Anthropic wrapper with retries
├── api/app.py                   FastAPI surface
├── config.py                    pydantic-settings config loader
└── utils/                       logging, shared types

benchmarks/
├── tasks.py                     50 tasks with graders
├── langgraph_baseline.py        Naive LangGraph agent, same tools
└── run_benchmark.py             Runner + aggregator + reporter

tests/                           Offline tests — stub LLM, stub tools
```
