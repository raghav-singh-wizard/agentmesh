# AgentMesh

**An MCP-native multi-agent orchestrator with hierarchical memory and critic-based validation.**

AgentMesh consumes tools from any [Model Context Protocol](https://modelcontextprotocol.io/)
server, plans with Claude, validates each tool result with a second LLM, and
remembers what it learns across sessions in three distinct memory tiers.

---

## Why this project exists

Most public agent examples are fine for a demo and fall over in production.
The usual failure modes:

- Tools are hardcoded into the agent, so adding a new data source means
  a code change.
- The agent accepts whatever a tool returns — empty strings, errors,
  off-topic blobs — and papers over the failure in its final answer.
- "Memory" is a single list, which means recent context, long-term facts,
  and reusable task recipes all compete for the same slots.

AgentMesh addresses these three things directly:

1. **MCP as the tool interface.** Any MCP server plugs in. Tools aren't
   compiled into the agent.
2. **A critic loop.** A second LLM reviews every tool result before the
   planner sees it. Errors get caught at the step they happen, not five
   steps later.
3. **Three memory tiers** — episodic (what happened this session),
   semantic (facts across sessions, vector-searchable), and procedural
   (reusable how-to patterns). Mirrors the Soar/ACT-R distinction applied
   to LLM agents.

---

## What's in the repo

```
agentmesh/
├── agentmesh/            core library (orchestrator, memory, critic, MCP client)
├── benchmarks/           50-task suite + LangGraph baseline + runner
├── examples/             quickstart, memory demo, custom MCP server wiring
├── tests/                offline unit/integration tests (no network)
├── docs/
│   ├── ARCHITECTURE.md   design explanation, control flow, diagrams
│   └── BENCHMARKS.md     methodology, how to reproduce
├── Dockerfile, docker-compose.yml
├── Makefile              common commands
└── pyproject.toml
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design walkthrough
and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the benchmark methodology.

---

## Quick start

**Requirements:** Python 3.10+, Redis, an Anthropic API key.

```bash
# 1. clone + install
git clone https://github.com/<you>/agentmesh.git
cd agentmesh
pip install -r requirements.txt
pip install -e .

# 2. configure
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY

# 3. start redis (via docker)
make redis-up

# 4. run a task
python -m agentmesh "Compute 237 * 418, then search the corpus for MCP."
```

That's it. The `-m agentmesh` command runs the orchestrator, connects to
the bundled demo MCP server, and prints a final answer plus the execution
trace.

### Or: one-line Python

```python
import asyncio
from agentmesh import Orchestrator
from agentmesh.config import get_settings

async def main():
    orch = Orchestrator.from_settings(get_settings())
    await orch.initialize()
    try:
        result = await orch.run(
            task="What is 15 * 23?",
            session_id="my-session",
        )
        print(result.final_answer)
    finally:
        await orch.close()

asyncio.run(main())
```

### Or: as an HTTP service

```bash
make serve
# → POST http://localhost:8000/run
#    { "task": "...", "session_id": "...", "critic_enabled": true }
```

`GET /health` returns the registered tool list. Full OpenAPI docs at
`/docs`.

---

## How it works (one-screen version)

```
task ──▶ retrieve memory ──▶ planner LLM ──┬── tool_use ──▶ MCP server
                                           │                    │
                                           │               tool result
                                           │                    │
                                           │                 critic LLM
                                           │                    │
                                           │     accept / retry (annotate) / abort
                                           │                    │
                                           │◀────────────── feedback loop
                                           │
                                           └── final text ──▶ promote to semantic memory ──▶ done
```

Control is a ReAct loop, hard-capped at `AGENTMESH_MAX_STEPS` iterations so
pathological plans always terminate.

Each tier of memory is a separate store because they have different access
patterns:

| Tier       | Backend    | Lifetime   | Why not merge into one |
| ---------- | ---------- | ---------- | ---------------------- |
| Episodic   | Redis list | Session    | Tail reads, no similarity needed |
| Semantic   | Chroma     | Forever    | Cross-session similarity search |
| Procedural | JSON       | Forever    | Ordered recipes — structure matters |

The critic is a *second* Claude call with `temperature=0` and a narrow job:
return one of `accept | retry | abort` for a `(task, tool_call, result)`
triple. If the critic itself fails (timeout, malformed output), we default
to `accept` so the main loop never deadlocks on a helper.

Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Benchmarks

The `benchmarks/` directory contains 50 tasks across five categories (math,
search, composite multi-tool, stateful, adversarial) and a LangGraph ReAct
baseline that uses the exact same tools.

```bash
# full head-to-head: AgentMesh vs LangGraph baseline
make bench

# ablation: AgentMesh with and without the critic loop
python -m benchmarks.run_benchmark --ablation

# quick smoke — 10 tasks
make bench-quick
```

The runner writes a JSON report to `benchmarks/results/run_<timestamp>.json`
(schema in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)) and prints a table
like:

```
                     Benchmark Summary
┌───────────────┬────┬────────┬───────────┬────────────┬───────────┐
│ Agent         │ N  │ Passed │ Pass rate │ Avg steps  │ Tokens in │
├───────────────┼────┼────────┼───────────┼────────────┼───────────┤
│ agentmesh     │ 50 │  ...   │   ...     │    ...     │   ...     │
│ baseline      │ 50 │  ...   │   ...     │    ...     │   ...     │
└───────────────┴────┴────────┴───────────┴────────────┴───────────┘
```

### Results (Claude Sonnet 4.5, 50 tasks, April 2026)

| Agent              | Pass rate  | Avg tool calls | Input tokens |
|--------------------|------------|----------------|--------------|
| AgentMesh (full)   | **88.0%**  | 1.56           | 149k         |
| LangGraph baseline | 40.0%      | 5.34           | 361k         |

**2.20× pass-rate improvement**, strongest on stateful (8/8 vs 1/8) and 
composite multi-tool (9/10 vs 3/10) tasks. AgentMesh also used **3.4× fewer 
tool calls** and **2.4× fewer input tokens** per task.

---

## Pluggable MCP servers

The bundled demo server (calculator, search, KV store, text stats) exists
for examples and tests. To plug in a real server, construct `MCPServerSpec`s
and pass them to `Orchestrator.from_settings`:

```python
from agentmesh.mcp_client.client import MCPServerSpec

servers = [
    MCPServerSpec(
        name="fs",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
    ),
    MCPServerSpec(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "..."},
    ),
]
orch = Orchestrator.from_settings(settings, mcp_servers=servers)
```

Tools are namespaced as `<server>__<tool>` so they never collide.
Discovery happens at connect time — the agent never has hardcoded
knowledge of what tools exist.

See [`examples/custom_mcp_server.py`](examples/custom_mcp_server.py) for
a complete example.

---

## Development

```bash
make dev           # install in editable mode + dev deps
make test          # pytest, offline (no LLM calls, no network)
make lint          # ruff
make format        # ruff --fix
make clean         # remove caches
```

The test suite runs fully offline using a stub LLM. Real-LLM end-to-end
verification happens through the benchmark runner.

### Running with Docker

```bash
# just Redis
make redis-up

# full stack (Redis + API server)
docker compose --profile full up
```

---

## Configuration reference

All configuration is via environment variables, loaded from `.env` if
present. See [`.env.example`](.env.example) for the complete list.

Most commonly tuned:

| Variable                          | Default            | What it does                           |
| --------------------------------- | ------------------ | -------------------------------------- |
| `ANTHROPIC_API_KEY`               | *(required)*       | Your Claude API key.                   |
| `AGENTMESH_PLANNER_MODEL`         | `claude-sonnet-4-5`| Model used for planning + final answer.|
| `AGENTMESH_CRITIC_MODEL`          | `claude-sonnet-4-5`| Model used for the critic loop.        |
| `AGENTMESH_CRITIC_ENABLED`        | `true`             | Set `false` for ablation / cost cuts.  |
| `AGENTMESH_MAX_STEPS`             | `12`               | Hard cap on tool-call iterations.      |
| `REDIS_URL`                       | `redis://localhost:6379/0` | Episodic memory backend.       |
| `CHROMA_PERSIST_DIR`              | `./data/chroma`    | Semantic memory vector store path.     |
| `PROCEDURAL_STORE_PATH`           | `./data/procedural.json` | Procedural memory JSON path.    |

---

## Roadmap / known limitations

In rough priority order:

- **Tool retrieval step** for large tool sets (100+). Right now every tool
  schema goes into every planner call, which is fine for 6 tools and bad
  for 100.
- **Streaming** final answers — currently the API materialises the full
  response before returning.
- **Token-budgeted memory context** — today the episodic/semantic/procedural
  retrieval uses fixed counts. A smarter version would budget by tokens.
- **Richer procedural memory** — currently keyword-overlap retrieval; a
  parameterised template format would be more useful.
- **Multi-user isolation** — session IDs are just strings. Production use
  needs real auth and rate limits.

---

## License

MIT. See [`LICENSE`](LICENSE).
