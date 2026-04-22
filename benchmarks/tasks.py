"""50 agentic-task benchmark suite.

Each task has:
  - id            : stable short id for results
  - category      : math | search | composite | stateful | adversarial
  - prompt        : natural-language instruction given to the agent
  - grader        : function(final_answer: str, trace: list[Step]) -> bool

Graders are deliberately lenient (case-insensitive substring, numeric tolerance)
so small phrasing differences don't tank the score. The *tasks* are the signal;
the graders exist only to automate counting.

The tasks are designed so they:
  - can all be solved using the demo MCP server's 6 tools;
  - span single-tool and multi-tool paths;
  - include adversarial cases where the naive path fails (tool errors,
    misleading prompts, state dependencies) — this is where critic+memory
    should produce measurable wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from agentmesh.utils.types import Step


Grader = Callable[[str, list[Step]], bool]


@dataclass
class Task:
    id: str
    category: str
    prompt: str
    grader: Grader


# --------------------------- grader helpers ---------------------------------


def contains(*needles: str) -> Grader:
    """Pass if any needle appears in the final answer (case-insensitive)."""
    lowered = [n.lower() for n in needles]

    def _g(answer: str, _steps: list[Step]) -> bool:
        a = (answer or "").lower()
        return any(n in a for n in lowered)

    return _g


def contains_all(*needles: str) -> Grader:
    lowered = [n.lower() for n in needles]

    def _g(answer: str, _steps: list[Step]) -> bool:
        a = (answer or "").lower()
        return all(n in a for n in lowered)

    return _g


def number_close(target: float, tol: float = 1e-3) -> Grader:
    """Pass if any number in the final answer is within `tol` of `target`."""

    def _g(answer: str, _steps: list[Step]) -> bool:
        nums = re.findall(r"-?\d+(?:\.\d+)?", answer or "")
        return any(abs(float(n) - target) <= tol for n in nums)

    return _g


def used_tool(*tool_names: str) -> Grader:
    """Pass if the trace called at least one of the listed tools (flattened names)."""
    names = set(tool_names)

    def _g(_answer: str, steps: list[Step]) -> bool:
        for s in steps:
            if s.kind == "tool_call":
                tool = s.payload.get("tool", "")
                if tool in names or tool.split("__", 1)[-1] in names:
                    return True
        return False

    return _g


def both(g1: Grader, g2: Grader) -> Grader:
    def _g(a: str, s: list[Step]) -> bool:
        return g1(a, s) and g2(a, s)

    return _g


# ---------------------------- task list -------------------------------------


# Note: 50 tasks, split roughly 10 / 15 / 10 / 8 / 7 across categories.
TASKS: list[Task] = [
    # ---------- math (10) ----------
    Task("math_01", "math", "Compute 237 + 418.", number_close(655)),
    Task("math_02", "math", "What is 89 * 137?", number_close(12193)),
    Task("math_03", "math", "Calculate 7! (seven factorial) using the calculator.",
         number_close(5040)),
    Task("math_04", "math", "What is 2 ** 16?", number_close(65536)),
    Task("math_05", "math", "Evaluate (123 + 77) / 4.", number_close(50)),
    Task("math_06", "math", "What is 999 - 333 + 111?", number_close(777)),
    Task("math_07", "math", "Compute 15 * 15 * 15.", number_close(3375)),
    Task("math_08", "math", "What is 1000 divided by 8?", number_close(125)),
    Task("math_09", "math", "Calculate 2 ** 10 + 2 ** 8.", number_close(1280)),
    Task("math_10", "math", "Compute (50 * 4) - (30 * 3) + (20 * 2).",
         number_close(150)),

    # ---------- search (15) ----------
    Task("search_01", "search",
         "Use the search tool to tell me what MCP stands for.",
         contains("model context protocol")),
    Task("search_02", "search",
         "Search the knowledge corpus for information about Anthropic.",
         contains("anthropic", "ai safety")),
    Task("search_03", "search",
         "What is ReAct according to the knowledge corpus?",
         contains("reasoning", "action")),
    Task("search_04", "search",
         "Search for LangGraph and describe what it is.",
         contains("graph", "stateful")),
    Task("search_05", "search",
         "What does the corpus say about Redis?",
         contains("in-memory", "redis")),
    Task("search_06", "search",
         "Search for Chroma and tell me its primary purpose.",
         contains("embedding", "retrieval")),
    Task("search_07", "search",
         "When was MCP introduced, according to the knowledge corpus?",
         contains("2024")),
    Task("search_08", "search",
         "Who introduced the Model Context Protocol?",
         contains("anthropic")),
    Task("search_09", "search",
         "Search for 'react' and summarise in one sentence.",
         contains("react")),
    Task("search_10", "search",
         "Find information about a company called Anthropic.",
         contains("anthropic")),
    Task("search_11", "search",
         "Search the corpus with the keyword 'protocol'.",
         contains("protocol")),
    Task("search_12", "search",
         "Find what you can about 'embedding database' in the corpus.",
         contains("chroma")),
    Task("search_13", "search",
         "Search for 'agent' patterns in the corpus.",
         contains("react")),
    Task("search_14", "search",
         "Look up 'Claude' in the corpus.",
         contains("anthropic", "claude")),
    Task("search_15", "search",
         "Search for 'data store' concepts in the corpus.",
         contains("redis", "chroma")),

    # ---------- composite (10) — require >1 tool ----------
    Task("comp_01", "composite",
         "Search for MCP in the corpus, then report the character count of "
         "the returned description. Return just the number.",
         number_close(160, tol=60)),
    Task("comp_02", "composite",
         "Compute 45 * 12, then store the result in the KV store under key "
         "'comp_02'. Confirm by reading it back.",
         contains("540")),
    Task("comp_03", "composite",
         "Search for 'langgraph', then store its description under key 'lg'. "
         "Then retrieve it and report what was saved.",
         contains("graph")),
    Task("comp_04", "composite",
         "Get text stats for the phrase 'Model Context Protocol'. "
         "Report the word count.",
         number_close(3)),
    Task("comp_05", "composite",
         "Compute 17 * 23, then search the corpus for 'anthropic'. "
         "Finally, return both the product and one fact about Anthropic.",
         both(number_close(391), contains("anthropic"))),
    Task("comp_06", "composite",
         "Store key='name' value='Raghav' in KV. Store key='role' value='AI Engineer'. "
         "Then list all keys.",
         contains_all("name", "role")),
    Task("comp_07", "composite",
         "Get character and word counts for 'AgentMesh is an MCP orchestrator'. "
         "Report the word count.",
         number_close(4)),
    Task("comp_08", "composite",
         "Compute 100 + 200 + 300 + 400 + 500 and store the result under key "
         "'sum'. Then retrieve and confirm.",
         contains("1500")),
    Task("comp_09", "composite",
         "Search for 'react' and 'langgraph', then summarise the difference "
         "between them in one sentence.",
         contains_all("react", "langgraph")),
    Task("comp_10", "composite",
         "Compute 12 ** 3, then search for 'mcp'. Return both results.",
         both(number_close(1728), contains("protocol"))),

    # ---------- stateful (8) — depend on memory across calls ----------
    Task("stateful_01", "stateful",
         "Store key='capital' value='Tokyo' in KV. Then retrieve the value for "
         "key 'capital'.",
         contains("tokyo")),
    Task("stateful_02", "stateful",
         "Save key='pi_approx' value='3.14159' to KV. Then list keys to "
         "confirm it was saved.",
         contains("pi_approx")),
    Task("stateful_03", "stateful",
         "Store three key/value pairs: a=1, b=2, c=3. Then list all keys.",
         contains_all("a", "b", "c")),
    Task("stateful_04", "stateful",
         "Save key='task' value='benchmark' in the KV store, then retrieve it.",
         contains("benchmark")),
    Task("stateful_05", "stateful",
         "Compute 7 * 8 and save the result under key 'mult'. Then retrieve "
         "the value for 'mult'.",
         contains("56")),
    Task("stateful_06", "stateful",
         "Store key='agent' value='AgentMesh' and key='proto' value='MCP'. "
         "Then retrieve the value for 'proto'.",
         contains("mcp")),
    Task("stateful_07", "stateful",
         "Save key='k1' value='hello' then key='k2' value='world'. "
         "Retrieve k1 and k2 and return both.",
         contains_all("hello", "world")),
    Task("stateful_08", "stateful",
         "Store key='result' value='42', then overwrite with value='43'. "
         "Retrieve the final value.",
         contains("43")),

    # ---------- adversarial (7) — designed to fail naive agents ----------
    # These use ambiguous phrasing, require the critic to catch errors, or
    # tempt the agent to skip tool use. They're how we measure the critic's
    # and memory's actual contribution.
    Task("adv_01", "adversarial",
         "Do NOT guess. Use the calculator tool to compute 127 * 389 exactly.",
         both(number_close(49403), used_tool("calculator"))),
    Task("adv_02", "adversarial",
         "Evaluate the expression 2 + 2 * 3 using the calculator. Remember "
         "operator precedence.",
         number_close(8)),
    Task("adv_03", "adversarial",
         "First try to use the calculator with a clearly invalid expression "
         "like 'hello world'. If it fails, recover by computing 15 + 27 instead "
         "and return that result.",
         number_close(42)),
    Task("adv_04", "adversarial",
         "Search for a term that does not exist in the corpus: 'zzzqqq'. "
         "If no results, report that clearly.",
         contains("no", "not found", "empty", "no results")),
    Task("adv_05", "adversarial",
         "Get the value for key 'nonexistent_key_xyz' from the KV store. "
         "Report what happens.",
         contains("null", "none", "not found", "empty")),
    Task("adv_06", "adversarial",
         "Compute (50 + 50) * 2 and then divide that by 4. Return only the "
         "final number.",
         number_close(50)),
    Task("adv_07", "adversarial",
         "Using the text_stats tool, measure the reading time of the text "
         "'one two three four five'. Return the reading time in seconds.",
         number_close(1.5, tol=0.5)),
]


assert len(TASKS) == 50, f"Expected 50 tasks, got {len(TASKS)}"


def tasks_by_category() -> dict[str, list[Task]]:
    out: dict[str, list[Task]] = {}
    for t in TASKS:
        out.setdefault(t.category, []).append(t)
    return out
