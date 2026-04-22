"""Critic: a second LLM that validates tool results before they're accepted.

Why bother?
- The planner is motivated to make progress; it will happily accept a tool
  result that's plausible-looking but wrong (empty, error, off-topic).
- Separating *generate* from *verify* is the oldest trick in the book for
  improving reliability. It's cheap (one extra call per tool use) compared to
  the cost of chasing a wrong answer for five more steps.

Output protocol: the critic returns structured JSON with a verdict in
{accept, retry, abort}. We parse it strictly; any malformed response defaults
to `accept` so we never deadlock — the critic is an assist, not a gate.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentmesh.llm.anthropic_client import AnthropicLLM
from agentmesh.utils.logging import get_logger
from agentmesh.utils.types import Critique, ToolCall, ToolResult

log = get_logger(__name__)

_SYSTEM_PROMPT = """You are a strict validator for an AI agent's tool-call results.

Your job: judge whether the tool output actually satisfies what the caller asked for, \
and return a structured verdict.

VERDICTS:
- "accept" : the result is correct and useful; the agent should proceed.
- "retry"  : the call failed or produced an obviously wrong/empty/off-topic \
result, but a different call could plausibly succeed. Explain what to change.
- "abort"  : the task cannot be completed with the available tools / inputs; \
stop trying. Use this sparingly.

OUTPUT FORMAT (exact JSON, no prose around it):
{"verdict": "accept" | "retry" | "abort",
 "reasoning": "<one short sentence>",
 "suggested_fix": "<optional: what the agent should do differently, or null>"}

Be terse. Be conservative — bias toward "accept" when the result is plausibly \
correct. Only "retry" on clear failure modes (errors, empty results, wrong type, \
nonsensical output). "abort" only when further tool calls are obviously hopeless."""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class Critic:
    """Wraps an LLM as a structured-output validator."""

    def __init__(self, llm: AnthropicLLM) -> None:
        self._llm = llm

    async def review(
        self,
        task: str,
        call: ToolCall,
        result: ToolResult,
    ) -> Critique:
        """Return a Critique for a single (call, result) pair."""

        user_prompt = self._build_user_prompt(task=task, call=call, result=result)

        try:
            resp = await self._llm.call(
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            log.warning("critic.llm_failed", error=str(e))
            return Critique(verdict="accept", reasoning=f"critic unavailable: {e}")

        return self._parse(resp.text)

    @staticmethod
    def _build_user_prompt(task: str, call: ToolCall, result: ToolResult) -> str:
        # Truncate very long tool outputs so the critic doesn't blow up latency.
        content = result.content
        if len(content) > 2000:
            content = content[:2000] + "...[truncated]"
        return (
            f"ORIGINAL TASK:\n{task}\n\n"
            f"TOOL CALL:\n  name: {call.name}\n  arguments: {json.dumps(call.arguments)}\n\n"
            f"TOOL RESULT:\n  is_error: {result.is_error}\n  content: {content}\n\n"
            "Return ONLY the JSON verdict object described in the system prompt."
        )

    @staticmethod
    def _parse(text: str) -> Critique:
        """Parse the critic's JSON output; fall back to 'accept' on any error."""
        match = _JSON_RE.search(text or "")
        if not match:
            log.debug("critic.no_json_in_response", text=text[:200])
            return Critique(verdict="accept", reasoning="unparseable critic output")

        try:
            data: dict[str, Any] = json.loads(match.group(0))
        except json.JSONDecodeError:
            return Critique(verdict="accept", reasoning="malformed JSON from critic")

        verdict = data.get("verdict", "accept")
        if verdict not in ("accept", "retry", "abort"):
            verdict = "accept"
        return Critique(
            verdict=verdict,
            reasoning=str(data.get("reasoning", ""))[:500],
            suggested_fix=(
                str(data["suggested_fix"])[:500]
                if data.get("suggested_fix")
                else None
            ),
        )
