"""Memory hierarchy composing episodic + semantic + procedural memory.

Terminology follows the Soar / ACT-R tradition as applied to LLM agents:

- *Episodic*   : what happened this session (short-term trace).
- *Semantic*   : facts learned across sessions (long-term, embedding-searchable).
- *Procedural* : how-to patterns (reusable tool sequences for recurring tasks).

The hierarchy is *composed*, not inherited: each tier is a small, self-contained
store. The orchestrator reads from all three when planning a step and writes to
the appropriate tier as new information appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentmesh.memory.episodic import EpisodicMemory
from agentmesh.memory.procedural import ProceduralMemory
from agentmesh.memory.semantic import SemanticMemory
from agentmesh.utils.logging import get_logger
from agentmesh.utils.types import MemoryRecord, Step

log = get_logger(__name__)


@dataclass
class MemoryContext:
    """Block of relevant memories retrieved for a single planning turn."""

    episodic: list[str]
    semantic: list[str]
    procedural: list[str]

    def as_prompt_block(self) -> str:
        """Render as a human-readable section for the system prompt."""
        parts: list[str] = []
        if self.episodic:
            parts.append("<episodic_memory>\n" + "\n".join(self.episodic) + "\n</episodic_memory>")
        if self.semantic:
            parts.append("<semantic_memory>\n" + "\n".join(self.semantic) + "\n</semantic_memory>")
        if self.procedural:
            parts.append(
                "<procedural_memory>\n" + "\n".join(self.procedural) + "\n</procedural_memory>"
            )
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return not (self.episodic or self.semantic or self.procedural)


class MemoryHierarchy:
    """Facade over the three memory tiers."""

    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
    ) -> None:
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

    async def initialize(self) -> None:
        await self.episodic.initialize()
        await self.semantic.initialize()
        await self.procedural.initialize()

    async def close(self) -> None:
        await self.episodic.close()
        await self.semantic.close()
        await self.procedural.close()

    # ------------------------------------------------------------------
    # read-side: pull context for planning
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        task: str,
        session_id: str,
        episodic_k: int = 10,
        semantic_k: int = 5,
        procedural_k: int = 3,
    ) -> MemoryContext:
        """Return the memory context relevant to `task` for `session_id`."""
        ep = await self.episodic.recent(session_id=session_id, k=episodic_k)
        sem = await self.semantic.search(query=task, k=semantic_k)
        proc = await self.procedural.search(task=task, k=procedural_k)
        return MemoryContext(
            episodic=[e.content for e in ep],
            semantic=[s.content for s in sem],
            procedural=[p.render() for p in proc],
        )

    # ------------------------------------------------------------------
    # write-side: called by orchestrator as a run progresses
    # ------------------------------------------------------------------

    async def record_step(self, session_id: str, step: Step) -> None:
        """Append a step to episodic memory."""
        rec = MemoryRecord(
            session_id=session_id,
            kind="episode",
            content=f"[{step.kind}] {step.summary}",
            metadata={"timestamp": step.timestamp, "payload_keys": list(step.payload)},
        )
        await self.episodic.append(rec)

    async def learn_fact(self, fact: str, metadata: dict[str, Any] | None = None) -> None:
        """Promote a durable fact into semantic memory."""
        rec = MemoryRecord(kind="fact", content=fact, metadata=metadata or {})
        await self.semantic.store(rec)
        log.info("memory.fact_learned", content=fact[:80])

    async def learn_procedure(
        self, task_pattern: str, steps: list[str], metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a reusable procedure (ordered tool-use recipe)."""
        await self.procedural.store(task_pattern=task_pattern, steps=steps, metadata=metadata)
        log.info("memory.procedure_learned", pattern=task_pattern[:80], steps=len(steps))
