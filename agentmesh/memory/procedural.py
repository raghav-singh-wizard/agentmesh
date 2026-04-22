"""Procedural memory: reusable "how to accomplish X" patterns.

We store each procedure as:
    { task_pattern, steps: [str], metadata: {...} }

Lookup is keyword-based over `task_pattern` — deliberately simple. The job of
procedural memory is to surface *candidate* recipes; the planner LLM decides
whether to apply them. Fancy retrieval isn't what makes this tier useful — the
structure (ordered step list) is.

Backed by a JSON file for persistence. Thread-safe via an asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from agentmesh.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Procedure:
    id: str
    task_pattern: str          # e.g. "summarise a document"
    steps: list[str]           # ordered tool-use description strings
    metadata: dict[str, Any] = field(default_factory=dict)
    uses: int = 0

    def render(self) -> str:
        steps_block = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.steps))
        return f"Pattern: {self.task_pattern}\nSteps:\n{steps_block}"


def _tokenise(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


class ProceduralMemory:
    """JSON-backed procedural memory with token-overlap retrieval."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._procs: list[Procedure] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._procs = [Procedure(**p) for p in raw]
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("memory.procedural.load_failed", error=str(e))
                self._procs = []
        log.info("memory.procedural.ready", path=str(self._path), count=len(self._procs))

    async def close(self) -> None:
        await self._flush()

    async def _flush(self) -> None:
        self._path.write_text(json.dumps([asdict(p) for p in self._procs], indent=2))

    async def store(
        self,
        task_pattern: str,
        steps: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        async with self._lock:
            # Merge if an identical pattern already exists — bump uses.
            for p in self._procs:
                if p.task_pattern.strip().lower() == task_pattern.strip().lower():
                    p.steps = steps
                    p.uses += 1
                    if metadata:
                        p.metadata.update(metadata)
                    await self._flush()
                    return p.id
            proc = Procedure(
                id=uuid4().hex,
                task_pattern=task_pattern.strip(),
                steps=steps,
                metadata=metadata or {},
            )
            self._procs.append(proc)
            await self._flush()
            return proc.id

    async def search(self, task: str, k: int = 3) -> list[Procedure]:
        if not self._procs:
            return []
        q_toks = _tokenise(task)
        if not q_toks:
            return []
        scored: list[tuple[float, Procedure]] = []
        for p in self._procs:
            p_toks = _tokenise(p.task_pattern)
            if not p_toks:
                continue
            overlap = len(q_toks & p_toks)
            if overlap == 0:
                continue
            # Jaccard + small usage bonus (popular recipes float up).
            score = overlap / len(q_toks | p_toks) + 0.02 * min(p.uses, 10)
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:k]]

    async def mark_used(self, proc_id: str) -> None:
        async with self._lock:
            for p in self._procs:
                if p.id == proc_id:
                    p.uses += 1
                    await self._flush()
                    return

    async def all(self) -> list[Procedure]:
        return list(self._procs)
