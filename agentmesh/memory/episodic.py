"""Episodic memory: short-term, session-scoped trace of what happened.

Backed by Redis lists. One list per session. Bounded in length so the prompt
context doesn't explode on long-running sessions — we keep the most recent N
entries (configurable).

If Redis isn't available, callers can substitute `InMemoryEpisodicMemory`
(used by the test suite and the fallback path).
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Deque

import redis.asyncio as redis_async

from agentmesh.utils.logging import get_logger
from agentmesh.utils.types import MemoryRecord

log = get_logger(__name__)


class EpisodicMemory:
    """Redis-backed per-session episodic log."""

    def __init__(self, redis_url: str, max_per_session: int = 200) -> None:
        self._url = redis_url
        self._max = max_per_session
        self._client: redis_async.Redis | None = None

    async def initialize(self) -> None:
        self._client = redis_async.from_url(self._url, decode_responses=True)
        await self._client.ping()
        log.info("memory.episodic.ready", backend="redis", url=self._url)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _key(self, session_id: str) -> str:
        return f"agentmesh:ep:{session_id}"

    async def append(self, record: MemoryRecord) -> None:
        if self._client is None:
            raise RuntimeError("EpisodicMemory not initialized")
        assert record.session_id is not None, "episodic record requires session_id"
        key = self._key(record.session_id)
        payload = json.dumps(record.model_dump())
        pipe = self._client.pipeline()
        pipe.rpush(key, payload)
        pipe.ltrim(key, -self._max, -1)
        await pipe.execute()

    async def recent(self, session_id: str, k: int = 10) -> list[MemoryRecord]:
        if self._client is None:
            raise RuntimeError("EpisodicMemory not initialized")
        raw = await self._client.lrange(self._key(session_id), -k, -1)
        return [MemoryRecord(**json.loads(x)) for x in raw]

    async def clear(self, session_id: str) -> None:
        if self._client is None:
            return
        await self._client.delete(self._key(session_id))


class InMemoryEpisodicMemory:
    """Drop-in replacement for tests / no-Redis environments."""

    def __init__(self, max_per_session: int = 200) -> None:
        self._max = max_per_session
        self._store: dict[str, Deque[MemoryRecord]] = defaultdict(
            lambda: deque(maxlen=max_per_session)
        )

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        self._store.clear()

    async def append(self, record: MemoryRecord) -> None:
        assert record.session_id is not None
        self._store[record.session_id].append(record)

    async def recent(self, session_id: str, k: int = 10) -> list[MemoryRecord]:
        return list(self._store.get(session_id, []))[-k:]

    async def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
