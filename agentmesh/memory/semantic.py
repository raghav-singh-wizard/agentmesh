"""Semantic memory: cross-session durable facts, retrievable by similarity.

Backed by a persistent Chroma collection. Uses Chroma's default embedder
(all-MiniLM-L6-v2 via `onnxruntime`) so there's no extra API call per write —
this matters for benchmark reproducibility and cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentmesh.utils.logging import get_logger
from agentmesh.utils.types import MemoryRecord

if TYPE_CHECKING:  # pragma: no cover — only for type-checkers
    import chromadb

log = get_logger(__name__)

_COLLECTION = "agentmesh_semantic"


class SemanticMemory:
    """Vector-indexed long-term memory for facts learned across sessions."""

    def __init__(self, persist_dir: str, embedding_function: Any = None) -> None:
        self._persist_dir = persist_dir
        self._embedding_function = embedding_function
        self._client: Any = None
        self._col: Any = None

    async def initialize(self) -> None:
        # Lazy import: chromadb is heavy (pulls in onnxruntime). Keeping it
        # out of module scope lets the rest of the package load even when
        # chromadb isn't installed — useful for lightweight deployments
        # that swap in a different semantic backend.
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "SemanticMemory requires `chromadb`. Install with "
                "`pip install chromadb` or disable semantic memory."
            ) from e

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        col_kwargs: dict[str, Any] = {"name": _COLLECTION}
        if self._embedding_function is not None:
            col_kwargs["embedding_function"] = self._embedding_function
        self._col = self._client.get_or_create_collection(**col_kwargs)
        log.info("memory.semantic.ready", backend="chroma", path=self._persist_dir)

    async def close(self) -> None:
        # chromadb PersistentClient flushes on write; nothing to close explicitly.
        self._client = None
        self._col = None

    async def store(self, record: MemoryRecord) -> None:
        if self._col is None:
            raise RuntimeError("SemanticMemory not initialized")
        # Chroma rejects empty metadata dicts silently, so always give it something.
        metadata = dict(record.metadata) or {"_": ""}
        metadata["timestamp"] = record.timestamp
        self._col.add(
            ids=[record.id],
            documents=[record.content],
            metadatas=[metadata],
        )

    async def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        if self._col is None or not query.strip():
            return []

        # Protect against querying an empty collection (Chroma throws).
        try:
            count = self._col.count()
        except Exception:
            count = 0
        if count == 0:
            return []

        n_results = min(k, count)
        res = self._col.query(query_texts=[query], n_results=n_results)

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0] or [{}] * len(ids)

        out: list[MemoryRecord] = []
        for rec_id, doc, meta in zip(ids, docs, metas, strict=False):
            meta = dict(meta or {})
            ts = meta.pop("timestamp", None)
            out.append(
                MemoryRecord(
                    id=rec_id,
                    kind="fact",
                    content=doc,
                    metadata=meta,
                    timestamp=ts or "",
                )
            )
        return out

    async def reset(self) -> None:
        if self._client is None or self._col is None:
            return
        self._client.delete_collection(_COLLECTION)
        self._col = self._client.get_or_create_collection(name=_COLLECTION)
