"""Memory tier tests. Uses InMemoryEpisodicMemory + real file-backed
procedural + real Chroma in a tmpdir. Chroma import is heavy; we mark the
semantic tests so they can be skipped if that's undesirable."""

from __future__ import annotations

import pytest

from agentmesh.memory.base import MemoryHierarchy
from agentmesh.memory.episodic import InMemoryEpisodicMemory
from agentmesh.memory.procedural import ProceduralMemory
from agentmesh.memory.semantic import SemanticMemory
from agentmesh.utils.types import MemoryRecord, Step

# ---------------------- episodic (in-memory variant) ------------------------


@pytest.mark.asyncio
async def test_episodic_append_and_recent():
    ep = InMemoryEpisodicMemory(max_per_session=5)
    await ep.initialize()
    for i in range(7):
        await ep.append(
            MemoryRecord(session_id="s1", kind="episode", content=f"step {i}")
        )
    recent = await ep.recent("s1", k=3)
    assert [r.content for r in recent] == ["step 4", "step 5", "step 6"]
    all_recent = await ep.recent("s1", k=50)
    assert len(all_recent) == 5  # capped at max_per_session
    await ep.close()


@pytest.mark.asyncio
async def test_episodic_session_isolation():
    ep = InMemoryEpisodicMemory()
    await ep.initialize()
    await ep.append(MemoryRecord(session_id="a", kind="episode", content="from a"))
    await ep.append(MemoryRecord(session_id="b", kind="episode", content="from b"))
    assert (await ep.recent("a", 10))[0].content == "from a"
    assert (await ep.recent("b", 10))[0].content == "from b"


# ---------------------- procedural ------------------------------------------


@pytest.mark.asyncio
async def test_procedural_store_and_search(tmp_procedural_path: str):
    pm = ProceduralMemory(path=tmp_procedural_path)
    await pm.initialize()

    await pm.store(
        task_pattern="summarise a document",
        steps=["read file", "compute stats", "write summary"],
    )
    await pm.store(
        task_pattern="compute arithmetic expression",
        steps=["call calculator tool"],
    )

    hits = await pm.search("please summarise this document for me", k=3)
    assert len(hits) >= 1
    assert hits[0].task_pattern == "summarise a document"

    # Non-matching query returns nothing (no spurious hits).
    assert await pm.search("xyz123 unrelated", k=3) == []

    await pm.close()


@pytest.mark.asyncio
async def test_procedural_persists_across_reload(tmp_procedural_path: str):
    pm1 = ProceduralMemory(path=tmp_procedural_path)
    await pm1.initialize()
    await pm1.store(task_pattern="ping a server", steps=["call ping tool"])
    await pm1.close()

    pm2 = ProceduralMemory(path=tmp_procedural_path)
    await pm2.initialize()
    assert len(await pm2.all()) == 1


@pytest.mark.asyncio
async def test_procedural_merges_identical_patterns(tmp_procedural_path: str):
    pm = ProceduralMemory(path=tmp_procedural_path)
    await pm.initialize()
    id1 = await pm.store(task_pattern="do thing", steps=["step a"])
    id2 = await pm.store(task_pattern="Do Thing", steps=["step b"])  # case-insens.
    assert id1 == id2
    all_ = await pm.all()
    assert len(all_) == 1
    assert all_[0].steps == ["step b"]
    assert all_[0].uses == 1


# ---------------------- semantic (Chroma, real but in tmpdir) ---------------


@pytest.mark.asyncio
async def test_semantic_store_and_retrieve(tmp_chroma_dir: str, test_embedder):
    sm = SemanticMemory(persist_dir=tmp_chroma_dir, embedding_function=test_embedder)
    await sm.initialize()

    await sm.store(MemoryRecord(kind="fact", content="Paris is the capital of France."))
    await sm.store(MemoryRecord(kind="fact", content="Redis is an in-memory data store."))
    await sm.store(MemoryRecord(kind="fact", content="MCP stands for Model Context Protocol."))

    hits = await sm.search("what is the capital of France", k=2)
    assert any("Paris" in h.content for h in hits)

    await sm.close()


@pytest.mark.asyncio
async def test_semantic_empty_query_returns_empty(tmp_chroma_dir: str, test_embedder):
    sm = SemanticMemory(persist_dir=tmp_chroma_dir, embedding_function=test_embedder)
    await sm.initialize()
    assert await sm.search("", k=3) == []
    assert await sm.search("anything", k=3) == []  # empty collection
    await sm.close()


# ---------------------- hierarchy facade ------------------------------------


@pytest.mark.asyncio
async def test_hierarchy_retrieve_combines_all_tiers(
    tmp_procedural_path: str, tmp_chroma_dir: str, test_embedder
):
    mem = MemoryHierarchy(
        episodic=InMemoryEpisodicMemory(),
        semantic=SemanticMemory(
            persist_dir=tmp_chroma_dir, embedding_function=test_embedder
        ),
        procedural=ProceduralMemory(path=tmp_procedural_path),
    )
    await mem.initialize()

    await mem.record_step("s1", Step(kind="plan", summary="first plan"))
    await mem.learn_fact("Python was created by Guido van Rossum.")
    await mem.learn_procedure(
        task_pattern="write code in python",
        steps=["draft", "test", "commit"],
    )

    ctx = await mem.retrieve(task="I want to write code in Python", session_id="s1")
    assert len(ctx.episodic) >= 1
    # Semantic + procedural may hit depending on query overlap; at least one should.
    assert ctx.semantic or ctx.procedural
    assert not ctx.is_empty
    assert "episodic_memory" in ctx.as_prompt_block()

    await mem.close()
