"""Unit tests for IInteractionStore implementations."""

from __future__ import annotations

import pytest

from agentic_trading.llm.envelope import (
    LLMEnvelope,
    LLMInteraction,
    LLMResult,
)
from agentic_trading.llm.store import (
    IInteractionStore,
    JsonFileInteractionStore,
    MemoryInteractionStore,
)


def _make_interaction(
    *,
    instructions: str = "Test prompt",
    trace_id: str = "",
    raw_output: str = '{"ok": true}',
) -> LLMInteraction:
    """Factory for test interactions."""
    kwargs = {"instructions": instructions}
    if trace_id:
        kwargs["trace_id"] = trace_id
    envelope = LLMEnvelope(**kwargs)
    result = LLMResult(
        envelope_id=envelope.envelope_id,
        raw_output=raw_output,
        validation_passed=True,
    )
    return LLMInteraction(envelope=envelope, result=result)


# ---------------------------------------------------------------------------
# MemoryInteractionStore
# ---------------------------------------------------------------------------


class TestMemoryInteractionStore:
    """In-memory store tests."""

    def test_implements_protocol(self):
        assert isinstance(MemoryInteractionStore(), IInteractionStore)

    @pytest.mark.asyncio
    async def test_store_and_retrieve_by_envelope_id(self):
        store = MemoryInteractionStore()
        interaction = _make_interaction()
        await store.store(interaction)

        retrieved = await store.get_by_envelope_id(
            interaction.envelope.envelope_id,
        )
        assert retrieved is not None
        assert retrieved.interaction_id == interaction.interaction_id

    @pytest.mark.asyncio
    async def test_get_by_envelope_id_returns_none_for_missing(self):
        store = MemoryInteractionStore()
        result = await store.get_by_envelope_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_trace_id(self):
        store = MemoryInteractionStore()
        i1 = _make_interaction(trace_id="trace-A")
        i2 = _make_interaction(trace_id="trace-A")
        i3 = _make_interaction(trace_id="trace-B")
        await store.store(i1)
        await store.store(i2)
        await store.store(i3)

        results = await store.get_by_trace_id("trace-A")
        assert len(results) == 2
        assert all(
            r.envelope.trace_id == "trace-A" for r in results
        )

    @pytest.mark.asyncio
    async def test_get_by_trace_id_returns_empty_for_missing(self):
        store = MemoryInteractionStore()
        results = await store.get_by_trace_id("nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_recent_returns_most_recent_first(self):
        store = MemoryInteractionStore()
        interactions = [
            _make_interaction(instructions=f"Prompt {i}")
            for i in range(5)
        ]
        for interaction in interactions:
            await store.store(interaction)

        recent = await store.recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].envelope.instructions == "Prompt 4"
        assert recent[1].envelope.instructions == "Prompt 3"
        assert recent[2].envelope.instructions == "Prompt 2"

    @pytest.mark.asyncio
    async def test_recent_with_limit_larger_than_items(self):
        store = MemoryInteractionStore()
        await store.store(_make_interaction())
        recent = await store.recent(limit=100)
        assert len(recent) == 1

    def test_items_property(self):
        store = MemoryInteractionStore()
        assert store.items == []

    def test_clear(self):
        store = MemoryInteractionStore()
        store._items.append(_make_interaction())  # noqa: SLF001
        store.clear()
        assert store.items == []


# ---------------------------------------------------------------------------
# JsonFileInteractionStore
# ---------------------------------------------------------------------------


class TestJsonFileInteractionStore:
    """JSONL file-backed store tests."""

    def test_implements_protocol(self):
        store = JsonFileInteractionStore(path="/tmp/test_dummy.jsonl")
        assert isinstance(store, IInteractionStore)

    @pytest.mark.asyncio
    async def test_store_persists_to_file(self, tmp_path):
        path = tmp_path / "interactions.jsonl"
        store = JsonFileInteractionStore(path=path)
        interaction = _make_interaction()
        await store.store(interaction)

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_load_on_init_roundtrip(self, tmp_path):
        path = tmp_path / "interactions.jsonl"

        # Store some interactions
        store1 = JsonFileInteractionStore(path=path)
        i1 = _make_interaction(instructions="First")
        i2 = _make_interaction(instructions="Second")
        await store1.store(i1)
        await store1.store(i2)

        # Load fresh instance
        store2 = JsonFileInteractionStore(path=path)
        assert len(store2.items) == 2
        assert store2.items[0].envelope.instructions == "First"
        assert store2.items[1].envelope.instructions == "Second"

    @pytest.mark.asyncio
    async def test_retrieve_by_envelope_id_after_reload(self, tmp_path):
        path = tmp_path / "interactions.jsonl"

        store1 = JsonFileInteractionStore(path=path)
        interaction = _make_interaction()
        await store1.store(interaction)

        store2 = JsonFileInteractionStore(path=path)
        retrieved = await store2.get_by_envelope_id(
            interaction.envelope.envelope_id,
        )
        assert retrieved is not None
        assert retrieved.interaction_id == interaction.interaction_id

    @pytest.mark.asyncio
    async def test_recent_ordering(self, tmp_path):
        path = tmp_path / "interactions.jsonl"
        store = JsonFileInteractionStore(path=path)

        for i in range(5):
            await store.store(
                _make_interaction(instructions=f"Prompt {i}"),
            )

        recent = await store.recent(limit=3)
        assert len(recent) == 3
        assert recent[0].envelope.instructions == "Prompt 4"

    @pytest.mark.asyncio
    async def test_get_by_trace_id_after_reload(self, tmp_path):
        path = tmp_path / "interactions.jsonl"

        store1 = JsonFileInteractionStore(path=path)
        await store1.store(_make_interaction(trace_id="trace-X"))
        await store1.store(_make_interaction(trace_id="trace-X"))
        await store1.store(_make_interaction(trace_id="trace-Y"))

        store2 = JsonFileInteractionStore(path=path)
        results = await store2.get_by_trace_id("trace-X")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_file_loads_cleanly(self, tmp_path):
        path = tmp_path / "interactions.jsonl"
        path.write_text("")
        store = JsonFileInteractionStore(path=path)
        assert store.items == []

    @pytest.mark.asyncio
    async def test_nonexistent_file_loads_cleanly(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        store = JsonFileInteractionStore(path=path)
        assert store.items == []

    @pytest.mark.asyncio
    async def test_malformed_line_skipped(self, tmp_path):
        path = tmp_path / "interactions.jsonl"
        # Write a valid interaction then a malformed line
        store = JsonFileInteractionStore(path=path)
        interaction = _make_interaction()
        await store.store(interaction)

        # Append malformed line
        with open(path, "a") as f:
            f.write("not valid json\n")

        # Reload — should have 1 valid item, skip malformed
        store2 = JsonFileInteractionStore(path=path)
        assert len(store2.items) == 1

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "interactions.jsonl"
        store = JsonFileInteractionStore(path=path)
        await store.store(_make_interaction())
        assert path.exists()
