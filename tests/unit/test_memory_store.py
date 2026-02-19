"""Tests for the MemoryStore component."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentic_trading.context.memory_store import (
    InMemoryMemoryStore,
    JsonFileMemoryStore,
    MemoryEntry,
    _compute_decayed_relevance,
)
from agentic_trading.core.enums import MemoryEntryType


def _make_entry(
    symbol: str = "BTC/USDT",
    entry_type: MemoryEntryType = MemoryEntryType.CMT_ASSESSMENT,
    **kwargs,
) -> MemoryEntry:
    """Create a test memory entry."""
    defaults = {
        "symbol": symbol,
        "entry_type": entry_type,
        "content": {"test": True},
        "summary": "Test entry",
        "tags": ["test"],
    }
    defaults.update(kwargs)
    return MemoryEntry(**defaults)


class TestRelevanceDecay:
    def test_no_decay_at_zero_age(self):
        entry = _make_entry(relevance_score=1.0, ttl_hours=24.0)
        score = _compute_decayed_relevance(entry, entry.timestamp)
        assert abs(score - 1.0) < 0.01

    def test_decays_to_tenth_at_ttl(self):
        entry = _make_entry(relevance_score=1.0, ttl_hours=24.0)
        future = entry.timestamp + timedelta(hours=24)
        score = _compute_decayed_relevance(entry, future)
        assert abs(score - 0.1) < 0.01

    def test_half_life_decay(self):
        entry = _make_entry(relevance_score=1.0, ttl_hours=24.0)
        half = entry.timestamp + timedelta(hours=12)
        score = _compute_decayed_relevance(entry, half)
        assert 0.2 < score < 0.5

    def test_zero_ttl(self):
        entry = _make_entry(relevance_score=1.0, ttl_hours=0.0)
        future = entry.timestamp + timedelta(hours=1)
        score = _compute_decayed_relevance(entry, future)
        assert score == 0.0


class TestInMemoryMemoryStore:
    def test_store_and_query_by_symbol(self):
        store = InMemoryMemoryStore()
        store.store(_make_entry(symbol="BTC/USDT"))
        store.store(_make_entry(symbol="ETH/USDT"))

        results = store.query(symbol="BTC/USDT")
        assert len(results) == 1
        assert results[0].symbol == "BTC/USDT"

    def test_query_by_entry_type(self):
        store = InMemoryMemoryStore()
        store.store(_make_entry(entry_type=MemoryEntryType.CMT_ASSESSMENT))
        store.store(_make_entry(entry_type=MemoryEntryType.SIGNAL))

        results = store.query(entry_type=MemoryEntryType.SIGNAL)
        assert len(results) == 1
        assert results[0].entry_type == MemoryEntryType.SIGNAL

    def test_query_by_tags(self):
        store = InMemoryMemoryStore()
        store.store(_make_entry(tags=["bullish", "high_conf"]))
        store.store(_make_entry(tags=["bearish"]))

        results = store.query(tags=["bullish"])
        assert len(results) == 1

    def test_query_by_strategy_id(self):
        store = InMemoryMemoryStore()
        store.store(_make_entry(strategy_id="trend_following"))
        store.store(_make_entry(strategy_id="mean_reversion"))

        results = store.query(strategy_id="trend_following")
        assert len(results) == 1

    def test_query_limit(self):
        store = InMemoryMemoryStore()
        for i in range(10):
            store.store(_make_entry(symbol="BTC/USDT"))

        results = store.query(symbol="BTC/USDT", limit=3)
        assert len(results) == 3

    def test_query_min_relevance_filter(self):
        store = InMemoryMemoryStore()
        # Old entry should have decayed relevance
        old_entry = _make_entry(ttl_hours=1.0)
        old_entry.timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        store.store(old_entry)

        # Recent entry should pass
        store.store(_make_entry(ttl_hours=24.0))

        results = store.query(min_relevance=0.1)
        # The old entry (2h past 1h TTL) should be filtered out
        assert len(results) == 1

    def test_query_since_filter(self):
        store = InMemoryMemoryStore()
        old = _make_entry()
        old.timestamp = datetime.now(timezone.utc) - timedelta(hours=48)
        store.store(old)

        recent = _make_entry()
        store.store(recent)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        results = store.query(since=cutoff)
        assert len(results) == 1

    def test_max_entries_eviction(self):
        store = InMemoryMemoryStore(max_entries=5)
        for i in range(10):
            store.store(_make_entry(summary=f"Entry {i}"))

        assert store.entry_count == 5

    def test_clear(self):
        store = InMemoryMemoryStore()
        store.store(_make_entry())
        store.clear()
        assert store.entry_count == 0

    def test_query_results_sorted_by_relevance(self):
        store = InMemoryMemoryStore()
        e1 = _make_entry(relevance_score=0.5)
        e2 = _make_entry(relevance_score=1.0)
        store.store(e1)
        store.store(e2)

        results = store.query()
        assert results[0].relevance_score >= results[1].relevance_score


class TestJsonFileMemoryStore:
    def test_store_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.jsonl"
            store = JsonFileMemoryStore(path)
            store.store(_make_entry(symbol="BTC/USDT"))

            results = store.query(symbol="BTC/USDT")
            assert len(results) == 1

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.jsonl"

            # Write with first instance
            store1 = JsonFileMemoryStore(path)
            store1.store(_make_entry(symbol="BTC/USDT", summary="persisted"))
            del store1

            # Read with second instance
            store2 = JsonFileMemoryStore(path)
            results = store2.query(symbol="BTC/USDT")
            assert len(results) == 1
            assert results[0].summary == "persisted"

    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "memory.jsonl"
            store = JsonFileMemoryStore(path)
            store.store(_make_entry())
            assert path.exists()

    def test_handles_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.jsonl"
            # Write a valid entry then a bad line
            entry = _make_entry()
            with open(path, "w") as f:
                f.write(entry.model_dump_json() + "\n")
                f.write("not valid json\n")

            store = JsonFileMemoryStore(path)
            assert store.entry_count == 1
