"""Tests for NarrationStore — ring buffer + retrieval."""

from __future__ import annotations

import pytest

from agentic_trading.narration.schema import NarrationItem
from agentic_trading.narration.store import NarrationStore


def _make_item(script_id: str = "s1", text: str = "Hello world") -> NarrationItem:
    return NarrationItem(script_id=script_id, script_text=text)


class TestNarrationStore:
    def test_add_and_get(self):
        store = NarrationStore(max_items=10)
        item = _make_item("id-1", "Test narration")
        store.add(item)
        assert store.get("id-1") is not None
        assert store.get("id-1").script_text == "Test narration"

    def test_latest_returns_newest_first(self):
        store = NarrationStore(max_items=10)
        store.add(_make_item("a", "First"))
        store.add(_make_item("b", "Second"))
        store.add(_make_item("c", "Third"))

        latest = store.latest(limit=3)
        assert len(latest) == 3
        assert latest[0].script_id == "c"
        assert latest[1].script_id == "b"
        assert latest[2].script_id == "a"

    def test_latest_one(self):
        store = NarrationStore()
        store.add(_make_item("a", "First"))
        store.add(_make_item("b", "Second"))
        assert store.latest_one().script_id == "b"

    def test_latest_one_empty(self):
        store = NarrationStore()
        assert store.latest_one() is None

    def test_count(self):
        store = NarrationStore(max_items=10)
        assert store.count == 0
        store.add(_make_item("a"))
        assert store.count == 1
        store.add(_make_item("b"))
        assert store.count == 2

    def test_max_items_eviction(self):
        store = NarrationStore(max_items=3)
        store.add(_make_item("a"))
        store.add(_make_item("b"))
        store.add(_make_item("c"))
        store.add(_make_item("d"))  # Evicts "a"

        assert store.count == 3
        assert store.get("a") is None  # Evicted
        assert store.get("d") is not None

    def test_clear(self):
        store = NarrationStore()
        store.add(_make_item("a"))
        store.add(_make_item("b"))
        store.clear()
        assert store.count == 0
        assert store.latest_one() is None

    def test_to_json_list(self):
        store = NarrationStore()
        item = _make_item("j1", "JSON test")
        item.metadata = {"action": "ENTER", "symbol": "BTC/USDT", "regime": "trend"}
        store.add(item)

        result = store.to_json_list(limit=10)
        assert len(result) == 1
        assert result[0]["script_id"] == "j1"
        assert result[0]["script_text"] == "JSON test"
        assert result[0]["action"] == "ENTER"
        assert result[0]["symbol"] == "BTC/USDT"

    def test_to_json_list_limit(self):
        store = NarrationStore()
        for i in range(20):
            store.add(_make_item(f"item-{i}", f"Text {i}"))

        result = store.to_json_list(limit=5)
        assert len(result) == 5

    def test_latest_respects_limit(self):
        store = NarrationStore()
        for i in range(50):
            store.add(_make_item(f"item-{i}"))
        assert len(store.latest(limit=10)) == 10

    def test_get_nonexistent_returns_none(self):
        store = NarrationStore()
        assert store.get("nonexistent") is None
