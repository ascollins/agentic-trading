"""Tests for the event store (``infrastructure/event_store.py``).

Covers:
- InMemoryEventStore: append, read, replay, idempotency, correlation.
- JsonFileEventStore: append, read, replay, idempotency, persistence.
- Event ordering invariant.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from agentic_trading.domain.events import (
    DomainEvent,
    FillReceived,
    PositionUpdated,
    SignalCreated,
)
from agentic_trading.infrastructure.event_store import (
    InMemoryEventStore,
    JsonFileEventStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(event_id: str = "", **kw) -> SignalCreated:
    defaults = dict(
        source="signal",
        strategy_id="s1",
        symbol="BTCUSDT",
        direction="LONG",
        confidence=0.9,
    )
    defaults.update(kw)
    if event_id:
        defaults["event_id"] = event_id
    return SignalCreated(**defaults)


def _make_fill(event_id: str = "", **kw) -> FillReceived:
    defaults = dict(
        source="reconciliation",
        fill_id="f1",
        order_id="o1",
        client_order_id="co1",
        symbol="BTCUSDT",
        exchange="bybit",
        side="buy",
        price=Decimal("50000"),
        qty=Decimal("0.1"),
        fee=Decimal("0.01"),
        fee_currency="USDT",
    )
    defaults.update(kw)
    if event_id:
        defaults["event_id"] = event_id
    return FillReceived(**defaults)


# ===========================================================================
# InMemoryEventStore
# ===========================================================================

class TestInMemoryEventStore:
    @pytest.fixture
    def store(self) -> InMemoryEventStore:
        return InMemoryEventStore()

    @pytest.mark.asyncio
    async def test_append_and_read(self, store: InMemoryEventStore):
        sig = _make_signal()
        await store.append(sig)
        events = await store.read()
        assert len(events) == 1
        assert events[0] is sig

    @pytest.mark.asyncio
    async def test_idempotent_on_event_id(self, store: InMemoryEventStore):
        sig = _make_signal(event_id="dup-1")
        await store.append(sig)
        await store.append(sig)  # duplicate
        await store.append(_make_signal(event_id="dup-1"))  # same id, different instance
        events = await store.read()
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_ordering_preserved(self, store: InMemoryEventStore):
        for i in range(10):
            await store.append(_make_signal(event_id=f"ev-{i}"))
        events = await store.read()
        ids = [e.event_id for e in events]
        assert ids == [f"ev-{i}" for i in range(10)]

    @pytest.mark.asyncio
    async def test_read_filter_by_type(self, store: InMemoryEventStore):
        await store.append(_make_signal())
        await store.append(_make_fill())
        await store.append(_make_signal())

        signals = await store.read(event_type=SignalCreated)
        fills = await store.read(event_type=FillReceived)
        assert len(signals) == 2
        assert len(fills) == 1

    @pytest.mark.asyncio
    async def test_read_filter_by_correlation(self, store: InMemoryEventStore):
        await store.append(_make_signal(correlation_id="corr-A"))
        await store.append(_make_signal(correlation_id="corr-B"))
        await store.append(_make_fill(correlation_id="corr-A"))

        events = await store.read(correlation_id="corr-A")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_read_with_limit(self, store: InMemoryEventStore):
        for i in range(20):
            await store.append(_make_signal(event_id=f"ev-{i}"))
        events = await store.read(limit=5)
        assert len(events) == 5
        assert events[0].event_id == "ev-0"

    @pytest.mark.asyncio
    async def test_read_after_sequence(self, store: InMemoryEventStore):
        for i in range(10):
            await store.append(_make_signal(event_id=f"ev-{i}"))
        events = await store.read(after_sequence=5)
        assert len(events) == 5  # indices 5, 6, 7, 8, 9
        assert events[0].event_id == "ev-5"

    @pytest.mark.asyncio
    async def test_replay(self, store: InMemoryEventStore):
        await store.append(_make_signal(event_id="ev-1"))
        await store.append(_make_fill(event_id="ev-2"))
        await store.append(_make_signal(event_id="ev-3"))

        replayed = []
        async for event in store.replay():
            replayed.append(event)
        assert len(replayed) == 3

    @pytest.mark.asyncio
    async def test_replay_filter_by_type(self, store: InMemoryEventStore):
        await store.append(_make_signal())
        await store.append(_make_fill())

        replayed = []
        async for event in store.replay(event_type=FillReceived):
            replayed.append(event)
        assert len(replayed) == 1
        assert isinstance(replayed[0], FillReceived)

    @pytest.mark.asyncio
    async def test_replay_filter_by_timestamp(self, store: InMemoryEventStore):
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, tzinfo=timezone.utc)
        t3 = datetime(2025, 12, 1, tzinfo=timezone.utc)

        await store.append(_make_signal(event_id="old", timestamp=t1))
        await store.append(_make_signal(event_id="mid", timestamp=t2))
        await store.append(_make_signal(event_id="new", timestamp=t3))

        replayed = []
        cutoff = datetime(2025, 5, 1, tzinfo=timezone.utc)
        async for event in store.replay(from_timestamp=cutoff):
            replayed.append(event)
        assert len(replayed) == 2
        assert replayed[0].event_id == "mid"

    @pytest.mark.asyncio
    async def test_get_by_correlation(self, store: InMemoryEventStore):
        await store.append(_make_signal(correlation_id="corr-X"))
        await store.append(_make_fill(correlation_id="corr-X"))
        await store.append(_make_signal(correlation_id="corr-Y"))

        events = await store.get_by_correlation("corr-X")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_clear(self, store: InMemoryEventStore):
        await store.append(_make_signal())
        assert len(store) == 1
        store.clear()
        assert len(store) == 0
        events = await store.read()
        assert events == []

    @pytest.mark.asyncio
    async def test_len(self, store: InMemoryEventStore):
        assert len(store) == 0
        await store.append(_make_signal(event_id="a"))
        await store.append(_make_fill(event_id="b"))
        assert len(store) == 2


# ===========================================================================
# JsonFileEventStore
# ===========================================================================

class TestJsonFileEventStore:
    @pytest.fixture
    def store(self, tmp_path) -> JsonFileEventStore:
        return JsonFileEventStore(tmp_path / "events.jsonl")

    @pytest.mark.asyncio
    async def test_append_and_read(self, store: JsonFileEventStore):
        sig = _make_signal(event_id="s1")
        await store.append(sig)
        events = await store.read()
        assert len(events) == 1
        assert events[0].event_id == "s1"
        assert isinstance(events[0], SignalCreated)

    @pytest.mark.asyncio
    async def test_idempotent_on_event_id(self, store: JsonFileEventStore):
        sig = _make_signal(event_id="dup-1")
        await store.append(sig)
        await store.append(sig)
        events = await store.read()
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_decimal_roundtrip(self, store: JsonFileEventStore):
        fill = _make_fill(
            event_id="f1",
            price=Decimal("50000.12345"),
            qty=Decimal("0.00001"),
        )
        await store.append(fill)
        events = await store.read()
        assert len(events) == 1
        restored = events[0]
        assert isinstance(restored, FillReceived)
        assert restored.price == Decimal("50000.12345")
        assert restored.qty == Decimal("0.00001")

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "events.jsonl"
        store1 = JsonFileEventStore(path)
        await store1.append(_make_signal(event_id="s1"))
        await store1.append(_make_fill(event_id="f1"))

        # New instance reads the same file
        store2 = JsonFileEventStore(path)
        events = await store2.read()
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_idempotency_across_restarts(self, tmp_path):
        path = tmp_path / "events.jsonl"
        store1 = JsonFileEventStore(path)
        await store1.append(_make_signal(event_id="s1"))

        # Restart: new instance, append same event_id
        store2 = JsonFileEventStore(path)
        await store2.append(_make_signal(event_id="s1"))  # dup
        events = await store2.read()
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_filter_by_type(self, store: JsonFileEventStore):
        await store.append(_make_signal(event_id="s1"))
        await store.append(_make_fill(event_id="f1"))
        signals = await store.read(event_type=SignalCreated)
        assert len(signals) == 1

    @pytest.mark.asyncio
    async def test_replay(self, store: JsonFileEventStore):
        await store.append(_make_signal(event_id="s1"))
        await store.append(_make_fill(event_id="f1"))

        replayed = []
        async for event in store.replay():
            replayed.append(event)
        assert len(replayed) == 2
        assert replayed[0].event_id == "s1"
        assert replayed[1].event_id == "f1"

    @pytest.mark.asyncio
    async def test_get_by_correlation(self, store: JsonFileEventStore):
        await store.append(_make_signal(event_id="s1", correlation_id="c1"))
        await store.append(_make_fill(event_id="f1", correlation_id="c1"))
        await store.append(_make_signal(event_id="s2", correlation_id="c2"))

        events = await store.get_by_correlation("c1")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_len(self, store: JsonFileEventStore):
        assert len(store) == 0
        await store.append(_make_signal(event_id="s1"))
        assert len(store) == 1


# ===========================================================================
# Cross-store: replay reconstructs same events
# ===========================================================================

class TestReplayReconstruction:
    """The core invariant: replaying the store yields the same events."""

    @pytest.mark.asyncio
    async def test_memory_store_replay_matches_read(self):
        store = InMemoryEventStore()
        events_in = [
            _make_signal(event_id="s1"),
            _make_fill(event_id="f1"),
            _make_signal(event_id="s2"),
        ]
        for e in events_in:
            await store.append(e)

        read_events = await store.read()
        replay_events = []
        async for e in store.replay():
            replay_events.append(e)

        assert len(read_events) == len(replay_events) == 3
        for r, p in zip(read_events, replay_events):
            assert r.event_id == p.event_id

    @pytest.mark.asyncio
    async def test_file_store_replay_matches_read(self, tmp_path):
        store = JsonFileEventStore(tmp_path / "events.jsonl")
        events_in = [
            _make_signal(event_id="s1"),
            _make_fill(event_id="f1"),
            _make_signal(event_id="s2"),
        ]
        for e in events_in:
            await store.append(e)

        read_events = await store.read()
        replay_events = []
        async for e in store.replay():
            replay_events.append(e)

        assert len(read_events) == len(replay_events) == 3
        for r, p in zip(read_events, replay_events):
            assert r.event_id == p.event_id
