"""Bus ↔ Store integration tests.

Covers:
- Publishing via bus auto-appends to store.
- Store idempotency: duplicate event_id → stored once.
- Replay from store matches bus history.
- Ownership violation → event NOT stored.
- Store failure → event still delivered to handlers (best-effort).
- End-to-end: multi-event lifecycle → full replay.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentic_trading.domain.events import (
    DecisionApproved,
    DecisionProposed,
    FillReceived,
    OrderPlanned,
    OrderSubmitted,
    PositionUpdated,
    SignalCreated,
)
from agentic_trading.infrastructure.event_bus import (
    InMemoryEventBus,
    WriteOwnershipError,
)
from agentic_trading.infrastructure.event_store import InMemoryEventStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> InMemoryEventStore:
    return InMemoryEventStore()


@pytest.fixture
def bus(store: InMemoryEventStore) -> InMemoryEventBus:
    return InMemoryEventBus(enforce_ownership=True, event_store=store)


@pytest.fixture
def bus_no_enforce(store: InMemoryEventStore) -> InMemoryEventBus:
    return InMemoryEventBus(enforce_ownership=False, event_store=store)


# ---------------------------------------------------------------------------
# Auto-append on publish
# ---------------------------------------------------------------------------

class TestAutoAppend:
    @pytest.mark.asyncio
    async def test_publish_appends_to_store(
        self, bus: InMemoryEventBus, store: InMemoryEventStore
    ) -> None:
        sig = SignalCreated(source="signal", strategy_id="s1", symbol="X")
        await bus.publish(sig)

        stored = await store.read()
        assert len(stored) == 1
        assert stored[0].event_id == sig.event_id

    @pytest.mark.asyncio
    async def test_multiple_publishes(
        self, bus: InMemoryEventBus, store: InMemoryEventStore
    ) -> None:
        for i in range(5):
            await bus.publish(
                SignalCreated(
                    source="signal",
                    event_id=f"sig-{i}",
                    strategy_id="s1",
                    symbol="X",
                )
            )
        stored = await store.read()
        assert len(stored) == 5

    @pytest.mark.asyncio
    async def test_store_idempotency_via_bus(
        self, bus: InMemoryEventBus, store: InMemoryEventStore
    ) -> None:
        """Same event_id published twice → stored once."""
        sig = SignalCreated(
            source="signal",
            event_id="dup-1",
            strategy_id="s1",
            symbol="X",
        )
        await bus.publish(sig)

        # Bus history records both publishes, store deduplicates.
        sig2 = SignalCreated(
            source="signal",
            event_id="dup-1",
            strategy_id="s2",
            symbol="Y",
        )
        await bus.publish(sig2)

        assert len(bus.get_history()) == 2  # bus records all
        stored = await store.read()
        assert len(stored) == 1  # store deduplicates

    @pytest.mark.asyncio
    async def test_no_store_means_no_append(self) -> None:
        """Bus without store → no error, no store interaction."""
        bus = InMemoryEventBus(enforce_ownership=True, event_store=None)
        sig = SignalCreated(source="signal", strategy_id="s1", symbol="X")
        await bus.publish(sig)
        assert len(bus.get_history()) == 1
        assert bus.event_store is None


# ---------------------------------------------------------------------------
# Ownership violation → not stored
# ---------------------------------------------------------------------------

class TestOwnershipPreventsStorage:
    @pytest.mark.asyncio
    async def test_rejected_event_not_stored(
        self, bus: InMemoryEventBus, store: InMemoryEventStore
    ) -> None:
        """If ownership enforcement rejects, event must NOT reach the store."""
        bad_signal = SignalCreated(
            source="execution",  # wrong owner
            strategy_id="s1",
            symbol="X",
        )
        with pytest.raises(WriteOwnershipError):
            await bus.publish(bad_signal)

        stored = await store.read()
        assert len(stored) == 0


# ---------------------------------------------------------------------------
# Store failure → handlers still run (best-effort)
# ---------------------------------------------------------------------------

class _FailingStore:
    """Store that always raises on append."""

    async def append(self, event):
        raise RuntimeError("store is down")

    async def read(self, **kw):
        return []

    async def replay(self, **kw):
        return
        yield  # noqa: unreachable — makes it an async generator

    async def get_by_correlation(self, cid):
        return []


class TestStoreFailureResilience:
    @pytest.mark.asyncio
    async def test_handlers_still_run_when_store_fails(self) -> None:
        """Store failure must not block handler dispatch."""
        failing_store = _FailingStore()
        bus = InMemoryEventBus(
            enforce_ownership=True,
            event_store=failing_store,  # type: ignore[arg-type]
        )
        received: list = []

        async def handler(event):
            received.append(event)

        bus.subscribe(SignalCreated, handler)

        sig = SignalCreated(source="signal", strategy_id="s1", symbol="X")
        await bus.publish(sig)

        # Handler ran despite store failure
        assert len(received) == 1
        # Event still in bus history
        assert len(bus.get_history()) == 1


# ---------------------------------------------------------------------------
# Replay from store matches bus history
# ---------------------------------------------------------------------------

class TestReplayFromStore:
    @pytest.mark.asyncio
    async def test_store_replay_matches_history(
        self, bus: InMemoryEventBus, store: InMemoryEventStore
    ) -> None:
        events = [
            SignalCreated(source="signal", event_id="e1", strategy_id="s1", symbol="X"),
            SignalCreated(source="signal", event_id="e2", strategy_id="s2", symbol="Y"),
        ]
        for e in events:
            await bus.publish(e)

        # Bus history
        history_ids = [e.event_id for e in bus.get_history()]

        # Store replay
        replay_ids = []
        async for event in store.replay():
            replay_ids.append(event.event_id)

        assert history_ids == replay_ids


# ---------------------------------------------------------------------------
# End-to-end lifecycle: multi-event chain
# ---------------------------------------------------------------------------

class TestEndToEndLifecycle:
    """Simulate a mini trade lifecycle: signal → decision → approval →
    order → fill → position.  All events land in the store in order."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_in_store(
        self, bus_no_enforce: InMemoryEventBus, store: InMemoryEventStore
    ) -> None:
        """Use bus_no_enforce to publish from multiple 'sources' in one test."""
        corr = "trade-1"

        sig = SignalCreated(
            source="signal",
            event_id="ev-1",
            correlation_id=corr,
            strategy_id="trend_v1",
            symbol="BTCUSDT",
            direction="LONG",
            confidence=0.9,
        )
        dec = DecisionProposed(
            source="signal",
            event_id="ev-2",
            correlation_id=corr,
            causation_id=sig.event_id,
            strategy_id="trend_v1",
            symbol="BTCUSDT",
            side="buy",
            qty=Decimal("0.1"),
            signal_event_id=sig.event_id,
        )
        approved = DecisionApproved(
            source="policy_gate",
            event_id="ev-3",
            correlation_id=corr,
            causation_id=dec.event_id,
            decision_event_id=dec.event_id,
            sizing_multiplier=0.8,
            maturity_level="L2_gated",
            impact_tier="low",
            checks_passed=("kill_switch", "position_limit"),
        )
        planned = OrderPlanned(
            source="execution",
            event_id="ev-4",
            correlation_id=corr,
            causation_id=approved.event_id,
            decision_event_id=dec.event_id,
            order_id="ord-1",
            client_order_id="coid-1",
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            qty=Decimal("0.08"),  # 0.1 * 0.8 sizing
        )
        submitted = OrderSubmitted(
            source="execution",
            event_id="ev-5",
            correlation_id=corr,
            causation_id=planned.event_id,
            order_id="ord-1",
            client_order_id="coid-1",
            symbol="BTCUSDT",
            exchange="bybit",
        )
        fill = FillReceived(
            source="reconciliation",
            event_id="ev-6",
            correlation_id=corr,
            causation_id=submitted.event_id,
            fill_id="fill-1",
            order_id="ord-1",
            client_order_id="coid-1",
            symbol="BTCUSDT",
            exchange="bybit",
            side="buy",
            price=Decimal("50000"),
            qty=Decimal("0.08"),
            fee=Decimal("0.004"),
            fee_currency="USDT",
        )
        position = PositionUpdated(
            source="reconciliation",
            event_id="ev-7",
            correlation_id=corr,
            causation_id=fill.event_id,
            symbol="BTCUSDT",
            exchange="bybit",
            qty=Decimal("0.08"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50100"),
            unrealized_pnl=Decimal("8"),
            leverage=10,
        )

        lifecycle = [sig, dec, approved, planned, submitted, fill, position]
        for event in lifecycle:
            await bus_no_enforce.publish(event)

        # ---- Verify store contents ----
        stored = await store.read()
        assert len(stored) == 7

        stored_types = [type(e).__name__ for e in stored]
        assert stored_types == [
            "SignalCreated",
            "DecisionProposed",
            "DecisionApproved",
            "OrderPlanned",
            "OrderSubmitted",
            "FillReceived",
            "PositionUpdated",
        ]

        # ---- Verify correlation filtering ----
        correlated = await store.get_by_correlation(corr)
        assert len(correlated) == 7

        # ---- Verify causality chain ----
        assert stored[1].causation_id == stored[0].event_id  # dec ← sig
        assert stored[2].causation_id == stored[1].event_id  # approved ← dec
        assert stored[3].causation_id == stored[2].event_id  # planned ← approved
        assert stored[4].causation_id == stored[3].event_id  # submitted ← planned
        assert stored[5].causation_id == stored[4].event_id  # fill ← submitted
        assert stored[6].causation_id == stored[5].event_id  # position ← fill

    @pytest.mark.asyncio
    async def test_replay_reconstructs_lifecycle(
        self, bus_no_enforce: InMemoryEventBus, store: InMemoryEventStore
    ) -> None:
        """Replay from store yields identical event stream."""
        events = [
            SignalCreated(source="signal", event_id="a", strategy_id="s1", symbol="X"),
            FillReceived(
                source="reconciliation",
                event_id="b",
                fill_id="f1",
                order_id="o1",
                client_order_id="co1",
                symbol="X",
                exchange="bybit",
                side="buy",
                price=Decimal("100"),
                qty=Decimal("1"),
                fee=Decimal("0"),
                fee_currency="USDT",
            ),
            PositionUpdated(
                source="reconciliation",
                event_id="c",
                symbol="X",
                exchange="bybit",
                qty=Decimal("1"),
                entry_price=Decimal("100"),
                mark_price=Decimal("101"),
                unrealized_pnl=Decimal("1"),
            ),
        ]
        for e in events:
            await bus_no_enforce.publish(e)

        # Read back from store
        stored = await store.read()
        assert [e.event_id for e in stored] == ["a", "b", "c"]

        # Replay
        replayed = []
        async for event in store.replay():
            replayed.append(event)
        assert [e.event_id for e in replayed] == ["a", "b", "c"]
        assert type(replayed[0]) is SignalCreated
        assert type(replayed[1]) is FillReceived
        assert type(replayed[2]) is PositionUpdated
