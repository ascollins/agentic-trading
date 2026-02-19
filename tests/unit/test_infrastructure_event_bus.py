"""Tests for the new event bus (``infrastructure/event_bus.py``).

Covers:
- publish/subscribe type-based routing.
- Write-ownership enforcement (happy + violation paths).
- Handler error isolation (dead-letter tracking, error counts).
- History and observability helpers.
- Multiple subscribers for the same event type.
"""

from __future__ import annotations

import pytest

from agentic_trading.domain.events import (
    DomainEvent,
    FillReceived,
    OrderSubmitted,
    SignalCreated,
    WRITE_OWNERSHIP,
)
from agentic_trading.infrastructure.event_bus import (
    InMemoryEventBus,
    WriteOwnershipError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus() -> InMemoryEventBus:
    return InMemoryEventBus(enforce_ownership=True)


@pytest.fixture
def bus_no_enforce() -> InMemoryEventBus:
    return InMemoryEventBus(enforce_ownership=False)


def _make_signal(**overrides) -> SignalCreated:
    defaults = dict(
        source="signal",
        strategy_id="trend_v1",
        symbol="BTCUSDT",
        direction="LONG",
        confidence=0.9,
    )
    defaults.update(overrides)
    return SignalCreated(**defaults)


def _make_fill(**overrides) -> FillReceived:
    from decimal import Decimal
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
    defaults.update(overrides)
    return FillReceived(**defaults)


# ---------------------------------------------------------------------------
# Publish / Subscribe
# ---------------------------------------------------------------------------

class TestPublishSubscribe:
    @pytest.mark.asyncio
    async def test_handler_receives_event(self, bus: InMemoryEventBus):
        received: list[DomainEvent] = []

        async def handler(event: DomainEvent) -> None:
            received.append(event)

        bus.subscribe(SignalCreated, handler)
        sig = _make_signal()
        await bus.publish(sig)

        assert len(received) == 1
        assert received[0] is sig

    @pytest.mark.asyncio
    async def test_no_crosstalk(self, bus: InMemoryEventBus):
        """Subscribing to SignalCreated doesn't receive FillReceived."""
        received: list[DomainEvent] = []

        async def handler(event: DomainEvent) -> None:
            received.append(event)

        bus.subscribe(SignalCreated, handler)
        await bus.publish(_make_fill())

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus: InMemoryEventBus):
        results_a: list[DomainEvent] = []
        results_b: list[DomainEvent] = []

        async def handler_a(event: DomainEvent) -> None:
            results_a.append(event)

        async def handler_b(event: DomainEvent) -> None:
            results_b.append(event)

        bus.subscribe(SignalCreated, handler_a)
        bus.subscribe(SignalCreated, handler_b)
        await bus.publish(_make_signal())

        assert len(results_a) == 1
        assert len(results_b) == 1

    @pytest.mark.asyncio
    async def test_publish_without_subscribers(self, bus: InMemoryEventBus):
        """Publishing to an event with no subscribers is a no-op."""
        sig = _make_signal()
        await bus.publish(sig)  # should not raise
        assert len(bus.get_history()) == 1


# ---------------------------------------------------------------------------
# Write Ownership
# ---------------------------------------------------------------------------

class TestWriteOwnership:
    @pytest.mark.asyncio
    async def test_correct_owner_succeeds(self, bus: InMemoryEventBus):
        sig = _make_signal(source="signal")
        await bus.publish(sig)
        assert len(bus.get_history()) == 1

    @pytest.mark.asyncio
    async def test_wrong_owner_raises(self, bus: InMemoryEventBus):
        sig = _make_signal(source="execution")  # wrong owner
        with pytest.raises(WriteOwnershipError, match="signal"):
            await bus.publish(sig)

    @pytest.mark.asyncio
    async def test_enforcement_disabled(self, bus_no_enforce: InMemoryEventBus):
        sig = _make_signal(source="anyone")
        await bus_no_enforce.publish(sig)  # should not raise
        assert len(bus_no_enforce.get_history()) == 1

    @pytest.mark.asyncio
    async def test_unknown_event_type_passes(self, bus: InMemoryEventBus):
        """Event types not in WRITE_OWNERSHIP are allowed through."""

        class UnregisteredEvent(DomainEvent):
            custom: str = "value"

        e = UnregisteredEvent(source="whatever")
        await bus.publish(e)
        assert len(bus.get_history()) == 1


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_handler_error_produces_dead_letter(self, bus: InMemoryEventBus):
        async def bad_handler(event: DomainEvent) -> None:
            raise ValueError("boom")

        bus.subscribe(SignalCreated, bad_handler)
        await bus.publish(_make_signal())

        assert len(bus.dead_letters) == 1
        dead_event, dead_error = bus.dead_letters[0]
        assert isinstance(dead_event, SignalCreated)
        assert "boom" in dead_error

    @pytest.mark.asyncio
    async def test_error_count_incremented(self, bus: InMemoryEventBus):
        async def bad_handler(event: DomainEvent) -> None:
            raise RuntimeError("fail")

        bus.subscribe(SignalCreated, bad_handler)
        await bus.publish(_make_signal())
        await bus.publish(_make_signal())

        counts = bus.get_error_counts()
        assert counts["SignalCreated"] == 2

    @pytest.mark.asyncio
    async def test_one_bad_handler_doesnt_block_others(self, bus: InMemoryEventBus):
        results: list[DomainEvent] = []

        async def bad_handler(event: DomainEvent) -> None:
            raise RuntimeError("fail")

        async def good_handler(event: DomainEvent) -> None:
            results.append(event)

        bus.subscribe(SignalCreated, bad_handler)
        bus.subscribe(SignalCreated, good_handler)
        await bus.publish(_make_signal())

        # Good handler still runs
        assert len(results) == 1
        # Bad handler recorded in dead letters
        assert len(bus.dead_letters) == 1

    @pytest.mark.asyncio
    async def test_clear_dead_letters(self, bus: InMemoryEventBus):
        async def bad_handler(event: DomainEvent) -> None:
            raise RuntimeError("fail")

        bus.subscribe(SignalCreated, bad_handler)
        await bus.publish(_make_signal())

        drained = bus.clear_dead_letters()
        assert len(drained) == 1
        assert len(bus.dead_letters) == 0


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    @pytest.mark.asyncio
    async def test_history_records_all(self, bus: InMemoryEventBus):
        await bus.publish(_make_signal())
        await bus.publish(_make_fill())
        assert len(bus.get_history()) == 2

    @pytest.mark.asyncio
    async def test_history_filter_by_type(self, bus: InMemoryEventBus):
        await bus.publish(_make_signal())
        await bus.publish(_make_fill())
        assert len(bus.get_history(SignalCreated)) == 1
        assert len(bus.get_history(FillReceived)) == 1

    @pytest.mark.asyncio
    async def test_clear_history(self, bus: InMemoryEventBus):
        await bus.publish(_make_signal())
        bus.clear_history()
        assert len(bus.get_history()) == 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self, bus: InMemoryEventBus):
        await bus.start()
        assert bus._running is True
        await bus.stop()
        assert bus._running is False

    @pytest.mark.asyncio
    async def test_messages_processed_count(self, bus: InMemoryEventBus):
        results: list[DomainEvent] = []

        async def handler(event: DomainEvent) -> None:
            results.append(event)

        bus.subscribe(SignalCreated, handler)
        await bus.publish(_make_signal())
        await bus.publish(_make_signal())
        assert bus.messages_processed == 2
