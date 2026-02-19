"""Tests for the BusManager facade.

Tests cover:
1. BusManager.from_config construction and wiring
2. Component accessors
3. Lifecycle (start/stop)
4. Legacy bus — publish / subscribe
5. Domain bus — publish / subscribe
6. Observability — legacy bus
7. Observability — domain bus
8. Unified metrics
9. Schema registry helpers
10. Package-level import
"""

from __future__ import annotations

import pytest

from agentic_trading.core.enums import Mode
from agentic_trading.core.events import CandleEvent, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal() -> Signal:
    """Create a minimal Signal event."""
    from agentic_trading.core.enums import SignalDirection, Timeframe

    return Signal(
        strategy_id="test",
        symbol="BTC/USDT",
        direction=SignalDirection.LONG,
        confidence=0.8,
        timeframe=Timeframe.M1,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestBusManagerFactory:
    """BusManager.from_config produces correctly wired instances."""

    def test_from_config_backtest(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config(mode=Mode.BACKTEST)
        assert mgr is not None
        assert mgr.legacy_bus is not None

    def test_from_config_default_is_backtest(self):
        from agentic_trading.bus.manager import BusManager
        from agentic_trading.bus.memory_bus import MemoryEventBus

        mgr = BusManager.from_config()
        assert isinstance(mgr.legacy_bus, MemoryEventBus)

    def test_from_config_no_domain_bus_by_default(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        assert mgr.domain_bus is None

    def test_from_config_with_domain_bus(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config(enable_domain_bus=True)
        assert mgr.domain_bus is not None

    def test_from_config_domain_bus_ownership(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config(
            enable_domain_bus=True,
            enforce_ownership=False,
        )
        assert mgr.domain_bus is not None
        assert not mgr.domain_bus._enforce_ownership


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestBusManagerLifecycle:
    """BusManager lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        await mgr.start()
        assert mgr.is_running
        await mgr.stop()
        assert not mgr.is_running

    @pytest.mark.asyncio
    async def test_start_stop_with_domain_bus(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config(enable_domain_bus=True)
        await mgr.start()
        assert mgr.is_running
        await mgr.stop()
        assert not mgr.is_running


# ---------------------------------------------------------------------------
# Legacy bus — publish / subscribe
# ---------------------------------------------------------------------------

class TestLegacyBusPubSub:
    """BusManager delegates publish/subscribe to legacy bus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        await mgr.start()

        received = []

        async def handler(event):
            received.append(event)

        await mgr.subscribe("strategy.signal", "test_group", handler)

        sig = _make_signal()
        await mgr.publish("strategy.signal", sig)

        assert len(received) == 1
        assert received[0].strategy_id == "test"

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_publish_multiple_handlers(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        await mgr.start()

        received_a = []
        received_b = []

        async def handler_a(event):
            received_a.append(event)

        async def handler_b(event):
            received_b.append(event)

        await mgr.subscribe("strategy.signal", "group_a", handler_a)
        await mgr.subscribe("strategy.signal", "group_b", handler_b)

        await mgr.publish("strategy.signal", _make_signal())

        assert len(received_a) == 1
        assert len(received_b) == 1

        await mgr.stop()


# ---------------------------------------------------------------------------
# Domain bus — publish / subscribe
# ---------------------------------------------------------------------------

class TestDomainBusPubSub:
    """BusManager delegates domain event operations."""

    @pytest.mark.asyncio
    async def test_publish_domain_raises_when_disabled(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        with pytest.raises(RuntimeError, match="Domain bus is not enabled"):
            await mgr.publish_domain(object())

    def test_subscribe_domain_raises_when_disabled(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        with pytest.raises(RuntimeError, match="Domain bus is not enabled"):
            mgr.subscribe_domain(object, lambda e: None)

    @pytest.mark.asyncio
    async def test_domain_bus_publish_subscribe(self):
        from agentic_trading.bus.manager import BusManager
        from agentic_trading.domain.events import SignalCreated

        mgr = BusManager.from_config(
            enable_domain_bus=True,
            enforce_ownership=False,
        )
        await mgr.start()

        received = []

        async def handler(event):
            received.append(event)

        mgr.subscribe_domain(SignalCreated, handler)

        evt = SignalCreated(
            source="signal",
            strategy_id="trend",
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,
        )
        await mgr.publish_domain(evt)
        assert len(received) == 1
        assert received[0].strategy_id == "trend"

        await mgr.stop()


# ---------------------------------------------------------------------------
# Observability — legacy bus
# ---------------------------------------------------------------------------

class TestLegacyObservability:
    """BusManager exposes legacy bus observability."""

    @pytest.mark.asyncio
    async def test_messages_processed_counts(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        await mgr.start()

        async def handler(event):
            pass

        await mgr.subscribe("strategy.signal", "test", handler)
        await mgr.publish("strategy.signal", _make_signal())

        assert mgr.messages_processed >= 1

        await mgr.stop()

    def test_error_counts_empty(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        assert mgr.get_error_counts() == {}

    def test_dead_letters_empty(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        assert mgr.get_dead_letters() == []

    def test_clear_dead_letters_empty(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        result = mgr.clear_dead_letters()
        assert result == []


# ---------------------------------------------------------------------------
# Observability — domain bus
# ---------------------------------------------------------------------------

class TestDomainObservability:
    """BusManager exposes domain bus observability."""

    def test_domain_error_counts_disabled(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        assert mgr.get_domain_error_counts() == {}

    def test_domain_dead_letters_disabled(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        assert mgr.get_domain_dead_letters() == []

    def test_domain_messages_processed_disabled(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        assert mgr.domain_messages_processed == 0

    @pytest.mark.asyncio
    async def test_domain_messages_processed_enabled(self):
        from agentic_trading.bus.manager import BusManager
        from agentic_trading.domain.events import SignalCreated

        mgr = BusManager.from_config(
            enable_domain_bus=True,
            enforce_ownership=False,
        )
        await mgr.start()

        async def handler(event):
            pass

        mgr.subscribe_domain(SignalCreated, handler)
        evt = SignalCreated(
            source="signal",
            strategy_id="trend",
            symbol="BTC/USDT",
            direction="long",
            confidence=0.8,
        )
        await mgr.publish_domain(evt)
        assert mgr.domain_messages_processed >= 1

        await mgr.stop()


# ---------------------------------------------------------------------------
# Unified metrics
# ---------------------------------------------------------------------------

class TestUnifiedMetrics:
    """BusManager.get_metrics combines both buses."""

    def test_metrics_legacy_only(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config()
        metrics = mgr.get_metrics()
        assert "legacy_messages_processed" in metrics
        assert "legacy_error_counts" in metrics
        assert "legacy_dead_letter_count" in metrics
        assert "domain_messages_processed" not in metrics

    def test_metrics_with_domain(self):
        from agentic_trading.bus.manager import BusManager

        mgr = BusManager.from_config(enable_domain_bus=True)
        metrics = mgr.get_metrics()
        assert "legacy_messages_processed" in metrics
        assert "domain_messages_processed" in metrics
        assert "domain_error_counts" in metrics
        assert "domain_dead_letter_count" in metrics


# ---------------------------------------------------------------------------
# Schema registry helpers
# ---------------------------------------------------------------------------

class TestSchemaRegistryHelpers:
    """BusManager exposes schema registry helpers."""

    def test_get_topic_for_event(self):
        from agentic_trading.bus.manager import BusManager

        sig = _make_signal()
        topic = BusManager.get_topic_for_event(sig)
        assert topic == "strategy.signal"

    def test_get_event_class(self):
        from agentic_trading.bus.manager import BusManager

        cls = BusManager.get_event_class("Signal")
        assert cls is Signal

    def test_get_event_class_unknown(self):
        from agentic_trading.bus.manager import BusManager

        cls = BusManager.get_event_class("NonexistentEvent")
        assert cls is None

    def test_list_topics(self):
        from agentic_trading.bus.manager import BusManager

        topics = BusManager.list_topics()
        assert isinstance(topics, list)
        assert "strategy.signal" in topics
        assert "market.candle" in topics
        assert "execution.fill" in topics
        assert topics == sorted(topics)  # sorted


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageImport:
    """BusManager is importable from package __init__."""

    def test_import_from_package(self):
        from agentic_trading.bus import BusManager

        assert BusManager is not None

    def test_import_from_module(self):
        from agentic_trading.bus.manager import BusManager

        assert BusManager is not None

    def test_import_identity(self):
        from agentic_trading.bus import BusManager as PkgBM
        from agentic_trading.bus.manager import BusManager as ModBM

        assert PkgBM is ModBM
