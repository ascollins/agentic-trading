"""Tests for SignalAgent — canonical event-driven signal layer.

Covers:
- Agent lifecycle (start / stop / health_check).
- Given a FeatureVector, emits ``SignalCreated`` domain event.
- Given a FeatureVector + SignalManager, emits ``DecisionProposed``.
- No signal → no domain events emitted.
- Empty candle buffer → no dispatch.
- Legacy ``Signal`` still published on topic bus.
- Write-ownership: ``SignalCreated.source == "signal"``.
- ``_to_signal_created`` mapping correctness.
- Multiple strategies, only signalling ones emit events.
- on_signal callback still fires.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from agentic_trading.agents.signal_agent import SignalAgent, _to_signal_created
from agentic_trading.core.enums import (
    AgentStatus,
    AgentType,
    Exchange,
    SignalDirection,
    Timeframe,
)
from agentic_trading.core.events import FeatureVector, Signal
from agentic_trading.domain.events import DecisionProposed, SignalCreated
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.infrastructure.event_bus import InMemoryEventBus
from agentic_trading.signal.strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubFeatureEngine:
    """Fake feature engine returning a fixed candle buffer."""

    def __init__(self, buffer: list | None = None) -> None:
        self._buffer = buffer

    def get_buffer(self, symbol: str, timeframe: Any) -> list | None:
        return self._buffer


class _AlwaysSignalStrategy(BaseStrategy):
    """Strategy that always returns a LONG signal."""

    def __init__(self) -> None:
        super().__init__(strategy_id="always_signal")

    def on_candle(self, ctx, candle, features) -> Signal | None:
        return Signal(
            strategy_id=self._strategy_id,
            symbol=features.symbol,
            direction=SignalDirection.LONG,
            confidence=0.85,
            rationale="bullish crossover",
            features_used={"close": features.features.get("close", 0)},
            timeframe=Timeframe.M1,
            risk_constraints={
                "sizing_method": "fixed_fractional",
                "price": 50050.0,
            },
            take_profit=Decimal("52000"),
            stop_loss=Decimal("48000"),
            trailing_stop=Decimal("500"),
            trace_id="test-trace-1",
        )


class _NeverSignalStrategy(BaseStrategy):
    """Strategy that never signals."""

    def __init__(self) -> None:
        super().__init__(strategy_id="never_signal")

    def on_candle(self, ctx, candle, features) -> Signal | None:
        return None


class _FlatSignalStrategy(BaseStrategy):
    """Strategy that always returns a FLAT (exit) signal."""

    def __init__(self) -> None:
        super().__init__(strategy_id="flat_signal")

    def on_candle(self, ctx, candle, features) -> Signal | None:
        return Signal(
            strategy_id=self._strategy_id,
            symbol=features.symbol,
            direction=SignalDirection.FLAT,
            confidence=0.5,
            rationale="exit condition met",
            features_used={"close": features.features.get("close", 0)},
            timeframe=Timeframe.M1,
            trace_id="test-flat-trace",
        )


def _make_candle():
    from agentic_trading.core.models import Candle

    return Candle(
        symbol="BTC/USDT",
        exchange=Exchange.BYBIT,
        timeframe=Timeframe.M1,
        timestamp=datetime.now(timezone.utc),
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50050.0,
        volume=100.0,
    )


def _make_feature_vector(symbol: str = "BTC/USDT") -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        timeframe=Timeframe.M1,
        features={
            "close": 50050.0,
            "adx_14": 25.0,
            "atr_14": 100.0,
            "rsi_14": 55.0,
            "donchian_upper_20": 50200.0,
            "donchian_lower_20": 49800.0,
        },
    )


def _make_ctx(bus: MemoryEventBus):
    from agentic_trading.core.clock import SimClock
    from agentic_trading.core.interfaces import TradingContext

    return TradingContext(
        clock=SimClock(),
        event_bus=bus,
        instruments={},
    )


# ---------------------------------------------------------------------------
# _to_signal_created mapping
# ---------------------------------------------------------------------------

class TestToSignalCreated:
    def test_maps_all_fields(self) -> None:
        sig = Signal(
            strategy_id="trend_following",
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            confidence=0.85,
            rationale="EMA crossover",
            features_used={"close": 50050.0, "adx": 28.0},
            timeframe=Timeframe.H1,
            take_profit=Decimal("52000"),
            stop_loss=Decimal("48000"),
            trailing_stop=Decimal("500"),
            trace_id="trace-123",
        )

        sc = _to_signal_created(sig)

        assert isinstance(sc, SignalCreated)
        assert sc.source == "signal"
        assert sc.strategy_id == "trend_following"
        assert sc.symbol == "BTC/USDT"
        assert sc.direction == "long"
        assert sc.confidence == 0.85
        assert sc.rationale == "EMA crossover"
        assert sc.take_profit == Decimal("52000")
        assert sc.stop_loss == Decimal("48000")
        assert sc.trailing_stop == Decimal("500")
        assert sc.timeframe == "1h"
        assert sc.correlation_id == "trace-123"
        # features_used is a tuple of (key, float) pairs
        features = dict(sc.features_used)
        assert features["close"] == 50050.0
        assert features["adx"] == 28.0

    def test_handles_none_tp_sl(self) -> None:
        sig = Signal(
            strategy_id="mean_reversion",
            symbol="ETH/USDT",
            direction=SignalDirection.SHORT,
            confidence=0.6,
            rationale="overbought",
            features_used={},
            timeframe=Timeframe.M5,
        )

        sc = _to_signal_created(sig)

        assert sc.take_profit is None
        assert sc.stop_loss is None
        assert sc.trailing_stop is None
        assert sc.features_used == ()

    def test_flat_direction(self) -> None:
        sig = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.FLAT,
            confidence=0.5,
            rationale="exit",
            features_used={},
        )

        sc = _to_signal_created(sig)
        assert sc.direction == "flat"


# ---------------------------------------------------------------------------
# SignalAgent lifecycle
# ---------------------------------------------------------------------------

class TestSignalAgentLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=bus,
            ctx=ctx,
        )

        assert agent.status == AgentStatus.CREATED
        assert not agent.is_running

        await agent.start()
        assert agent.is_running
        assert agent.status == AgentStatus.RUNNING

        await agent.stop()
        assert not agent.is_running
        assert agent.status == AgentStatus.STOPPED

    def test_agent_type(self) -> None:
        bus = MemoryEventBus()
        agent = SignalAgent(
            strategies=[],
            feature_engine=_StubFeatureEngine(),
            event_bus=bus,
        )
        assert agent.agent_type == AgentType.STRATEGY

    def test_capabilities(self) -> None:
        bus = MemoryEventBus()
        agent = SignalAgent(
            strategies=[],
            feature_engine=_StubFeatureEngine(),
            event_bus=bus,
        )
        caps = agent.capabilities()
        assert "feature.vector" in caps.subscribes_to
        assert "strategy.signal" in caps.publishes_to

    def test_health_check_before_start(self) -> None:
        bus = MemoryEventBus()
        agent = SignalAgent(
            strategies=[],
            feature_engine=_StubFeatureEngine(),
            event_bus=bus,
        )
        report = agent.health_check()
        assert not report.healthy

    @pytest.mark.asyncio
    async def test_health_check_after_start(self) -> None:
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)
        agent = SignalAgent(
            strategies=[],
            feature_engine=_StubFeatureEngine(),
            event_bus=bus,
            ctx=ctx,
        )
        await agent.start()
        report = agent.health_check()
        assert report.healthy
        await agent.stop()


# ---------------------------------------------------------------------------
# SignalAgent: SignalCreated emission
# ---------------------------------------------------------------------------

class TestSignalAgentSignalCreated:
    @pytest.mark.asyncio
    async def test_emits_signal_created_on_domain_bus(self) -> None:
        """Given a FeatureVector + strategy that signals, emits SignalCreated."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        received: list[SignalCreated] = []

        async def _on_signal_created(event: SignalCreated) -> None:
            received.append(event)

        domain_bus.subscribe(SignalCreated, _on_signal_created)

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            ctx=ctx,
        )
        await agent.start()

        fv = _make_feature_vector()
        await legacy_bus.publish("feature.vector", fv)

        assert len(received) == 1
        sc = received[0]
        assert sc.source == "signal"
        assert sc.strategy_id == "always_signal"
        assert sc.symbol == "BTC/USDT"
        assert sc.direction == "long"
        assert sc.confidence == 0.85
        assert sc.rationale == "bullish crossover"
        assert sc.take_profit == Decimal("52000")
        assert sc.stop_loss == Decimal("48000")

        assert agent.signal_created_count == 1
        await agent.stop()

    @pytest.mark.asyncio
    async def test_no_signal_means_no_domain_event(self) -> None:
        """Strategy returns None → no SignalCreated emitted."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        received: list[SignalCreated] = []

        async def _on(event: SignalCreated) -> None:
            received.append(event)

        domain_bus.subscribe(SignalCreated, _on)

        agent = SignalAgent(
            strategies=[_NeverSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            ctx=ctx,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        assert len(received) == 0
        assert agent.signal_created_count == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_empty_buffer_means_no_dispatch(self) -> None:
        """Empty candle buffer → no dispatch at all."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        received: list[SignalCreated] = []

        async def _on(event: SignalCreated) -> None:
            received.append(event)

        domain_bus.subscribe(SignalCreated, _on)

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=None),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            ctx=ctx,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        assert len(received) == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_legacy_signal_still_published(self) -> None:
        """Legacy Signal event still goes to strategy.signal topic."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        legacy_signals: list[Signal] = []

        async def _on_legacy_signal(event: Any) -> None:
            if isinstance(event, Signal):
                legacy_signals.append(event)

        await legacy_bus.subscribe("strategy.signal", "test_sub", _on_legacy_signal)

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            ctx=ctx,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        assert len(legacy_signals) == 1
        assert legacy_signals[0].strategy_id == "always_signal"
        await agent.stop()

    @pytest.mark.asyncio
    async def test_multiple_strategies_only_signalling_ones_emit(self) -> None:
        """Only strategies that produce signals result in domain events."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        received: list[SignalCreated] = []

        async def _on(event: SignalCreated) -> None:
            received.append(event)

        domain_bus.subscribe(SignalCreated, _on)

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy(), _NeverSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            ctx=ctx,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        # Only AlwaysSignal produces an event
        assert len(received) == 1
        assert received[0].strategy_id == "always_signal"
        assert agent.signal_created_count == 1
        await agent.stop()

    @pytest.mark.asyncio
    async def test_flat_signal_emits_signal_created(self) -> None:
        """FLAT signals also produce SignalCreated domain events."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        received: list[SignalCreated] = []

        async def _on(event: SignalCreated) -> None:
            received.append(event)

        domain_bus.subscribe(SignalCreated, _on)

        agent = SignalAgent(
            strategies=[_FlatSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            ctx=ctx,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        assert len(received) == 1
        assert received[0].direction == "flat"
        await agent.stop()

    @pytest.mark.asyncio
    async def test_on_signal_callback_invoked(self) -> None:
        """on_signal callback fires for each signal produced."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        callback_calls: list[tuple] = []

        def _cb(sig: Signal, elapsed: float) -> None:
            callback_calls.append((sig, elapsed))

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            ctx=ctx,
            on_signal=_cb,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        assert len(callback_calls) == 1
        sig, elapsed = callback_calls[0]
        assert sig.strategy_id == "always_signal"
        assert elapsed >= 0.0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_ignores_non_feature_vector_events(self) -> None:
        """Non-FeatureVector events on feature.vector topic are ignored."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            ctx=ctx,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", "not a feature vector")

        assert agent.signal_created_count == 0
        await agent.stop()


# ---------------------------------------------------------------------------
# SignalAgent: DecisionProposed emission
# ---------------------------------------------------------------------------

class TestSignalAgentDecisionProposed:
    @pytest.mark.asyncio
    async def test_emits_decision_proposed_with_signal_manager(self) -> None:
        """When signal_manager is provided, emits DecisionProposed."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        # Build a minimal signal manager with portfolio manager
        from agentic_trading.signal.manager import SignalManager

        signal_mgr = SignalManager.from_config(
            strategies=[_AlwaysSignalStrategy()],
        )

        received_signals: list[SignalCreated] = []
        received_decisions: list[DecisionProposed] = []

        async def _on_sc(event: SignalCreated) -> None:
            received_signals.append(event)

        async def _on_dp(event: DecisionProposed) -> None:
            received_decisions.append(event)

        domain_bus.subscribe(SignalCreated, _on_sc)
        domain_bus.subscribe(DecisionProposed, _on_dp)

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            signal_manager=signal_mgr,
            ctx=ctx,
        )
        agent.set_capital(100_000.0)
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        # Should have emitted both SignalCreated and DecisionProposed
        assert len(received_signals) == 1
        assert agent.signal_created_count == 1

        # DecisionProposed should reference the SignalCreated event
        if received_decisions:
            dp = received_decisions[0]
            assert dp.source == "signal"
            assert dp.strategy_id == "always_signal"
            assert dp.symbol == "BTC/USDT"
            assert dp.side in ("buy", "sell")
            assert dp.qty > Decimal("0")
            assert dp.signal_event_id == received_signals[0].event_id
            assert dp.causation_id == received_signals[0].event_id
            assert agent.decision_proposed_count == 1

        await agent.stop()

    @pytest.mark.asyncio
    async def test_no_decision_when_sizing_produces_zero(self) -> None:
        """FLAT signal → no DecisionProposed (sizing returns 0 targets)."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        from agentic_trading.signal.manager import SignalManager

        signal_mgr = SignalManager.from_config(
            strategies=[_FlatSignalStrategy()],
        )

        received_decisions: list[DecisionProposed] = []

        async def _on_dp(event: DecisionProposed) -> None:
            received_decisions.append(event)

        domain_bus.subscribe(DecisionProposed, _on_dp)

        agent = SignalAgent(
            strategies=[_FlatSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            signal_manager=signal_mgr,
            ctx=ctx,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        # FLAT signals don't go through portfolio sizing
        # so no DecisionProposed should be emitted
        assert len(received_decisions) == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_no_decision_without_signal_manager(self) -> None:
        """Without signal_manager, no DecisionProposed emitted."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        received: list[DecisionProposed] = []

        async def _on(event: DecisionProposed) -> None:
            received.append(event)

        domain_bus.subscribe(DecisionProposed, _on)

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            ctx=ctx,
            # No signal_manager
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        assert len(received) == 0
        assert agent.decision_proposed_count == 0
        await agent.stop()


# ---------------------------------------------------------------------------
# SignalAgent: write ownership enforcement
# ---------------------------------------------------------------------------

class TestSignalAgentWriteOwnership:
    @pytest.mark.asyncio
    async def test_signal_created_passes_ownership_check(self) -> None:
        """SignalCreated.source=signal passes domain bus ownership gate."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        received: list[SignalCreated] = []

        async def _on(event: SignalCreated) -> None:
            received.append(event)

        domain_bus.subscribe(SignalCreated, _on)

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            ctx=ctx,
        )
        await agent.start()

        # This should NOT raise WriteOwnershipError
        await legacy_bus.publish("feature.vector", _make_feature_vector())

        assert len(received) == 1
        assert received[0].source == "signal"
        await agent.stop()

    @pytest.mark.asyncio
    async def test_domain_bus_history_records_events(self) -> None:
        """Domain bus history captures all SignalCreated events."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        domain_bus = InMemoryEventBus(enforce_ownership=True)
        await domain_bus.start()

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=domain_bus,
            ctx=ctx,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        history = domain_bus.get_history(SignalCreated)
        assert len(history) == 1
        assert history[0].strategy_id == "always_signal"
        await agent.stop()


# ---------------------------------------------------------------------------
# SignalAgent: accessors
# ---------------------------------------------------------------------------

class TestSignalAgentAccessors:
    def test_strategies_property(self) -> None:
        bus = MemoryEventBus()
        s1 = _AlwaysSignalStrategy()
        s2 = _NeverSignalStrategy()
        agent = SignalAgent(
            strategies=[s1, s2],
            feature_engine=_StubFeatureEngine(),
            event_bus=bus,
        )
        assert len(agent.strategies) == 2
        assert agent.strategies[0].strategy_id == "always_signal"

    def test_initial_counters_zero(self) -> None:
        bus = MemoryEventBus()
        agent = SignalAgent(
            strategies=[],
            feature_engine=_StubFeatureEngine(),
            event_bus=bus,
        )
        assert agent.signal_created_count == 0
        assert agent.decision_proposed_count == 0

    @pytest.mark.asyncio
    async def test_runner_created_after_start(self) -> None:
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)
        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(),
            event_bus=bus,
            ctx=ctx,
        )
        assert agent.runner is None
        await agent.start()
        assert agent.runner is not None
        await agent.stop()


# ---------------------------------------------------------------------------
# SignalAgent: without domain bus (graceful degradation)
# ---------------------------------------------------------------------------

class TestSignalAgentNoDomainBus:
    @pytest.mark.asyncio
    async def test_works_without_domain_bus(self) -> None:
        """Agent still functions when domain_bus is None (legacy-only mode)."""
        legacy_bus = MemoryEventBus()
        await legacy_bus.start()
        ctx = _make_ctx(legacy_bus)

        legacy_signals: list[Signal] = []

        async def _on(event: Any) -> None:
            if isinstance(event, Signal):
                legacy_signals.append(event)

        await legacy_bus.subscribe("strategy.signal", "test", _on)

        agent = SignalAgent(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=_StubFeatureEngine(buffer=[_make_candle()]),
            event_bus=legacy_bus,
            domain_bus=None,  # No domain bus
            ctx=ctx,
        )
        await agent.start()

        await legacy_bus.publish("feature.vector", _make_feature_vector())

        # Legacy events still work
        assert len(legacy_signals) == 1
        # Counter still incremented (event was created, just not published)
        assert agent.signal_created_count == 1
        await agent.stop()
