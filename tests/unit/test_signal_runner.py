"""Tests for StrategyRunner.

Covers:
- Feature vector dispatch to strategies.
- Indicator aliasing.
- Signal publication to event bus.
- on_signal callback invocation.
- Signal count tracking.
- Empty buffer → no dispatch.
- No signal → nothing published.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic_trading.core.enums import SignalDirection, Timeframe
from agentic_trading.core.events import FeatureVector, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.signal.runner import StrategyRunner, alias_features
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
    """Strategy that always returns a signal."""

    def __init__(self) -> None:
        super().__init__(strategy_id="always_signal")

    def on_candle(self, ctx, candle, features) -> Signal | None:
        return Signal(
            strategy_id=self._strategy_id,
            symbol=features.symbol,
            direction=SignalDirection.LONG,
            confidence=0.85,
            rationale="test signal",
            features_used={"close": features.features.get("close", 0)},
            trace_id="test-trace-1",
        )


class _NeverSignalStrategy(BaseStrategy):
    """Strategy that never signals."""

    def __init__(self) -> None:
        super().__init__(strategy_id="never_signal")

    def on_candle(self, ctx, candle, features) -> Signal | None:
        return None


def _make_candle() -> Candle:
    return Candle(
        symbol="BTC/USDT",
        exchange="binance",
        timeframe="1m",
        timestamp=datetime.now(timezone.utc),
        open=Decimal("50000"),
        high=Decimal("50100"),
        low=Decimal("49900"),
        close=Decimal("50050"),
        volume=Decimal("100"),
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


def _make_ctx(bus: MemoryEventBus) -> TradingContext:
    from agentic_trading.core.clock import SimClock

    return TradingContext(
        clock=SimClock(),
        event_bus=bus,
        instruments={},
    )


# ---------------------------------------------------------------------------
# alias_features
# ---------------------------------------------------------------------------

class TestAliasFeatures:
    def test_adds_short_aliases(self) -> None:
        features = {"adx_14": 25.0, "atr_14": 100.0, "rsi_14": 55.0}
        out = alias_features(features)
        assert out["adx"] == 25.0
        assert out["atr"] == 100.0
        assert out["rsi"] == 55.0
        # Originals preserved
        assert out["adx_14"] == 25.0

    def test_does_not_overwrite_existing_alias(self) -> None:
        features = {"adx_14": 25.0, "adx": 99.0}
        out = alias_features(features)
        assert out["adx"] == 99.0  # Not overwritten

    def test_donchian_aliases(self) -> None:
        features = {"donchian_upper_20": 100.0, "donchian_lower_20": 50.0}
        out = alias_features(features)
        assert out["donchian_upper"] == 100.0
        assert out["donchian_lower"] == 50.0

    def test_returns_copy(self) -> None:
        features = {"close": 100.0}
        out = alias_features(features)
        out["new_key"] = 999
        assert "new_key" not in features


# ---------------------------------------------------------------------------
# StrategyRunner
# ---------------------------------------------------------------------------

class TestStrategyRunner:
    @pytest.mark.asyncio
    async def test_dispatch_produces_signal(self) -> None:
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)
        engine = _StubFeatureEngine(buffer=[_make_candle()])
        runner = StrategyRunner(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=engine,
            event_bus=bus,
        )
        await runner.start(ctx)

        # Publish a feature vector → triggers strategy dispatch
        fv = _make_feature_vector()
        await bus.publish("feature.vector", fv)

        # Signal should have been published to "strategy.signal"
        assert runner.signal_count == 1

    @pytest.mark.asyncio
    async def test_no_buffer_means_no_dispatch(self) -> None:
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)
        engine = _StubFeatureEngine(buffer=None)
        runner = StrategyRunner(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=engine,
            event_bus=bus,
        )
        await runner.start(ctx)

        fv = _make_feature_vector()
        await bus.publish("feature.vector", fv)
        assert runner.signal_count == 0

    @pytest.mark.asyncio
    async def test_no_signal_means_nothing_published(self) -> None:
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)
        engine = _StubFeatureEngine(buffer=[_make_candle()])
        runner = StrategyRunner(
            strategies=[_NeverSignalStrategy()],
            feature_engine=engine,
            event_bus=bus,
        )
        await runner.start(ctx)

        fv = _make_feature_vector()
        await bus.publish("feature.vector", fv)
        assert runner.signal_count == 0

    @pytest.mark.asyncio
    async def test_on_signal_callback_invoked(self) -> None:
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)
        engine = _StubFeatureEngine(buffer=[_make_candle()])

        callback_calls: list[tuple] = []

        def _cb(sig: Signal, elapsed: float) -> None:
            callback_calls.append((sig, elapsed))

        runner = StrategyRunner(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=engine,
            event_bus=bus,
            on_signal=_cb,
        )
        await runner.start(ctx)

        fv = _make_feature_vector()
        await bus.publish("feature.vector", fv)

        assert len(callback_calls) == 1
        sig, elapsed = callback_calls[0]
        assert sig.strategy_id == "always_signal"
        assert sig.direction == SignalDirection.LONG
        assert elapsed >= 0.0

    @pytest.mark.asyncio
    async def test_multiple_strategies(self) -> None:
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)
        engine = _StubFeatureEngine(buffer=[_make_candle()])
        runner = StrategyRunner(
            strategies=[_AlwaysSignalStrategy(), _NeverSignalStrategy()],
            feature_engine=engine,
            event_bus=bus,
        )
        await runner.start(ctx)

        fv = _make_feature_vector()
        await bus.publish("feature.vector", fv)

        # Only AlwaysSignal produces a signal
        assert runner.signal_count == 1

    @pytest.mark.asyncio
    async def test_strategies_property(self) -> None:
        bus = MemoryEventBus()
        s1 = _AlwaysSignalStrategy()
        s2 = _NeverSignalStrategy()
        runner = StrategyRunner(
            strategies=[s1, s2],
            feature_engine=_StubFeatureEngine(),
            event_bus=bus,
        )
        assert len(runner.strategies) == 2
        assert runner.strategies[0].strategy_id == "always_signal"
        assert runner.strategies[1].strategy_id == "never_signal"

    @pytest.mark.asyncio
    async def test_ignores_non_feature_vector_events(self) -> None:
        """Events that aren't FeatureVector should be silently ignored."""
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)
        engine = _StubFeatureEngine(buffer=[_make_candle()])
        runner = StrategyRunner(
            strategies=[_AlwaysSignalStrategy()],
            feature_engine=engine,
            event_bus=bus,
        )
        await runner.start(ctx)

        # Publish a non-FeatureVector event on the same topic
        await bus.publish("feature.vector", "not a feature vector")
        assert runner.signal_count == 0

    @pytest.mark.asyncio
    async def test_aliased_features_passed_to_strategy(self) -> None:
        """Strategy should receive aliased features."""
        bus = MemoryEventBus()
        await bus.start()
        ctx = _make_ctx(bus)
        engine = _StubFeatureEngine(buffer=[_make_candle()])

        received_features: list[dict] = []

        class _CapturingStrategy(BaseStrategy):
            def __init__(self):
                super().__init__(strategy_id="capture")

            def on_candle(self, ctx, candle, features) -> Signal | None:
                received_features.append(dict(features.features))
                return None

        runner = StrategyRunner(
            strategies=[_CapturingStrategy()],
            feature_engine=engine,
            event_bus=bus,
        )
        await runner.start(ctx)

        fv = _make_feature_vector()
        await bus.publish("feature.vector", fv)

        assert len(received_features) == 1
        assert "adx" in received_features[0]
        assert "atr" in received_features[0]
        assert "rsi" in received_features[0]
        assert "donchian_upper" in received_features[0]
        assert "donchian_lower" in received_features[0]
