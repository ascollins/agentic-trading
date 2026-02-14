"""Tests for the funding rate arbitrage strategy."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agentic_trading.core.enums import Exchange, SignalDirection, Timeframe
from agentic_trading.core.events import FeatureVector, Signal
from agentic_trading.core.interfaces import PortfolioState, TradingContext
from agentic_trading.core.clock import SimClock
from agentic_trading.core.models import Candle
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.strategies.funding_arb import FundingArbStrategy


def _make_candle(
    close: float = 100.0,
    symbol: str = "BTC/USDT",
) -> Candle:
    return Candle(
        symbol=symbol,
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1000.0,
    )


def _make_features(
    funding_rate: float | None = None,
    atr: float = 2.0,
    symbol: str = "BTC/USDT",
) -> FeatureVector:
    features = {"atr": atr, "close": 100.0}
    if funding_rate is not None:
        features["funding_rate"] = funding_rate
    return FeatureVector(
        symbol=symbol,
        timeframe=Timeframe.M1,
        features=features,
    )


def _make_ctx() -> TradingContext:
    return TradingContext(
        clock=SimClock(),
        event_bus=MemoryEventBus(),
        instruments={},
        portfolio_state=PortfolioState(),
    )


class TestFundingArbStrategy:
    """Test FundingArbStrategy signal generation."""

    def test_no_signal_without_funding_rate(self):
        """Should return None when funding_rate is missing from features."""
        strategy = FundingArbStrategy()
        ctx = _make_ctx()
        candle = _make_candle()
        features = _make_features(funding_rate=None)

        signal = strategy.on_candle(ctx, candle, features)
        assert signal is None

    def test_no_signal_below_threshold(self):
        """Should return None when |funding_rate| < threshold."""
        strategy = FundingArbStrategy(params={"funding_threshold": 0.0001})
        ctx = _make_ctx()
        candle = _make_candle()
        features = _make_features(funding_rate=0.00005)  # Below 1 bps

        signal = strategy.on_candle(ctx, candle, features)
        assert signal is None

    def test_short_on_positive_funding(self):
        """Should go SHORT when funding_rate > threshold (longs pay shorts)."""
        strategy = FundingArbStrategy(params={"funding_threshold": 0.0001})
        ctx = _make_ctx()
        candle = _make_candle()
        features = _make_features(funding_rate=0.0005)  # 5 bps

        signal = strategy.on_candle(ctx, candle, features)
        assert signal is not None
        assert signal.direction == SignalDirection.SHORT
        assert "Shorts earn funding" in signal.rationale

    def test_long_on_negative_funding(self):
        """Should go LONG when funding_rate < -threshold (shorts pay longs)."""
        strategy = FundingArbStrategy(params={"funding_threshold": 0.0001})
        ctx = _make_ctx()
        candle = _make_candle()
        features = _make_features(funding_rate=-0.0003)  # -3 bps

        signal = strategy.on_candle(ctx, candle, features)
        assert signal is not None
        assert signal.direction == SignalDirection.LONG
        assert "Longs earn funding" in signal.rationale

    def test_high_confidence_on_extreme_funding(self):
        """High funding rate should produce high confidence."""
        strategy = FundingArbStrategy(params={
            "funding_threshold": 0.0001,
            "high_funding_threshold": 0.0005,
        })
        ctx = _make_ctx()
        candle = _make_candle()
        features = _make_features(funding_rate=0.001)  # 10 bps — very high

        signal = strategy.on_candle(ctx, candle, features)
        assert signal is not None
        assert signal.confidence >= 0.8
        assert "HIGH funding" in signal.rationale

    def test_moderate_confidence_on_moderate_funding(self):
        """Moderate funding rate should produce moderate confidence."""
        strategy = FundingArbStrategy(params={
            "funding_threshold": 0.0001,
            "high_funding_threshold": 0.0005,
        })
        ctx = _make_ctx()
        candle = _make_candle()
        features = _make_features(funding_rate=0.00015)  # Just above threshold

        signal = strategy.on_candle(ctx, candle, features)
        assert signal is not None
        assert signal.confidence <= 0.6

    def test_strategy_registered(self):
        """Strategy should be in the registry."""
        from agentic_trading.strategies.registry import list_strategies

        assert "funding_arb" in list_strategies()

    def test_risk_constraints_include_atr(self):
        """Risk constraints should include ATR-based stop."""
        strategy = FundingArbStrategy(params={
            "funding_threshold": 0.0001,
            "atr_stop_multiplier": 3.0,
        })
        ctx = _make_ctx()
        candle = _make_candle()
        features = _make_features(funding_rate=0.0005, atr=2.0)

        signal = strategy.on_candle(ctx, candle, features)
        assert signal is not None
        rc = signal.risk_constraints
        assert "atr" in rc
        assert rc["stop_distance_atr"] == pytest.approx(6.0)  # 2.0 * 3.0

    def test_features_used_includes_funding_rate(self):
        """features_used should include the funding rate value."""
        strategy = FundingArbStrategy(params={"funding_threshold": 0.0001})
        ctx = _make_ctx()
        candle = _make_candle()
        features = _make_features(funding_rate=0.0003)

        signal = strategy.on_candle(ctx, candle, features)
        assert signal is not None
        assert "funding_rate" in signal.features_used
        assert signal.features_used["funding_rate"] == pytest.approx(0.0003)
