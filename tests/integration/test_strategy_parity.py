"""Integration test: strategy parity across all 3 modes.

Verifies that TrendFollowingStrategy.on_candle() produces identical output
when given identical inputs, regardless of how the TradingContext is wired.
This proves that the strategy is truly mode-agnostic.
"""

import pytest

from datetime import datetime, timezone

from agentic_trading.core.clock import WallClock, SimClock
from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.events import FeatureVector
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.strategies.trend_following import TrendFollowingStrategy


def _make_candle() -> Candle:
    """Create a deterministic test candle."""
    return Candle(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        open=65000.0,
        high=65500.0,
        low=64800.0,
        close=65200.0,
        volume=150.0,
        is_closed=True,
    )


def _make_features() -> FeatureVector:
    """Create a deterministic feature vector that triggers a signal.

    The features are set so that:
    - fast_ema (12) > slow_ema (26)  -> LONG direction
    - adx > 25 (threshold)           -> trend is strong enough
    - volume_ratio >= 0.8            -> volume filter passes
    - atr > 0                        -> sizing is possible
    """
    return FeatureVector(
        symbol="BTC/USDT",
        timeframe=Timeframe.M1,
        features={
            "ema_12": 65300.0,
            "ema_26": 65000.0,
            "adx": 35.0,
            "atr": 200.0,
            "volume_ratio": 1.2,
            "close": 65200.0,
        },
    )


def _make_context(clock, bus) -> TradingContext:
    """Create a minimal TradingContext with no instruments or portfolio."""
    return TradingContext(
        clock=clock,
        event_bus=bus,
        instruments={},
    )


class TestStrategyParity:
    """Verify that TrendFollowingStrategy produces identical results
    regardless of which clock/bus combination is used (simulating
    backtest, paper, and live modes)."""

    def test_same_strategy_same_result_across_modes(self):
        """Instantiate the strategy three times with different clocks,
        feed the same candle + features, and assert identical signals."""
        candle = _make_candle()
        features = _make_features()

        # Mode 1: Backtest (SimClock)
        bus1 = MemoryEventBus()
        ctx1 = _make_context(
            SimClock(start=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)),
            bus1,
        )
        strat1 = TrendFollowingStrategy(strategy_id="tf_backtest")
        signal1 = strat1.on_candle(ctx1, candle, features)

        # Mode 2: Paper (WallClock)
        bus2 = MemoryEventBus()
        ctx2 = _make_context(WallClock(), bus2)
        strat2 = TrendFollowingStrategy(strategy_id="tf_paper")
        signal2 = strat2.on_candle(ctx2, candle, features)

        # Mode 3: Live (WallClock, separate instance)
        bus3 = MemoryEventBus()
        ctx3 = _make_context(WallClock(), bus3)
        strat3 = TrendFollowingStrategy(strategy_id="tf_live")
        signal3 = strat3.on_candle(ctx3, candle, features)

        # All three should produce a signal (not None)
        assert signal1 is not None, "Backtest mode produced no signal"
        assert signal2 is not None, "Paper mode produced no signal"
        assert signal3 is not None, "Live mode produced no signal"

        # All three should have the same direction
        assert signal1.direction == signal2.direction == signal3.direction

        # All three should have the same confidence
        assert signal1.confidence == signal2.confidence == signal3.confidence

        # All three should have the same symbol
        assert signal1.symbol == signal2.symbol == signal3.symbol

        # All three should use the same features
        assert signal1.features_used == signal2.features_used == signal3.features_used

    def test_no_signal_case_is_also_consistent(self):
        """When ADX is below threshold, all modes should return None."""
        candle = _make_candle()
        weak_features = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M1,
            features={
                "ema_12": 65300.0,
                "ema_26": 65000.0,
                "adx": 15.0,  # Below threshold of 25
                "atr": 200.0,
                "volume_ratio": 1.2,
            },
        )

        bus1 = MemoryEventBus()
        ctx1 = _make_context(SimClock(), bus1)
        strat1 = TrendFollowingStrategy(strategy_id="tf_backtest")

        bus2 = MemoryEventBus()
        ctx2 = _make_context(WallClock(), bus2)
        strat2 = TrendFollowingStrategy(strategy_id="tf_paper")

        assert strat1.on_candle(ctx1, candle, weak_features) is None
        assert strat2.on_candle(ctx2, candle, weak_features) is None

    def test_parameters_are_identical(self):
        """All instances share the same default parameters."""
        strat1 = TrendFollowingStrategy(strategy_id="a")
        strat2 = TrendFollowingStrategy(strategy_id="b")
        strat3 = TrendFollowingStrategy(strategy_id="c")

        assert strat1.get_parameters() == strat2.get_parameters() == strat3.get_parameters()
