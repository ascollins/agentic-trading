"""Test trading strategies and strategy registry."""

from agentic_trading.core.clock import SimClock
from agentic_trading.core.enums import Exchange, SignalDirection, Timeframe
from agentic_trading.core.events import FeatureVector, RegimeState, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle, Instrument
from agentic_trading.event_bus.memory_bus import MemoryEventBus

# Import strategies to trigger registration
from agentic_trading.strategies.trend_following import TrendFollowingStrategy
from agentic_trading.strategies.mean_reversion import MeanReversionStrategy
from agentic_trading.strategies.breakout import BreakoutStrategy
from agentic_trading.strategies.registry import create_strategy, list_strategies


def _make_ctx(bus=None) -> TradingContext:
    """Create a minimal TradingContext for testing."""
    bus = bus or MemoryEventBus()
    clock = SimClock()
    return TradingContext(
        clock=clock,
        event_bus=bus,
        instruments={},
    )


def _make_candle(close: float = 67000.0) -> Candle:
    from datetime import datetime, timezone
    return Candle(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M5,
        timestamp=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        open=close - 50,
        high=close + 100,
        low=close - 100,
        close=close,
        volume=10.0,
    )


class TestTrendFollowingStrategy:
    def test_returns_none_without_features(self):
        strategy = TrendFollowingStrategy()
        ctx = _make_ctx()
        candle = _make_candle()
        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={},
        )
        result = strategy.on_candle(ctx, candle, fv)
        assert result is None

    def test_returns_signal_with_sufficient_features(self):
        strategy = TrendFollowingStrategy(
            params={"adx_threshold": 20, "min_confidence": 0.1}
        )
        ctx = _make_ctx()
        candle = _make_candle()
        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={
                "ema_12": 67100.0,
                "ema_26": 66800.0,  # fast > slow -> LONG
                "adx": 35.0,  # Above threshold
                "atr": 500.0,
                "volume_ratio": 1.2,
            },
        )
        result = strategy.on_candle(ctx, candle, fv)
        assert result is not None
        assert isinstance(result, Signal)
        assert result.direction == SignalDirection.LONG

    def test_returns_short_signal(self):
        strategy = TrendFollowingStrategy(
            params={"adx_threshold": 20, "min_confidence": 0.1}
        )
        ctx = _make_ctx()
        candle = _make_candle()
        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={
                "ema_12": 66500.0,
                "ema_26": 67000.0,  # fast < slow -> SHORT
                "adx": 30.0,
                "atr": 500.0,
                "volume_ratio": 1.2,
            },
        )
        result = strategy.on_candle(ctx, candle, fv)
        assert result is not None
        assert result.direction == SignalDirection.SHORT

    def test_no_signal_when_adx_low(self):
        strategy = TrendFollowingStrategy()
        ctx = _make_ctx()
        candle = _make_candle()
        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={
                "ema_12": 67100.0,
                "ema_26": 66800.0,
                "adx": 15.0,  # Below threshold
                "atr": 500.0,
            },
        )
        result = strategy.on_candle(ctx, candle, fv)
        assert result is None


class TestMeanReversionStrategy:
    def test_returns_none_without_features(self):
        strategy = MeanReversionStrategy()
        ctx = _make_ctx()
        candle = _make_candle()
        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={},
        )
        result = strategy.on_candle(ctx, candle, fv)
        assert result is None

    def test_returns_long_signal_oversold(self):
        strategy = MeanReversionStrategy(
            params={
                "require_range_regime": False,
                "mean_reversion_score_threshold": 0.0,
                "min_confidence": 0.0,
            }
        )
        ctx = _make_ctx()
        # Price well below lower band, RSI very oversold
        candle = _make_candle(close=65000.0)
        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={
                "bb_upper": 68000.0,
                "bb_lower": 66000.0,
                "bb_middle": 67000.0,
                "rsi": 20.0,
                "atr": 500.0,
            },
        )
        result = strategy.on_candle(ctx, candle, fv)
        assert result is not None
        assert result.direction == SignalDirection.LONG

    def test_returns_short_signal_overbought(self):
        strategy = MeanReversionStrategy(
            params={
                "require_range_regime": False,
                "mean_reversion_score_threshold": 0.0,
                "min_confidence": 0.0,
            }
        )
        ctx = _make_ctx()
        # Price well above upper band, RSI very overbought
        candle = _make_candle(close=69000.0)
        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={
                "bb_upper": 68000.0,
                "bb_lower": 66000.0,
                "bb_middle": 67000.0,
                "rsi": 85.0,
                "atr": 500.0,
            },
        )
        result = strategy.on_candle(ctx, candle, fv)
        assert result is not None
        assert result.direction == SignalDirection.SHORT


class TestBreakoutStrategy:
    def test_returns_none_without_features(self):
        strategy = BreakoutStrategy()
        ctx = _make_ctx()
        candle = _make_candle()
        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={},
        )
        result = strategy.on_candle(ctx, candle, fv)
        assert result is None

    def test_returns_long_on_breakout_above_donchian(self):
        strategy = BreakoutStrategy(
            params={
                "volume_confirmation_multiplier": 1.0,
                "min_confidence": 0.0,
                "min_liquidity_score": 0.0,
            }
        )
        ctx = _make_ctx()
        candle = _make_candle(close=69000.0)
        fv = FeatureVector(
            symbol="BTC/USDT",
            timeframe=Timeframe.M5,
            features={
                "donchian_upper": 68500.0,
                "donchian_lower": 65500.0,
                "atr": 500.0,
                "volume_ratio": 2.0,  # High volume confirms
            },
        )
        # Seed the prev_atr so ATR expansion check works
        strategy._prev_atr = 450.0
        result = strategy.on_candle(ctx, candle, fv)
        assert result is not None
        assert result.direction == SignalDirection.LONG


class TestStrategyRegistry:
    def test_list_strategies_includes_all(self):
        strategies = list_strategies()
        assert "trend_following" in strategies
        assert "mean_reversion" in strategies
        assert "breakout" in strategies

    def test_create_strategy_by_id(self):
        strategy = create_strategy("trend_following")
        assert strategy.strategy_id == "trend_following"
        assert isinstance(strategy, TrendFollowingStrategy)

    def test_create_unknown_strategy_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy("nonexistent_strategy")

    def test_get_parameters(self):
        strategy = TrendFollowingStrategy()
        params = strategy.get_parameters()
        assert "fast_ema" in params
        assert "slow_ema" in params
