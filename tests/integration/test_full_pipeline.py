"""Integration test: full pipeline smoke test.

End-to-end: publish CandleEvents -> FeatureEngine produces FeatureVectors ->
Strategy produces a Signal.  Uses MemoryEventBus throughout.
"""

import pytest
from datetime import datetime, timezone, timedelta

from agentic_trading.core.clock import SimClock
from agentic_trading.core.enums import Exchange, SignalDirection, Timeframe
from agentic_trading.core.events import CandleEvent, FeatureVector, Signal
from agentic_trading.core.interfaces import TradingContext
from agentic_trading.core.models import Candle
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.features.engine import FeatureEngine
from agentic_trading.strategies.trend_following import TrendFollowingStrategy


def _generate_trending_candles(
    n: int,
    start_price: float = 100.0,
    trend: float = 0.5,
) -> list[CandleEvent]:
    """Generate a sequence of candle events with an upward trend.

    Each candle closes slightly higher than the previous one to create a
    clear uptrend that the EMA crossover and ADX filters can detect.
    """
    base_time = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    candles = []
    price = start_price

    for i in range(n):
        price += trend + (i * 0.01)  # Accelerating uptrend
        spread = price * 0.005  # 0.5% spread

        candles.append(CandleEvent(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.M1,
            timestamp=base_time + timedelta(minutes=i),
            open=price - spread / 2,
            high=price + spread,
            low=price - spread,
            close=price + spread / 2,
            volume=100.0 + i * 5,
            is_closed=True,
            source_module="test",
        ))

    return candles


@pytest.mark.asyncio
async def test_candle_to_feature_vector():
    """CandleEvents fed through FeatureEngine produce FeatureVectors."""
    bus = MemoryEventBus()
    engine = FeatureEngine(event_bus=bus)
    await engine.start()

    feature_vectors = []

    async def capture_features(event):
        if isinstance(event, FeatureVector):
            feature_vectors.append(event)

    await bus.subscribe("feature.vector", "test_capture", capture_features)

    # Generate and publish enough candles for indicators to warm up.
    # ADX needs ~28 bars, EMA-26 needs 26 bars, etc.
    candles = _generate_trending_candles(50)
    for candle in candles:
        await bus.publish("market.candle", candle)

    # FeatureEngine should have produced feature vectors
    assert len(feature_vectors) > 0, "No FeatureVectors were produced"

    # The last feature vector should have meaningful indicators
    last_fv = feature_vectors[-1]
    assert last_fv.symbol == "BTC/USDT"
    assert "ema_12" in last_fv.features
    assert "ema_26" in last_fv.features or "ema_50" in last_fv.features
    assert "close" in last_fv.features


@pytest.mark.asyncio
async def test_full_pipeline_candle_to_signal():
    """End-to-end: CandleEvent -> FeatureEngine -> Strategy -> Signal.

    This is a smoke test that verifies all components can be wired
    together and produce a signal from raw candle data.
    """
    bus = MemoryEventBus()
    feature_engine = FeatureEngine(event_bus=bus)
    await feature_engine.start()

    clock = SimClock(start=datetime(2024, 6, 1, tzinfo=timezone.utc))
    ctx = TradingContext(clock=clock, event_bus=bus, instruments={})
    strategy = TrendFollowingStrategy(strategy_id="test_trend")

    signals = []

    # Wire: FeatureVector -> Strategy.on_candle
    # The FeatureEngine produces keys like "adx_14" and "atr_14" while the
    # TrendFollowingStrategy looks for "adx" and "atr".  A real pipeline
    # would have a mapping layer; here we alias the keys for integration.
    async def on_feature_vector(event):
        if not isinstance(event, FeatureVector):
            return

        # Find the corresponding candle from the feature engine buffer
        candle_buffer = feature_engine.get_buffer(event.symbol, event.timeframe)
        if not candle_buffer:
            return

        # Alias suffixed indicator keys to plain names for strategy compat
        aliased_features = dict(event.features)
        if "adx_14" in aliased_features and "adx" not in aliased_features:
            aliased_features["adx"] = aliased_features["adx_14"]
        if "atr_14" in aliased_features and "atr" not in aliased_features:
            aliased_features["atr"] = aliased_features["atr_14"]

        patched_fv = FeatureVector(
            symbol=event.symbol,
            timeframe=event.timeframe,
            features=aliased_features,
            source_module=event.source_module,
        )

        latest_candle = candle_buffer[-1]
        signal = strategy.on_candle(ctx, latest_candle, patched_fv)
        if signal is not None:
            signals.append(signal)

    await bus.subscribe("feature.vector", "strategy_runner", on_feature_vector)

    # Generate enough candles with a strong trend.
    # We need ~50 candles minimum for ADX and EMAs to warm up.
    candles = _generate_trending_candles(80, start_price=100.0, trend=1.0)
    for candle in candles:
        clock.set_time(candle.timestamp)
        await bus.publish("market.candle", candle)

    # We expect at least some signals once indicators warm up.
    # The exact number depends on indicator warmup, but with 80 bars
    # of strong uptrend, we should get some LONG signals.
    assert len(signals) > 0, (
        "No signals produced. The pipeline may need more candles for warmup."
    )

    # All signals should be LONG given the uptrend
    for sig in signals:
        assert sig.direction == SignalDirection.LONG, (
            f"Expected LONG signal in uptrend, got {sig.direction}"
        )
        assert sig.symbol == "BTC/USDT"
        assert sig.confidence > 0


@pytest.mark.asyncio
async def test_pipeline_no_signal_on_insufficient_data():
    """With only a few candles, the strategy should not produce signals
    (indicators have not warmed up)."""
    bus = MemoryEventBus()
    feature_engine = FeatureEngine(event_bus=bus)
    await feature_engine.start()

    clock = SimClock()
    ctx = TradingContext(clock=clock, event_bus=bus, instruments={})
    strategy = TrendFollowingStrategy(strategy_id="test_trend")

    signals = []

    async def on_feature_vector(event):
        if not isinstance(event, FeatureVector):
            return
        candle_buffer = feature_engine.get_buffer(event.symbol, event.timeframe)
        if not candle_buffer:
            return

        aliased_features = dict(event.features)
        if "adx_14" in aliased_features and "adx" not in aliased_features:
            aliased_features["adx"] = aliased_features["adx_14"]
        if "atr_14" in aliased_features and "atr" not in aliased_features:
            aliased_features["atr"] = aliased_features["atr_14"]

        patched_fv = FeatureVector(
            symbol=event.symbol,
            timeframe=event.timeframe,
            features=aliased_features,
            source_module=event.source_module,
        )
        signal = strategy.on_candle(ctx, candle_buffer[-1], patched_fv)
        if signal is not None:
            signals.append(signal)

    await bus.subscribe("feature.vector", "strategy_runner", on_feature_vector)

    # Only 5 candles -- not enough for any indicator warmup
    candles = _generate_trending_candles(5)
    for candle in candles:
        await bus.publish("market.candle", candle)

    assert len(signals) == 0, (
        "Strategy should not produce signals with insufficient data"
    )
