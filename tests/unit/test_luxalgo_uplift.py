"""Tests for LuxAlgo feature-parity uplift.

Covers all 7 features added in the uplift:
1. Ichimoku Cloud indicator
2. HyperWave momentum oscillator
3. Session-based entry/exit features
4. Previous session/day/week high-low levels
5. Partial position exits
6. Tick-based TP/SL calculator
7. External notification dispatcher
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agentic_trading.core.enums import (
    Exchange,
    OrderType,
    RiskAlertSeverity,
    Side,
    Timeframe,
    TimeInForce,
)
from agentic_trading.core.events import OrderIntent, RiskAlert
from agentic_trading.core.models import Candle


# ===================================================================
# Helpers
# ===================================================================

def _make_candles(
    n: int,
    start: datetime | None = None,
    base_price: float = 100.0,
    interval_minutes: int = 60,
) -> list[Candle]:
    """Generate N synthetic candles with incrementing prices."""
    if start is None:
        start = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    candles = []
    for i in range(n):
        ts = start + timedelta(minutes=interval_minutes * i)
        price = base_price + i * 0.5
        candles.append(Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BYBIT,
            timeframe=Timeframe.H1,
            timestamp=ts,
            open=price - 0.2,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=100.0 + i,
        ))
    return candles


def _make_ohlcv(n: int, base: float = 100.0) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Return (high, low, close) arrays for N bars."""
    rng = np.random.default_rng(42)
    close = base + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    return high, low, close


# ===================================================================
# 1. Ichimoku Cloud
# ===================================================================

class TestIchimokuCloud:
    """Tests for compute_ichimoku()."""

    def test_output_keys(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_ichimoku

        high, low, close = _make_ohlcv(100)
        result = compute_ichimoku(high, low, close)
        assert set(result.keys()) == {
            "tenkan_sen", "kijun_sen", "senkou_span_a",
            "senkou_span_b", "chikou_span",
        }

    def test_array_lengths_match_input(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_ichimoku

        high, low, close = _make_ohlcv(100)
        result = compute_ichimoku(high, low, close)
        for key, arr in result.items():
            assert len(arr) == 100, f"{key} length mismatch"

    def test_tenkan_shorter_warmup_than_kijun(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_ichimoku

        high, low, close = _make_ohlcv(100)
        result = compute_ichimoku(high, low, close, tenkan_period=9, kijun_period=26)
        tenkan_valid = np.sum(~np.isnan(result["tenkan_sen"]))
        kijun_valid = np.sum(~np.isnan(result["kijun_sen"]))
        assert tenkan_valid > kijun_valid

    def test_senkou_a_is_average_of_tenkan_kijun(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_ichimoku

        high, low, close = _make_ohlcv(100)
        result = compute_ichimoku(high, low, close)
        # At the last bar, senkou_a should be (tenkan + kijun) / 2
        t = result["tenkan_sen"][-1]
        k = result["kijun_sen"][-1]
        sa = result["senkou_span_a"][-1]
        if not (np.isnan(t) or np.isnan(k)):
            assert abs(sa - (t + k) / 2.0) < 1e-10

    def test_chikou_span_is_close_shifted_back(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_ichimoku

        high, low, close = _make_ohlcv(100)
        result = compute_ichimoku(high, low, close, displacement=26)
        # chikou[0] should equal close[26]
        assert abs(result["chikou_span"][0] - close[26]) < 1e-10

    def test_short_input_returns_nans(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_ichimoku

        high, low, close = _make_ohlcv(5)
        result = compute_ichimoku(high, low, close, senkou_b_period=52)
        # With only 5 bars, senkou_b should be all NaN
        assert np.all(np.isnan(result["senkou_span_b"]))


# ===================================================================
# 2. HyperWave Momentum Oscillator
# ===================================================================

class TestHyperWave:
    """Tests for compute_hyperwave()."""

    def test_output_shapes(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_hyperwave

        high, low, close = _make_ohlcv(100)
        wave, signal, hist = compute_hyperwave(high, low, close)
        assert len(wave) == 100
        assert len(signal) == 100
        assert len(hist) == 100

    def test_histogram_is_wave_minus_signal(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_hyperwave

        high, low, close = _make_ohlcv(200)
        wave, signal, hist = compute_hyperwave(high, low, close)
        # Where both are valid, histogram = wave - signal
        valid = ~np.isnan(wave) & ~np.isnan(signal)
        if np.any(valid):
            np.testing.assert_allclose(
                hist[valid], wave[valid] - signal[valid], atol=1e-10
            )

    def test_short_input_returns_nans(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_hyperwave

        high, low, close = _make_ohlcv(10)
        wave, signal, hist = compute_hyperwave(high, low, close, slow_period=34)
        assert np.all(np.isnan(wave))
        assert np.all(np.isnan(signal))

    def test_values_are_bounded(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_hyperwave

        high, low, close = _make_ohlcv(200)
        wave, signal, hist = compute_hyperwave(high, low, close)
        valid_wave = wave[~np.isnan(wave)]
        # Normalized momentum oscillator should be bounded (typically ±3)
        assert np.all(np.abs(valid_wave) < 10.0)


# ===================================================================
# 3. Session-Based Entry/Exit Features
# ===================================================================

class TestSessionFeatures:
    """Tests for session time features in the feature engine."""

    def test_session_flags_in_feature_vector(self) -> None:
        from agentic_trading.intelligence.features.engine import FeatureEngine

        engine = FeatureEngine(indicator_config={"smc_enabled": False})
        # Create candles starting at 10:00 UTC, 1h apart, 30 bars
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        candles = _make_candles(30, start=start)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        f = fv.features

        # Last candle is at 10+29=39 → 39 mod 24 = 15 UTC
        last_hour = candles[-1].timestamp.hour
        assert f["hour_utc"] == float(last_hour)
        assert f["hour_utc"] == 15.0

    def test_asia_session_flag(self) -> None:
        from agentic_trading.intelligence.features.engine import FeatureEngine

        engine = FeatureEngine(indicator_config={"smc_enabled": False})
        start = datetime(2024, 1, 15, 3, 0, tzinfo=timezone.utc)
        candles = _make_candles(2, start=start, interval_minutes=5)
        fv = engine.compute_features("BTC/USDT", Timeframe.M5, candles)
        assert fv.features["is_asia_session"] == 1.0

    def test_london_ny_overlap_flag(self) -> None:
        from agentic_trading.intelligence.features.engine import FeatureEngine

        engine = FeatureEngine(indicator_config={"smc_enabled": False})
        start = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)
        candles = _make_candles(2, start=start, interval_minutes=5)
        fv = engine.compute_features("BTC/USDT", Timeframe.M5, candles)
        assert fv.features["is_london_ny_overlap"] == 1.0
        assert fv.features["is_london_session"] == 1.0
        assert fv.features["is_new_york_session"] == 1.0

    def test_day_of_week(self) -> None:
        from agentic_trading.intelligence.features.engine import FeatureEngine

        engine = FeatureEngine(indicator_config={"smc_enabled": False})
        # Jan 15 2024 is a Monday
        start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        candles = _make_candles(2, start=start, interval_minutes=5)
        fv = engine.compute_features("BTC/USDT", Timeframe.M5, candles)
        assert fv.features["day_of_week"] == 1.0  # Monday


# ===================================================================
# 4. Previous Session / Day / Week High-Low Levels
# ===================================================================

class TestSessionLevels:
    """Tests for compute_session_levels()."""

    def test_prev_day_high_low(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_session_levels

        # Day 1: bars 0-23 (Jan 15), Day 2: bars 24-25 (Jan 16)
        start = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=i) for i in range(26)]
        highs = np.array([100.0 + i for i in range(26)])
        lows = np.array([90.0 + i for i in range(26)])
        closes = np.array([95.0 + i for i in range(26)])

        result = compute_session_levels(timestamps, highs, lows, closes)

        # Previous day = Jan 15, bars 0-23
        assert result["prev_day_high"] == 123.0  # 100 + 23
        assert result["prev_day_low"] == 90.0    # 90 + 0
        assert "prev_day_close" in result

    def test_prev_session_levels(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_session_levels

        start = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        timestamps = [start + timedelta(hours=i) for i in range(26)]
        highs = np.array([100.0 + i for i in range(26)])
        lows = np.array([90.0 + i for i in range(26)])
        closes = np.array([95.0 + i for i in range(26)])

        result = compute_session_levels(timestamps, highs, lows, closes)

        assert "prev_asia_high" in result
        assert "prev_london_high" in result
        assert "prev_new_york_high" in result

    def test_insufficient_data_returns_nans(self) -> None:
        from agentic_trading.intelligence.features.indicators import compute_session_levels

        ts = [datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)]
        result = compute_session_levels(ts, np.array([100.0]), np.array([90.0]), np.array([95.0]))
        assert np.isnan(result["prev_day_high"])
        assert np.isnan(result["prev_week_high"])


# ===================================================================
# 5. Partial Position Exits
# ===================================================================

class TestPartialExits:
    """Tests for build_partial_exit_intents()."""

    def test_three_part_exit(self) -> None:
        from agentic_trading.signal.portfolio.intent_converter import (
            build_partial_exit_intents,
        )

        ts = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        intents = build_partial_exit_intents(
            symbol="BTC/USDT",
            strategy_id="trend_following",
            exchange=Exchange.BYBIT,
            current_qty=Decimal("1.0"),
            portions=[0.5, 0.25, 0.25],
            timestamp=ts,
        )

        assert len(intents) == 3
        assert intents[0].qty == Decimal("0.500000")
        assert intents[1].qty == Decimal("0.250000")
        assert intents[2].qty == Decimal("0.250000")

    def test_all_intents_are_reduce_only(self) -> None:
        from agentic_trading.signal.portfolio.intent_converter import (
            build_partial_exit_intents,
        )

        ts = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        intents = build_partial_exit_intents(
            symbol="ETH/USDT",
            strategy_id="mean_reversion",
            exchange=Exchange.BYBIT,
            current_qty=Decimal("10.0"),
            portions=[0.5, 0.5],
            timestamp=ts,
        )
        for intent in intents:
            assert intent.reduce_only is True

    def test_exit_portion_is_set(self) -> None:
        from agentic_trading.signal.portfolio.intent_converter import (
            build_partial_exit_intents,
        )

        ts = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        intents = build_partial_exit_intents(
            symbol="BTC/USDT",
            strategy_id="breakout",
            exchange=Exchange.BYBIT,
            current_qty=Decimal("2.0"),
            portions=[0.5, 0.5],
            timestamp=ts,
        )
        assert intents[0].exit_portion == 0.5
        assert intents[1].exit_portion == 0.5

    def test_unique_dedupe_keys(self) -> None:
        from agentic_trading.signal.portfolio.intent_converter import (
            build_partial_exit_intents,
        )

        ts = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        intents = build_partial_exit_intents(
            symbol="BTC/USDT",
            strategy_id="trend_following",
            exchange=Exchange.BYBIT,
            current_qty=Decimal("1.0"),
            portions=[0.5, 0.3, 0.2],
            timestamp=ts,
        )
        keys = [i.dedupe_key for i in intents]
        assert len(keys) == len(set(keys)), "Dedupe keys must be unique"

    def test_invalid_portions_skipped(self) -> None:
        from agentic_trading.signal.portfolio.intent_converter import (
            build_partial_exit_intents,
        )

        ts = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
        intents = build_partial_exit_intents(
            symbol="BTC/USDT",
            strategy_id="trend_following",
            exchange=Exchange.BYBIT,
            current_qty=Decimal("1.0"),
            portions=[0.0, -0.5, 1.5, 0.5],
            timestamp=ts,
        )
        assert len(intents) == 1
        assert intents[0].qty == Decimal("0.500000")

    def test_exit_portion_on_order_intent_model(self) -> None:
        """Verify the exit_portion field exists on OrderIntent."""
        intent = OrderIntent(
            dedupe_key="test123",
            strategy_id="s1",
            symbol="BTC/USDT",
            exchange=Exchange.BYBIT,
            side=Side.SELL,
            qty=Decimal("0.5"),
            exit_portion=0.5,
        )
        assert intent.exit_portion == 0.5

    def test_exit_portion_defaults_to_none(self) -> None:
        intent = OrderIntent(
            dedupe_key="test456",
            strategy_id="s1",
            symbol="BTC/USDT",
            exchange=Exchange.BYBIT,
            side=Side.BUY,
            qty=Decimal("1.0"),
        )
        assert intent.exit_portion is None


# ===================================================================
# 6. Tick-Based TP/SL Calculator
# ===================================================================

class TestTpSlCalculator:
    """Tests for compute_tpsl()."""

    def test_ticks_method_long(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        levels = compute_tpsl(
            entry_price=Decimal("50000"),
            side="long",
            method="ticks",
            tp_distance=200,
            sl_distance=100,
            tick_size=Decimal("0.01"),
        )
        assert levels.take_profit == Decimal("50002.00")
        assert levels.stop_loss == Decimal("49999.00")

    def test_ticks_method_short(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        levels = compute_tpsl(
            entry_price=Decimal("50000"),
            side="short",
            method="ticks",
            tp_distance=200,
            sl_distance=100,
            tick_size=Decimal("0.01"),
        )
        assert levels.take_profit == Decimal("49998.00")
        assert levels.stop_loss == Decimal("50001.00")

    def test_percentage_method_long(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        levels = compute_tpsl(
            entry_price=Decimal("1000"),
            side="long",
            method="percentage",
            tp_distance=2.0,
            sl_distance=1.0,
        )
        assert levels.take_profit == Decimal("1020")
        assert levels.stop_loss == Decimal("990")

    def test_atr_method(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        levels = compute_tpsl(
            entry_price=Decimal("100"),
            side="long",
            method="atr",
            tp_distance=3.0,
            sl_distance=1.5,
            atr_value=Decimal("2.0"),
        )
        assert levels.take_profit == Decimal("106.0")
        assert levels.stop_loss == Decimal("97.0")

    def test_currency_method(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        levels = compute_tpsl(
            entry_price=Decimal("50000"),
            side="long",
            method="currency",
            tp_distance=500,
            sl_distance=250,
        )
        assert levels.take_profit == Decimal("50500")
        assert levels.stop_loss == Decimal("49750")

    def test_price_method(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        levels = compute_tpsl(
            entry_price=Decimal("50000"),
            side="long",
            method="price",
            tp_distance=51000,
            sl_distance=49000,
        )
        assert levels.take_profit == Decimal("51000")
        assert levels.stop_loss == Decimal("49000")

    def test_trailing_stop_ticks(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        levels = compute_tpsl(
            entry_price=Decimal("50000"),
            side="long",
            method="ticks",
            trailing_distance=50,
            tick_size=Decimal("0.01"),
        )
        assert levels.trailing_stop_distance == Decimal("0.50")
        assert levels.take_profit is None
        assert levels.stop_loss is None

    def test_missing_tick_size_raises(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        with pytest.raises(ValueError, match="tick_size"):
            compute_tpsl(
                entry_price=Decimal("100"),
                side="long",
                method="ticks",
                tp_distance=10,
            )

    def test_missing_atr_raises(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        with pytest.raises(ValueError, match="atr_value"):
            compute_tpsl(
                entry_price=Decimal("100"),
                side="long",
                method="atr",
                tp_distance=2.0,
            )

    def test_unknown_method_raises(self) -> None:
        from agentic_trading.execution.tpsl_calculator import compute_tpsl

        with pytest.raises(ValueError, match="Unknown"):
            compute_tpsl(
                entry_price=Decimal("100"),
                side="long",
                method="magic",
            )


# ===================================================================
# 7. External Notification Dispatcher
# ===================================================================

class TestAlertNotifier:
    """Tests for AlertNotifier."""

    def _make_alert(
        self,
        severity: RiskAlertSeverity = RiskAlertSeverity.WARNING,
    ) -> RiskAlert:
        return RiskAlert(
            severity=severity,
            alert_type="test_rule",
            message="Test alert message",
            details={"key": "value"},
        )

    def test_no_channels_dispatches_zero(self) -> None:
        from agentic_trading.execution.risk.notifier import AlertNotifier

        notifier = AlertNotifier()
        alert = self._make_alert()
        result = asyncio.get_event_loop().run_until_complete(
            notifier.dispatch(alert)
        )
        assert result == 0
        assert notifier.channel_count == 0

    def test_callback_channel_dispatches(self) -> None:
        from agentic_trading.execution.risk.notifier import AlertNotifier

        notifier = AlertNotifier()
        received: list[RiskAlert] = []

        async def mock_callback(alert: RiskAlert) -> bool:
            received.append(alert)
            return True

        notifier.add_callback("test_cb", mock_callback)
        alert = self._make_alert()
        result = asyncio.get_event_loop().run_until_complete(
            notifier.dispatch(alert)
        )
        assert result == 1
        assert len(received) == 1
        assert received[0].alert_type == "test_rule"

    def test_severity_filter_skips_low_severity(self) -> None:
        from agentic_trading.execution.risk.notifier import AlertNotifier

        notifier = AlertNotifier()
        received: list[RiskAlert] = []

        async def mock_callback(alert: RiskAlert) -> bool:
            received.append(alert)
            return True

        # Only receive CRITICAL and above
        notifier.add_callback(
            "critical_only", mock_callback,
            severity_min=RiskAlertSeverity.CRITICAL.value,
        )
        warning_alert = self._make_alert(RiskAlertSeverity.WARNING)
        result = asyncio.get_event_loop().run_until_complete(
            notifier.dispatch(warning_alert)
        )
        assert result == 0
        assert len(received) == 0

    def test_severity_filter_passes_high_severity(self) -> None:
        from agentic_trading.execution.risk.notifier import AlertNotifier

        notifier = AlertNotifier()
        received: list[RiskAlert] = []

        async def mock_callback(alert: RiskAlert) -> bool:
            received.append(alert)
            return True

        notifier.add_callback(
            "critical_only", mock_callback,
            severity_min=RiskAlertSeverity.CRITICAL.value,
        )
        critical_alert = self._make_alert(RiskAlertSeverity.CRITICAL)
        result = asyncio.get_event_loop().run_until_complete(
            notifier.dispatch(critical_alert)
        )
        assert result == 1
        assert len(received) == 1

    def test_webhook_channel_registration(self) -> None:
        from agentic_trading.execution.risk.notifier import AlertNotifier

        notifier = AlertNotifier()
        notifier.add_webhook("https://hooks.example.com/alert")
        assert notifier.channel_count == 1

    def test_email_channel_registration(self) -> None:
        from agentic_trading.execution.risk.notifier import AlertNotifier

        notifier = AlertNotifier()
        notifier.add_email(
            smtp_host="smtp.example.com",
            from_addr="alerts@example.com",
            to_addrs=["ops@example.com"],
        )
        assert notifier.channel_count == 1

    def test_multiple_channels(self) -> None:
        from agentic_trading.execution.risk.notifier import AlertNotifier

        notifier = AlertNotifier()
        notifier.add_webhook("https://hooks.example.com/alert")
        notifier.add_email(smtp_host="smtp.example.com")

        async def noop(alert: RiskAlert) -> bool:
            return True

        notifier.add_callback("noop", noop)
        assert notifier.channel_count == 3

    def test_channel_stats(self) -> None:
        from agentic_trading.execution.risk.notifier import AlertNotifier

        notifier = AlertNotifier()
        notifier.add_webhook("https://hooks.example.com")

        async def ok_cb(alert: RiskAlert) -> bool:
            return True

        notifier.add_callback("ok", ok_cb)

        stats = notifier.get_channel_stats()
        assert len(stats) == 2
        assert stats[0]["type"] == "webhook"
        assert stats[1]["type"] == "callback"

    def test_callback_error_counted(self) -> None:
        from agentic_trading.execution.risk.notifier import AlertNotifier

        notifier = AlertNotifier()

        async def failing_cb(alert: RiskAlert) -> bool:
            raise RuntimeError("boom")

        notifier.add_callback("failing", failing_cb)
        alert = self._make_alert()
        result = asyncio.get_event_loop().run_until_complete(
            notifier.dispatch(alert)
        )
        assert result == 0
        assert notifier.total_errors == 1


# ===================================================================
# Integration: Feature Engine includes new indicators
# ===================================================================

class TestFeatureEngineUplift:
    """Verify new indicators appear in feature vectors."""

    def test_ichimoku_features_present(self) -> None:
        from agentic_trading.intelligence.features.engine import FeatureEngine

        engine = FeatureEngine(indicator_config={"smc_enabled": False})
        candles = _make_candles(60, base_price=50000.0)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        f = fv.features

        assert "ichimoku_tenkan" in f
        assert "ichimoku_kijun" in f
        assert "ichimoku_senkou_a" in f
        assert "ichimoku_senkou_b" in f
        assert "ichimoku_cloud_sign" in f
        assert "ichimoku_price_location" in f

    def test_hyperwave_features_present(self) -> None:
        from agentic_trading.intelligence.features.engine import FeatureEngine

        engine = FeatureEngine(indicator_config={"smc_enabled": False})
        candles = _make_candles(60, base_price=50000.0)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        f = fv.features

        assert "hyperwave" in f
        assert "hyperwave_signal" in f
        assert "hyperwave_histogram" in f

    def test_session_time_features_present(self) -> None:
        from agentic_trading.intelligence.features.engine import FeatureEngine

        engine = FeatureEngine(indicator_config={"smc_enabled": False})
        candles = _make_candles(10, base_price=50000.0)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        f = fv.features

        assert "hour_utc" in f
        assert "minute_utc" in f
        assert "day_of_week" in f
        assert "is_asia_session" in f
        assert "is_london_session" in f
        assert "is_new_york_session" in f
        assert "is_london_ny_overlap" in f

    def test_session_levels_present(self) -> None:
        from agentic_trading.intelligence.features.engine import FeatureEngine

        engine = FeatureEngine(indicator_config={"smc_enabled": False})
        # Need candles spanning 2+ days for prev_day levels
        start = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        candles = _make_candles(30, start=start, base_price=50000.0)
        fv = engine.compute_features("BTC/USDT", Timeframe.H1, candles)
        f = fv.features

        assert "prev_day_high" in f
        assert "prev_day_low" in f
        assert "prev_day_close" in f
        assert "prev_asia_high" in f
        assert "prev_london_high" in f
        assert "prev_new_york_high" in f
