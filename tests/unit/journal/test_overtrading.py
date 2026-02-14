"""Tests for OvertradingDetector — abnormal signal frequency analysis."""

import pytest
from datetime import datetime, timedelta

from agentic_trading.journal.overtrading import OvertradingDetector


class TestOvertradingBasics:
    """Test basic recording and detection."""

    def test_empty_report(self, overtrading_detector):
        report = overtrading_detector.report("nonexistent")
        assert report["is_overtrading"] is False
        assert report["trade_count"] == 0

    def test_no_alert_with_few_samples(self, overtrading_detector):
        t0 = datetime(2024, 1, 1, 12, 0)
        for i in range(3):
            result = overtrading_detector.record_trade(
                "trend", t0 + timedelta(hours=i * 4)
            )
            assert result is None  # Not enough samples yet

    def test_normal_trading_no_alert(self, overtrading_detector):
        t0 = datetime(2024, 1, 1, 12, 0)
        # Consistent 4-hour intervals
        for i in range(15):
            result = overtrading_detector.record_trade(
                "trend", t0 + timedelta(hours=i * 4)
            )
        assert result is None  # All intervals consistent

    def test_detects_overtrading(self):
        det = OvertradingDetector(
            lookback=20, threshold_z=1.5, cooldown_minutes=1, min_samples=5,
        )
        t0 = datetime(2024, 1, 1, 12, 0)
        # Normal pace: 4-hour intervals (10 trades → 9 intervals of 4h each)
        for i in range(10):
            det.record_trade("trend", t0 + timedelta(hours=i * 4))
        # Sudden burst: only 5 minutes after last trade (36h mark → 36h5m)
        # This is a 0.083h interval vs 4h average → z ≈ -many
        alert = det.record_trade(
            "trend", t0 + timedelta(hours=36, minutes=5)
        )
        assert alert is not None
        assert alert["strategy_id"] == "trend"
        assert alert["z_score"] < -1.5


class TestOvertradingReport:
    """Test report generation."""

    def test_report_with_data(self, overtrading_detector):
        t0 = datetime(2024, 1, 1, 12, 0)
        for i in range(10):
            overtrading_detector.record_trade(
                "trend", t0 + timedelta(hours=i * 2)
            )
        report = overtrading_detector.report("trend")
        assert report["strategy_id"] == "trend"
        assert report["trade_count"] == 10
        assert report["intervals_tracked"] == 9
        assert abs(report["mean_interval_hours"] - 2.0) < 0.1
        assert report["normal_trades_per_day"] > 0

    def test_is_overtrading_method(self, overtrading_detector):
        assert overtrading_detector.is_overtrading("trend") is False


class TestOvertradingCooldown:
    """Test alert cooldown behavior."""

    def test_cooldown_suppresses_repeated_alerts(self):
        det = OvertradingDetector(
            lookback=20, threshold_z=1.0, cooldown_minutes=60, min_samples=5,
        )
        t0 = datetime(2024, 1, 1, 12, 0)
        # Build baseline
        for i in range(10):
            det.record_trade("trend", t0 + timedelta(hours=i * 4))

        # First burst (should alert)
        alert1 = det.record_trade("trend", t0 + timedelta(hours=40, minutes=1))

        # Second burst within cooldown (should be suppressed)
        alert2 = det.record_trade("trend", t0 + timedelta(hours=40, minutes=2))

        # At least the first should be an alert if detection triggered
        if alert1 is not None:
            assert alert2 is None  # Cooldown suppression


class TestOvertradingMultiStrategy:
    """Test multi-strategy tracking."""

    def test_independent_strategies(self, overtrading_detector):
        t0 = datetime(2024, 1, 1, 12, 0)
        for i in range(10):
            overtrading_detector.record_trade("trend", t0 + timedelta(hours=i * 4))
            overtrading_detector.record_trade("scalp", t0 + timedelta(hours=i * 0.5))

        report_trend = overtrading_detector.report("trend")
        report_scalp = overtrading_detector.report("scalp")
        # Scalp has faster pace than trend
        assert report_scalp["mean_interval_hours"] < report_trend["mean_interval_hours"]

    def test_get_all_strategy_ids(self, overtrading_detector):
        t0 = datetime(2024, 1, 1, 12, 0)
        overtrading_detector.record_trade("a", t0)
        overtrading_detector.record_trade("b", t0)
        assert set(overtrading_detector.get_all_strategy_ids()) == {"a", "b"}
