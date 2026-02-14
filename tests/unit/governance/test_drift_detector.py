"""Tests for governance.drift_detector — Live vs Backtest Drift Detection."""

import pytest

from agentic_trading.core.config import DriftDetectorConfig
from agentic_trading.core.enums import GovernanceAction
from agentic_trading.governance.drift_detector import DriftDetector


class TestBaselineManagement:
    """Baseline registration and querying."""

    def test_set_baseline(self, drift_detector):
        drift_detector.set_baseline("s1", {"win_rate": 0.55, "sharpe": 1.8})
        assert drift_detector.has_baseline("s1")

    def test_no_baseline_by_default(self, drift_detector):
        assert drift_detector.has_baseline("unknown") is False

    def test_only_tracked_metrics_stored(self, drift_detector):
        """Metrics not in config.metrics_tracked should be ignored."""
        drift_detector.set_baseline(
            "s1", {"win_rate": 0.55, "irrelevant_metric": 999.0}
        )
        status = drift_detector.get_status("s1")
        assert "irrelevant_metric" not in status["metrics"]

    def test_baseline_from_backtest(self, drift_detector):
        """load_baseline_from_backtest extracts named attributes."""

        class MockResult:
            win_rate = 0.52
            sharpe = 1.5
            profit_factor = 1.3
            avg_rr = 1.2

        drift_detector.load_baseline_from_backtest("s1", MockResult())
        assert drift_detector.has_baseline("s1")
        status = drift_detector.get_status("s1")
        assert status["metrics"]["win_rate"]["baseline"] == 0.52


class TestLiveMetricUpdates:
    """Live metric tracking."""

    def test_update_live_metric(self, drift_detector):
        drift_detector.update_live_metric("s1", "win_rate", 0.45)
        status = drift_detector.get_status("s1")
        assert status["metrics"]["win_rate"]["live"] == 0.45

    def test_batch_update(self, drift_detector):
        drift_detector.update_live_metrics("s1", {
            "win_rate": 0.50,
            "sharpe": 1.0,
        })
        status = drift_detector.get_status("s1")
        assert status["metrics"]["win_rate"]["live"] == 0.50
        assert status["metrics"]["sharpe"]["live"] == 1.0

    def test_untracked_metric_ignored(self, drift_detector):
        drift_detector.update_live_metric("s1", "fake_metric", 999.0)
        status = drift_detector.get_status("s1")
        assert "fake_metric" not in status["metrics"]


class TestDriftDetection:
    """Drift detection and alert generation."""

    def test_no_alerts_without_baseline(self, drift_detector):
        drift_detector.update_live_metric("s1", "win_rate", 0.30)
        alerts = drift_detector.check_drift("s1")
        assert len(alerts) == 0

    def test_no_alerts_within_threshold(self, drift_detector):
        drift_detector.set_baseline("s1", {"win_rate": 0.55})
        drift_detector.update_live_metric("s1", "win_rate", 0.50)
        alerts = drift_detector.check_drift("s1")
        # ~9% deviation, below 30% threshold
        assert len(alerts) == 0

    def test_reduce_size_on_moderate_drift(self, drift_detector):
        """30–50% deviation should trigger REDUCE_SIZE."""
        drift_detector.set_baseline("s1", {"win_rate": 0.55})
        # 36% deviation: (0.55 - 0.35) / 0.55 ≈ 36.4%
        drift_detector.update_live_metric("s1", "win_rate", 0.35)
        alerts = drift_detector.check_drift("s1")
        assert len(alerts) == 1
        assert alerts[0].action_taken == GovernanceAction.REDUCE_SIZE
        assert alerts[0].deviation_pct > 30.0

    def test_pause_on_severe_drift(self, drift_detector):
        """50%+ deviation should trigger PAUSE."""
        drift_detector.set_baseline("s1", {"win_rate": 0.55})
        # 63.6% deviation
        drift_detector.update_live_metric("s1", "win_rate", 0.20)
        alerts = drift_detector.check_drift("s1")
        assert len(alerts) == 1
        assert alerts[0].action_taken == GovernanceAction.PAUSE

    def test_multiple_metrics_drift(self, drift_detector):
        """Multiple metrics can drift independently."""
        drift_detector.set_baseline("s1", {
            "win_rate": 0.55,
            "profit_factor": 1.5,
        })
        drift_detector.update_live_metric("s1", "win_rate", 0.30)
        drift_detector.update_live_metric("s1", "profit_factor", 0.70)
        alerts = drift_detector.check_drift("s1")
        assert len(alerts) == 2

    def test_alert_contains_correct_values(self, drift_detector):
        drift_detector.set_baseline("s1", {"win_rate": 0.50})
        drift_detector.update_live_metric("s1", "win_rate", 0.30)
        alerts = drift_detector.check_drift("s1")
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.strategy_id == "s1"
        assert alert.metric_name == "win_rate"
        assert alert.baseline_value == 0.50
        assert alert.live_value == 0.30
        assert alert.deviation_pct == 40.0

    def test_zero_baseline_handles_gracefully(self, drift_detector):
        """Zero baseline should not cause division error."""
        drift_detector.set_baseline("s1", {"win_rate": 0.0})
        drift_detector.update_live_metric("s1", "win_rate", 0.5)
        alerts = drift_detector.check_drift("s1")
        # 100% deviation from zero
        assert len(alerts) >= 1


class TestDriftStatus:
    """Status reporting."""

    def test_status_without_baseline(self, drift_detector):
        status = drift_detector.get_status("unknown")
        assert status["has_baseline"] is False

    def test_status_with_partial_data(self, drift_detector):
        drift_detector.set_baseline("s1", {"win_rate": 0.55})
        # No live data yet
        status = drift_detector.get_status("s1")
        assert status["metrics"]["win_rate"]["baseline"] == 0.55
        assert status["metrics"]["win_rate"]["live"] is None
        assert status["metrics"]["win_rate"]["deviation_pct"] is None

    def test_custom_thresholds(self):
        cfg = DriftDetectorConfig(
            deviation_threshold_pct=10.0,
            pause_threshold_pct=20.0,
        )
        detector = DriftDetector(cfg)
        detector.set_baseline("s1", {"win_rate": 0.50})
        detector.update_live_metric("s1", "win_rate", 0.42)
        # 16% deviation → above 10% but below 20%
        alerts = detector.check_drift("s1")
        assert len(alerts) == 1
        assert alerts[0].action_taken == GovernanceAction.REDUCE_SIZE
