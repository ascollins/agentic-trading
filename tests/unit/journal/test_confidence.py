"""Tests for ConfidenceCalibrator — signal quality vs outcome."""

import pytest

from agentic_trading.journal.confidence import ConfidenceCalibrator


class TestConfidenceCalibratorBasics:
    """Test basic recording and reporting."""

    def test_empty_report(self, confidence_calibrator):
        report = confidence_calibrator.report("nonexistent")
        assert report["total_observations"] == 0
        assert report["brier_score"] == 1.0

    def test_single_observation(self, confidence_calibrator):
        confidence_calibrator.record("trend", confidence=0.8, won=True, r_multiple=2.0)
        report = confidence_calibrator.report("trend")
        assert report["total_observations"] == 1

    def test_confidence_clamped_to_0_1(self, confidence_calibrator):
        confidence_calibrator.record("trend", confidence=1.5, won=True)
        confidence_calibrator.record("trend", confidence=-0.3, won=False)
        report = confidence_calibrator.report("trend")
        assert report["total_observations"] == 2


class TestConfidenceCalibratorBuckets:
    """Test bucket computation and calibration metrics."""

    def test_bucket_count_matches_config(self):
        cal = ConfidenceCalibrator(n_buckets=4)
        # Add observations across the range
        for i in range(100):
            conf = i / 100.0
            won = conf > 0.5
            cal.record("trend", confidence=conf, won=won)
        report = cal.report("trend")
        assert len(report["buckets"]) == 4

    def test_well_calibrated_low_brier(self):
        cal = ConfidenceCalibrator(n_buckets=5)
        import random
        rng = random.Random(42)
        # Perfect calibration: P(win|conf) ≈ conf
        for _ in range(200):
            conf = rng.random()
            won = rng.random() < conf
            cal.record("perfect", confidence=conf, won=won, r_multiple=1.0 if won else -0.5)
        report = cal.report("perfect")
        # Brier score should be reasonable (< 0.3 for well-calibrated)
        assert report["brier_score"] < 0.30

    def test_overconfident_strategy(self):
        cal = ConfidenceCalibrator(n_buckets=5)
        # Always high confidence but only 30% win rate
        for i in range(100):
            cal.record("overconf", confidence=0.85, won=(i % 3 == 0))
        report = cal.report("overconf")
        assert report["overconfidence_ratio"] > 0.0

    def test_is_well_calibrated(self, confidence_calibrator):
        # Not enough data
        confidence_calibrator.record("trend", confidence=0.8, won=True)
        assert confidence_calibrator.is_well_calibrated("trend") is False
        assert confidence_calibrator.is_well_calibrated(
            "trend", min_observations=1, max_brier=0.5
        ) is True


class TestConfidenceEdge:
    """Test confidence-outcome correlation."""

    def test_positive_edge(self):
        """High confidence predicts better outcomes."""
        cal = ConfidenceCalibrator(n_buckets=5)
        import random
        rng = random.Random(123)
        for _ in range(200):
            conf = rng.random()
            # Higher confidence → higher win rate
            won = rng.random() < (0.3 + 0.5 * conf)
            cal.record("good", confidence=conf, won=won)
        edge = cal.get_confidence_edge("good")
        assert edge > 0  # Positive correlation

    def test_no_edge(self):
        """Confidence is uncorrelated with outcomes."""
        cal = ConfidenceCalibrator(n_buckets=5)
        import random
        rng = random.Random(456)
        for _ in range(200):
            conf = rng.random()
            won = rng.random() < 0.5  # 50/50 regardless of confidence
            cal.record("random", confidence=conf, won=won)
        edge = cal.get_confidence_edge("random")
        assert abs(edge) < 0.5  # Should be near zero

    def test_insufficient_data_edge(self, confidence_calibrator):
        confidence_calibrator.record("trend", 0.8, True)
        assert confidence_calibrator.get_confidence_edge("trend") == 0.0

    def test_get_all_strategy_ids(self, confidence_calibrator):
        confidence_calibrator.record("a", 0.5, True)
        confidence_calibrator.record("b", 0.5, False)
        assert set(confidence_calibrator.get_all_strategy_ids()) == {"a", "b"}


class TestConfidenceEviction:
    """Test FIFO observation eviction."""

    def test_max_observations(self):
        cal = ConfidenceCalibrator(max_observations=10)
        for i in range(20):
            cal.record("trend", confidence=0.5, won=True)
        report = cal.report("trend")
        assert report["total_observations"] == 10
