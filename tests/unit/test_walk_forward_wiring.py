"""Tests for walk-forward validation wiring."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from agentic_trading.strategies.research.walk_forward import (
    WalkForwardValidator,
    WalkForwardResult,
    WalkForwardReport,
    WalkForwardWindow,
)


class TestWalkForwardValidator:
    """Test WalkForwardValidator window creation and evaluation."""

    def _make_timestamps(self, n: int = 1000) -> list[datetime]:
        """Create N timestamps starting from 2024-01-01."""
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return [base + timedelta(minutes=i) for i in range(n)]

    def test_create_windows_basic(self):
        """Should create the expected number of windows."""
        timestamps = self._make_timestamps(1000)
        validator = WalkForwardValidator(n_folds=5, train_pct=0.7)
        windows = validator.create_windows(timestamps)

        assert len(windows) > 0
        assert len(windows) <= 5

    def test_create_windows_requires_minimum_data(self):
        """Should raise if fewer than 100 data points."""
        timestamps = self._make_timestamps(50)
        validator = WalkForwardValidator(n_folds=5)

        with pytest.raises(ValueError, match="at least 100"):
            validator.create_windows(timestamps)

    def test_windows_temporal_ordering(self):
        """Train end should be before test start in each window."""
        timestamps = self._make_timestamps(1000)
        validator = WalkForwardValidator(n_folds=3, train_pct=0.7, gap_periods=1)
        windows = validator.create_windows(timestamps)

        for w in windows:
            assert w.train_start <= w.train_end
            assert w.train_end < w.test_start
            assert w.test_start <= w.test_end

    def test_anchored_windows(self):
        """Anchored windows should all start from the first timestamp."""
        timestamps = self._make_timestamps(1000)
        validator = WalkForwardValidator(n_folds=3, anchored=True)
        windows = validator.create_windows(timestamps)

        first_ts = timestamps[0]
        for w in windows:
            assert w.train_start == first_ts

    def test_evaluate_no_overfit(self):
        """When test Sharpe ≈ train Sharpe, overfit score should be low."""
        results = [
            WalkForwardResult(
                fold_index=i,
                train_sharpe=1.5,
                test_sharpe=1.4,
                train_return=0.1,
                test_return=0.09,
                train_max_dd=-0.05,
                test_max_dd=-0.06,
            )
            for i in range(3)
        ]

        validator = WalkForwardValidator(n_folds=3)
        report = validator.evaluate(results)

        assert report.overfit_score < 0.2
        assert not report.is_overfit
        assert report.avg_train_sharpe == pytest.approx(1.5, abs=0.01)
        assert report.avg_test_sharpe == pytest.approx(1.4, abs=0.01)

    def test_evaluate_overfit_detected(self):
        """When test Sharpe << train Sharpe, overfit should be detected."""
        results = [
            WalkForwardResult(
                fold_index=i,
                train_sharpe=3.0,
                test_sharpe=0.5,
                train_return=0.5,
                test_return=0.02,
                train_max_dd=-0.05,
                test_max_dd=-0.15,
            )
            for i in range(3)
        ]

        validator = WalkForwardValidator(n_folds=3)
        report = validator.evaluate(results)

        assert report.overfit_score > 0.5
        assert report.is_overfit
        assert report.degradation_pct > 50

    def test_evaluate_empty(self):
        """Empty fold results should return an empty report."""
        validator = WalkForwardValidator()
        report = validator.evaluate([])
        assert len(report.folds) == 0

    def test_compute_sharpe(self):
        """Sharpe computation should be correct."""
        returns = [0.01, 0.02, -0.005, 0.015, 0.01]
        sharpe = WalkForwardValidator.compute_sharpe(returns)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive returns should give positive Sharpe

    def test_compute_sharpe_empty(self):
        """Empty returns should give zero Sharpe."""
        assert WalkForwardValidator.compute_sharpe([]) == 0.0
        assert WalkForwardValidator.compute_sharpe([0.01]) == 0.0  # Need at least 2

    def test_compute_max_drawdown(self):
        """Max drawdown should be negative for a drawdown."""
        returns = [0.1, -0.2, 0.05, -0.15]  # Has a drawdown
        dd = WalkForwardValidator.compute_max_drawdown(returns)
        assert dd < 0
