"""Test compute_metrics returns valid Sharpe, Sortino, max drawdown from sample equity curves."""

import math

import numpy as np
import pytest

from agentic_trading.backtester.results import BacktestResult, compute_metrics


class TestComputeMetrics:
    def test_empty_equity_returns_default(self):
        result = compute_metrics([], [])
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0

    def test_single_point_returns_default(self):
        result = compute_metrics([100_000.0], [])
        assert result.total_return == 0.0

    def test_positive_return(self):
        eq = [100_000.0, 101_000.0, 102_000.0, 103_000.0, 104_000.0]
        result = compute_metrics(eq, [0.01, 0.01, 0.01, 0.01])
        assert result.total_return > 0
        assert result.total_return == pytest.approx(0.04, abs=0.001)

    def test_negative_return(self):
        eq = [100_000.0, 99_000.0, 98_000.0, 97_000.0]
        result = compute_metrics(eq, [-0.01, -0.01, -0.01])
        assert result.total_return < 0

    def test_sharpe_ratio_computed(self):
        # Consistent positive returns -> positive Sharpe
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.005, 100)
        eq = [100_000.0]
        for r in returns:
            eq.append(eq[-1] * (1 + r))
        result = compute_metrics(eq, returns.tolist())
        assert result.sharpe_ratio != 0.0
        # Positive drift should give positive Sharpe
        assert result.sharpe_ratio > 0

    def test_sortino_ratio_computed(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.005, 100)
        eq = [100_000.0]
        for r in returns:
            eq.append(eq[-1] * (1 + r))
        result = compute_metrics(eq, returns.tolist())
        assert result.sortino_ratio != 0.0

    def test_max_drawdown_is_negative(self):
        eq = [100_000.0, 105_000.0, 95_000.0, 98_000.0, 103_000.0]
        result = compute_metrics(eq, [0.05, -0.095, 0.03, 0.05])
        assert result.max_drawdown < 0
        # Max drawdown should be around (95000 - 105000) / 105000 = -0.0952
        assert result.max_drawdown < -0.05

    def test_trade_metrics_with_wins_and_losses(self):
        eq = [100_000.0, 101_000.0, 99_500.0, 102_000.0]
        trade_returns = [500.0, -300.0, 800.0, -100.0, 200.0]
        result = compute_metrics(eq, trade_returns)
        assert result.total_trades == 5
        assert result.winning_trades == 3
        assert result.losing_trades == 2
        assert 0 < result.win_rate < 1

    def test_profit_factor(self):
        trade_returns = [500.0, -300.0, 800.0, -100.0]
        eq = [100_000.0, 101_000.0, 100_500.0, 101_500.0]
        result = compute_metrics(eq, trade_returns)
        # Gross profit: 1300, Gross loss: 400
        assert result.profit_factor == pytest.approx(1300 / 400, abs=0.01)

    def test_costs_stored(self):
        eq = [100_000.0, 101_000.0]
        result = compute_metrics(eq, [0.01], fees=50.0, slippage=10.0, funding=-5.0)
        assert result.total_fees == 50.0
        assert result.total_slippage == 10.0
        assert result.total_funding == -5.0

    def test_equity_curve_stored(self):
        eq = [100_000.0, 101_000.0, 102_000.0]
        result = compute_metrics(eq, [0.01, 0.01])
        assert result.equity_curve == eq

    def test_peak_equity(self):
        eq = [100_000.0, 110_000.0, 105_000.0, 115_000.0, 108_000.0]
        result = compute_metrics(eq, [0.1, -0.05, 0.095, -0.06])
        assert result.peak_equity == 115_000.0

    def test_daily_returns_length(self):
        eq = [100_000.0, 101_000.0, 102_000.0, 103_000.0]
        result = compute_metrics(eq, [])
        assert len(result.daily_returns) == 3  # len(eq) - 1


class TestBacktestResult:
    def test_summary_returns_dict(self):
        result = BacktestResult(
            total_return=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.08,
            total_trades=100,
            win_rate=0.55,
            profit_factor=1.8,
            total_fees=500.0,
            total_funding=-50.0,
        )
        summary = result.summary()
        assert isinstance(summary, dict)
        assert "total_return" in summary
        assert "sharpe" in summary
        assert "sortino" in summary
        assert "max_dd" in summary
