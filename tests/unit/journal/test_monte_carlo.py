"""Tests for MonteCarloProjector — equity projection and ruin analysis."""

import pytest

from agentic_trading.journal.monte_carlo import MonteCarloProjector


class TestMonteCarloBasics:
    """Test basic projection and data input."""

    def test_insufficient_data(self, monte_carlo):
        result = monte_carlo.project("trend")
        assert "error" in result
        assert result["error"] == "insufficient_data"

    def test_insufficient_data_few_trades(self, monte_carlo):
        monte_carlo.set_trades("trend", [10.0, -5.0, 8.0])
        result = monte_carlo.project("trend")
        assert result["error"] == "insufficient_data"
        assert result["current_trades"] == 3

    def test_basic_projection(self, monte_carlo):
        # Strategy with positive expectancy
        pnl = [50.0, -20.0, 30.0, -10.0, 40.0, -15.0, 25.0, -5.0, 35.0, 20.0]
        monte_carlo.set_trades("trend", pnl)
        result = monte_carlo.project("trend", initial_equity=100_000, n_trades=100)
        assert result["strategy_id"] == "trend"
        assert result["initial_equity"] == 100_000
        assert result["n_trades"] == 100
        assert result["source_trades"] == 10
        assert "ruin_probability" in result
        assert "median_terminal_equity" in result
        assert "percentile_5" in result
        assert "percentile_95" in result

    def test_add_trade_incremental(self, monte_carlo):
        for pnl in [50.0, -20.0, 30.0, -10.0, 40.0]:
            monte_carlo.add_trade("trend", pnl)
        result = monte_carlo.project("trend")
        assert result["source_trades"] == 5


class TestMonteCarloProjection:
    """Test projection quality and statistical properties."""

    def test_positive_expectancy_has_low_ruin(self):
        mc = MonteCarloProjector(n_simulations=500, seed=42)
        # Strongly positive edge
        pnl = [100.0, 80.0, -20.0, 120.0, 90.0, -10.0, 70.0, 110.0, -30.0, 60.0]
        mc.set_trades("winner", pnl)
        result = mc.project("winner", initial_equity=100_000, n_trades=200)
        assert result["ruin_probability"] < 0.1
        assert result["median_terminal_equity"] > 100_000
        assert result["profit_probability"] > 0.5

    def test_negative_expectancy_has_high_ruin(self):
        mc = MonteCarloProjector(n_simulations=500, seed=42, ruin_threshold_pct=0.3)
        # Strongly negative edge: avg -320/trade, enough to blow through 30% ruin
        pnl = [-500.0, -600.0, 100.0, -400.0, -700.0, 50.0, -300.0, -550.0, 150.0, -450.0]
        mc.set_trades("loser", pnl)
        result = mc.project("loser", initial_equity=100_000, n_trades=500)
        assert result["ruin_probability"] > 0.3
        assert result["median_terminal_equity"] < 100_000

    def test_percentile_ordering(self, monte_carlo):
        pnl = [50.0, -20.0, 30.0, -10.0, 40.0, -15.0, 25.0, -5.0, 35.0, 20.0]
        monte_carlo.set_trades("trend", pnl)
        result = monte_carlo.project("trend", initial_equity=100_000, n_trades=100)
        assert result["percentile_5"] <= result["percentile_25"]
        assert result["percentile_25"] <= result["median_terminal_equity"]
        assert result["median_terminal_equity"] <= result["percentile_75"]
        assert result["percentile_75"] <= result["percentile_95"]

    def test_equity_bands_generated(self, monte_carlo):
        pnl = [50.0, -20.0, 30.0, -10.0, 40.0, -15.0, 25.0, -5.0, 35.0, 20.0]
        monte_carlo.set_trades("trend", pnl)
        result = monte_carlo.project("trend", initial_equity=100_000, n_trades=100)
        bands = result["equity_bands"]
        assert len(bands) > 0
        # Each band should have p5 <= p25 <= p50 <= p75 <= p95
        for band in bands:
            assert band["p5"] <= band["p25"]
            assert band["p25"] <= band["p50"]
            assert band["p50"] <= band["p75"]
            assert band["p75"] <= band["p95"]

    def test_drawdown_distribution(self, monte_carlo):
        pnl = [50.0, -20.0, 30.0, -10.0, 40.0, -15.0, 25.0, -5.0, 35.0, 20.0]
        monte_carlo.set_trades("trend", pnl)
        result = monte_carlo.project("trend", initial_equity=100_000, n_trades=200)
        assert result["mean_max_drawdown"] >= 0
        assert result["median_max_drawdown"] >= 0
        assert result["worst_case_drawdown_95"] >= result["median_max_drawdown"]


class TestKellyFraction:
    """Test Kelly criterion computation."""

    def test_kelly_positive_edge(self):
        mc = MonteCarloProjector(seed=42)
        # 60% win rate, 2:1 avg win/loss
        pnl = ([100.0] * 6 + [-50.0] * 4) * 5  # 50 trades
        mc.set_trades("edge", pnl)
        kelly = mc.kelly_fraction("edge")
        assert kelly > 0.0
        assert kelly < 1.0

    def test_kelly_zero_for_negative_edge(self):
        mc = MonteCarloProjector(seed=42)
        # 30% win rate, 1:1 avg win/loss
        pnl = ([50.0] * 3 + [-50.0] * 7) * 5  # 50 trades
        mc.set_trades("loser", pnl)
        kelly = mc.kelly_fraction("loser")
        assert kelly == 0.0

    def test_kelly_insufficient_data(self, monte_carlo):
        monte_carlo.set_trades("new", [10.0, -5.0])
        assert monte_carlo.kelly_fraction("new") == 0.0

    def test_kelly_nonexistent_strategy(self, monte_carlo):
        assert monte_carlo.kelly_fraction("ghost") == 0.0


class TestMonteCarloReproducibility:
    """Test that seeded MC produces consistent results."""

    def test_reproducible_with_seed(self):
        mc1 = MonteCarloProjector(n_simulations=100, seed=42)
        mc2 = MonteCarloProjector(n_simulations=100, seed=42)
        pnl = [50.0, -20.0, 30.0, -10.0, 40.0, -15.0, 25.0, -5.0, 35.0, 20.0]
        mc1.set_trades("t", pnl)
        mc2.set_trades("t", pnl)
        r1 = mc1.project("t", n_trades=50)
        r2 = mc2.project("t", n_trades=50)
        assert r1["median_terminal_equity"] == r2["median_terminal_equity"]
        assert r1["ruin_probability"] == r2["ruin_probability"]
