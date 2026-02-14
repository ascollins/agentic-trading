"""Tests for CoinFlipBaseline — statistical edge verification."""

import pytest

from agentic_trading.journal.coin_flip import CoinFlipBaseline


class TestCoinFlipBasics:
    """Test basic evaluation."""

    def test_insufficient_data(self, coin_flip):
        result = coin_flip.evaluate("trend")
        assert result["has_edge"] is False
        assert "error" in result

    def test_few_trades(self, coin_flip):
        coin_flip.add_trades("trend", [10.0, -5.0, 8.0])
        result = coin_flip.evaluate("trend")
        assert result["has_edge"] is False

    def test_basic_evaluation_structure(self, coin_flip):
        pnl = [50.0, -20.0, 30.0, -10.0, 40.0, -15.0, 25.0, -5.0, 35.0, 20.0]
        coin_flip.add_trades("trend", pnl)
        result = coin_flip.evaluate("trend")
        assert "has_edge" in result
        assert "p_value_win_rate" in result
        assert "p_value_sign_test" in result
        assert "p_value_bootstrap" in result
        assert "cohens_d" in result
        assert "effect_size" in result
        assert "significant_tests" in result


class TestCoinFlipEdgeDetection:
    """Test that strong edges are detected and noise is rejected."""

    def test_strong_positive_edge(self):
        """A very profitable strategy should show statistical significance."""
        baseline = CoinFlipBaseline(n_simulations=5000, seed=42)
        # Highly profitable: 80% win rate with good ratio
        pnl = ([100.0] * 8 + [-50.0] * 2) * 10  # 100 trades
        baseline.add_trades("winner", pnl)
        result = baseline.evaluate("winner")
        assert result["has_edge"] is True
        assert result["p_value_win_rate"] < 0.05
        assert result["win_rate"] > 0.5
        assert result["random_better_pct"] < 0.1

    def test_no_edge_random(self):
        """A 50/50 strategy should not show an edge."""
        baseline = CoinFlipBaseline(n_simulations=5000, seed=42)
        import random
        rng = random.Random(99)
        pnl = [rng.choice([50.0, -50.0]) for _ in range(100)]
        baseline.add_trades("random", pnl)
        result = baseline.evaluate("random")
        # Should not confidently declare an edge
        assert result["p_value_bootstrap"] > 0.01

    def test_negative_edge_detected(self):
        """A losing strategy should not have an edge."""
        baseline = CoinFlipBaseline(n_simulations=5000, seed=42)
        # 30% win rate
        pnl = ([50.0] * 3 + [-50.0] * 7) * 10
        baseline.add_trades("loser", pnl)
        result = baseline.evaluate("loser")
        assert result["has_edge"] is False
        assert result["mean_pnl"] < 0


class TestCoinFlipStatistics:
    """Test individual statistical tests."""

    def test_effect_size_categories(self, coin_flip):
        # Large effect
        pnl = ([200.0] * 8 + [-20.0] * 2) * 10
        coin_flip.add_trades("large", pnl)
        result = coin_flip.evaluate("large")
        assert result["effect_size"] in ("large", "medium")

    def test_sample_adequacy(self, coin_flip):
        # With 60% win rate, need ~200 trades for power=0.80
        pnl = ([50.0] * 6 + [-50.0] * 4) * 2  # Only 20 trades
        coin_flip.add_trades("small", pnl)
        result = coin_flip.evaluate("small")
        assert "min_trades_for_power" in result

    def test_has_edge_method(self, coin_flip):
        pnl = ([100.0] * 8 + [-50.0] * 2) * 10
        coin_flip.add_trades("winner", pnl)
        # This is a very strong edge — should return True
        assert coin_flip.has_edge("winner") is True


class TestCoinFlipMultiStrategy:
    """Test multi-strategy comparison."""

    def test_rank_strategies(self):
        baseline = CoinFlipBaseline(n_simulations=3000, seed=42)
        # Strong strategy
        baseline.add_trades("strong", ([100.0] * 8 + [-50.0] * 2) * 10)
        # Weak strategy
        baseline.add_trades("weak", ([50.0] * 5 + [-50.0] * 5) * 10)
        ranked = baseline.rank_strategies()
        assert len(ranked) == 2
        # Strong should rank first (lower p-value)
        assert ranked[0]["strategy_id"] == "strong"

    def test_get_all_strategy_ids(self, coin_flip):
        coin_flip.add_trade("a", 10.0)
        coin_flip.add_trade("b", -5.0)
        assert set(coin_flip.get_all_strategy_ids()) == {"a", "b"}

    def test_add_trade_incremental(self, coin_flip):
        for i in range(15):
            coin_flip.add_trade("trend", 50.0 if i % 2 == 0 else -20.0)
        result = coin_flip.evaluate("trend")
        assert result["trade_count"] == 15


class TestCoinFlipReproducibility:
    """Test reproducibility with seed."""

    def test_reproducible_results(self):
        b1 = CoinFlipBaseline(n_simulations=1000, seed=42)
        b2 = CoinFlipBaseline(n_simulations=1000, seed=42)
        pnl = [50.0, -20.0, 30.0, -10.0, 40.0, -15.0, 25.0, -5.0, 35.0, 20.0]
        b1.add_trades("t", pnl)
        b2.add_trades("t", pnl)
        r1 = b1.evaluate("t")
        r2 = b2.evaluate("t")
        assert r1["p_value_bootstrap"] == r2["p_value_bootstrap"]
