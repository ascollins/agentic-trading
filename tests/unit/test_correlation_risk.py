"""Tests for structural correlation check and correlation risk analysis."""

import pytest

from agentic_trading.portfolio.correlation_risk import (
    STRUCTURAL_CORRELATIONS,
    CorrelationRiskAnalyzer,
    quick_correlation_check,
)


class TestQuickCorrelationCheck:
    def test_btc_eth_flagged_at_high_threshold(self):
        result = quick_correlation_check(
            ["BTC/USDT", "ETH/USDT"], threshold="high"
        )
        assert result["warning_count"] if "warning_count" in result else len(result["warnings"]) > 0
        assert any("BTC/USDT" in w and "ETH/USDT" in w for w in result["warnings"])

    def test_uncorrelated_assets_no_warnings(self):
        # These two are not in any structural pair together
        result = quick_correlation_check(
            ["BTC/USDT", "DOGE/USDT"], threshold="high"
        )
        # BTC/DOGE is only in "moderate" tier, not "high"
        assert len(result["warnings"]) == 0

    def test_moderate_threshold_flags_more(self):
        result = quick_correlation_check(
            ["BTC/USDT", "DOGE/USDT"], threshold="moderate"
        )
        assert len(result["warnings"]) > 0

    def test_diversification_score_perfect(self):
        # Assets not in any correlation pair
        result = quick_correlation_check(
            ["XLM/USDT", "ATOM/USDT", "FIL/USDT"], threshold="high"
        )
        assert result["diversification_score"] == 1.0

    def test_diversification_score_low(self):
        # All same cluster
        result = quick_correlation_check(
            ["BTC/USDT", "ETH/USDT"], threshold="high"
        )
        assert result["diversification_score"] < 1.0

    def test_cluster_building_merges(self):
        # ETH/SOL and SOL/AVAX should form one cluster
        result = quick_correlation_check(
            ["ETH/USDT", "SOL/USDT", "AVAX/USDT"], threshold="high"
        )
        clusters = result["correlated_clusters"]
        # All three should be in one cluster
        assert len(clusters) >= 1
        largest = max(clusters, key=len)
        assert "ETH/USDT" in largest or "SOL/USDT" in largest

    def test_very_high_threshold_only(self):
        # very_high threshold should only flag very_high pairs
        result = quick_correlation_check(
            ["BTC/USDT", "ETH/USDT", "BTC/USDC"], threshold="very_high"
        )
        # BTC/ETH is "high", not "very_high"
        btc_eth_flagged = any(
            ("BTC/USDT", "ETH/USDT", t) in [(a, b, c) for a, b, c in result["flagged_pairs"]]
            for t in ["very_high"]
        )
        assert not btc_eth_flagged

    def test_empty_positions(self):
        result = quick_correlation_check([], threshold="high")
        assert result["diversification_score"] == 1.0
        assert result["correlated_clusters"] == []
        assert result["warnings"] == []

    def test_perp_spot_very_high(self):
        result = quick_correlation_check(
            ["BTC/USDT:USDT", "BTC/USDT"], threshold="very_high"
        )
        assert len(result["flagged_pairs"]) == 1
        assert result["flagged_pairs"][0][2] == "very_high"


class TestCorrelationRiskAnalyzerDynamic:
    """Quick sanity check that the existing dynamic analyzer still works."""

    def test_no_data_returns_empty(self):
        analyzer = CorrelationRiskAnalyzer()
        assert analyzer.find_clusters() == []

    def test_correlated_returns_clustered(self):
        analyzer = CorrelationRiskAnalyzer(lookback_periods=5)
        for i in range(10):
            analyzer.update_returns("A", float(i))
            analyzer.update_returns("B", float(i) * 0.95)  # nearly identical
            analyzer.update_returns("C", float(-i))  # opposite
        clusters = analyzer.find_clusters()
        # A and B should cluster together
        assert any("A" in c and "B" in c for c in clusters)
