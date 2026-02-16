"""Tests for multi-timeframe SMC confluence scoring."""

from __future__ import annotations

import pytest

from agentic_trading.analysis.smc_confluence import (
    SMCConfluenceResult,
    SMCConfluenceScorer,
    SMCTimeframeSummary,
)
from agentic_trading.core.enums import MarketStructureBias, Timeframe


def _make_bullish_features(prefix: str = "") -> dict[str, float]:
    """Create a set of bullish SMC features with optional TF prefix."""
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}smc_swing_bias": 0.7,
        f"{p}smc_swing_count": 8.0,
        f"{p}smc_hh_count": 3.0,
        f"{p}smc_hl_count": 2.0,
        f"{p}smc_lh_count": 0.0,
        f"{p}smc_ll_count": 0.0,
        f"{p}smc_ob_count_bullish": 3.0,
        f"{p}smc_ob_count_bearish": 1.0,
        f"{p}smc_ob_unmitigated_bullish": 2.0,
        f"{p}smc_ob_unmitigated_bearish": 0.0,
        f"{p}smc_nearest_demand_distance": 0.01,
        f"{p}smc_nearest_supply_distance": 0.05,
        f"{p}smc_fvg_count_bullish": 2.0,
        f"{p}smc_fvg_count_bearish": 0.0,
        f"{p}smc_fvg_count_total": 2.0,
        f"{p}smc_bos_bullish": 2.0,
        f"{p}smc_bos_bearish": 0.0,
        f"{p}smc_choch_bullish": 1.0,
        f"{p}smc_choch_bearish": 0.0,
        f"{p}smc_last_break_direction": 1.0,
        f"{p}smc_last_break_is_choch": 1.0,
        f"{p}smc_bsl_count": 1.0,
        f"{p}smc_ssl_count": 0.0,
        f"{p}smc_bsl_confirmed_count": 1.0,
        f"{p}smc_ssl_confirmed_count": 0.0,
        f"{p}smc_last_sweep_type": 1.0,
        f"{p}smc_last_sweep_bars_ago": 5.0,
        f"{p}smc_last_sweep_penetration": 0.002,
        f"{p}smc_sweep_reversal_confirmed": 1.0,
        f"{p}smc_equilibrium": 100.0,
        f"{p}smc_dealing_range_high": 110.0,
        f"{p}smc_dealing_range_low": 90.0,
        f"{p}smc_price_zone": -1.0,
        f"{p}smc_deviation_from_eq": -0.15,
        f"{p}smc_range_position": 0.35,
        f"{p}smc_in_ote": 1.0,
        f"{p}smc_ote_alignment": 1.0,
        f"{p}smc_confluence_score": 10.0,
    }


def _make_bearish_features(prefix: str = "") -> dict[str, float]:
    """Create bearish SMC features."""
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}smc_swing_bias": -0.6,
        f"{p}smc_swing_count": 6.0,
        f"{p}smc_hh_count": 0.0,
        f"{p}smc_hl_count": 0.0,
        f"{p}smc_lh_count": 2.0,
        f"{p}smc_ll_count": 3.0,
        f"{p}smc_ob_count_bullish": 1.0,
        f"{p}smc_ob_count_bearish": 3.0,
        f"{p}smc_ob_unmitigated_bullish": 0.0,
        f"{p}smc_ob_unmitigated_bearish": 2.0,
        f"{p}smc_nearest_demand_distance": 0.05,
        f"{p}smc_nearest_supply_distance": 0.01,
        f"{p}smc_fvg_count_bullish": 0.0,
        f"{p}smc_fvg_count_bearish": 2.0,
        f"{p}smc_fvg_count_total": 2.0,
        f"{p}smc_bos_bullish": 0.0,
        f"{p}smc_bos_bearish": 2.0,
        f"{p}smc_choch_bullish": 0.0,
        f"{p}smc_choch_bearish": 1.0,
        f"{p}smc_last_break_direction": -1.0,
        f"{p}smc_last_break_is_choch": 0.0,
        f"{p}smc_bsl_count": 0.0,
        f"{p}smc_ssl_count": 1.0,
        f"{p}smc_bsl_confirmed_count": 0.0,
        f"{p}smc_ssl_confirmed_count": 1.0,
        f"{p}smc_last_sweep_type": -1.0,
        f"{p}smc_last_sweep_bars_ago": 3.0,
        f"{p}smc_last_sweep_penetration": 0.003,
        f"{p}smc_sweep_reversal_confirmed": 1.0,
        f"{p}smc_equilibrium": 100.0,
        f"{p}smc_dealing_range_high": 110.0,
        f"{p}smc_dealing_range_low": 90.0,
        f"{p}smc_price_zone": 1.0,
        f"{p}smc_deviation_from_eq": 0.15,
        f"{p}smc_range_position": 0.65,
        f"{p}smc_in_ote": 1.0,
        f"{p}smc_ote_alignment": -1.0,
        f"{p}smc_confluence_score": 9.0,
    }


class TestSMCConfluenceScorer:
    """Tests for SMCConfluenceScorer."""

    def test_score_fully_aligned_bullish(self):
        """All TFs bullish produces high confluence score."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))
        features.update(_make_bullish_features("15m"))

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4, Timeframe.H1, Timeframe.M15],
        )
        assert result.htf_bias == MarketStructureBias.BULLISH
        assert result.ltf_confirmation is True
        assert result.bias_alignment_score == 1.0
        assert result.total_confluence_points > 8.0

    def test_score_fully_aligned_bearish(self):
        """All TFs bearish produces high confluence score."""
        features = {}
        features.update(_make_bearish_features("4h"))
        features.update(_make_bearish_features("1h"))

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert result.htf_bias == MarketStructureBias.BEARISH
        assert result.ltf_confirmation is True

    def test_score_divergent_timeframes(self):
        """HTF bullish, LTF bearish = low alignment."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bearish_features("1h"))

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert result.ltf_confirmation is False
        assert result.bias_alignment_score < 0.5
        assert len(result.conflicts) > 0

    def test_htf_ob_at_ltf_entry_detection(self):
        """Detects when HTF OB is near LTF price."""
        features = _make_bullish_features("4h")
        features["4h_smc_nearest_demand_distance"] = 0.005  # < 0.02 threshold

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4],
        )
        assert result.htf_ob_at_ltf_entry is True

    def test_fvg_confluence_across_tfs(self):
        """FVGs on multiple TFs detected as confluence."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert result.fvg_confluence is True

    def test_liquidity_sweep_bonus(self):
        """Confirmed sweep adds points."""
        features = _make_bullish_features("4h")
        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4],
        )
        assert result.liquidity_swept is True
        # Score should include sweep contribution
        assert any("sweep" in f.lower() for f in result.confluence_factors)

    def test_no_smc_features_returns_zero_score(self):
        """Empty features produce zero confluence."""
        scorer = SMCConfluenceScorer()
        result = scorer.score("BTC/USDT", {})
        assert result.total_confluence_points == 0.0
        # No features: swing_bias defaults to 0.0 -> NEUTRAL (|0| < 0.1)
        assert result.htf_bias in (
            MarketStructureBias.UNCLEAR,
            MarketStructureBias.NEUTRAL,
        )

    def test_single_timeframe_scoring(self):
        """Works with just one TF (non-prefixed features)."""
        features = _make_bullish_features("")
        scorer = SMCConfluenceScorer()
        result = scorer.score("BTC/USDT", features)
        assert len(result.timeframe_summaries) >= 1
        assert result.total_confluence_points > 0

    def test_confluence_factors_listed(self):
        """Factor names are human-readable."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert len(result.confluence_factors) > 0
        for factor in result.confluence_factors:
            assert isinstance(factor, str)
            assert len(factor) > 5

    def test_score_bounded_0_to_14(self):
        """Score is within valid range."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert 0.0 <= result.total_confluence_points <= 14.0

    def test_structure_alignment_aligned(self):
        """All TFs same bias and no CHoCH = aligned."""
        features = {}
        # Bullish features without CHoCH
        bull = _make_bullish_features("4h")
        bull["4h_smc_choch_bullish"] = 0.0
        bull["4h_smc_choch_bearish"] = 0.0
        features.update(bull)

        bull2 = _make_bullish_features("1h")
        bull2["1h_smc_choch_bullish"] = 0.0
        bull2["1h_smc_choch_bearish"] = 0.0
        features.update(bull2)

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert result.structure_alignment == "aligned"

    def test_structure_alignment_transitioning(self):
        """CHoCH on any TF = transitioning."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))
        # 1h has a CHoCH
        features["1h_smc_choch_bearish"] = 1.0

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert result.structure_alignment == "transitioning"

    def test_structure_alignment_divergent(self):
        """HTF bullish and LTF bearish without CHoCH = divergent."""
        features = {}
        bull = _make_bullish_features("4h")
        bull["4h_smc_choch_bullish"] = 0.0
        bull["4h_smc_choch_bearish"] = 0.0
        features.update(bull)

        bear = _make_bearish_features("1h")
        bear["1h_smc_choch_bullish"] = 0.0
        bear["1h_smc_choch_bearish"] = 0.0
        features.update(bear)

        scorer = SMCConfluenceScorer()
        result = scorer.score(
            "BTC/USDT", features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert result.structure_alignment == "divergent"

    def test_extract_smc_summary(self):
        """Correctly parses prefixed features into summary."""
        features = _make_bullish_features("4h")
        scorer = SMCConfluenceScorer()
        summary = scorer._extract_smc_summary("4h", Timeframe.H4, features)
        assert summary.timeframe == Timeframe.H4
        assert summary.swing_bias == pytest.approx(0.7)
        assert summary.trend_label == MarketStructureBias.BULLISH
        assert summary.unmitigated_demand_zones == 2
        assert summary.in_ote is True
        assert summary.bsl_sweeps == 1
