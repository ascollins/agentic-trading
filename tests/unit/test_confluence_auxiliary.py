"""Tests for auxiliary confluence assessments (funding, OI, orderbook, volume delta, correlation)."""

from __future__ import annotations

import math

import pytest

from agentic_trading.intelligence.analysis.smc_confluence import (
    CorrelationAssessment,
    FundingAssessment,
    MAX_CONFLUENCE_POINTS,
    OIAssessment,
    OrderbookAssessment,
    SMCConfluenceResult,
    SMCConfluenceScorer,
    VolumeDeltaAssessment,
)
from agentic_trading.core.enums import MarketStructureBias, Timeframe


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_base_features(
    swing_bias: float = 0.5,
    prefix: str = "4h",
) -> dict[str, float]:
    """Minimal feature dict that produces a BULLISH HTF bias."""
    return {
        f"{prefix}_smc_swing_bias": swing_bias,
        f"{prefix}_smc_bos_bullish": 1.0,
        f"{prefix}_smc_bos_bearish": 0.0,
        f"{prefix}_smc_choch_bullish": 0.0,
        f"{prefix}_smc_choch_bearish": 0.0,
        f"{prefix}_smc_last_break_direction": 1.0,
        f"{prefix}_smc_last_break_is_choch": 0.0,
        f"{prefix}_smc_ob_unmitigated_bullish": 2.0,
        f"{prefix}_smc_ob_unmitigated_bearish": 0.0,
        f"{prefix}_smc_nearest_demand_distance": 0.01,
        f"{prefix}_smc_nearest_supply_distance": 0.03,
        f"{prefix}_smc_fvg_count_bullish": 1.0,
        f"{prefix}_smc_fvg_count_bearish": 0.0,
        f"{prefix}_smc_price_zone": -1.0,
        f"{prefix}_smc_equilibrium": 100.0,
        f"{prefix}_smc_dealing_range_high": 110.0,
        f"{prefix}_smc_dealing_range_low": 90.0,
        f"{prefix}_smc_in_ote": 1.0,
        f"{prefix}_smc_ote_alignment": 1.0,
        f"{prefix}_smc_bsl_count": 0.0,
        f"{prefix}_smc_ssl_count": 1.0,
        f"{prefix}_smc_last_sweep_type": -1.0,
        f"{prefix}_smc_confluence_score": 8.0,
    }


class TestFundingAssessment:
    """Tests for _assess_funding()."""

    def test_no_funding_data_returns_none(self):
        result = SMCConfluenceScorer._assess_funding({}, MarketStructureBias.BULLISH)
        assert result is None

    def test_neutral_funding(self):
        features = {"funding_rate": 0.0001, "funding_zscore": 0.3}
        result = SMCConfluenceScorer._assess_funding(features, MarketStructureBias.BULLISH)
        assert result is not None
        assert result.sentiment == "NEUTRAL"
        assert not result.is_crowded

    def test_overheated_long(self):
        features = {"funding_rate": 0.001, "funding_zscore": 2.5}
        result = SMCConfluenceScorer._assess_funding(features, MarketStructureBias.BULLISH)
        assert result is not None
        assert result.sentiment == "OVERHEATED_LONG"
        assert result.is_crowded

    def test_overheated_short(self):
        features = {"funding_rate": -0.001, "funding_zscore": -2.5}
        result = SMCConfluenceScorer._assess_funding(features, MarketStructureBias.BEARISH)
        assert result is not None
        assert result.sentiment == "OVERHEATED_SHORT"
        assert result.is_crowded

    def test_slightly_bullish(self):
        features = {"funding_rate": 0.0003, "funding_zscore": 1.5}
        result = SMCConfluenceScorer._assess_funding(features, MarketStructureBias.BULLISH)
        assert result is not None
        assert result.sentiment == "SLIGHTLY_BULLISH"
        assert not result.is_crowded

    def test_nan_zscore_treated_as_zero(self):
        features = {"funding_rate": 0.0001, "funding_zscore": float("nan")}
        result = SMCConfluenceScorer._assess_funding(features, MarketStructureBias.BULLISH)
        assert result is not None
        assert result.sentiment == "NEUTRAL"


class TestOIAssessment:
    """Tests for _assess_open_interest()."""

    def test_no_oi_data_returns_none(self):
        result = SMCConfluenceScorer._assess_open_interest({}, MarketStructureBias.BULLISH)
        assert result is None

    def test_fresh_money_entering(self):
        features = {"oi_current": 1_000_000.0, "oi_change_pct_24h": 5.0, "oi_trend": 1.0}
        result = SMCConfluenceScorer._assess_open_interest(features, MarketStructureBias.BULLISH)
        assert result is not None
        assert result.interpretation == "FRESH_MONEY_ENTERING"

    def test_unwinding(self):
        features = {"oi_current": 1_000_000.0, "oi_change_pct_24h": -3.0, "oi_trend": -1.0}
        result = SMCConfluenceScorer._assess_open_interest(features, MarketStructureBias.BEARISH)
        assert result is not None
        assert result.interpretation == "UNWINDING"

    def test_stable(self):
        features = {"oi_current": 1_000_000.0, "oi_change_pct_24h": 0.5, "oi_trend": 0.0}
        result = SMCConfluenceScorer._assess_open_interest(features, MarketStructureBias.BULLISH)
        assert result is not None
        assert result.interpretation == "STABLE"


class TestOrderbookAssessment:
    """Tests for _assess_orderbook()."""

    def test_no_orderbook_data_returns_none(self):
        result = SMCConfluenceScorer._assess_orderbook({}, MarketStructureBias.BULLISH)
        assert result is None

    def test_bid_heavy_imbalance(self):
        features = {
            "ob_imbalance": 2.0,
            "ob_spread_pct": 0.01,
            "ob_bid_wall_price": 67000.0,
            "ob_bid_wall_size": 100.0,
            "ob_bid_wall_persistence": 0.85,
            "ob_ask_wall_price": 68000.0,
            "ob_ask_wall_size": 50.0,
            "ob_ask_wall_persistence": 0.6,
        }
        result = SMCConfluenceScorer._assess_orderbook(features, MarketStructureBias.BULLISH)
        assert result is not None
        assert result.imbalance == 2.0
        assert "Bid-heavy" in result.description

    def test_ask_heavy_imbalance(self):
        features = {"ob_imbalance": 0.5, "ob_spread_pct": 0.02}
        result = SMCConfluenceScorer._assess_orderbook(features, MarketStructureBias.BEARISH)
        assert result is not None
        assert "Ask-heavy" in result.description

    def test_balanced(self):
        features = {"ob_imbalance": 1.0, "ob_spread_pct": 0.01}
        result = SMCConfluenceScorer._assess_orderbook(features, MarketStructureBias.BULLISH)
        assert result is not None
        assert "Balanced" in result.description


class TestVolumeDeltaAssessment:
    """Tests for _assess_volume_delta()."""

    def test_no_volume_delta_returns_none(self):
        result = SMCConfluenceScorer._assess_volume_delta({}, MarketStructureBias.BULLISH)
        assert result is None

    def test_bullish_delta(self):
        features = {
            "volume_delta": 1000.0,
            "volume_delta_cumulative": 5000.0,
            "volume_delta_ratio": 1.5,
            "volume_delta_trend": 1.0,
        }
        result = SMCConfluenceScorer._assess_volume_delta(features, MarketStructureBias.BULLISH)
        assert result is not None
        assert result.delta > 0
        assert result.trend == "INCREASING"
        assert "Bullish" in result.description

    def test_bearish_delta(self):
        features = {
            "volume_delta": -500.0,
            "volume_delta_cumulative": -2000.0,
            "volume_delta_ratio": 0.7,
            "volume_delta_trend": -1.0,
        }
        result = SMCConfluenceScorer._assess_volume_delta(features, MarketStructureBias.BEARISH)
        assert result is not None
        assert result.delta < 0
        assert "Bearish" in result.description


class TestCorrelationAssessment:
    """Tests for _assess_correlation()."""

    def test_no_correlation_data_returns_none(self):
        result = SMCConfluenceScorer._assess_correlation({})
        assert result is None

    def test_high_correlation(self):
        result = SMCConfluenceScorer._assess_correlation({"btc_correlation": 0.85})
        assert result is not None
        assert result.independence_level == "HIGH"

    def test_moderate_correlation(self):
        result = SMCConfluenceScorer._assess_correlation({"btc_correlation": 0.5})
        assert result is not None
        assert result.independence_level == "MODERATE"

    def test_low_correlation(self):
        result = SMCConfluenceScorer._assess_correlation({"btc_correlation": 0.1})
        assert result is not None
        assert result.independence_level == "LOW"


class TestAuxiliaryScoring:
    """Tests for _score_auxiliary()."""

    def test_neutral_funding_adds_points(self):
        funding = FundingAssessment(rate=0.0001, z_score=0.3, sentiment="NEUTRAL")
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, funding, None, None, None,
        )
        assert pts == 1.0
        assert len(factors) == 1
        assert "neutral" in factors[0].lower()

    def test_crowded_against_position_adds_2pts(self):
        # Overheated SHORT while we're LONG = favorable
        funding = FundingAssessment(
            rate=-0.001, z_score=-2.5,
            sentiment="OVERHEATED_SHORT", is_crowded=True,
        )
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, funding, None, None, None,
        )
        assert pts == 2.0

    def test_crowded_with_position_subtracts_points(self):
        # Overheated LONG while we're LONG = bad
        funding = FundingAssessment(
            rate=0.001, z_score=2.5,
            sentiment="OVERHEATED_LONG", is_crowded=True,
        )
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, funding, None, None, None,
        )
        assert pts == 0.0  # max(0, -1)
        assert len(conflicts) == 1

    def test_oi_fresh_money_adds_2pts(self):
        oi = OIAssessment(
            current=1_000_000, change_pct_24h=5.0,
            trend=1, interpretation="FRESH_MONEY_ENTERING",
        )
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, None, oi, None, None,
        )
        assert pts == 2.0

    def test_oi_unwinding_adds_conflict(self):
        oi = OIAssessment(
            current=1_000_000, change_pct_24h=-3.0,
            trend=-1, interpretation="UNWINDING",
        )
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, None, oi, None, None,
        )
        assert pts == 0.0
        assert len(conflicts) == 1

    def test_orderbook_supports_bias(self):
        ob = OrderbookAssessment(imbalance=1.8)
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, None, None, ob, None,
        )
        assert pts == 1.0

    def test_orderbook_against_bias(self):
        ob = OrderbookAssessment(imbalance=0.5)
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, None, None, ob, None,
        )
        assert pts == 0.0
        assert len(conflicts) == 1

    def test_volume_delta_confirms(self):
        vd = VolumeDeltaAssessment(delta=1000.0, trend="INCREASING")
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, None, None, None, vd,
        )
        assert pts == 1.0

    def test_volume_delta_diverges(self):
        vd = VolumeDeltaAssessment(delta=-500.0, trend="DECREASING")
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, None, None, None, vd,
        )
        assert pts == 0.0
        assert len(conflicts) == 1

    def test_all_favorable_max_6pts(self):
        funding = FundingAssessment(
            rate=-0.001, z_score=-2.5,
            sentiment="OVERHEATED_SHORT", is_crowded=True,
        )
        oi = OIAssessment(
            current=1_000_000, change_pct_24h=5.0,
            trend=1, interpretation="FRESH_MONEY_ENTERING",
        )
        ob = OrderbookAssessment(imbalance=2.0)
        vd = VolumeDeltaAssessment(delta=1000.0, trend="INCREASING")

        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.BULLISH, funding, oi, ob, vd,
        )
        assert pts == 6.0
        assert len(conflicts) == 0

    def test_no_direction_no_scoring(self):
        """Auxiliary scoring requires a directional bias."""
        funding = FundingAssessment(rate=0.0001, z_score=0.3, sentiment="NEUTRAL")
        pts, factors, conflicts = SMCConfluenceScorer._score_auxiliary(
            MarketStructureBias.UNCLEAR, funding, None, None, None,
        )
        assert pts == 0.0


class TestConfluenceIntegration:
    """End-to-end tests for the full scoring pipeline with auxiliary data."""

    def test_max_confluence_is_20(self):
        assert MAX_CONFLUENCE_POINTS == 20.0

    def test_score_includes_auxiliary_data(self):
        features = _make_base_features()
        features["funding_rate"] = 0.0001
        features["funding_zscore"] = 0.3
        features["oi_current"] = 1_000_000.0
        features["oi_change_pct_24h"] = 5.0
        features["oi_trend"] = 1.0

        scorer = SMCConfluenceScorer()
        result = scorer.score("BTC/USDT", features, [Timeframe.H4])

        assert result.funding_assessment is not None
        assert result.oi_assessment is not None
        assert result.total_confluence_points > 0
        assert result.max_confluence_points == 20.0

    def test_result_has_all_assessment_fields(self):
        features = _make_base_features()
        features["funding_rate"] = 0.0001
        features["funding_zscore"] = 0.3
        features["oi_current"] = 1_000_000.0
        features["oi_change_pct_24h"] = 2.0
        features["oi_trend"] = 1.0
        features["ob_imbalance"] = 1.8
        features["ob_spread_pct"] = 0.01
        features["volume_delta"] = 500.0
        features["volume_delta_cumulative"] = 2000.0
        features["volume_delta_ratio"] = 1.3
        features["volume_delta_trend"] = 1.0
        features["btc_correlation"] = 0.5

        scorer = SMCConfluenceScorer()
        result = scorer.score("BTC/USDT", features, [Timeframe.H4])

        assert result.funding_assessment is not None
        assert result.oi_assessment is not None
        assert result.orderbook_assessment is not None
        assert result.volume_delta_assessment is not None
        assert result.correlation_assessment is not None


class TestOpenInterestEngine:
    """Tests for the OpenInterestEngine."""

    def test_compute_basic_features(self):
        from agentic_trading.intelligence.features.open_interest import (
            OpenInterestEngine,
        )

        engine = OpenInterestEngine()
        features = engine.compute_oi_features(
            symbol="BTC/USDT",
            open_interest=500_000_000.0,
        )
        assert features["oi_current"] == 500_000_000.0
        assert features["oi_trend"] == 0.0

    def test_change_pct_after_multiple_observations(self):
        from agentic_trading.intelligence.features.open_interest import (
            OpenInterestEngine,
        )

        engine = OpenInterestEngine()
        engine.compute_oi_features("BTC/USDT", 1_000_000.0)
        features = engine.compute_oi_features("BTC/USDT", 1_050_000.0)
        assert "oi_change_pct_latest" in features
        assert features["oi_change_pct_latest"] == pytest.approx(5.0, abs=0.1)


class TestOrderbookEngine:
    """Tests for the OrderbookEngine."""

    def test_compute_basic_features(self):
        from agentic_trading.intelligence.features.orderbook import OrderbookEngine

        engine = OrderbookEngine()
        features = engine.compute_orderbook_features(
            symbol="BTC/USDT",
            bids=[[67000.0, 5.0], [66990.0, 3.0], [66980.0, 10.0]],
            asks=[[67010.0, 4.0], [67020.0, 2.0], [67030.0, 1.0]],
        )
        assert "ob_spread" in features
        assert "ob_imbalance" in features
        assert "ob_bid_wall_price" in features
        assert features["ob_spread"] == pytest.approx(10.0)
        assert features["ob_bid_wall_price"] == pytest.approx(66980.0)  # Largest bid

    def test_empty_orderbook(self):
        from agentic_trading.intelligence.features.orderbook import OrderbookEngine

        engine = OrderbookEngine()
        features = engine.compute_orderbook_features("BTC/USDT", [], [])
        assert features == {}

    def test_wall_persistence_tracking(self):
        from agentic_trading.intelligence.features.orderbook import OrderbookEngine

        engine = OrderbookEngine()
        bids = [[67000.0, 5.0], [66900.0, 50.0]]
        asks = [[67010.0, 4.0], [67100.0, 30.0]]

        # First snapshot
        f1 = engine.compute_orderbook_features("BTC/USDT", bids, asks)
        # Second snapshot (same wall)
        f2 = engine.compute_orderbook_features("BTC/USDT", bids, asks)
        # Persistence should increase
        assert f2["ob_bid_wall_persistence"] >= f1.get("ob_bid_wall_persistence", 0)


class TestVolumeDeltaIndicator:
    """Tests for the volume delta indicator function."""

    def test_compute_volume_delta_basic(self):
        import numpy as np
        from agentic_trading.intelligence.features.indicators import compute_volume_delta

        opens = np.array([100.0] * 25, dtype=np.float64)
        closes = np.array([101.0] * 15 + [99.0] * 10, dtype=np.float64)
        volumes = np.array([100.0] * 25, dtype=np.float64)

        delta, cumulative, ratio, trend = compute_volume_delta(
            opens, closes, volumes, cumulative_period=20,
        )

        assert len(delta) == 25
        # First 15 bars: close > open = positive delta
        assert delta[0] > 0
        # Last 10 bars: close < open = negative delta
        assert delta[-1] < 0

    def test_doji_bars_zero_delta(self):
        import numpy as np
        from agentic_trading.intelligence.features.indicators import compute_volume_delta

        opens = np.array([100.0] * 25, dtype=np.float64)
        closes = np.array([100.0] * 25, dtype=np.float64)
        volumes = np.array([100.0] * 25, dtype=np.float64)

        delta, cumulative, ratio, trend = compute_volume_delta(
            opens, closes, volumes, cumulative_period=20,
        )

        assert delta[0] == 0.0
        # Ratio should be 1.0 for doji bars (50/50 split)
        assert ratio[-1] == pytest.approx(1.0, abs=0.01)


class TestTradePlanNarrative:
    """Test that the new format_report includes auxiliary data sections."""

    def test_format_report_includes_funding(self):
        from agentic_trading.intelligence.analysis.smc_trade_plan import (
            SMCTradePlanGenerator,
            SMCAnalysisReport,
        )
        from agentic_trading.intelligence.analysis.smc_confluence import (
            SMCConfluenceResult,
            FundingAssessment,
        )

        confluence = SMCConfluenceResult(
            symbol="BTC/USDT",
            htf_bias=MarketStructureBias.BULLISH,
            total_confluence_points=12.0,
            funding_assessment=FundingAssessment(
                rate=0.0001,
                z_score=0.3,
                sentiment="NEUTRAL",
                description="+0.0100% (Neutral) - Not overcrowded on either side",
            ),
        )
        report = SMCAnalysisReport(
            symbol="BTC/USDT",
            current_price=67000.0,
            overall_bias=MarketStructureBias.BULLISH,
            confluence_result=confluence,
            trade_verdict="POTENTIAL LONG SETUP (High Confluence)",
            setup_direction=SignalDirection.LONG,
        )

        gen = SMCTradePlanGenerator()
        text = gen.format_report(report)

        assert "Binance Data" in text
        assert "Funding Rate" in text
        assert "VERDICT" in text

    def test_format_report_includes_oi(self):
        from agentic_trading.intelligence.analysis.smc_trade_plan import (
            SMCTradePlanGenerator,
            SMCAnalysisReport,
        )
        from agentic_trading.intelligence.analysis.smc_confluence import (
            SMCConfluenceResult,
            OIAssessment,
        )

        confluence = SMCConfluenceResult(
            symbol="BTC/USDT",
            htf_bias=MarketStructureBias.BULLISH,
            total_confluence_points=14.0,
            oi_assessment=OIAssessment(
                current=500_000_000,
                change_pct_24h=2.6,
                trend=1,
                interpretation="FRESH_MONEY_ENTERING",
                description="+2.6% increase in 24h - Fresh Money Entering",
            ),
        )
        report = SMCAnalysisReport(
            symbol="BTC/USDT",
            current_price=67000.0,
            overall_bias=MarketStructureBias.BULLISH,
            confluence_result=confluence,
            trade_verdict="POTENTIAL LONG",
            setup_direction=SignalDirection.LONG,
        )

        gen = SMCTradePlanGenerator()
        text = gen.format_report(report)

        assert "Open Interest" in text
        assert "Fresh Money" in text


# Need this import for test above
from agentic_trading.core.enums import SignalDirection

# Import the SMCAnalysisReport for the narrative tests
from agentic_trading.intelligence.analysis.smc_trade_plan import SMCAnalysisReport
