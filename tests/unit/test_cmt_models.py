"""Tests for CMT analysis Pydantic models."""

from __future__ import annotations

import pytest

from agentic_trading.intelligence.analysis.cmt_models import (
    CMTAssessmentRequest,
    CMTAssessmentResponse,
    CMTConfluenceScore,
    CMTLayerResult,
    CMTTarget,
    CMTTradePlan,
)


# ---------------------------------------------------------------------------
# CMTLayerResult
# ---------------------------------------------------------------------------


class TestCMTLayerResult:
    def test_defaults(self):
        lr = CMTLayerResult(layer=1, name="Trend Identification")
        assert lr.layer == 1
        assert lr.name == "Trend Identification"
        assert lr.direction == ""
        assert lr.confidence == ""
        assert lr.score == 0.0
        assert lr.key_findings == []
        assert lr.warnings == []

    def test_full_construction(self):
        lr = CMTLayerResult(
            layer=3,
            name="Pattern Recognition",
            direction="bullish",
            confidence="high",
            score=1.5,
            key_findings=["Cup and handle forming", "Neckline at 105k"],
            warnings=["Low volume on breakout"],
        )
        assert lr.direction == "bullish"
        assert lr.score == 1.5
        assert len(lr.key_findings) == 2
        assert len(lr.warnings) == 1


# ---------------------------------------------------------------------------
# CMTConfluenceScore
# ---------------------------------------------------------------------------


class TestCMTConfluenceScore:
    def test_defaults_all_zero(self):
        cs = CMTConfluenceScore()
        assert cs.total == 0.0
        assert cs.threshold_met is False
        assert cs.veto is False

    def test_compute_total_sums_all_dimensions(self):
        cs = CMTConfluenceScore(
            trend_alignment=2.0,
            key_level_proximity=1.5,
            pattern_signal=1.0,
            indicator_consensus=1.5,
            sentiment_alignment=0.5,
            volatility_regime=0.5,
            macro_alignment=0.5,
        )
        result = cs.compute_total()
        assert result == pytest.approx(7.5)
        assert cs.total == pytest.approx(7.5)

    def test_compute_total_negative(self):
        cs = CMTConfluenceScore(
            trend_alignment=-2.0,
            key_level_proximity=0.0,
            pattern_signal=-2.0,
            indicator_consensus=-2.0,
            sentiment_alignment=-1.0,
            volatility_regime=-1.0,
            macro_alignment=-1.0,
        )
        result = cs.compute_total()
        assert result == pytest.approx(-9.0)

    def test_check_veto_triggered_by_trend_alignment(self):
        cs = CMTConfluenceScore(trend_alignment=-2.0)
        assert cs.check_veto() is True
        assert cs.veto is True

    def test_check_veto_triggered_by_pattern_signal(self):
        cs = CMTConfluenceScore(pattern_signal=-2.0)
        assert cs.check_veto() is True

    def test_check_veto_triggered_by_indicator_consensus(self):
        cs = CMTConfluenceScore(indicator_consensus=-2.0)
        assert cs.check_veto() is True

    def test_check_veto_not_triggered_by_sentiment(self):
        """Sentiment at -1 (its min) should NOT trigger veto."""
        cs = CMTConfluenceScore(sentiment_alignment=-1.0)
        assert cs.check_veto() is False

    def test_check_veto_not_triggered_by_volatility(self):
        cs = CMTConfluenceScore(volatility_regime=-1.0)
        assert cs.check_veto() is False

    def test_check_veto_not_triggered_by_macro(self):
        cs = CMTConfluenceScore(macro_alignment=-1.0)
        assert cs.check_veto() is False

    def test_no_veto_at_minus_one_point_nine(self):
        """Values close to but not at -2 should not veto."""
        cs = CMTConfluenceScore(trend_alignment=-1.9)
        assert cs.check_veto() is False

    def test_threshold_met_with_high_total(self):
        cs = CMTConfluenceScore(
            trend_alignment=2.0,
            key_level_proximity=1.0,
            pattern_signal=1.0,
            indicator_consensus=1.5,
            sentiment_alignment=0.5,
        )
        cs.compute_total()
        cs.check_veto()
        cs.threshold_met = cs.total >= 5 and not cs.veto
        assert cs.threshold_met is True

    def test_threshold_not_met_with_veto(self):
        """Even high total should fail if veto is active."""
        cs = CMTConfluenceScore(
            trend_alignment=-2.0,
            key_level_proximity=2.0,
            pattern_signal=2.0,
            indicator_consensus=2.0,
            sentiment_alignment=1.0,
            volatility_regime=1.0,
            macro_alignment=1.0,
        )
        cs.compute_total()
        cs.check_veto()
        cs.threshold_met = cs.total >= 5 and not cs.veto
        assert cs.total == pytest.approx(7.0)
        assert cs.veto is True
        assert cs.threshold_met is False


# ---------------------------------------------------------------------------
# CMTTarget
# ---------------------------------------------------------------------------


class TestCMTTarget:
    def test_construction(self):
        t = CMTTarget(price=105_000.0, pct=50.0, source="fib_extension")
        assert t.price == 105_000.0
        assert t.pct == 50.0
        assert t.source == "fib_extension"

    def test_defaults(self):
        t = CMTTarget(price=100.0, pct=100.0)
        assert t.source == ""


# ---------------------------------------------------------------------------
# CMTTradePlan
# ---------------------------------------------------------------------------


class TestCMTTradePlan:
    def _make_plan(self, **overrides) -> CMTTradePlan:
        defaults = {
            "direction": "LONG",
            "entry_price": 100_000.0,
            "entry_trigger": "Break above 100k resistance",
            "stop_loss": 97_000.0,
            "stop_reasoning": "Below swing low",
            "targets": [CMTTarget(price=105_000.0, pct=50.0, source="measured_move")],
            "rr_ratio": 1.67,
            "blended_rr": 1.5,
            "position_size_pct": 2.0,
            "invalidation": "Close below 96k",
            "thesis": "Bullish breakout from consolidation",
        }
        defaults.update(overrides)
        return CMTTradePlan(**defaults)

    def test_full_construction(self):
        plan = self._make_plan()
        assert plan.direction == "LONG"
        assert plan.entry_price == 100_000.0
        assert plan.stop_loss == 97_000.0
        assert len(plan.targets) == 1
        assert plan.rr_ratio == 1.67

    def test_short_direction(self):
        plan = self._make_plan(
            direction="SHORT",
            stop_loss=103_000.0,
            targets=[CMTTarget(price=95_000.0, pct=50.0, source="measured_move")],
        )
        assert plan.direction == "SHORT"

    def test_multiple_targets(self):
        plan = self._make_plan(
            targets=[
                CMTTarget(price=105_000.0, pct=50.0, source="sr_level"),
                CMTTarget(price=110_000.0, pct=30.0, source="fib_extension"),
                CMTTarget(price=115_000.0, pct=20.0, source="measured_move"),
            ]
        )
        assert len(plan.targets) == 3
        assert sum(t.pct for t in plan.targets) == 100.0

    def test_defaults(self):
        plan = CMTTradePlan(direction="LONG", entry_price=100.0, stop_loss=95.0)
        assert plan.targets == []
        assert plan.rr_ratio == 0.0
        assert plan.thesis == ""


# ---------------------------------------------------------------------------
# CMTAssessmentRequest
# ---------------------------------------------------------------------------


class TestCMTAssessmentRequest:
    def test_minimal(self):
        req = CMTAssessmentRequest(symbol="BTC/USDT")
        assert req.symbol == "BTC/USDT"
        assert req.timeframes == []
        assert req.ohlcv_summary == {}
        assert req.indicator_values == {}

    def test_full_construction(self):
        req = CMTAssessmentRequest(
            symbol="ETH/USDT",
            timeframes=["H1", "H4", "D1"],
            ohlcv_summary={"H1": [{"c": 3500.0}]},
            indicator_values={"H1_rsi": 55.0, "H4_macd_hist": 0.5},
            htf_assessment={"bias": "bullish"},
            smc_confluence={"score": 8},
        )
        assert len(req.timeframes) == 3
        assert "H1_rsi" in req.indicator_values


# ---------------------------------------------------------------------------
# CMTAssessmentResponse
# ---------------------------------------------------------------------------


class TestCMTAssessmentResponse:
    def test_minimal(self):
        resp = CMTAssessmentResponse(symbol="BTC/USDT")
        assert resp.symbol == "BTC/USDT"
        assert resp.layers == []
        assert resp.confluence.total == 0.0
        assert resp.trade_plan is None
        assert resp.system_health == "green"

    def test_layer_dict(self):
        resp = CMTAssessmentResponse(
            symbol="BTC/USDT",
            layers=[
                CMTLayerResult(
                    layer=1,
                    name="Trend Identification",
                    direction="bullish",
                    score=1.5,
                ),
                CMTLayerResult(
                    layer=2,
                    name="Support Resistance",
                    direction="neutral",
                    score=0.5,
                ),
            ],
        )
        ld = resp.layer_dict()
        assert "layer_1_trend_identification" in ld
        assert "layer_2_support_resistance" in ld
        assert ld["layer_1_trend_identification"]["score"] == 1.5

    def test_with_trade_plan(self):
        resp = CMTAssessmentResponse(
            symbol="BTC/USDT",
            trade_plan=CMTTradePlan(
                direction="LONG",
                entry_price=100_000.0,
                stop_loss=97_000.0,
            ),
            thesis="Bullish setup",
            system_health="green",
        )
        assert resp.trade_plan is not None
        assert resp.trade_plan.direction == "LONG"

    def test_amber_health(self):
        resp = CMTAssessmentResponse(
            symbol="BTC/USDT",
            system_health="amber",
        )
        assert resp.system_health == "amber"
