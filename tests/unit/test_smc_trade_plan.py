"""Tests for SMC trade plan generator and narrative formatter."""

from __future__ import annotations

import pytest

from agentic_trading.analysis.smc_confluence import SMCConfluenceScorer
from agentic_trading.analysis.smc_trade_plan import (
    InvalidationCondition,
    SMCAnalysisReport,
    SMCTradePlanGenerator,
)
from agentic_trading.core.enums import (
    ConvictionLevel,
    MarketStructureBias,
    SignalDirection,
    Timeframe,
)


def _make_bullish_features(prefix: str = "") -> dict[str, float]:
    """Create bullish SMC features."""
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
        f"{p}smc_equilibrium": 625.0,
        f"{p}smc_dealing_range_high": 650.0,
        f"{p}smc_dealing_range_low": 600.0,
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
        f"{p}smc_equilibrium": 625.0,
        f"{p}smc_dealing_range_high": 650.0,
        f"{p}smc_dealing_range_low": 600.0,
        f"{p}smc_price_zone": 1.0,
        f"{p}smc_deviation_from_eq": 0.15,
        f"{p}smc_range_position": 0.65,
        f"{p}smc_in_ote": 1.0,
        f"{p}smc_ote_alignment": -1.0,
        f"{p}smc_confluence_score": 9.0,
    }


def _make_weak_features(prefix: str = "") -> dict[str, float]:
    """Create weak/unclear SMC features (no strong signal)."""
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}smc_swing_bias": 0.05,
        f"{p}smc_swing_count": 4.0,
        f"{p}smc_hh_count": 1.0,
        f"{p}smc_hl_count": 1.0,
        f"{p}smc_lh_count": 1.0,
        f"{p}smc_ll_count": 1.0,
        f"{p}smc_ob_count_bullish": 1.0,
        f"{p}smc_ob_count_bearish": 1.0,
        f"{p}smc_ob_unmitigated_bullish": 0.0,
        f"{p}smc_ob_unmitigated_bearish": 0.0,
        f"{p}smc_nearest_demand_distance": 0.0,
        f"{p}smc_nearest_supply_distance": 0.0,
        f"{p}smc_fvg_count_bullish": 0.0,
        f"{p}smc_fvg_count_bearish": 0.0,
        f"{p}smc_fvg_count_total": 0.0,
        f"{p}smc_bos_bullish": 0.0,
        f"{p}smc_bos_bearish": 0.0,
        f"{p}smc_choch_bullish": 0.0,
        f"{p}smc_choch_bearish": 0.0,
        f"{p}smc_last_break_direction": 0.0,
        f"{p}smc_last_break_is_choch": 0.0,
        f"{p}smc_bsl_count": 0.0,
        f"{p}smc_ssl_count": 0.0,
        f"{p}smc_bsl_confirmed_count": 0.0,
        f"{p}smc_ssl_confirmed_count": 0.0,
        f"{p}smc_last_sweep_type": 0.0,
        f"{p}smc_last_sweep_bars_ago": 0.0,
        f"{p}smc_last_sweep_penetration": 0.0,
        f"{p}smc_sweep_reversal_confirmed": 0.0,
        f"{p}smc_equilibrium": 0.0,
        f"{p}smc_dealing_range_high": 0.0,
        f"{p}smc_dealing_range_low": 0.0,
        f"{p}smc_price_zone": 0.0,
        f"{p}smc_deviation_from_eq": 0.0,
        f"{p}smc_range_position": 0.5,
        f"{p}smc_in_ote": 0.0,
        f"{p}smc_ote_alignment": 0.0,
        f"{p}smc_confluence_score": 1.0,
    }


class TestSMCTradePlanGenerator:
    """Tests for SMCTradePlanGenerator."""

    def test_generate_report_bullish_setup(self):
        """Aligned bullish features produce bullish report."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator()
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert report.overall_bias == MarketStructureBias.BULLISH
        assert report.setup_direction == SignalDirection.LONG
        assert "LONG" in report.trade_verdict

    def test_generate_report_bearish_setup(self):
        """Aligned bearish features produce bearish report."""
        features = {}
        features.update(_make_bearish_features("4h"))
        features.update(_make_bearish_features("1h"))

        gen = SMCTradePlanGenerator()
        report = gen.generate_report(
            "BNB/USDT", 640.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert report.overall_bias == MarketStructureBias.BEARISH
        assert report.setup_direction == SignalDirection.SHORT
        assert "SHORT" in report.trade_verdict

    def test_generate_report_no_setup(self):
        """Conflicting signals produce FLAT verdict."""
        features = {}
        features.update(_make_weak_features("4h"))
        features.update(_make_weak_features("1h"))

        gen = SMCTradePlanGenerator()
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert report.setup_direction == SignalDirection.FLAT
        assert "NO TRADE" in report.trade_verdict

    def test_generate_trade_plan_from_report(self):
        """Report converts to valid TradePlan."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator(min_confluence_score=5.0, min_rr_ratio=1.0)
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        plan = gen.generate_trade_plan(report)
        assert plan is not None
        assert plan.direction == SignalDirection.LONG
        assert plan.symbol == "BNB/USDT"
        assert plan.entry.primary_entry > 0
        assert plan.stop_loss > 0
        assert len(plan.targets) >= 1
        assert plan.blended_rr > 0

    def test_generate_trade_plan_returns_none_low_confluence(self):
        """Score below threshold returns None."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator(min_confluence_score=50.0)  # Very high threshold
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        plan = gen.generate_trade_plan(report)
        assert plan is None

    def test_generate_trade_plan_returns_none_flat(self):
        """FLAT verdict returns None."""
        features = _make_weak_features("4h")

        gen = SMCTradePlanGenerator()
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4],
        )
        plan = gen.generate_trade_plan(report)
        assert plan is None

    def test_entry_zone_near_demand_for_longs(self):
        """Long entry zone is near the demand/support level."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator(min_confluence_score=5.0, min_rr_ratio=1.0)
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        plan = gen.generate_trade_plan(report)
        assert plan is not None
        # Entry should be near current price or demand zone
        assert plan.entry.primary_entry > 0
        assert plan.entry.entry_low is not None
        assert plan.entry.entry_high is not None
        assert plan.entry.entry_low <= plan.entry.primary_entry <= plan.entry.entry_high

    def test_stop_loss_below_support_for_longs(self):
        """Long SL is below the key support level."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator(min_confluence_score=5.0, min_rr_ratio=1.0)
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        plan = gen.generate_trade_plan(report)
        assert plan is not None
        assert plan.stop_loss < plan.entry.primary_entry

    def test_targets_ascending_for_longs(self):
        """Long targets are in ascending order."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator(min_confluence_score=5.0, min_rr_ratio=1.0)
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        plan = gen.generate_trade_plan(report)
        assert plan is not None
        assert len(plan.targets) >= 2
        for i in range(len(plan.targets) - 1):
            assert plan.targets[i].price <= plan.targets[i + 1].price

    def test_rr_ratio_positive(self):
        """R:R is positive for valid setups."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator(min_confluence_score=5.0, min_rr_ratio=1.0)
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        plan = gen.generate_trade_plan(report)
        assert plan is not None
        assert plan.blended_rr > 0

    def test_structure_narrative_format(self):
        """Narrative contains BOS/CHoCH descriptions."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator()
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert len(report.htf_structure_narrative) > 0
        # Should mention BOS or CHoCH
        assert "BOS" in report.htf_structure_narrative or "CHoCH" in report.htf_structure_narrative

    def test_format_report_contains_sections(self):
        """Output text has symbol, HTF Bias, Trade Verdict, and Score sections."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator(min_confluence_score=5.0, min_rr_ratio=1.0)
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        plan = gen.generate_trade_plan(report)
        text = gen.format_report(report, plan)

        assert "BNB/USDT" in text
        assert "HTF Bias" in text
        assert "Trade Verdict" in text
        assert "Score:" in text

    def test_format_report_bullish_long_setup(self):
        """Full integration: bullish features produce formatted long setup."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator(min_confluence_score=5.0, min_rr_ratio=1.0)
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        plan = gen.generate_trade_plan(report)
        text = gen.format_report(report, plan)

        text_upper = text.upper()
        assert "BULLISH" in text_upper or "LONG" in text_upper
        assert "Entry" in text or "Entry Zone" in text
        assert "Stop Loss" in text or "SL" in text
        assert "Target" in text or "TP" in text

    def test_trade_plan_to_signal_bridge(self):
        """TradePlan.to_signal() produces valid Signal kwargs."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator(min_confluence_score=5.0, min_rr_ratio=1.0)
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        plan = gen.generate_trade_plan(report)
        assert plan is not None

        signal_kwargs = plan.to_signal()
        assert signal_kwargs["direction"] == SignalDirection.LONG
        assert signal_kwargs["symbol"] == "BNB/USDT"
        assert signal_kwargs["confidence"] > 0
        assert "risk_constraints" in signal_kwargs

    def test_invalidation_conditions_populated(self):
        """At least one invalidation condition for directional setups."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator()
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        assert len(report.invalidation_conditions) >= 1
        for ic in report.invalidation_conditions:
            assert isinstance(ic, InvalidationCondition)
            assert len(ic.description) > 5

    def test_conviction_from_confluence_score(self):
        """High confluence = HIGH conviction."""
        features = {}
        features.update(_make_bullish_features("4h"))
        features.update(_make_bullish_features("1h"))

        gen = SMCTradePlanGenerator()
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4, Timeframe.H1],
        )
        # Bullish features have high confluence
        assert report.conviction in (ConvictionLevel.HIGH, ConvictionLevel.MODERATE)

    def test_format_report_no_plan(self):
        """Format works even without a trade plan."""
        features = _make_weak_features("4h")

        gen = SMCTradePlanGenerator()
        report = gen.generate_report(
            "BNB/USDT", 612.0, features,
            available_timeframes=[Timeframe.H4],
        )
        text = gen.format_report(report, None)
        assert "BNB/USDT" in text
        assert "Trade Verdict" in text
