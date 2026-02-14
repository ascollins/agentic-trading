"""Tests for structured trade plan model."""

import pytest

from agentic_trading.analysis.trade_plan import (
    EntryZone,
    TargetSpec,
    TradePlan,
)
from agentic_trading.core.enums import (
    ConvictionLevel,
    MarketStructureBias,
    SetupGrade,
    SignalDirection,
    Timeframe,
)


class TestEntryZone:
    def test_primary_entry_only(self):
        ez = EntryZone(primary_entry=95000.0)
        assert ez.primary_entry == 95000.0
        assert ez.scaled_entries == []

    def test_with_scaled_entries(self):
        ez = EntryZone(
            primary_entry=95000.0,
            entry_low=93000.0,
            entry_high=95000.0,
            scaled_entries=[(95000.0, 0.4), (93000.0, 0.35), (91000.0, 0.25)],
        )
        assert len(ez.scaled_entries) == 3


class TestTargetSpec:
    def test_basic_construction(self):
        ts = TargetSpec(price=100000.0, rr_ratio=2.5, scale_out_pct=0.4)
        assert ts.price == 100000.0
        assert ts.rr_ratio == 2.5


class TestTradePlan:
    def _make_plan(self, **overrides) -> TradePlan:
        defaults = dict(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            entry=EntryZone(primary_entry=95000.0),
            stop_loss=92000.0,
            strategy_id="test_strat",
            confidence=0.75,
            conviction=ConvictionLevel.HIGH,
            setup_grade=SetupGrade.A,
            blended_rr=2.5,
            targets=[
                TargetSpec(price=98000.0, rr_ratio=1.0, scale_out_pct=0.4),
                TargetSpec(price=102000.0, rr_ratio=2.33, scale_out_pct=0.35),
                TargetSpec(price=108000.0, rr_ratio=4.33, scale_out_pct=0.25),
            ],
            rationale="Bullish OB test at 95k",
        )
        defaults.update(overrides)
        return TradePlan(**defaults)

    def test_construction_with_required_fields(self):
        plan = TradePlan(
            symbol="ETH/USDT",
            direction=SignalDirection.SHORT,
            entry=EntryZone(primary_entry=3200.0),
            stop_loss=3400.0,
        )
        assert plan.symbol == "ETH/USDT"
        assert plan.direction == SignalDirection.SHORT

    def test_to_signal_risk_constraints(self):
        plan = self._make_plan()
        rc = plan.to_signal_risk_constraints()
        assert rc["sizing_method"] == "stop_loss_based"
        assert rc["entry"] == 95000.0
        assert rc["stop_loss"] == 92000.0
        assert rc["risk_pct"] == 0.01
        assert rc["blended_rr"] == 2.5
        assert rc["setup_grade"] == "A"
        assert rc["conviction"] == "high"
        assert len(rc["targets"]) == 3
        assert len(rc["scale_out_pcts"]) == 3

    def test_to_signal_with_scaled_entries(self):
        plan = self._make_plan(
            entry=EntryZone(
                primary_entry=95000.0,
                scaled_entries=[(95000.0, 0.5), (93000.0, 0.5)],
            ),
        )
        rc = plan.to_signal_risk_constraints()
        assert rc["sizing_method"] == "scaled_entry"
        assert len(rc["scaled_entries"]) == 2

    def test_to_signal_kwargs(self):
        plan = self._make_plan()
        sig = plan.to_signal()
        assert sig["strategy_id"] == "test_strat"
        assert sig["symbol"] == "BTC/USDT"
        assert sig["direction"] == SignalDirection.LONG
        assert sig["confidence"] == 0.75
        assert sig["rationale"] == "Bullish OB test at 95k"
        assert sig["timeframe"] == Timeframe.H1
        assert "risk_constraints" in sig
        assert "features_used" in sig

    def test_default_values(self):
        plan = TradePlan(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            entry=EntryZone(primary_entry=100.0),
            stop_loss=90.0,
        )
        assert plan.conviction == ConvictionLevel.MODERATE
        assert plan.setup_grade == SetupGrade.C
        assert plan.confidence == 0.0
        assert plan.risk_pct == 0.01
        assert plan.htf_bias == MarketStructureBias.UNCLEAR

    def test_metadata_preserved(self):
        plan = self._make_plan(metadata={"source": "manual", "chart_id": "abc123"})
        assert plan.metadata["source"] == "manual"

    def test_indicators_snapshot_in_signal(self):
        plan = self._make_plan(
            indicators_snapshot={"rsi_14": 45.2, "ema_21": 94800.0}
        )
        sig = plan.to_signal()
        assert sig["features_used"]["rsi_14"] == 45.2
