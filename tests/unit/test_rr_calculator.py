"""Tests for risk-reward calculator."""

import pytest

from agentic_trading.analysis.rr_calculator import (
    RRResult,
    TargetLevel,
    calculate_rr,
    project_pnl,
)
from agentic_trading.core.enums import SetupGrade, SignalDirection


class TestCalculateRR:
    def test_basic_long_rr(self):
        result = calculate_rr(
            entry=100.0, stop_loss=90.0, targets=[110.0, 120.0]
        )
        assert result.is_valid
        assert result.direction == SignalDirection.LONG
        assert len(result.targets) == 2
        assert result.targets[0].rr_ratio == 1.0  # (110-100)/(100-90) = 1.0
        assert result.targets[1].rr_ratio == 2.0  # (120-100)/(100-90) = 2.0

    def test_basic_short_rr(self):
        result = calculate_rr(
            entry=100.0, stop_loss=110.0, targets=[90.0, 80.0]
        )
        assert result.is_valid
        assert result.direction == SignalDirection.SHORT
        assert result.targets[0].rr_ratio == 1.0  # (100-90)/(110-100) = 1.0
        assert result.targets[1].rr_ratio == 2.0

    def test_custom_scale_out_pcts(self):
        result = calculate_rr(
            entry=100.0, stop_loss=90.0,
            targets=[110.0, 120.0],
            scale_out_pcts=[0.6, 0.4],
        )
        assert result.targets[0].scale_out_pct == pytest.approx(0.6, abs=0.01)
        assert result.targets[1].scale_out_pct == pytest.approx(0.4, abs=0.01)
        # Blended RR = 1.0*0.6 + 2.0*0.4 = 1.4
        assert result.blended_rr == pytest.approx(1.4, abs=0.05)

    def test_zero_entry_returns_invalid(self):
        result = calculate_rr(entry=0, stop_loss=90.0, targets=[110.0])
        assert not result.is_valid
        assert "Invalid prices" in result.invalidation_reason

    def test_entry_equals_stop_returns_invalid(self):
        result = calculate_rr(entry=100.0, stop_loss=100.0, targets=[110.0])
        assert not result.is_valid
        assert "equals" in result.invalidation_reason.lower()

    def test_no_targets_returns_invalid(self):
        result = calculate_rr(entry=100.0, stop_loss=90.0, targets=[])
        assert not result.is_valid

    def test_single_target(self):
        result = calculate_rr(
            entry=100.0, stop_loss=90.0, targets=[130.0]
        )
        assert len(result.targets) == 1
        assert result.targets[0].rr_ratio == 3.0
        assert result.targets[0].scale_out_pct == pytest.approx(1.0, abs=0.01)

    def test_many_targets(self):
        result = calculate_rr(
            entry=100.0, stop_loss=95.0,
            targets=[105.0, 110.0, 115.0, 120.0, 130.0],
        )
        assert len(result.targets) == 5
        # First target: (105-100)/(100-95) = 1.0
        assert result.targets[0].rr_ratio == 1.0

    def test_direction_auto_inference_long(self):
        result = calculate_rr(entry=100.0, stop_loss=90.0, targets=[110.0])
        assert result.direction == SignalDirection.LONG

    def test_direction_auto_inference_short(self):
        result = calculate_rr(entry=90.0, stop_loss=100.0, targets=[80.0])
        assert result.direction == SignalDirection.SHORT

    def test_setup_grade_a_plus(self):
        # Very high RR setup
        result = calculate_rr(
            entry=100.0, stop_loss=99.0, targets=[105.0, 110.0]
        )
        # 5R and 10R targets -> blended should be very high
        assert result.setup_grade in (SetupGrade.A_PLUS, SetupGrade.A)

    def test_setup_grade_f(self):
        # Terrible RR — target barely moves, large stop
        result = calculate_rr(
            entry=100.0, stop_loss=80.0, targets=[101.0]
        )
        assert result.targets[0].rr_ratio < 1.0
        assert result.setup_grade in (SetupGrade.D, SetupGrade.F)

    def test_risk_per_unit(self):
        result = calculate_rr(entry=100.0, stop_loss=90.0, targets=[110.0])
        assert result.risk_per_unit == pytest.approx(10.0, abs=0.001)

    def test_scale_out_pcts_normalised(self):
        # Pcts sum to 200 not 100 -> should be normalised
        result = calculate_rr(
            entry=100.0, stop_loss=90.0,
            targets=[110.0, 120.0],
            scale_out_pcts=[100.0, 100.0],
        )
        assert result.is_valid
        assert result.targets[0].scale_out_pct == pytest.approx(0.5, abs=0.01)

    def test_blended_rr_computed(self):
        result = calculate_rr(
            entry=100.0, stop_loss=90.0, targets=[120.0, 130.0]
        )
        # Equal weight: (2.0 + 3.0) / 2 = 2.5
        assert result.blended_rr == pytest.approx(2.5, abs=0.05)


class TestProjectPnl:
    def test_basic_pnl_projection(self):
        rr = calculate_rr(entry=100.0, stop_loss=90.0, targets=[120.0])
        pnl = project_pnl(account_size=100_000, risk_pct=0.01, rr_result=rr)
        assert pnl["risk_amount"] == 1000.0
        assert pnl["max_loss"] == -1000.0

    def test_risk_amount_correct(self):
        rr = calculate_rr(entry=100.0, stop_loss=90.0, targets=[110.0])
        pnl = project_pnl(account_size=50_000, risk_pct=0.02, rr_result=rr)
        assert pnl["risk_amount"] == 1000.0  # 50k * 2%

    def test_scenarios_count_matches_targets(self):
        rr = calculate_rr(
            entry=100.0, stop_loss=90.0, targets=[110.0, 120.0, 130.0]
        )
        pnl = project_pnl(account_size=100_000, risk_pct=0.01, rr_result=rr)
        assert len(pnl["scenarios"]) == 3

    def test_full_target_profit_positive(self):
        rr = calculate_rr(entry=100.0, stop_loss=90.0, targets=[120.0])
        pnl = project_pnl(account_size=100_000, risk_pct=0.01, rr_result=rr)
        assert pnl["full_target_profit"] > 0

    def test_setup_grade_present(self):
        rr = calculate_rr(entry=100.0, stop_loss=90.0, targets=[120.0])
        pnl = project_pnl(account_size=100_000, risk_pct=0.01, rr_result=rr)
        assert "setup_grade" in pnl
        assert pnl["setup_grade"] in ("A+", "A", "B", "C", "D", "F")
