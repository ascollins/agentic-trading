"""Tests for governance.maturity — Strategy Maturity Manager."""

import pytest

from agentic_trading.core.config import MaturityConfig
from agentic_trading.core.enums import MaturityLevel
from agentic_trading.governance.maturity import MaturityManager


class TestMaturityManager:
    """MaturityManager level tracking and transitions."""

    def test_default_level_from_config(self, maturity_manager):
        """New strategies start at the config default level."""
        level = maturity_manager.get_level("new_strategy")
        assert level == MaturityLevel.L1_PAPER

    def test_custom_default_level(self):
        cfg = MaturityConfig(default_level="L0_shadow")
        mgr = MaturityManager(cfg)
        assert mgr.get_level("s1") == MaturityLevel.L0_SHADOW

    def test_set_level_direct(self, maturity_manager):
        """Admin override should directly set the level."""
        transition = maturity_manager.set_level(
            "s1", MaturityLevel.L4_AUTONOMOUS, reason="test"
        )
        assert transition is not None
        assert transition.from_level == MaturityLevel.L1_PAPER
        assert transition.to_level == MaturityLevel.L4_AUTONOMOUS
        assert maturity_manager.get_level("s1") == MaturityLevel.L4_AUTONOMOUS

    def test_set_same_level_returns_none(self, maturity_manager):
        """Setting the same level should be a no-op."""
        assert maturity_manager.set_level("s1", MaturityLevel.L1_PAPER) is None

    def test_can_execute_l0_false(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L0_SHADOW)
        assert maturity_manager.can_execute("s1") is False

    def test_can_execute_l1_false(self, maturity_manager):
        assert maturity_manager.can_execute("s1") is False  # default L1

    def test_can_execute_l2_true(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L2_GATED)
        assert maturity_manager.can_execute("s1") is True

    def test_can_execute_l4_true(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        assert maturity_manager.can_execute("s1") is True


class TestSizingCaps:
    """Maturity-based sizing caps."""

    def test_l0_sizing_cap_zero(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L0_SHADOW)
        assert maturity_manager.get_sizing_cap("s1") == 0.0

    def test_l1_sizing_cap_zero(self, maturity_manager):
        assert maturity_manager.get_sizing_cap("s1") == 0.0  # default L1

    def test_l2_sizing_cap(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L2_GATED)
        assert maturity_manager.get_sizing_cap("s1") == 0.10

    def test_l3_sizing_cap_from_config(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L3_CONSTRAINED)
        assert maturity_manager.get_sizing_cap("s1") == 0.25

    def test_l4_sizing_cap_full(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        assert maturity_manager.get_sizing_cap("s1") == 1.0


class TestPromotion:
    """Slow promotion logic."""

    def _good_metrics(self):
        return {"total_trades": 60, "win_rate": 0.55, "profit_factor": 1.5}

    def test_promotion_one_level(self, maturity_manager):
        """Should promote exactly one level."""
        maturity_manager.set_level("s1", MaturityLevel.L2_GATED)
        t = maturity_manager.evaluate_promotion("s1", self._good_metrics())
        assert t is not None
        assert t.to_level == MaturityLevel.L3_CONSTRAINED

    def test_no_promotion_if_already_l4(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        assert maturity_manager.evaluate_promotion("s1", self._good_metrics()) is None

    def test_no_promotion_insufficient_trades(self, maturity_manager):
        metrics = {"total_trades": 10, "win_rate": 0.6, "profit_factor": 2.0}
        assert maturity_manager.evaluate_promotion("s1", metrics) is None

    def test_no_promotion_low_win_rate(self, maturity_manager):
        metrics = {"total_trades": 100, "win_rate": 0.30, "profit_factor": 2.0}
        assert maturity_manager.evaluate_promotion("s1", metrics) is None


class TestDemotion:
    """Fast demotion logic."""

    def test_demotion_on_drawdown(self, maturity_manager):
        """Severe drawdown should demote to L1."""
        maturity_manager.set_level("s1", MaturityLevel.L4_AUTONOMOUS)
        metrics = {"drawdown_pct": 0.15, "loss_streak": 0}
        t = maturity_manager.evaluate_demotion("s1", metrics)
        assert t is not None
        assert t.to_level == MaturityLevel.L1_PAPER

    def test_demotion_on_loss_streak(self, maturity_manager):
        """Loss streak demotes one level."""
        maturity_manager.set_level("s1", MaturityLevel.L3_CONSTRAINED)
        metrics = {"drawdown_pct": 0.05, "loss_streak": 12}
        t = maturity_manager.evaluate_demotion("s1", metrics)
        assert t is not None
        assert t.to_level == MaturityLevel.L2_GATED

    def test_no_demotion_below_l0(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L0_SHADOW)
        metrics = {"drawdown_pct": 0.50, "loss_streak": 100}
        assert maturity_manager.evaluate_demotion("s1", metrics) is None

    def test_no_demotion_within_threshold(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L3_CONSTRAINED)
        metrics = {"drawdown_pct": 0.05, "loss_streak": 3}
        assert maturity_manager.evaluate_demotion("s1", metrics) is None

    def test_all_levels_introspection(self, maturity_manager):
        maturity_manager.set_level("s1", MaturityLevel.L3_CONSTRAINED)
        maturity_manager.set_level("s2", MaturityLevel.L4_AUTONOMOUS)
        levels = maturity_manager.all_levels()
        assert levels["s1"] == MaturityLevel.L3_CONSTRAINED
        assert levels["s2"] == MaturityLevel.L4_AUTONOMOUS
