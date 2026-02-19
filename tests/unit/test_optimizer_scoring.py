"""Tests for the enhanced optimizer composite scoring, recommendations,
and strategy discovery.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agentic_trading.core.config import OptimizerSchedulerConfig, StrategyParamConfig
from agentic_trading.core.enums import OptimizationRecommendation
from agentic_trading.optimizer.param_grid import (
    CMT_STRATEGIES,
    list_strategies_with_grids,
    strategies_missing_grids,
)
from agentic_trading.optimizer.report import (
    CompositeScoreWeights,
    OptimizationCycleReport,
    StrategyRecommendation,
    StrategyResult,
)
from agentic_trading.optimizer.scheduler import OptimizerScheduler


def _make_result(**kwargs) -> StrategyResult:
    """Factory for StrategyResult with sensible defaults."""
    defaults = {
        "params": {},
        "total_return": 0.10,
        "sharpe_ratio": 1.0,
        "sortino_ratio": 1.5,
        "calmar_ratio": 2.0,
        "max_drawdown": -0.08,
        "total_trades": 100,
        "win_rate": 0.55,
        "profit_factor": 1.8,
        "avg_win": 0.02,
        "avg_loss": 0.01,
        "annualized_return": 0.20,
        "total_fees": 50.0,
    }
    defaults.update(kwargs)
    return StrategyResult(**defaults)


def _make_scheduler(**kwargs) -> OptimizerScheduler:
    """Factory for OptimizerScheduler with test config."""
    config = kwargs.pop("config", None)
    if config is None:
        config = OptimizerSchedulerConfig(enabled=True)
    return OptimizerScheduler(
        config=config,
        data_dir="data/historical",
        agent_id="test-optimizer",
        **kwargs,
    )


# ===========================================================================
# CompositeScoreWeights tests
# ===========================================================================


class TestCompositeScoreWeights:
    """Tests for multi-objective composite scoring."""

    def test_positive_result_gets_positive_score(self):
        weights = CompositeScoreWeights()
        result = _make_result(
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.5,
            profit_factor=1.8,
            max_drawdown=-0.08,
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
        )
        score = weights.compute(result)
        assert score > 0

    def test_negative_result_gets_low_score(self):
        weights = CompositeScoreWeights()
        result = _make_result(
            sharpe_ratio=-1.0,
            sortino_ratio=-1.5,
            calmar_ratio=-0.5,
            profit_factor=0.5,
            max_drawdown=-0.35,
            win_rate=0.30,
            avg_win=0.005,
            avg_loss=0.015,
        )
        score = weights.compute(result)
        assert score < 0

    def test_profit_factor_capped_at_five(self):
        weights = CompositeScoreWeights()
        r1 = _make_result(profit_factor=5.0)
        r2 = _make_result(profit_factor=100.0)
        assert weights.compute(r1) == weights.compute(r2)

    def test_higher_drawdown_produces_lower_score(self):
        weights = CompositeScoreWeights()
        r1 = _make_result(max_drawdown=-0.05)
        r2 = _make_result(max_drawdown=-0.20)
        assert weights.compute(r1) > weights.compute(r2)

    def test_higher_sortino_produces_higher_score(self):
        weights = CompositeScoreWeights()
        r1 = _make_result(sortino_ratio=3.0)
        r2 = _make_result(sortino_ratio=1.0)
        assert weights.compute(r1) > weights.compute(r2)

    def test_higher_calmar_produces_higher_score(self):
        weights = CompositeScoreWeights()
        r1 = _make_result(calmar_ratio=3.0)
        r2 = _make_result(calmar_ratio=1.0)
        assert weights.compute(r1) > weights.compute(r2)

    def test_positive_expectancy_boosts_score(self):
        weights = CompositeScoreWeights()
        # Positive expectancy: 0.55 * 0.03 - 0.45 * 0.01 = 0.012
        r1 = _make_result(win_rate=0.55, avg_win=0.03, avg_loss=0.01)
        # Negative expectancy: 0.30 * 0.01 - 0.70 * 0.02 = -0.011
        r2 = _make_result(win_rate=0.30, avg_win=0.01, avg_loss=0.02)
        assert weights.compute(r1) > weights.compute(r2)

    def test_custom_weights(self):
        # Only care about Sharpe
        weights = CompositeScoreWeights(
            sortino_weight=0.0,
            calmar_weight=0.0,
            max_drawdown_penalty=0.0,
            profit_factor_weight=0.0,
            expectancy_weight=0.0,
            sharpe_weight=1.0,
        )
        r1 = _make_result(sharpe_ratio=2.0, sortino_ratio=0.0)
        r2 = _make_result(sharpe_ratio=1.0, sortino_ratio=5.0)
        assert weights.compute(r1) > weights.compute(r2)

    def test_zero_avg_loss_handles_gracefully(self):
        weights = CompositeScoreWeights()
        result = _make_result(avg_loss=0.0, win_rate=1.0, avg_win=0.05)
        # Should not raise
        score = weights.compute(result)
        assert isinstance(score, float)


# ===========================================================================
# Recommendation decision logic tests
# ===========================================================================


class TestDetermineRecommendation:
    """Tests for the KEEP/UPDATE/DISABLE decision tree."""

    def test_disable_for_terrible_metrics(self):
        scheduler = _make_scheduler()
        best_result = _make_result(
            sortino_ratio=-1.0,
            calmar_ratio=-0.3,
            max_drawdown=-0.35,
            win_rate=0.25,
            avg_win=0.005,
            avg_loss=0.02,
        )
        rec, rationale = scheduler._determine_recommendation(
            strategy_id="test",
            optimized_score=-1.0,
            current_score=0.5,
            improvement_pct=-200.0,
            is_overfit=False,
            wf_passed=True,
            best_result=best_result,
            current_metrics={},
            optimized_metrics={},
        )
        assert rec == OptimizationRecommendation.DISABLE
        assert "not viable" in rationale

    def test_keep_for_insufficient_improvement(self):
        scheduler = _make_scheduler()
        best_result = _make_result()
        rec, rationale = scheduler._determine_recommendation(
            strategy_id="test",
            optimized_score=1.1,
            current_score=1.0,
            improvement_pct=5.0,  # Below default 10% threshold
            is_overfit=False,
            wf_passed=True,
            best_result=best_result,
            current_metrics={},
            optimized_metrics={},
        )
        assert rec == OptimizationRecommendation.KEEP
        assert "below threshold" in rationale

    def test_keep_for_overfit(self):
        scheduler = _make_scheduler()
        best_result = _make_result()
        rec, rationale = scheduler._determine_recommendation(
            strategy_id="test",
            optimized_score=2.0,
            current_score=1.0,
            improvement_pct=50.0,
            is_overfit=True,
            wf_passed=False,
            best_result=best_result,
            current_metrics={},
            optimized_metrics={},
        )
        assert rec == OptimizationRecommendation.KEEP
        assert "overfitting" in rationale

    def test_update_for_meaningful_improvement(self):
        scheduler = _make_scheduler()
        best_result = _make_result(
            sortino_ratio=2.0,
            calmar_ratio=2.5,
            profit_factor=2.2,
        )
        optimized_metrics = {
            "sortino": 2.0,
            "calmar": 2.5,
            "max_drawdown": -0.08,
            "profit_factor": 2.2,
            "expectancy": 0.012,
        }
        rec, rationale = scheduler._determine_recommendation(
            strategy_id="test",
            optimized_score=2.0,
            current_score=1.0,
            improvement_pct=50.0,
            is_overfit=False,
            wf_passed=True,
            best_result=best_result,
            current_metrics={},
            optimized_metrics=optimized_metrics,
        )
        assert rec == OptimizationRecommendation.UPDATE
        assert "walk-forward passed" in rationale

    def test_update_without_wf_requirement(self):
        config = OptimizerSchedulerConfig(
            enabled=True,
            require_walk_forward_pass=False,
        )
        scheduler = _make_scheduler(config=config)
        best_result = _make_result()
        rec, _ = scheduler._determine_recommendation(
            strategy_id="test",
            optimized_score=2.0,
            current_score=1.0,
            improvement_pct=50.0,
            is_overfit=True,  # overfit but WF not required
            wf_passed=False,
            best_result=best_result,
            current_metrics={},
            optimized_metrics={},
        )
        assert rec == OptimizationRecommendation.UPDATE


# ===========================================================================
# Strategy discovery tests
# ===========================================================================


class TestStrategyDiscovery:
    """Tests for CMT strategy discovery."""

    def test_default_config_returns_all_cmt_strategies(self):
        scheduler = _make_scheduler()
        strategies = scheduler._discover_strategies()
        # All 8 CMT strategies should be discovered since we added all grids
        assert len(strategies) == 8
        for s in CMT_STRATEGIES:
            assert s in strategies

    def test_discover_all_from_registry(self):
        config = OptimizerSchedulerConfig(
            enabled=True,
            discover_all_strategies=True,
        )
        scheduler = _make_scheduler(config=config)
        strategies = scheduler._discover_strategies()
        # Should discover at least the CMT strategies with grids
        assert len(strategies) >= 8

    def test_filters_out_strategies_without_grids(self):
        config = OptimizerSchedulerConfig(
            enabled=True,
            strategies=["multi_tf_ma", "nonexistent_strategy"],
        )
        scheduler = _make_scheduler(config=config)
        strategies = scheduler._discover_strategies()
        assert "multi_tf_ma" in strategies
        assert "nonexistent_strategy" not in strategies


# ===========================================================================
# Param grid tests
# ===========================================================================


class TestParamGrids:
    """Tests for param grid completeness."""

    def test_all_cmt_strategies_have_grids(self):
        grids = list_strategies_with_grids()
        for s in CMT_STRATEGIES:
            assert s in grids, f"CMT strategy {s} missing param grid"

    def test_no_missing_cmt_grids(self):
        missing = strategies_missing_grids(CMT_STRATEGIES)
        assert missing == [], f"CMT strategies without grids: {missing}"


# ===========================================================================
# Auto-apply guardrails tests
# ===========================================================================


class TestAutoApplyGuardrails:
    """Tests for auto-apply safety checks."""

    @pytest.mark.asyncio
    async def test_skips_non_update(self):
        scheduler = _make_scheduler()
        rec = StrategyRecommendation(
            strategy_id="test",
            recommendation=OptimizationRecommendation.KEEP,
        )
        # Should return without applying
        await scheduler._try_auto_apply(rec)

    @pytest.mark.asyncio
    async def test_skips_overfit(self):
        scheduler = _make_scheduler()
        rec = StrategyRecommendation(
            strategy_id="test",
            recommendation=OptimizationRecommendation.UPDATE,
            is_overfit=True,
        )
        await scheduler._try_auto_apply(rec)
        # No apply should happen (no strategy config to check)

    @pytest.mark.asyncio
    async def test_skips_wf_failure(self):
        scheduler = _make_scheduler()
        rec = StrategyRecommendation(
            strategy_id="test",
            recommendation=OptimizationRecommendation.UPDATE,
            is_overfit=False,
            walk_forward_passed=False,
        )
        await scheduler._try_auto_apply(rec)

    @pytest.mark.asyncio
    async def test_applies_when_guardrails_pass(self):
        config = OptimizerSchedulerConfig(
            enabled=True,
            require_governance_approval=False,
        )
        strategy_config = [
            StrategyParamConfig(
                strategy_id="test",
                params={"old_param": 1},
            )
        ]
        scheduler = _make_scheduler(
            config=config,
            strategy_config=strategy_config,
        )
        rec = StrategyRecommendation(
            strategy_id="test",
            recommendation=OptimizationRecommendation.UPDATE,
            optimized_params={"new_param": 2},
            is_overfit=False,
            walk_forward_passed=True,
            improvement_pct=25.0,
        )
        await scheduler._try_auto_apply(rec)
        # Check that params were updated
        assert strategy_config[0].params.get("new_param") == 2

    @pytest.mark.asyncio
    async def test_skips_when_governance_required(self):
        config = OptimizerSchedulerConfig(
            enabled=True,
            require_governance_approval=True,
        )
        mock_gate = MagicMock()
        strategy_config = [
            StrategyParamConfig(
                strategy_id="test",
                params={"old_param": 1},
            )
        ]
        scheduler = _make_scheduler(
            config=config,
            strategy_config=strategy_config,
            governance_gate=mock_gate,
        )
        rec = StrategyRecommendation(
            strategy_id="test",
            recommendation=OptimizationRecommendation.UPDATE,
            optimized_params={"new_param": 2},
            is_overfit=False,
            walk_forward_passed=True,
        )
        await scheduler._try_auto_apply(rec)
        # Params should NOT be updated (governance required)
        assert "new_param" not in strategy_config[0].params


# ===========================================================================
# Introspection tests
# ===========================================================================


class TestIntrospection:
    """Tests for history tracking and trajectory."""

    def test_get_current_params(self):
        strategy_config = [
            StrategyParamConfig(
                strategy_id="bb_squeeze",
                params={"adx_threshold": 25},
            )
        ]
        scheduler = _make_scheduler(strategy_config=strategy_config)
        params = scheduler._get_current_params("bb_squeeze")
        assert params == {"adx_threshold": 25}

    def test_get_current_params_missing(self):
        scheduler = _make_scheduler()
        params = scheduler._get_current_params("nonexistent")
        assert params == {}

    def test_get_latest_recommendation(self):
        scheduler = _make_scheduler()
        rec = StrategyRecommendation(
            strategy_id="bb_squeeze",
            recommendation=OptimizationRecommendation.UPDATE,
        )
        cycle = OptimizationCycleReport(
            run_number=1,
            recommendations=[rec],
        )
        scheduler._history.append(cycle)
        latest = scheduler.get_latest_recommendation("bb_squeeze")
        assert latest is not None
        assert latest.recommendation == OptimizationRecommendation.UPDATE

    def test_get_latest_recommendation_none(self):
        scheduler = _make_scheduler()
        assert scheduler.get_latest_recommendation("nonexistent") is None

    def test_improvement_trajectory(self):
        scheduler = _make_scheduler()
        for i in range(3):
            rec = StrategyRecommendation(
                strategy_id="bb_squeeze",
                recommendation=OptimizationRecommendation.KEEP,
                current_score=float(i),
                optimized_score=float(i + 1),
            )
            cycle = OptimizationCycleReport(
                run_number=i + 1,
                recommendations=[rec],
            )
            scheduler._history.append(cycle)

        trajectory = scheduler.get_improvement_trajectory("bb_squeeze")
        assert len(trajectory) == 3
        assert trajectory[0]["run_number"] == 1
        assert trajectory[2]["current_score"] == 2.0
