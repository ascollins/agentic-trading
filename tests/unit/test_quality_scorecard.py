"""Tests for the QualityScorecard — trader agent quality measurement."""

from __future__ import annotations

import pytest

from agentic_trading.journal.quality_scorecard import (
    Grade,
    MetricGrade,
    PortfolioQualityReport,
    QualityReport,
    QualityScorecard,
    StrategyType,
    classify_strategy,
    score_to_grade,
    _grade_sharpe,
    _grade_sortino,
    _grade_calmar,
    _grade_profit_factor,
    _grade_max_drawdown,
    _grade_win_rate,
    _grade_avg_r,
    _grade_expectancy,
    _grade_management_efficiency,
    _grade_statistical_edge,
    _grade_ruin_probability,
)


# ================================================================== #
# Helper factories                                                    #
# ================================================================== #


def _make_stats(
    *,
    total_trades: int = 50,
    wins: int = 25,
    win_rate: float = 0.5,
    profit_factor: float = 1.5,
    expectancy: float = 10.0,
    avg_r: float = 0.3,
    sharpe: float = 0.8,
    avg_management_efficiency: float = 0.5,
    **kwargs,
) -> dict:
    """Build a strategy stats dict."""
    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": total_trades - wins,
        "breakevens": 0,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_r": avg_r,
        "avg_winner": 30.0,
        "avg_loser": 20.0,
        "sharpe": sharpe,
        "total_pnl": expectancy * total_trades,
        "total_fees": 5.0,
        "best_trade": 100.0,
        "worst_trade": -50.0,
        "current_streak": 2,
        "max_win_streak": 5,
        "max_loss_streak": 4,
        "avg_hold_seconds": 3600.0,
        "avg_management_efficiency": avg_management_efficiency,
        **kwargs,
    }


def _make_backtest_result(
    *,
    sharpe_ratio: float = 1.2,
    sortino_ratio: float = 1.8,
    calmar_ratio: float = 1.5,
    max_drawdown: float = -0.10,
    total_trades: int = 100,
    win_rate: float = 0.45,
    profit_factor: float = 1.8,
    avg_trade_return: float = 0.005,
    total_return: float = 0.15,
):
    """Build a mock BacktestResult-like object."""

    class _MockResult:
        pass

    r = _MockResult()
    r.sharpe_ratio = sharpe_ratio
    r.sortino_ratio = sortino_ratio
    r.calmar_ratio = calmar_ratio
    r.max_drawdown = max_drawdown
    r.total_trades = total_trades
    r.win_rate = win_rate
    r.profit_factor = profit_factor
    r.avg_trade_return = avg_trade_return
    r.total_return = total_return
    return r


def _make_edge_result(
    *,
    has_edge: bool = True,
    p_value_bootstrap: float = 0.03,
    sample_adequate: bool = True,
) -> dict:
    """Build a CoinFlipBaseline.evaluate() result."""
    return {
        "strategy_id": "test",
        "has_edge": has_edge,
        "p_value_bootstrap": p_value_bootstrap,
        "sample_adequate": sample_adequate,
        "trade_count": 100,
    }


def _make_monte_carlo_result(
    *,
    ruin_probability: float = 0.02,
) -> dict:
    """Build a MonteCarloProjector.project() result."""
    return {
        "strategy_id": "test",
        "ruin_probability": ruin_probability,
        "mean_terminal_equity": 120000.0,
        "median_terminal_equity": 115000.0,
    }


# ================================================================== #
# Grade conversion tests                                              #
# ================================================================== #


class TestScoreToGrade:
    def test_a_plus(self):
        assert score_to_grade(97.0) == Grade.A_PLUS

    def test_a(self):
        assert score_to_grade(92.0) == Grade.A

    def test_a_minus(self):
        assert score_to_grade(87.0) == Grade.A_MINUS

    def test_b_plus(self):
        assert score_to_grade(82.0) == Grade.B_PLUS

    def test_b(self):
        assert score_to_grade(77.0) == Grade.B

    def test_b_minus(self):
        assert score_to_grade(72.0) == Grade.B_MINUS

    def test_c_plus(self):
        assert score_to_grade(67.0) == Grade.C_PLUS

    def test_c(self):
        assert score_to_grade(62.0) == Grade.C

    def test_c_minus(self):
        assert score_to_grade(57.0) == Grade.C_MINUS

    def test_d(self):
        assert score_to_grade(45.0) == Grade.D

    def test_f(self):
        assert score_to_grade(20.0) == Grade.F

    def test_zero(self):
        assert score_to_grade(0.0) == Grade.F

    def test_perfect(self):
        assert score_to_grade(100.0) == Grade.A_PLUS

    def test_clamp_above_100(self):
        assert score_to_grade(150.0) == Grade.A_PLUS

    def test_clamp_below_0(self):
        assert score_to_grade(-10.0) == Grade.F

    def test_boundary_95(self):
        assert score_to_grade(95.0) == Grade.A_PLUS

    def test_boundary_90(self):
        assert score_to_grade(90.0) == Grade.A


# ================================================================== #
# Strategy classification tests                                       #
# ================================================================== #


class TestClassifyStrategy:
    def test_trend_following(self):
        assert classify_strategy("trend_following") == StrategyType.TREND

    def test_mean_reversion(self):
        assert classify_strategy("mean_reversion") == StrategyType.MEAN_REVERSION

    def test_breakout(self):
        assert classify_strategy("breakout") == StrategyType.BREAKOUT

    def test_rsi_divergence(self):
        assert classify_strategy("rsi_divergence") == StrategyType.MOMENTUM

    def test_bb_squeeze(self):
        assert classify_strategy("bb_squeeze") == StrategyType.MEAN_REVERSION

    def test_fibonacci_confluence(self):
        assert classify_strategy("fibonacci_confluence") == StrategyType.STATISTICAL

    def test_unknown_defaults_to_hybrid(self):
        assert classify_strategy("unknown_strategy") == StrategyType.HYBRID


# ================================================================== #
# Individual metric grading tests                                     #
# ================================================================== #


class TestGradeSharpe:
    def test_excellent_sharpe(self):
        g = _grade_sharpe(2.5)
        assert g.score == 100.0
        assert g.grade == Grade.A_PLUS

    def test_very_good_sharpe(self):
        g = _grade_sharpe(1.5)
        assert g.score == 85.0
        assert g.grade == Grade.A_MINUS

    def test_good_sharpe(self):
        g = _grade_sharpe(1.0)
        assert g.score == 75.0
        assert g.grade == Grade.B

    def test_mediocre_sharpe(self):
        g = _grade_sharpe(0.5)
        assert g.score == 60.0
        assert g.grade == Grade.C

    def test_poor_sharpe(self):
        g = _grade_sharpe(0.0)
        assert g.score == 40.0
        assert g.grade == Grade.D

    def test_negative_sharpe(self):
        g = _grade_sharpe(-1.0)
        assert g.score == 0.0  # clamped

    def test_weight_is_high(self):
        g = _grade_sharpe(1.0)
        assert g.weight == 1.5


class TestGradeSortino:
    def test_excellent(self):
        g = _grade_sortino(3.0)
        assert g.score == 100.0

    def test_good(self):
        g = _grade_sortino(1.5)
        assert g.score == 75.0

    def test_poor(self):
        g = _grade_sortino(0.0)
        assert g.score == 35.0


class TestGradeCalmar:
    def test_excellent(self):
        g = _grade_calmar(3.0)
        assert g.score == 100.0

    def test_good(self):
        g = _grade_calmar(1.0)
        assert g.score == 70.0

    def test_poor(self):
        g = _grade_calmar(0.0)
        assert g.score == 35.0


class TestGradeProfitFactor:
    def test_excellent(self):
        g = _grade_profit_factor(3.0)
        assert g.score == 100.0

    def test_target(self):
        g = _grade_profit_factor(1.75)
        assert g.score == 75.0

    def test_breakeven(self):
        g = _grade_profit_factor(1.0)
        assert g.score == 40.0

    def test_losing(self):
        g = _grade_profit_factor(0.5)
        assert g.score == 20.0

    def test_infinity(self):
        g = _grade_profit_factor(float("inf"))
        assert g.score == 100.0


class TestGradeMaxDrawdown:
    def test_minimal_drawdown(self):
        g = _grade_max_drawdown(-0.03)
        assert g.score == 100.0

    def test_target_drawdown(self):
        g = _grade_max_drawdown(-0.15)
        assert g.score == 75.0

    def test_severe_drawdown(self):
        g = _grade_max_drawdown(-0.35)
        assert g.score == 20.0

    def test_displays_positive_percentage(self):
        g = _grade_max_drawdown(-0.15)
        assert g.value == 15.0  # Displayed as positive %

    def test_weight_is_high(self):
        g = _grade_max_drawdown(-0.10)
        assert g.weight == 1.5


class TestGradeWinRate:
    def test_trend_low_winrate_still_ok(self):
        """Trend strategies should pass with 40% win rate."""
        g = _grade_win_rate(0.40, StrategyType.TREND)
        assert g.score >= 60.0

    def test_mean_reversion_needs_high_winrate(self):
        """Mean reversion at 40% should be poor."""
        g = _grade_win_rate(0.40, StrategyType.MEAN_REVERSION)
        assert g.score == 40.0

    def test_mean_reversion_70pct_is_excellent(self):
        g = _grade_win_rate(0.70, StrategyType.MEAN_REVERSION)
        assert g.score == 90.0

    def test_breakout_30pct_acceptable(self):
        g = _grade_win_rate(0.30, StrategyType.BREAKOUT)
        assert g.score >= 55.0


class TestGradeAvgR:
    def test_excellent(self):
        g = _grade_avg_r(1.0)
        assert g.score == 100.0

    def test_target(self):
        g = _grade_avg_r(0.3)
        assert g.score == 70.0

    def test_negative(self):
        g = _grade_avg_r(-0.5)
        assert g.score == 0.0


class TestGradeExpectancy:
    def test_positive_high(self):
        g = _grade_expectancy(100.0)
        assert g.score == 95.0

    def test_positive_moderate(self):
        g = _grade_expectancy(20.0)
        assert g.score == 75.0

    def test_zero(self):
        g = _grade_expectancy(0.0)
        assert g.score == 40.0

    def test_negative(self):
        g = _grade_expectancy(-10.0)
        assert g.score == 30.0


class TestGradeManagementEfficiency:
    def test_excellent(self):
        g = _grade_management_efficiency(0.80)
        assert g.score == 95.0

    def test_target(self):
        g = _grade_management_efficiency(0.50)
        assert g.score == 65.0

    def test_poor(self):
        g = _grade_management_efficiency(0.20)
        assert g.score < 40.0


class TestGradeStatisticalEdge:
    def test_strong_edge(self):
        g = _grade_statistical_edge(True, 0.005, True)
        assert g.score == 95.0

    def test_significant_edge(self):
        g = _grade_statistical_edge(True, 0.03, True)
        assert g.score == 80.0

    def test_no_edge(self):
        g = _grade_statistical_edge(False, 0.50, True)
        assert g.score == 25.0

    def test_insufficient_sample(self):
        g = _grade_statistical_edge(False, 0.50, False)
        assert g.score == 50.0


class TestGradeRuinProbability:
    def test_very_safe(self):
        g = _grade_ruin_probability(0.01)
        assert g.score == 100.0

    def test_safe(self):
        g = _grade_ruin_probability(0.05)
        assert g.score == 75.0

    def test_dangerous(self):
        g = _grade_ruin_probability(0.30)
        assert g.score == 20.0


# ================================================================== #
# QualityScorecard integration tests                                  #
# ================================================================== #


class TestQualityScorecard:
    def setup_method(self):
        self.scorecard = QualityScorecard(min_trades=15, passing_score=60.0)

    def test_evaluate_with_stats_only(self):
        stats = _make_stats(
            total_trades=50,
            win_rate=0.50,
            profit_factor=1.8,
            expectancy=15.0,
            avg_r=0.4,
            sharpe=1.0,
            avg_management_efficiency=0.6,
        )
        report = self.scorecard.evaluate(
            strategy_id="trend_following",
            stats=stats,
        )
        assert report.strategy_id == "trend_following"
        assert report.strategy_type == StrategyType.TREND
        assert report.total_trades == 50
        assert report.sufficient_data is True
        assert len(report.metrics) >= 9  # Core metrics
        assert 0 <= report.overall_score <= 100

    def test_evaluate_with_backtest_result(self):
        result = _make_backtest_result()
        report = self.scorecard.evaluate(
            strategy_id="mean_reversion",
            backtest_result=result,
        )
        assert report.strategy_type == StrategyType.MEAN_REVERSION
        assert report.total_trades == 100
        assert report.sufficient_data is True
        # With good metrics, should pass
        assert report.overall_score > 60.0

    def test_evaluate_with_all_data_sources(self):
        stats = _make_stats()
        backtest = _make_backtest_result()
        edge = _make_edge_result()
        mc = _make_monte_carlo_result()

        report = self.scorecard.evaluate(
            strategy_id="breakout",
            stats=stats,
            backtest_result=backtest,
            edge_result=edge,
            monte_carlo_result=mc,
        )

        # Should have more metrics when all sources provided
        assert len(report.metrics) >= 11  # 9 core + edge + ruin
        assert report.sufficient_data is True

    def test_insufficient_trades(self):
        stats = _make_stats(total_trades=5)
        report = self.scorecard.evaluate(
            strategy_id="trend_following",
            stats=stats,
        )
        assert report.sufficient_data is False
        assert report.passing is False

    def test_excellent_strategy_passes(self):
        backtest = _make_backtest_result(
            sharpe_ratio=2.0,
            sortino_ratio=3.0,
            calmar_ratio=2.5,
            max_drawdown=-0.05,
            profit_factor=2.5,
            win_rate=0.55,
            avg_trade_return=0.01,
        )
        stats = _make_stats(
            avg_r=0.7,
            avg_management_efficiency=0.75,
        )
        edge = _make_edge_result(has_edge=True, p_value_bootstrap=0.005)
        mc = _make_monte_carlo_result(ruin_probability=0.005)

        report = self.scorecard.evaluate(
            strategy_id="trend_following",
            stats=stats,
            backtest_result=backtest,
            edge_result=edge,
            monte_carlo_result=mc,
        )

        assert report.passing is True
        assert report.overall_score >= 80.0
        assert report.overall_grade.value in ("A+", "A", "A-", "B+", "B")

    def test_terrible_strategy_fails(self):
        backtest = _make_backtest_result(
            sharpe_ratio=-0.5,
            sortino_ratio=-0.3,
            calmar_ratio=-0.2,
            max_drawdown=-0.40,
            profit_factor=0.5,
            win_rate=0.15,
            avg_trade_return=-0.02,
        )
        report = self.scorecard.evaluate(
            strategy_id="breakout",
            backtest_result=backtest,
        )

        assert report.passing is False
        assert report.overall_score < 40.0
        assert report.overall_grade == Grade.F

    def test_strategy_type_override(self):
        stats = _make_stats()
        report = self.scorecard.evaluate(
            strategy_id="custom_strategy",
            strategy_type="momentum",
            stats=stats,
        )
        assert report.strategy_type == StrategyType.MOMENTUM

    def test_strategy_type_auto_detect(self):
        stats = _make_stats()
        report = self.scorecard.evaluate(
            strategy_id="rsi_divergence",
            stats=stats,
        )
        assert report.strategy_type == StrategyType.MOMENTUM

    def test_report_has_strengths_and_weaknesses(self):
        backtest = _make_backtest_result(
            sharpe_ratio=2.0,
            max_drawdown=-0.30,  # Bad
        )
        report = self.scorecard.evaluate(
            strategy_id="trend_following",
            backtest_result=backtest,
        )
        # Should identify something about max drawdown as weakness
        assert len(report.weaknesses) > 0 or len(report.strengths) > 0

    def test_report_to_dict(self):
        stats = _make_stats()
        report = self.scorecard.evaluate(
            strategy_id="trend_following",
            stats=stats,
        )
        d = report.to_dict()
        assert "strategy_id" in d
        assert "overall_score" in d
        assert "overall_grade" in d
        assert "metrics" in d
        assert isinstance(d["metrics"], list)
        assert all("name" in m for m in d["metrics"])

    def test_summary_table(self):
        stats = _make_stats()
        report = self.scorecard.evaluate(
            strategy_id="trend_following",
            stats=stats,
        )
        table = report.summary_table()
        assert "TRADER QUALITY SCORECARD" in table
        assert "trend_following" in table
        assert "Sharpe Ratio" in table

    def test_edge_result_with_error_skipped(self):
        stats = _make_stats()
        edge = {"error": "insufficient_data", "min_trades": 10}
        report = self.scorecard.evaluate(
            strategy_id="trend_following",
            stats=stats,
            edge_result=edge,
        )
        # Should not include statistical edge metric
        metric_names = [m.name for m in report.metrics]
        assert "statistical_edge" not in metric_names

    def test_monte_carlo_with_error_skipped(self):
        stats = _make_stats()
        mc = {"error": "insufficient_data", "min_trades": 5}
        report = self.scorecard.evaluate(
            strategy_id="trend_following",
            stats=stats,
            monte_carlo_result=mc,
        )
        metric_names = [m.name for m in report.metrics]
        assert "ruin_probability" not in metric_names


# ================================================================== #
# Portfolio quality report tests                                      #
# ================================================================== #


class TestPortfolioQualityReport:
    def setup_method(self):
        self.scorecard = QualityScorecard()

    def test_empty_portfolio(self):
        report = self.scorecard.evaluate_portfolio([])
        assert report.overall_score == 0.0
        assert report.passing_strategies == 0

    def test_portfolio_with_strategies(self):
        stats_good = _make_stats(
            total_trades=100,
            win_rate=0.55,
            profit_factor=2.0,
            sharpe=1.5,
            expectancy=25.0,
            avg_r=0.5,
            avg_management_efficiency=0.7,
        )
        stats_bad = _make_stats(
            total_trades=50,
            win_rate=0.25,
            profit_factor=0.8,
            sharpe=-0.2,
            expectancy=-5.0,
            avg_r=-0.1,
            avg_management_efficiency=0.3,
        )

        report_good = self.scorecard.evaluate(
            strategy_id="trend_following",
            stats=stats_good,
        )
        report_bad = self.scorecard.evaluate(
            strategy_id="breakout",
            stats=stats_bad,
        )

        portfolio = self.scorecard.evaluate_portfolio([report_good, report_bad])

        assert portfolio.overall_score > 0
        assert portfolio.best_strategy == "trend_following"
        assert portfolio.worst_strategy == "breakout"
        assert portfolio.passing_strategies + portfolio.failing_strategies == 2

    def test_portfolio_weighted_by_trades(self):
        """Strategy with more trades should have more influence on score."""
        # Strategy with 100 trades and score 80
        stats_big = _make_stats(
            total_trades=100,
            win_rate=0.55,
            profit_factor=2.0,
            sharpe=1.5,
            expectancy=25.0,
            avg_r=0.5,
            avg_management_efficiency=0.7,
        )
        # Strategy with 10 trades and very poor score
        stats_small = _make_stats(
            total_trades=10,
            win_rate=0.10,
            profit_factor=0.3,
            sharpe=-1.0,
            expectancy=-20.0,
            avg_r=-0.5,
            avg_management_efficiency=0.1,
        )

        report_big = self.scorecard.evaluate("big_strat", stats=stats_big)
        report_small = self.scorecard.evaluate("small_strat", stats=stats_small)

        portfolio = self.scorecard.evaluate_portfolio([report_big, report_small])

        # Should be closer to big strategy's score due to weighting
        assert portfolio.overall_score > (report_big.overall_score + report_small.overall_score) / 2 - 5

    def test_portfolio_to_dict(self):
        stats = _make_stats()
        report = self.scorecard.evaluate("test", stats=stats)
        portfolio = self.scorecard.evaluate_portfolio([report])

        d = portfolio.to_dict()
        assert "overall_score" in d
        assert "overall_grade" in d
        assert "strategies" in d
        assert "recommendations" in d

    def test_portfolio_with_backtest(self):
        stats = _make_stats()
        report = self.scorecard.evaluate("test", stats=stats)
        backtest = _make_backtest_result()

        portfolio = self.scorecard.evaluate_portfolio(
            [report], portfolio_backtest=backtest
        )
        assert len(portfolio.portfolio_metrics) > 0


# ================================================================== #
# MetricGrade tests                                                   #
# ================================================================== #


class TestMetricGrade:
    def test_auto_status_passing(self):
        m = MetricGrade(
            name="test", display_name="Test", value=1.0,
            score=80.0, grade=Grade.B_PLUS, target="> 0.5",
        )
        assert m.status == "passing"

    def test_auto_status_warning(self):
        m = MetricGrade(
            name="test", display_name="Test", value=0.3,
            score=55.0, grade=Grade.C_MINUS, target="> 0.5",
        )
        assert m.status == "warning"

    def test_auto_status_failing(self):
        m = MetricGrade(
            name="test", display_name="Test", value=0.1,
            score=30.0, grade=Grade.F, target="> 0.5",
        )
        assert m.status == "failing"

    def test_explicit_status_overrides(self):
        m = MetricGrade(
            name="test", display_name="Test", value=1.0,
            score=80.0, grade=Grade.B_PLUS, target="> 0.5",
            status="custom",
        )
        assert m.status == "custom"


# ================================================================== #
# Grade enum tests                                                    #
# ================================================================== #


class TestGradeEnum:
    def test_grade_string_values(self):
        assert Grade.A_PLUS.value == "A+"
        assert Grade.A.value == "A"
        assert Grade.F.value == "F"

    def test_grade_is_str_enum(self):
        assert isinstance(Grade.A_PLUS, str)
        assert Grade.A_PLUS == "A+"
