"""Trader Agent Quality Scorecard — institutional-grade quality measurement.

Aggregates key performance metrics into a single graded scorecard that
measures whether the trader agent is performing at institutional standards.
Designed for operators who need a quick "go / no-go" assessment and for
automated governance to demote or promote strategies.

Each metric is graded from F (failing) through A+ (exceptional) using
target ranges derived from institutional trading benchmarks:

    Metric               Target Range
    ─────────────────────────────────────────────
    Sharpe Ratio         > 1.0 (1.5+ very good)
    Sortino Ratio        > 1.5
    Calmar Ratio         > 1.0 (2.0+ excellent)
    Profit Factor        > 1.75
    Max Drawdown         < 15-20% (conservative)
    Win Rate             Varies by strategy type
    Avg R-Multiple       > 0.3
    Expectancy           > 0 (positive expected value)
    Management Eff.      > 0.5 (capturing 50%+ of MFE)
    Statistical Edge     p-value < 0.05

The composite score is a weighted average across all metric grades,
producing an overall letter grade and numeric score (0-100).

Usage::

    scorecard = QualityScorecard()
    report = scorecard.evaluate(
        strategy_id="trend_following",
        strategy_type="trend",
        stats=journal.get_strategy_stats("trend_following"),
        backtest_result=result,
        rolling_snapshot=tracker.snapshot("trend_following"),
        edge_result=baseline.evaluate("trend_following"),
        monte_carlo_result=projector.project("trend_following"),
    )
    print(report.overall_grade)    # "B+"
    print(report.overall_score)    # 78.5
    print(report.passing)          # True
    print(report.summary_table())  # Formatted summary table
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ================================================================== #
# Grade system                                                        #
# ================================================================== #

class Grade(str, Enum):
    """Letter grade with numeric score equivalent."""

    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    F = "F"


# Numeric score bands for each grade
_GRADE_SCORES: dict[Grade, tuple[float, float]] = {
    Grade.A_PLUS: (95, 100),
    Grade.A: (90, 94.9),
    Grade.A_MINUS: (85, 89.9),
    Grade.B_PLUS: (80, 84.9),
    Grade.B: (75, 79.9),
    Grade.B_MINUS: (70, 74.9),
    Grade.C_PLUS: (65, 69.9),
    Grade.C: (60, 64.9),
    Grade.C_MINUS: (55, 59.9),
    Grade.D: (40, 54.9),
    Grade.F: (0, 39.9),
}

# Minimum score for each grade
_GRADE_THRESHOLDS: list[tuple[float, Grade]] = [
    (95.0, Grade.A_PLUS),
    (90.0, Grade.A),
    (85.0, Grade.A_MINUS),
    (80.0, Grade.B_PLUS),
    (75.0, Grade.B),
    (70.0, Grade.B_MINUS),
    (65.0, Grade.C_PLUS),
    (60.0, Grade.C),
    (55.0, Grade.C_MINUS),
    (40.0, Grade.D),
    (0.0, Grade.F),
]


def score_to_grade(score: float) -> Grade:
    """Convert a numeric score (0-100) to a letter grade."""
    score = max(0.0, min(100.0, score))
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return Grade.F


# ================================================================== #
# Strategy type classification                                        #
# ================================================================== #

class StrategyType(str, Enum):
    """Strategy archetype — determines win rate expectations."""

    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    STATISTICAL = "statistical"
    HYBRID = "hybrid"


# Map strategy_id patterns to types for automatic classification
_STRATEGY_TYPE_PATTERNS: dict[str, StrategyType] = {
    "trend_following": StrategyType.TREND,
    "multi_tf_ma": StrategyType.TREND,
    "mean_reversion": StrategyType.MEAN_REVERSION,
    "mean_reversion_enhanced": StrategyType.MEAN_REVERSION,
    "bb_squeeze": StrategyType.MEAN_REVERSION,
    "breakout": StrategyType.BREAKOUT,
    "supply_demand": StrategyType.BREAKOUT,
    "rsi_divergence": StrategyType.MOMENTUM,
    "stochastic_macd": StrategyType.MOMENTUM,
    "obv_divergence": StrategyType.MOMENTUM,
    "fibonacci_confluence": StrategyType.STATISTICAL,
    "funding_arb": StrategyType.STATISTICAL,
}

# Win rate targets by strategy type
_WIN_RATE_TARGETS: dict[StrategyType, dict[str, float]] = {
    StrategyType.TREND: {
        "excellent": 0.45,  # Trend following typically has lower win rate
        "good": 0.38,
        "acceptable": 0.30,
        "poor": 0.20,
    },
    StrategyType.MEAN_REVERSION: {
        "excellent": 0.70,
        "good": 0.60,
        "acceptable": 0.50,
        "poor": 0.40,
    },
    StrategyType.BREAKOUT: {
        "excellent": 0.50,
        "good": 0.40,
        "acceptable": 0.30,
        "poor": 0.20,
    },
    StrategyType.MOMENTUM: {
        "excellent": 0.55,
        "good": 0.45,
        "acceptable": 0.35,
        "poor": 0.25,
    },
    StrategyType.STATISTICAL: {
        "excellent": 0.60,
        "good": 0.50,
        "acceptable": 0.40,
        "poor": 0.30,
    },
    StrategyType.HYBRID: {
        "excellent": 0.55,
        "good": 0.45,
        "acceptable": 0.35,
        "poor": 0.25,
    },
}


def classify_strategy(strategy_id: str) -> StrategyType:
    """Auto-classify strategy type from its ID."""
    return _STRATEGY_TYPE_PATTERNS.get(strategy_id, StrategyType.HYBRID)


# ================================================================== #
# Metric grading functions                                            #
# ================================================================== #

@dataclass
class MetricGrade:
    """A single graded metric with score, grade, and context."""

    name: str
    display_name: str
    value: float
    score: float  # 0-100
    grade: Grade
    target: str  # Human-readable target description
    weight: float = 1.0
    status: str = ""  # "passing", "warning", "failing"
    detail: str = ""  # Additional context

    def __post_init__(self) -> None:
        if not self.status:
            if self.score >= 70:
                self.status = "passing"
            elif self.score >= 50:
                self.status = "warning"
            else:
                self.status = "failing"


def _grade_sharpe(sharpe: float) -> MetricGrade:
    """Grade Sharpe ratio. Target: > 1.0, 1.5+ very good."""
    if sharpe >= 2.5:
        score = 100.0
    elif sharpe >= 2.0:
        score = 95.0
    elif sharpe >= 1.5:
        score = 85.0
    elif sharpe >= 1.0:
        score = 75.0
    elif sharpe >= 0.5:
        score = 60.0
    elif sharpe >= 0.0:
        score = 40.0
    else:
        score = max(0.0, 20.0 + sharpe * 20.0)  # Negative Sharpe

    return MetricGrade(
        name="sharpe_ratio",
        display_name="Sharpe Ratio",
        value=round(sharpe, 4),
        score=score,
        grade=score_to_grade(score),
        target="> 1.0 (1.5+ very good)",
        weight=1.5,  # High weight — most important risk-adjusted metric
    )


def _grade_sortino(sortino: float) -> MetricGrade:
    """Grade Sortino ratio. Target: > 1.5."""
    if sortino >= 3.0:
        score = 100.0
    elif sortino >= 2.5:
        score = 95.0
    elif sortino >= 2.0:
        score = 85.0
    elif sortino >= 1.5:
        score = 75.0
    elif sortino >= 1.0:
        score = 65.0
    elif sortino >= 0.5:
        score = 50.0
    elif sortino >= 0.0:
        score = 35.0
    else:
        score = max(0.0, 15.0 + sortino * 15.0)

    return MetricGrade(
        name="sortino_ratio",
        display_name="Sortino Ratio",
        value=round(sortino, 4),
        score=score,
        grade=score_to_grade(score),
        target="> 1.5",
        weight=1.2,
    )


def _grade_calmar(calmar: float) -> MetricGrade:
    """Grade Calmar ratio. Target: > 1.0, 2.0+ excellent."""
    if calmar >= 3.0:
        score = 100.0
    elif calmar >= 2.0:
        score = 90.0
    elif calmar >= 1.5:
        score = 80.0
    elif calmar >= 1.0:
        score = 70.0
    elif calmar >= 0.5:
        score = 55.0
    elif calmar >= 0.0:
        score = 35.0
    else:
        score = max(0.0, 15.0 + calmar * 15.0)

    return MetricGrade(
        name="calmar_ratio",
        display_name="Calmar Ratio",
        value=round(calmar, 4),
        score=score,
        grade=score_to_grade(score),
        target="> 1.0 (2.0+ excellent)",
        weight=1.0,
    )


def _grade_profit_factor(pf: float) -> MetricGrade:
    """Grade profit factor. Target: > 1.75."""
    if pf == float("inf"):
        score = 100.0
    elif pf >= 3.0:
        score = 100.0
    elif pf >= 2.5:
        score = 95.0
    elif pf >= 2.0:
        score = 85.0
    elif pf >= 1.75:
        score = 75.0
    elif pf >= 1.5:
        score = 65.0
    elif pf >= 1.25:
        score = 55.0
    elif pf >= 1.0:
        score = 40.0
    else:
        score = max(0.0, pf * 40.0)

    return MetricGrade(
        name="profit_factor",
        display_name="Profit Factor",
        value=round(pf, 4) if pf != float("inf") else 999.0,
        score=score,
        grade=score_to_grade(score),
        target="> 1.75",
        weight=1.3,
    )


def _grade_max_drawdown(dd: float) -> MetricGrade:
    """Grade max drawdown. Target: < 15-20% (conservative).

    Note: dd should be a negative value (e.g., -0.15 for 15% drawdown).
    """
    abs_dd = abs(dd)
    if abs_dd <= 0.05:
        score = 100.0
    elif abs_dd <= 0.10:
        score = 90.0
    elif abs_dd <= 0.15:
        score = 75.0
    elif abs_dd <= 0.20:
        score = 60.0
    elif abs_dd <= 0.30:
        score = 40.0
    elif abs_dd <= 0.50:
        score = 20.0
    else:
        score = 0.0

    return MetricGrade(
        name="max_drawdown",
        display_name="Max Drawdown",
        value=round(abs_dd * 100, 2),  # Display as positive percentage
        score=score,
        grade=score_to_grade(score),
        target="< 15-20%",
        weight=1.5,  # High weight — capital preservation is critical
        detail=f"{abs_dd:.1%} peak-to-trough decline",
    )


def _grade_win_rate(
    win_rate: float,
    strategy_type: StrategyType,
) -> MetricGrade:
    """Grade win rate relative to strategy type expectations."""
    targets = _WIN_RATE_TARGETS[strategy_type]

    if win_rate >= targets["excellent"]:
        score = 90.0
    elif win_rate >= targets["good"]:
        score = 75.0
    elif win_rate >= targets["acceptable"]:
        score = 60.0
    elif win_rate >= targets["poor"]:
        score = 40.0
    else:
        score = max(0.0, win_rate * 100.0)

    target_str = (
        f"> {targets['good']:.0%} for {strategy_type.value} "
        f"(excellent: {targets['excellent']:.0%})"
    )

    return MetricGrade(
        name="win_rate",
        display_name="Win Rate",
        value=round(win_rate * 100, 1),  # Display as percentage
        score=score,
        grade=score_to_grade(score),
        target=target_str,
        weight=0.8,
        detail=f"{strategy_type.value} strategy — win rate expectations adjusted",
    )


def _grade_avg_r(avg_r: float) -> MetricGrade:
    """Grade average R-multiple. Target: > 0.3."""
    if avg_r >= 1.0:
        score = 100.0
    elif avg_r >= 0.7:
        score = 90.0
    elif avg_r >= 0.5:
        score = 80.0
    elif avg_r >= 0.3:
        score = 70.0
    elif avg_r >= 0.1:
        score = 55.0
    elif avg_r >= 0.0:
        score = 40.0
    else:
        score = max(0.0, 20.0 + avg_r * 40.0)

    return MetricGrade(
        name="avg_r_multiple",
        display_name="Avg R-Multiple",
        value=round(avg_r, 4),
        score=score,
        grade=score_to_grade(score),
        target="> 0.3",
        weight=1.0,
    )


def _grade_expectancy(expectancy: float) -> MetricGrade:
    """Grade expectancy (avg P&L per trade). Target: > 0."""
    if expectancy >= 100:
        score = 95.0
    elif expectancy >= 50:
        score = 85.0
    elif expectancy >= 20:
        score = 75.0
    elif expectancy >= 5:
        score = 65.0
    elif expectancy > 0:
        score = 55.0
    elif expectancy == 0:
        score = 40.0
    else:
        # Negative expectancy — scale down
        score = max(0.0, 40.0 + expectancy)  # $-40 = score 0

    return MetricGrade(
        name="expectancy",
        display_name="Expectancy",
        value=round(expectancy, 2),
        score=score,
        grade=score_to_grade(score),
        target="> $0 per trade",
        weight=1.2,
    )


def _grade_management_efficiency(eff: float) -> MetricGrade:
    """Grade management efficiency (actual P&L / MFE). Target: > 0.5."""
    if eff >= 0.80:
        score = 95.0
    elif eff >= 0.70:
        score = 85.0
    elif eff >= 0.60:
        score = 75.0
    elif eff >= 0.50:
        score = 65.0
    elif eff >= 0.40:
        score = 50.0
    elif eff >= 0.30:
        score = 35.0
    else:
        score = max(0.0, eff * 100.0)

    return MetricGrade(
        name="management_efficiency",
        display_name="Trade Management",
        value=round(eff * 100, 1),  # Display as percentage
        score=score,
        grade=score_to_grade(score),
        target="> 50% of max profit captured",
        weight=0.8,
    )


def _grade_statistical_edge(
    has_edge: bool,
    p_value: float,
    sample_adequate: bool,
) -> MetricGrade:
    """Grade statistical edge significance."""
    if has_edge and p_value < 0.01:
        score = 95.0
        detail = "Strong statistical evidence of genuine edge"
    elif has_edge and p_value < 0.05:
        score = 80.0
        detail = "Significant statistical edge detected"
    elif has_edge:
        score = 70.0
        detail = "Marginal statistical edge"
    elif not sample_adequate:
        score = 50.0
        detail = "Insufficient sample size to determine edge"
    elif p_value < 0.10:
        score = 45.0
        detail = "Near-significant — more trades needed"
    else:
        score = 25.0
        detail = "No statistical edge detected — could be luck"

    return MetricGrade(
        name="statistical_edge",
        display_name="Skill vs Luck",
        value=round(p_value, 4),
        score=score,
        grade=score_to_grade(score),
        target="p-value < 0.05",
        weight=0.7,
        detail=detail,
    )


def _grade_ruin_probability(ruin_prob: float) -> MetricGrade:
    """Grade probability of ruin from Monte Carlo. Target: < 5%."""
    if ruin_prob <= 0.01:
        score = 100.0
    elif ruin_prob <= 0.03:
        score = 90.0
    elif ruin_prob <= 0.05:
        score = 75.0
    elif ruin_prob <= 0.10:
        score = 60.0
    elif ruin_prob <= 0.20:
        score = 40.0
    elif ruin_prob <= 0.40:
        score = 20.0
    else:
        score = 0.0

    return MetricGrade(
        name="ruin_probability",
        display_name="Risk of Ruin",
        value=round(ruin_prob * 100, 2),  # Display as percentage
        score=score,
        grade=score_to_grade(score),
        target="< 5%",
        weight=1.3,
        detail=f"{ruin_prob:.1%} probability of 50%+ drawdown over next 500 trades",
    )


# ================================================================== #
# Quality Report                                                      #
# ================================================================== #

@dataclass
class QualityReport:
    """Complete quality assessment for one strategy or the portfolio."""

    strategy_id: str
    strategy_type: StrategyType
    total_trades: int
    metrics: list[MetricGrade] = field(default_factory=list)
    overall_score: float = 0.0
    overall_grade: Grade = Grade.F
    passing: bool = False
    sufficient_data: bool = False
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON/API use."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type.value,
            "total_trades": self.total_trades,
            "overall_score": round(self.overall_score, 1),
            "overall_grade": self.overall_grade.value,
            "passing": self.passing,
            "sufficient_data": self.sufficient_data,
            "metrics": [
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "value": m.value,
                    "score": round(m.score, 1),
                    "grade": m.grade.value,
                    "target": m.target,
                    "status": m.status,
                    "detail": m.detail,
                }
                for m in self.metrics
            ],
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
        }

    def summary_table(self) -> str:
        """Format as a human-readable summary table."""
        lines = [
            f"╔══════════════════════════════════════════════════════════════╗",
            f"║  TRADER QUALITY SCORECARD: {self.strategy_id:<33}║",
            f"║  Type: {self.strategy_type.value:<20} Trades: {self.total_trades:<18}║",
            f"╠══════════════════════════════════════════════════════════════╣",
            f"║  OVERALL: {self.overall_grade.value:<5} ({self.overall_score:.1f}/100)  "
            f"{'✓ PASSING' if self.passing else '✗ FAILING':<25}║",
            f"╠══════════════════════════════════════════════════════════════╣",
            f"║  {'Metric':<22} {'Value':>10} {'Grade':>6} {'Score':>6} {'Target':<10}║",
            f"║  {'─' * 56}║",
        ]

        for m in self.metrics:
            val_str = self._format_value(m)
            lines.append(
                f"║  {m.display_name:<22} {val_str:>10} {m.grade.value:>6} "
                f"{m.score:>5.0f}  {m.target:<10}║"
            )

        lines.append(f"╠══════════════════════════════════════════════════════════════╣")

        if self.strengths:
            lines.append(f"║  STRENGTHS:{'':49}║")
            for s in self.strengths[:3]:
                lines.append(f"║    ✓ {s:<54}║")

        if self.weaknesses:
            lines.append(f"║  WEAKNESSES:{'':48}║")
            for w in self.weaknesses[:3]:
                lines.append(f"║    ✗ {w:<54}║")

        if self.recommendations:
            lines.append(f"║  RECOMMENDATIONS:{'':42}║")
            for r in self.recommendations[:3]:
                lines.append(f"║    → {r:<54}║")

        lines.append(f"╚══════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    @staticmethod
    def _format_value(m: MetricGrade) -> str:
        """Format a metric value for display."""
        if m.name in ("win_rate", "max_drawdown", "management_efficiency",
                       "ruin_probability"):
            return f"{m.value:.1f}%"
        elif m.name == "expectancy":
            return f"${m.value:.2f}"
        elif m.name == "statistical_edge":
            return f"p={m.value:.3f}"
        else:
            return f"{m.value:.2f}"


# ================================================================== #
# Portfolio Quality Report                                            #
# ================================================================== #

@dataclass
class PortfolioQualityReport:
    """Aggregate quality report across all strategies."""

    strategy_reports: list[QualityReport] = field(default_factory=list)
    overall_score: float = 0.0
    overall_grade: Grade = Grade.F
    passing_strategies: int = 0
    failing_strategies: int = 0
    best_strategy: str = ""
    worst_strategy: str = ""
    portfolio_metrics: list[MetricGrade] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "overall_score": round(self.overall_score, 1),
            "overall_grade": self.overall_grade.value,
            "passing_strategies": self.passing_strategies,
            "failing_strategies": self.failing_strategies,
            "total_strategies": len(self.strategy_reports),
            "best_strategy": self.best_strategy,
            "worst_strategy": self.worst_strategy,
            "portfolio_metrics": [
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "value": m.value,
                    "score": round(m.score, 1),
                    "grade": m.grade.value,
                }
                for m in self.portfolio_metrics
            ],
            "strategies": [r.to_dict() for r in self.strategy_reports],
            "recommendations": self.recommendations,
        }


# ================================================================== #
# Quality Scorecard                                                   #
# ================================================================== #

# Minimum trades required for a reliable quality assessment
MIN_TRADES_FOR_ASSESSMENT = 15


class QualityScorecard:
    """Computes institutional-grade quality scores for trading strategies.

    Aggregates metrics from the journal, backtest results, rolling tracker,
    statistical edge analysis, and Monte Carlo simulations into a single
    graded scorecard per strategy.

    Parameters
    ----------
    min_trades : int
        Minimum trades before generating a graded report.
        Below this threshold, the report will indicate insufficient data.
    passing_score : float
        Minimum overall score (0-100) to be considered "passing".
        Default 60.0 (C grade).
    """

    def __init__(
        self,
        *,
        min_trades: int = MIN_TRADES_FOR_ASSESSMENT,
        passing_score: float = 60.0,
    ) -> None:
        self._min_trades = max(5, min_trades)
        self._passing_score = passing_score

    def evaluate(
        self,
        strategy_id: str,
        strategy_type: StrategyType | str | None = None,
        stats: dict[str, Any] | None = None,
        backtest_result: Any | None = None,
        rolling_snapshot: dict[str, Any] | None = None,
        edge_result: dict[str, Any] | None = None,
        monte_carlo_result: dict[str, Any] | None = None,
    ) -> QualityReport:
        """Evaluate a strategy and produce a graded quality report.

        Accepts data from multiple sources — use whatever is available.
        At minimum, either ``stats`` or ``backtest_result`` must be provided.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier.
        strategy_type : StrategyType | str | None
            Strategy archetype for win rate calibration.
            Auto-detected from strategy_id if not provided.
        stats : dict
            Output from ``TradeJournal.get_strategy_stats()``.
        backtest_result : BacktestResult
            Output from ``BacktestEngine.run()``.
        rolling_snapshot : dict
            Output from ``RollingTracker.snapshot()``.
        edge_result : dict
            Output from ``CoinFlipBaseline.evaluate()``.
        monte_carlo_result : dict
            Output from ``MonteCarloProjector.project()``.
        """
        # Resolve strategy type
        if isinstance(strategy_type, str):
            try:
                stype = StrategyType(strategy_type)
            except ValueError:
                stype = classify_strategy(strategy_id)
        elif strategy_type is not None:
            stype = strategy_type
        else:
            stype = classify_strategy(strategy_id)

        # Collect metrics from available data sources
        metrics: list[MetricGrade] = []
        total_trades = 0

        # --- Extract values from all available sources ---
        sharpe = 0.0
        sortino = 0.0
        calmar = 0.0
        profit_factor = 0.0
        max_drawdown = 0.0
        win_rate = 0.0
        avg_r = 0.0
        expectancy = 0.0
        mgmt_eff = 0.0

        # Priority: backtest_result > rolling_snapshot > stats
        if backtest_result is not None:
            sharpe = getattr(backtest_result, "sharpe_ratio", 0.0)
            sortino = getattr(backtest_result, "sortino_ratio", 0.0)
            calmar = getattr(backtest_result, "calmar_ratio", 0.0)
            profit_factor = getattr(backtest_result, "profit_factor", 0.0)
            max_drawdown = getattr(backtest_result, "max_drawdown", 0.0)
            win_rate = getattr(backtest_result, "win_rate", 0.0)
            total_trades = getattr(backtest_result, "total_trades", 0)
            avg_trade_return = getattr(backtest_result, "avg_trade_return", 0.0)
            expectancy = avg_trade_return * 100  # Convert to bps-like scale

        if stats:
            total_trades = max(total_trades, stats.get("total_trades", 0))
            if not backtest_result:
                win_rate = stats.get("win_rate", win_rate)
                profit_factor = stats.get("profit_factor", profit_factor)
                expectancy = stats.get("expectancy", expectancy)
            avg_r = stats.get("avg_r", avg_r)
            mgmt_eff = stats.get("avg_management_efficiency", mgmt_eff)
            # Stats-level Sharpe if backtest didn't provide one
            if sharpe == 0.0:
                sharpe = stats.get("sharpe", 0.0)

        if rolling_snapshot:
            total_trades = max(total_trades, rolling_snapshot.get("trade_count", 0))
            # Use rolling if more current than stats
            if not stats and not backtest_result:
                win_rate = rolling_snapshot.get("win_rate", win_rate)
                profit_factor = rolling_snapshot.get("profit_factor", profit_factor)
                expectancy = rolling_snapshot.get("expectancy", expectancy)
            if avg_r == 0.0:
                avg_r = rolling_snapshot.get("avg_r", avg_r)
            if mgmt_eff == 0.0:
                mgmt_eff = rolling_snapshot.get("avg_management_efficiency", mgmt_eff)
            if sortino == 0.0:
                sortino = rolling_snapshot.get("sortino", 0.0)
            if sharpe == 0.0:
                sharpe = rolling_snapshot.get("sharpe", 0.0)

        # Check data sufficiency
        sufficient_data = total_trades >= self._min_trades

        # --- Grade each metric ---
        metrics.append(_grade_sharpe(sharpe))
        metrics.append(_grade_sortino(sortino))
        metrics.append(_grade_calmar(calmar))
        metrics.append(_grade_profit_factor(profit_factor))
        metrics.append(_grade_max_drawdown(max_drawdown))
        metrics.append(_grade_win_rate(win_rate, stype))
        metrics.append(_grade_avg_r(avg_r))
        metrics.append(_grade_expectancy(expectancy))
        metrics.append(_grade_management_efficiency(mgmt_eff))

        # Statistical edge (if available)
        if edge_result and edge_result.get("error") is None:
            metrics.append(_grade_statistical_edge(
                has_edge=edge_result.get("has_edge", False),
                p_value=edge_result.get("p_value_bootstrap", 1.0),
                sample_adequate=edge_result.get("sample_adequate", False),
            ))

        # Ruin probability (if available)
        if monte_carlo_result and monte_carlo_result.get("error") is None:
            metrics.append(_grade_ruin_probability(
                ruin_prob=monte_carlo_result.get("ruin_probability", 1.0),
            ))

        # --- Compute overall weighted score ---
        total_weight = sum(m.weight for m in metrics)
        if total_weight > 0:
            overall_score = sum(m.score * m.weight for m in metrics) / total_weight
        else:
            overall_score = 0.0

        overall_grade = score_to_grade(overall_score)
        passing = overall_score >= self._passing_score and sufficient_data

        # --- Generate insights ---
        strengths = []
        weaknesses = []
        recommendations = []

        # Sort metrics by score
        sorted_metrics = sorted(metrics, key=lambda m: m.score, reverse=True)

        for m in sorted_metrics[:3]:
            if m.score >= 75:
                strengths.append(f"{m.display_name}: {m.grade.value} ({m.value})")

        for m in sorted_metrics[-3:]:
            if m.score < 60:
                weaknesses.append(f"{m.display_name}: {m.grade.value} ({m.value})")

        # Generate recommendations
        if not sufficient_data:
            recommendations.append(
                f"Need {self._min_trades - total_trades} more trades for "
                f"reliable assessment"
            )

        for m in metrics:
            if m.name == "max_drawdown" and m.score < 60:
                recommendations.append(
                    "Reduce position sizing or tighten stops to lower drawdown"
                )
            elif m.name == "profit_factor" and m.score < 60:
                recommendations.append(
                    "Improve trade selection or cut losers faster to raise "
                    "profit factor"
                )
            elif m.name == "management_efficiency" and m.score < 60:
                recommendations.append(
                    "Consider trailing stops or partial take-profits to "
                    "capture more of each move"
                )
            elif m.name == "statistical_edge" and m.score < 60:
                recommendations.append(
                    "Edge not statistically proven — more sample data needed "
                    "or strategy may be random"
                )
            elif m.name == "win_rate" and m.score < 50:
                recommendations.append(
                    f"Win rate below expectations for {stype.value} strategy — "
                    f"review entry criteria"
                )

        # Cap recommendations
        recommendations = recommendations[:5]

        report = QualityReport(
            strategy_id=strategy_id,
            strategy_type=stype,
            total_trades=total_trades,
            metrics=metrics,
            overall_score=round(overall_score, 1),
            overall_grade=overall_grade,
            passing=passing,
            sufficient_data=sufficient_data,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )

        logger.info(
            "Quality scorecard: strategy=%s grade=%s score=%.1f passing=%s trades=%d",
            strategy_id,
            overall_grade.value,
            overall_score,
            passing,
            total_trades,
        )

        return report

    def evaluate_portfolio(
        self,
        strategy_reports: list[QualityReport],
        portfolio_backtest: Any | None = None,
    ) -> PortfolioQualityReport:
        """Aggregate individual strategy scores into a portfolio assessment.

        Parameters
        ----------
        strategy_reports : list[QualityReport]
            Individual strategy quality reports.
        portfolio_backtest : BacktestResult | None
            Optional aggregate backtest result for the whole portfolio.
        """
        if not strategy_reports:
            return PortfolioQualityReport()

        passing = [r for r in strategy_reports if r.passing]
        failing = [r for r in strategy_reports if not r.passing]

        # Weighted average by trade count
        total_trades = sum(r.total_trades for r in strategy_reports)
        if total_trades > 0:
            weighted_score = sum(
                r.overall_score * r.total_trades for r in strategy_reports
            ) / total_trades
        else:
            weighted_score = sum(
                r.overall_score for r in strategy_reports
            ) / len(strategy_reports)

        # Best/worst
        sorted_reports = sorted(
            strategy_reports, key=lambda r: r.overall_score, reverse=True
        )
        best = sorted_reports[0].strategy_id if sorted_reports else ""
        worst = sorted_reports[-1].strategy_id if sorted_reports else ""

        # Portfolio-level metrics from backtest (if available)
        portfolio_metrics: list[MetricGrade] = []
        if portfolio_backtest is not None:
            portfolio_metrics = [
                _grade_sharpe(getattr(portfolio_backtest, "sharpe_ratio", 0.0)),
                _grade_sortino(getattr(portfolio_backtest, "sortino_ratio", 0.0)),
                _grade_calmar(getattr(portfolio_backtest, "calmar_ratio", 0.0)),
                _grade_max_drawdown(getattr(portfolio_backtest, "max_drawdown", 0.0)),
                _grade_profit_factor(getattr(portfolio_backtest, "profit_factor", 0.0)),
            ]

        # Portfolio recommendations
        recommendations = []
        if len(failing) > len(passing):
            recommendations.append(
                f"{len(failing)}/{len(strategy_reports)} strategies failing — "
                f"consider disabling weakest performers"
            )
        if worst and sorted_reports[-1].overall_score < 40:
            recommendations.append(
                f"'{worst}' scored {sorted_reports[-1].overall_score:.0f}/100 — "
                f"consider removing from portfolio"
            )
        if len(passing) > 0 and best:
            recommendations.append(
                f"'{best}' is the top performer at "
                f"{sorted_reports[0].overall_score:.0f}/100 — "
                f"consider increasing allocation"
            )

        return PortfolioQualityReport(
            strategy_reports=strategy_reports,
            overall_score=round(weighted_score, 1),
            overall_grade=score_to_grade(weighted_score),
            passing_strategies=len(passing),
            failing_strategies=len(failing),
            best_strategy=best,
            worst_strategy=worst,
            portfolio_metrics=portfolio_metrics,
            recommendations=recommendations,
        )
