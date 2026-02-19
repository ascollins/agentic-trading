"""Backward-compatibility shim — see ``agentic_trading.reconciliation.journal.quality_scorecard``.

Will be removed in PR 16.
"""

from agentic_trading.reconciliation.journal.quality_scorecard import *  # noqa: F401, F403
from agentic_trading.reconciliation.journal.quality_scorecard import (  # noqa: F811
    QualityScorecard,
    QualityReport,
    PortfolioQualityReport,
    MetricGrade,
    Grade,
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

__all__ = [
    "QualityScorecard",
    "QualityReport",
    "PortfolioQualityReport",
    "MetricGrade",
    "Grade",
    "StrategyType",
    "classify_strategy",
    "score_to_grade",
]
