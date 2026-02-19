"""Optimization report generation and TOML config writing.

Formats optimization results as structured console output and generates
updated TOML configuration files with optimal parameters.

Includes institutional-grade composite scoring (Sortino, Calmar, max DD,
profit factor, expectancy, Sharpe) and actionable strategy recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentic_trading.core.enums import OptimizationRecommendation
from agentic_trading.strategies.research.walk_forward import WalkForwardReport

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result of a single parameter combination backtest."""

    params: dict[str, Any]
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    annualized_return: float = 0.0
    total_fees: float = 0.0


@dataclass
class CompositeScoreWeights:
    """Configurable weights for multi-objective scoring.

    Aligned with institutional priorities:
    - Sortino preferred over Sharpe (doesn't penalise upside vol)
    - Calmar is what allocators check first (return / max drawdown)
    - Max drawdown is the most scrutinised single number
    - Profit factor more meaningful than win rate alone
    - Expectancy is the meta-metric: positive expectancy over
      large samples is the entire game
    - Sharpe is industry standard but penalises upside vol
    """

    sortino_weight: float = 0.25
    calmar_weight: float = 0.20
    max_drawdown_penalty: float = 0.20
    profit_factor_weight: float = 0.15
    expectancy_weight: float = 0.10
    sharpe_weight: float = 0.10

    def compute(self, result: StrategyResult) -> float:
        """Compute weighted composite score from a StrategyResult.

        Higher is better.  Max drawdown is a penalty (subtracted).
        Profit factor is capped at 5.0 to prevent distortion from
        low-trade-count strategies.  Expectancy is computed from
        win_rate, avg_win, and avg_loss.
        """
        capped_pf = min(result.profit_factor, 5.0)

        # Expectancy = (win% × avg_win) - (loss% × avg_loss)
        loss_rate = 1.0 - result.win_rate
        expectancy = (
            result.win_rate * result.avg_win
            - loss_rate * abs(result.avg_loss)
        )
        # Normalise expectancy to roughly same scale as other metrics.
        # Divide by avg_loss magnitude so it becomes an R-multiple.
        if abs(result.avg_loss) > 0:
            expectancy_r = expectancy / abs(result.avg_loss)
        else:
            expectancy_r = expectancy  # fallback if no losses

        score = (
            result.sortino_ratio * self.sortino_weight
            + result.calmar_ratio * self.calmar_weight
            - abs(result.max_drawdown) * self.max_drawdown_penalty * 10
            + capped_pf * self.profit_factor_weight
            + expectancy_r * self.expectancy_weight
            + result.sharpe_ratio * self.sharpe_weight
        )
        return round(score, 4)


@dataclass
class StrategyRecommendation:
    """Actionable recommendation for a single strategy."""

    strategy_id: str
    recommendation: OptimizationRecommendation
    current_params: dict[str, Any] = field(default_factory=dict)
    optimized_params: dict[str, Any] = field(default_factory=dict)
    current_score: float = 0.0
    optimized_score: float = 0.0
    improvement_pct: float = 0.0
    rationale: str = ""
    is_overfit: bool = False
    walk_forward_passed: bool = False
    metrics_current: dict[str, float] = field(default_factory=dict)
    metrics_optimized: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationCycleReport:
    """Full cycle report with per-strategy recommendations."""

    run_number: int
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    recommendations: list[StrategyRecommendation] = field(
        default_factory=list
    )
    duration_seconds: float = 0.0
    strategies_optimized: int = 0
    strategies_skipped: int = 0
    strategies_failed: int = 0


@dataclass
class OptimizationReport:
    """Complete optimization report for a single strategy."""

    strategy_id: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    results: list[StrategyResult] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_sharpe: float = 0.0
    best_return: float = 0.0
    best_composite_score: float = 0.0
    walk_forward: WalkForwardReport | None = None
    is_overfit: bool = False
    samples_tested: int = 0
    data_period: str = ""


def print_summary(report: OptimizationReport) -> None:
    """Print a formatted optimization summary to console."""
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION REPORT: {report.strategy_id}")
    print(f"{'=' * 70}")
    print(f"  Timestamp:       {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Data Period:     {report.data_period}")
    print(f"  Samples Tested:  {report.samples_tested}")

    if report.best_params:
        print("\n--- BEST PARAMETERS ---")
        for k, v in sorted(report.best_params.items()):
            print(f"  {k:35s}: {v}")

        print("\n--- BEST PERFORMANCE ---")
        print(f"  Composite Score: {report.best_composite_score:.4f}")
        print(f"  Sharpe Ratio:    {report.best_sharpe:.4f}")
        print(f"  Total Return:    {report.best_return:+.2f}%")

    # Top-5 results
    sorted_results = sorted(
        report.results, key=lambda r: r.sharpe_ratio, reverse=True
    )
    top_n = sorted_results[:5]

    if top_n:
        print("\n--- TOP 5 PARAMETER SETS ---")
        print(
            f"  {'Rank':<5} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} "
            f"{'Return':>9} {'MaxDD':>9} {'PF':>6}"
        )
        print(f"  {'-' * 60}")
        for i, r in enumerate(top_n, 1):
            print(
                f"  {i:<5} {r.sharpe_ratio:>8.4f} {r.sortino_ratio:>8.4f} "
                f"{r.calmar_ratio:>8.4f} {r.total_return:>+8.2f}% "
                f"{r.max_drawdown:>+8.2f}% {r.profit_factor:>5.2f}"
            )
            # Print differing params from best
            if i > 1 and report.best_params:
                diffs = {
                    k: v
                    for k, v in r.params.items()
                    if report.best_params.get(k) != v
                }
                if diffs:
                    diff_str = ", ".join(f"{k}={v}" for k, v in diffs.items())
                    print(f"        Differs: {diff_str}")

    # Walk-forward results
    if report.walk_forward:
        wf = report.walk_forward
        print("\n--- WALK-FORWARD VALIDATION ---")
        print(f"  Avg Train Sharpe:  {wf.avg_train_sharpe:.4f}")
        print(f"  Avg Test Sharpe:   {wf.avg_test_sharpe:.4f}")
        print(f"  Overfit Score:     {wf.overfit_score:.4f}")
        print(f"  Degradation:       {wf.degradation_pct:.1f}%")
        print(
            f"  Verdict:           "
            f"{'OVERFIT - USE WITH CAUTION' if wf.is_overfit else 'ACCEPTABLE'}"
        )

        for fold in wf.folds:
            print(
                f"    Fold {fold.fold_index}: "
                f"train={fold.train_sharpe:.3f} test={fold.test_sharpe:.3f} "
                f"ratio={fold.overfit_ratio:.2f}"
            )

    if report.is_overfit:
        print("\n  WARNING: Strategy shows signs of overfitting!")
        print("  Consider: fewer parameters, longer data, or simpler logic.")

    print(f"\n{'=' * 70}\n")


def to_toml_params(report: OptimizationReport) -> str:
    """Generate TOML strategy params section from optimization results.

    Returns a string like:
        [strategies.params]
        adx_threshold = 25
        atr_multiplier = 2.5
        ...
    """
    lines = [
        "[[strategies]]",
        f'strategy_id = "{report.strategy_id}"',
        "enabled = true",
        "",
        "[strategies.params]",
    ]

    for k, v in sorted(report.best_params.items()):
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, (float, int)):
            lines.append(f"{k} = {v}")
        else:
            lines.append(f"{k} = {v}")

    return "\n".join(lines)


def to_full_toml(reports: list[OptimizationReport]) -> str:
    """Generate a complete strategies section for all optimized strategies."""
    sections = []
    for report in reports:
        if report.best_params:
            sections.append(to_toml_params(report))

    return "\n\n".join(sections)
