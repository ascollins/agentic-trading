"""Optimization report generation and TOML config writing.

Formats optimization results as structured console output and generates
updated TOML configuration files with optimal parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentic_trading.strategies.research.walk_forward import WalkForwardReport

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result of a single parameter combination backtest."""

    params: dict[str, Any]
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_fees: float = 0.0


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
        print(f"\n--- BEST PARAMETERS ---")
        for k, v in sorted(report.best_params.items()):
            print(f"  {k:35s}: {v}")

        print(f"\n--- BEST PERFORMANCE ---")
        print(f"  Sharpe Ratio:    {report.best_sharpe:.4f}")
        print(f"  Total Return:    {report.best_return:+.2f}%")

    # Top-5 results
    sorted_results = sorted(
        report.results, key=lambda r: r.sharpe_ratio, reverse=True
    )
    top_n = sorted_results[:5]

    if top_n:
        print(f"\n--- TOP 5 PARAMETER SETS ---")
        print(
            f"  {'Rank':<5} {'Sharpe':>8} {'Return':>9} {'MaxDD':>9} "
            f"{'Trades':>7} {'WinRate':>8} {'PF':>6}"
        )
        print(f"  {'-' * 55}")
        for i, r in enumerate(top_n, 1):
            print(
                f"  {i:<5} {r.sharpe_ratio:>8.4f} {r.total_return:>+8.2f}% "
                f"{r.max_drawdown:>+8.2f}% {r.total_trades:>7} "
                f"{r.win_rate:>7.1f}% {r.profit_factor:>5.2f}"
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
        print(f"\n--- WALK-FORWARD VALIDATION ---")
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
        print(f"\n  WARNING: Strategy shows signs of overfitting!")
        print(f"  Consider: fewer parameters, longer data, or simpler logic.")

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
        f"[[strategies]]",
        f'strategy_id = "{report.strategy_id}"',
        f"enabled = true",
        f"",
        f"[strategies.params]",
    ]

    for k, v in sorted(report.best_params.items()):
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, float):
            lines.append(f"{k} = {v}")
        elif isinstance(v, int):
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
