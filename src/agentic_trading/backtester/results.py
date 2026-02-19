"""Backtest result computation and metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StrategyBreakdown:
    """Per-strategy backtest metrics."""

    strategy_id: str = ""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_return: float = 0.0
    total_pnl_pct: float = 0.0  # Sum of all trade returns (%)


@dataclass
class BacktestResult:
    """Comprehensive backtest performance metrics."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_returns: list[float] = field(default_factory=list)

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Cost breakdown
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0

    # Equity curve
    equity_curve: list[float] = field(default_factory=list)
    peak_equity: float = 0.0

    # Metadata
    start_date: str = ""
    end_date: str = ""
    symbols: list[str] = field(default_factory=list)
    strategy_id: str = ""
    config_hash: str = ""
    deterministic_hash: str = ""

    # Per-strategy breakdown
    per_strategy: list[StrategyBreakdown] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return summary dict for logging."""
        return {
            "total_return": f"{self.total_return:.2%}",
            "sharpe": f"{self.sharpe_ratio:.2f}",
            "sortino": f"{self.sortino_ratio:.2f}",
            "max_dd": f"{self.max_drawdown:.2%}",
            "trades": self.total_trades,
            "win_rate": f"{self.win_rate:.1%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "fees": f"{self.total_fees:.2f}",
            "funding": f"{self.total_funding:.2f}",
        }


def compute_metrics(
    equity_curve: list[float],
    trade_returns: list[float],
    fees: float = 0.0,
    slippage: float = 0.0,
    funding: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Compute all backtest metrics from equity curve and trade returns."""
    result = BacktestResult()

    if not equity_curve or len(equity_curve) < 2:
        return result

    eq = np.array(equity_curve)

    # Returns
    daily_returns = np.diff(eq) / eq[:-1]
    result.daily_returns = daily_returns.tolist()
    result.total_return = float((eq[-1] / eq[0]) - 1.0)

    n_days = len(daily_returns)
    if n_days > 0 and (1 + result.total_return) > 0:
        result.annualized_return = float(
            (1 + result.total_return) ** (periods_per_year / n_days) - 1
        )

    # Sharpe
    if n_days > 1:
        mean_ret = float(np.mean(daily_returns))
        std_ret = float(np.std(daily_returns, ddof=1))
        if std_ret > 0:
            result.sharpe_ratio = round(
                mean_ret / std_ret * np.sqrt(periods_per_year), 4
            )

    # Sortino (downside deviation)
    if n_days > 1:
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0:
            downside_std = float(np.std(downside, ddof=1))
            if downside_std > 0:
                result.sortino_ratio = round(
                    float(np.mean(daily_returns)) / downside_std * np.sqrt(periods_per_year),
                    4,
                )

    # Drawdown
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    result.max_drawdown = float(np.min(drawdowns))
    result.peak_equity = float(np.max(eq))

    # Max drawdown duration
    in_dd = drawdowns < 0
    if np.any(in_dd):
        dd_starts = np.where(np.diff(in_dd.astype(int)) == 1)[0]
        dd_ends = np.where(np.diff(in_dd.astype(int)) == -1)[0]
        if len(dd_starts) > 0 and len(dd_ends) > 0:
            durations = []
            for s in dd_starts:
                matching_ends = dd_ends[dd_ends > s]
                if len(matching_ends) > 0:
                    durations.append(matching_ends[0] - s)
            if durations:
                result.max_drawdown_duration_days = max(durations)

    # Calmar
    if result.max_drawdown < 0:
        result.calmar_ratio = round(
            result.annualized_return / abs(result.max_drawdown), 4
        )

    # Trade metrics
    if trade_returns:
        tr = np.array(trade_returns)
        result.total_trades = len(tr)
        result.winning_trades = int(np.sum(tr > 0))
        result.losing_trades = int(np.sum(tr < 0))
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        result.avg_trade_return = float(np.mean(tr))

        wins = tr[tr > 0]
        losses = tr[tr < 0]

        if len(wins) > 0:
            result.avg_win = float(np.mean(wins))
            result.largest_win = float(np.max(wins))
        if len(losses) > 0:
            result.avg_loss = float(np.mean(losses))
            result.largest_loss = float(np.min(losses))

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0
        gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0
        result.profit_factor = round(
            gross_profit / gross_loss if gross_loss > 0 else float("inf"), 4
        )

    # Costs
    result.total_fees = fees
    result.total_slippage = slippage
    result.total_funding = funding
    result.equity_curve = equity_curve

    return result


def compute_per_strategy_metrics(
    trade_returns: list[tuple[str, float]],
) -> list[StrategyBreakdown]:
    """Compute per-strategy metrics from (strategy_id, return) tuples."""
    if not trade_returns:
        return []

    by_strategy: dict[str, list[float]] = defaultdict(list)
    for strategy_id, ret in trade_returns:
        by_strategy[strategy_id].append(ret)

    results = []
    for strategy_id in sorted(by_strategy.keys()):
        returns = by_strategy[strategy_id]
        tr = np.array(returns)

        wins = tr[tr > 0]
        losses = tr[tr < 0]

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0

        results.append(StrategyBreakdown(
            strategy_id=strategy_id,
            total_trades=len(tr),
            winning_trades=int(np.sum(tr > 0)),
            losing_trades=int(np.sum(tr < 0)),
            win_rate=float(np.sum(tr > 0)) / len(tr) if len(tr) > 0 else 0.0,
            profit_factor=round(
                gross_profit / gross_loss if gross_loss > 0 else float("inf"), 4
            ),
            avg_return=float(np.mean(tr)),
            total_pnl_pct=float(np.sum(tr)) * 100,
        ))

    return results
