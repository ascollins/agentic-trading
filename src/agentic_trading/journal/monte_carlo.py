"""Monte Carlo equity projector — forward-looking risk analysis.

Performs bootstrap resampling of historical trade P&L to project
future equity curves and compute probability of ruin, expected
drawdown distributions, and confidence intervals for terminal equity.

Inspired by Edgewonk's "Trade Simulator" feature that answers
"given my historical edge, what's the range of possible outcomes
over the next N trades?"

Usage::

    projector = MonteCarloProjector()
    projector.set_trades("trend", pnl_series=[12.5, -8.0, 25.0, ...])
    result = projector.project("trend", initial_equity=100000, n_trades=500)
    print(result["ruin_probability"])       # 0.02
    print(result["median_terminal_equity"]) # 115_000
    print(result["percentile_5"])           # 85_000
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _StrategyPnL:
    """Historical P&L data for one strategy."""

    pnl_series: list[float] = field(default_factory=list)
    r_series: list[float] = field(default_factory=list)


class MonteCarloProjector:
    """Monte Carlo equity projection and probability-of-ruin estimator.

    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo paths to simulate.  Default 1000.
    ruin_threshold_pct : float
        Percentage of initial equity below which we declare "ruin".
        Default 0.5 (50% drawdown = ruin).
    seed : int | None
        Random seed for reproducibility.  None = non-deterministic.
    """

    def __init__(
        self,
        *,
        n_simulations: int = 1000,
        ruin_threshold_pct: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._n_sims = max(100, n_simulations)
        self._ruin_threshold = ruin_threshold_pct
        self._rng = random.Random(seed)
        self._data: dict[str, _StrategyPnL] = defaultdict(_StrategyPnL)

    # ------------------------------------------------------------------ #
    # Data input                                                           #
    # ------------------------------------------------------------------ #

    def set_trades(
        self,
        strategy_id: str,
        pnl_series: list[float],
        r_series: list[float] | None = None,
    ) -> None:
        """Set historical P&L (and optional R-multiple) series.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier.
        pnl_series : list[float]
            List of per-trade net P&L values.
        r_series : list[float] | None
            Optional list of per-trade R-multiples (same length).
        """
        data = self._data[strategy_id]
        data.pnl_series = list(pnl_series)
        data.r_series = list(r_series) if r_series else []

    def add_trade(
        self,
        strategy_id: str,
        pnl: float,
        r_multiple: float = 0.0,
    ) -> None:
        """Incrementally add a single trade outcome."""
        data = self._data[strategy_id]
        data.pnl_series.append(pnl)
        if r_multiple != 0.0:
            data.r_series.append(r_multiple)

    # ------------------------------------------------------------------ #
    # Projection                                                           #
    # ------------------------------------------------------------------ #

    def project(
        self,
        strategy_id: str,
        initial_equity: float = 100_000.0,
        n_trades: int = 500,
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation and return projection results.

        Parameters
        ----------
        strategy_id : str
            Strategy to project.
        initial_equity : float
            Starting capital.
        n_trades : int
            Number of future trades to simulate per path.

        Returns
        -------
        dict
            Projection results including ruin probability, percentiles,
            drawdown statistics, and equity curve percentile bands.
        """
        data = self._data.get(strategy_id)
        if data is None or len(data.pnl_series) < 5:
            return {
                "error": "insufficient_data",
                "min_trades": 5,
                "current_trades": len(data.pnl_series) if data else 0,
            }

        source = data.pnl_series
        ruin_level = initial_equity * (1.0 - self._ruin_threshold)

        terminal_equities: list[float] = []
        max_drawdowns: list[float] = []
        ruin_count = 0
        equity_paths: list[list[float]] = []

        # Store a subset of paths for percentile bands
        store_paths = min(100, self._n_sims)

        for i in range(self._n_sims):
            equity = initial_equity
            peak = equity
            worst_dd = 0.0
            ruined = False
            path: list[float] = [equity] if i < store_paths else []

            for _ in range(n_trades):
                trade_pnl = self._rng.choice(source)
                equity += trade_pnl

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0.0
                if dd > worst_dd:
                    worst_dd = dd

                if equity <= ruin_level:
                    ruined = True

                if i < store_paths:
                    path.append(equity)

            terminal_equities.append(equity)
            max_drawdowns.append(worst_dd)
            if ruined:
                ruin_count += 1
            if path:
                equity_paths.append(path)

        terminal_equities.sort()
        max_drawdowns.sort()

        n = self._n_sims

        # Compute percentile bands from stored paths
        bands = self._compute_percentile_bands(equity_paths, n_trades)

        return {
            "strategy_id": strategy_id,
            "initial_equity": initial_equity,
            "n_trades": n_trades,
            "n_simulations": n,
            "source_trades": len(source),
            "ruin_probability": round(ruin_count / n, 6),
            "ruin_threshold_pct": self._ruin_threshold,
            # Terminal equity distribution
            "mean_terminal_equity": round(
                sum(terminal_equities) / n, 2
            ),
            "median_terminal_equity": round(
                self._percentile(terminal_equities, 50), 2
            ),
            "percentile_5": round(
                self._percentile(terminal_equities, 5), 2
            ),
            "percentile_25": round(
                self._percentile(terminal_equities, 25), 2
            ),
            "percentile_75": round(
                self._percentile(terminal_equities, 75), 2
            ),
            "percentile_95": round(
                self._percentile(terminal_equities, 95), 2
            ),
            "min_terminal": round(terminal_equities[0], 2),
            "max_terminal": round(terminal_equities[-1], 2),
            # Drawdown distribution
            "mean_max_drawdown": round(
                sum(max_drawdowns) / n, 6
            ),
            "median_max_drawdown": round(
                self._percentile(max_drawdowns, 50), 6
            ),
            "worst_case_drawdown_95": round(
                self._percentile(max_drawdowns, 95), 6
            ),
            # Equity curve bands (trade index → {p5, p25, p50, p75, p95})
            "equity_bands": bands,
            # Profit probability
            "profit_probability": round(
                sum(1 for e in terminal_equities if e > initial_equity) / n,
                4,
            ),
        }

    # ------------------------------------------------------------------ #
    # Kelly criterion                                                      #
    # ------------------------------------------------------------------ #

    def kelly_fraction(self, strategy_id: str) -> float:
        """Compute optimal Kelly fraction from trade history.

        Returns the fraction of capital to risk per trade for geometric
        growth maximisation.  In practice, use half-Kelly or less.

        Returns 0.0 if insufficient data or negative expectancy.
        """
        data = self._data.get(strategy_id)
        if data is None or len(data.pnl_series) < 10:
            return 0.0

        wins = [p for p in data.pnl_series if p > 0]
        losses = [abs(p) for p in data.pnl_series if p < 0]

        if not wins or not losses:
            return 0.0

        win_rate = len(wins) / len(data.pnl_series)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        if avg_loss == 0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss

        # Kelly formula: f* = (p * b - q) / b
        # where p = win_rate, q = 1 - p, b = win_loss_ratio
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        return max(0.0, round(kelly, 6))

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _percentile(sorted_values: list[float], pct: float) -> float:
        """Compute percentile from a sorted list."""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * (pct / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    @staticmethod
    def _compute_percentile_bands(
        paths: list[list[float]], n_trades: int
    ) -> list[dict[str, float]]:
        """Compute percentile bands across stored equity paths.

        Returns data at 10 evenly-spaced checkpoints (not every trade)
        to keep the output compact.
        """
        if not paths:
            return []

        # Sample at ~10 points + start + end
        step = max(1, n_trades // 10)
        checkpoints = list(range(0, n_trades + 1, step))
        if checkpoints[-1] != n_trades:
            checkpoints.append(n_trades)

        bands = []
        for idx in checkpoints:
            values = sorted(p[idx] for p in paths if idx < len(p))
            if not values:
                continue
            bands.append({
                "trade_index": idx,
                "p5": round(MonteCarloProjector._percentile(values, 5), 2),
                "p25": round(MonteCarloProjector._percentile(values, 25), 2),
                "p50": round(MonteCarloProjector._percentile(values, 50), 2),
                "p75": round(MonteCarloProjector._percentile(values, 75), 2),
                "p95": round(MonteCarloProjector._percentile(values, 95), 2),
            })

        return bands
