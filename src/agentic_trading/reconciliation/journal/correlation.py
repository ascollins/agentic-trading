"""Cross-strategy and cross-asset correlation analysis.

Measures how strategy returns and trade outcomes co-move.
High positive correlation between strategies means less diversification
benefit; negative correlation is ideal for portfolio construction.

Usage::

    matrix = CorrelationMatrix()
    for trade in closed_trades:
        matrix.add_trade(trade)
    report = matrix.report()
    print(report["strategy_correlation"])  # {("a","b"): 0.45, ...}
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .record import TradePhase, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class _DailyReturn:
    """Accumulated daily return for one bucket."""

    total_pnl: float = 0.0
    trade_count: int = 0


class CorrelationMatrix:
    """Cross-strategy and cross-asset return correlation.

    Parameters
    ----------
    max_days : int
        Maximum number of daily return observations to track.
        Default 365.
    """

    def __init__(self, *, max_days: int = 365) -> None:
        self._max_days = max_days
        # {strategy_id: {date_str: _DailyReturn}}
        self._strategy_daily: dict[str, dict[str, _DailyReturn]] = defaultdict(
            lambda: defaultdict(_DailyReturn)
        )
        # {symbol: {date_str: _DailyReturn}}
        self._symbol_daily: dict[str, dict[str, _DailyReturn]] = defaultdict(
            lambda: defaultdict(_DailyReturn)
        )

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a closed trade for correlation tracking."""
        if trade.phase != TradePhase.CLOSED or trade.closed_at is None:
            return

        date_key = trade.closed_at.strftime("%Y-%m-%d")
        pnl = float(trade.net_pnl)

        # Strategy daily returns
        dr = self._strategy_daily[trade.strategy_id][date_key]
        dr.total_pnl += pnl
        dr.trade_count += 1

        # Symbol daily returns
        sr = self._symbol_daily[trade.symbol][date_key]
        sr.total_pnl += pnl
        sr.trade_count += 1

    # ------------------------------------------------------------------ #
    # Correlation computation                                              #
    # ------------------------------------------------------------------ #

    def report(self) -> dict[str, Any]:
        """Compute correlation matrices.

        Returns
        -------
        dict
            ``strategy_correlation`` : dict of (sid_a, sid_b) → pearson_r
            ``symbol_correlation`` : dict of (sym_a, sym_b) → pearson_r
            ``strategy_ids`` : list of strategy IDs
            ``symbols`` : list of symbols
            ``data_days`` : number of unique days tracked
        """
        strat_ids = sorted(self._strategy_daily.keys())
        symbols = sorted(self._symbol_daily.keys())

        # Get all unique dates across all strategies
        all_dates = set()
        for daily in self._strategy_daily.values():
            all_dates.update(daily.keys())
        all_dates = sorted(all_dates)[-self._max_days:]

        # Build strategy correlation matrix
        strat_corr: dict[tuple[str, str], float] = {}
        for i, sid_a in enumerate(strat_ids):
            for j, sid_b in enumerate(strat_ids):
                if j <= i:
                    continue
                r = self._pearson(
                    self._strategy_daily[sid_a],
                    self._strategy_daily[sid_b],
                    all_dates,
                )
                strat_corr[(sid_a, sid_b)] = round(r, 4)

        # Build symbol correlation matrix
        sym_corr: dict[tuple[str, str], float] = {}
        for i, sym_a in enumerate(symbols):
            for j, sym_b in enumerate(symbols):
                if j <= i:
                    continue
                r = self._pearson(
                    self._symbol_daily[sym_a],
                    self._symbol_daily[sym_b],
                    all_dates,
                )
                sym_corr[(sym_a, sym_b)] = round(r, 4)

        # Convert tuple keys to strings for JSON serialisation
        strat_corr_str = {
            f"{a}|{b}": v for (a, b), v in strat_corr.items()
        }
        sym_corr_str = {
            f"{a}|{b}": v for (a, b), v in sym_corr.items()
        }

        return {
            "strategy_correlation": strat_corr_str,
            "symbol_correlation": sym_corr_str,
            "strategy_ids": strat_ids,
            "symbols": symbols,
            "data_days": len(all_dates),
        }

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pearson(
        daily_a: dict[str, _DailyReturn],
        daily_b: dict[str, _DailyReturn],
        dates: list[str],
    ) -> float:
        """Compute Pearson correlation between two daily return series."""
        # Build aligned series (use 0.0 for missing dates)
        xs = [daily_a.get(d, _DailyReturn()).total_pnl for d in dates]
        ys = [daily_b.get(d, _DailyReturn()).total_pnl for d in dates]

        # Only use dates where at least one has data
        pairs = [(x, y) for x, y in zip(xs, ys) if x != 0 or y != 0]
        if len(pairs) < 5:
            return 0.0

        xs_f = [p[0] for p in pairs]
        ys_f = [p[1] for p in pairs]
        n = len(xs_f)

        mean_x = sum(xs_f) / n
        mean_y = sum(ys_f) / n

        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs_f, ys_f))
        var_x = sum((x - mean_x) ** 2 for x in xs_f)
        var_y = sum((y - mean_y) ** 2 for y in ys_f)

        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return 0.0
        return cov / denom
