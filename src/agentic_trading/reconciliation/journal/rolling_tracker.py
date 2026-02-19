"""Rolling performance tracker — sliding-window analytics.

Maintains a configurable-length window of recent trades and computes
rolling statistics that reveal trajectory changes before they show up
in all-time numbers.  Designed for Edgewonk-style "Am I getting
better or worse?" self-measurement.

Metrics are computed lazily (on read) and cached until the window
changes, so adding trades is O(1) and reading is O(N) only on
cache miss.

Example::

    tracker = RollingTracker(window_size=50)
    tracker.add_trade(closed_trade_record)
    snapshot = tracker.snapshot("trend_following")
    print(snapshot["win_rate"], snapshot["sharpe"])
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .record import TradeOutcome, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class _TradeSnapshot:
    """Lightweight summary extracted from a TradeRecord."""

    pnl: float
    r_multiple: float
    won: bool
    hold_seconds: float
    management_efficiency: float
    fees: float
    gross_pnl: float
    signal_confidence: float


@dataclass
class _StrategyWindow:
    """Sliding window of trade snapshots for one strategy."""

    trades: deque[_TradeSnapshot] = field(default_factory=deque)
    _dirty: bool = True
    _cache: dict[str, Any] = field(default_factory=dict)

    def add(self, snap: _TradeSnapshot, max_size: int) -> None:
        self.trades.append(snap)
        while len(self.trades) > max_size:
            self.trades.popleft()
        self._dirty = True

    def invalidate(self) -> None:
        self._dirty = True


class RollingTracker:
    """Sliding-window performance metrics across recent trades.

    Parameters
    ----------
    window_size : int
        Number of most recent trades to keep per strategy.  Default 100.
    """

    def __init__(self, *, window_size: int = 100) -> None:
        self._window_size = max(1, window_size)
        self._windows: dict[str, _StrategyWindow] = defaultdict(
            _StrategyWindow
        )

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a closed trade to the rolling window.

        Trades that are not in the CLOSED phase are silently ignored.
        """
        from .record import TradePhase

        if trade.phase != TradePhase.CLOSED:
            return

        snap = _TradeSnapshot(
            pnl=float(trade.net_pnl),
            r_multiple=trade.r_multiple,
            won=trade.outcome == TradeOutcome.WIN,
            hold_seconds=trade.hold_duration_seconds,
            management_efficiency=trade.management_efficiency,
            fees=float(trade.total_fees),
            gross_pnl=float(trade.gross_pnl),
            signal_confidence=trade.signal_confidence,
        )
        window = self._windows[trade.strategy_id]
        window.add(snap, self._window_size)

    # ------------------------------------------------------------------ #
    # Analytics                                                            #
    # ------------------------------------------------------------------ #

    def snapshot(self, strategy_id: str) -> dict[str, Any]:
        """Compute rolling metrics for a strategy.

        Returns an empty dict if no trades recorded.
        """
        window = self._windows.get(strategy_id)
        if window is None or len(window.trades) == 0:
            return {}

        if not window._dirty and window._cache:
            return window._cache

        trades = list(window.trades)
        n = len(trades)

        wins = sum(1 for t in trades if t.won)
        losses = n - wins  # breakevens counted as non-wins
        pnls = [t.pnl for t in trades]
        r_mults = [t.r_multiple for t in trades]

        gross_w = sum(t.pnl for t in trades if t.won and t.pnl > 0)
        gross_l = sum(abs(t.pnl) for t in trades if not t.won and t.pnl < 0)

        # Sharpe (trade-level)
        sharpe = 0.0
        if n >= 2:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl > 0:
                sharpe = mean_pnl / std_pnl

        # Sortino (downside deviation only)
        sortino = 0.0
        if n >= 2:
            mean_pnl = statistics.mean(pnls)
            downside = [min(0, p) for p in pnls]
            downside_std = math.sqrt(
                sum(d ** 2 for d in downside) / len(downside)
            )
            if downside_std > 0:
                sortino = mean_pnl / downside_std

        # Max drawdown within window
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Profit factor
        pf = 0.0
        if gross_l > 0:
            pf = gross_w / gross_l
        elif gross_w > 0:
            pf = float("inf")

        # Average hold & management efficiency
        efficiencies = [
            t.management_efficiency
            for t in trades
            if t.management_efficiency > 0
        ]

        result = {
            "trade_count": n,
            "window_size": self._window_size,
            "win_rate": wins / n if n else 0.0,
            "loss_rate": losses / n if n else 0.0,
            "profit_factor": pf,
            "expectancy": sum(pnls) / n if n else 0.0,
            "avg_r": statistics.mean(r_mults) if r_mults else 0.0,
            "median_r": statistics.median(r_mults) if r_mults else 0.0,
            "best_r": max(r_mults) if r_mults else 0.0,
            "worst_r": min(r_mults) if r_mults else 0.0,
            "sharpe": sharpe,
            "sortino": sortino,
            "total_pnl": sum(pnls),
            "avg_pnl": statistics.mean(pnls) if pnls else 0.0,
            "max_drawdown": max_dd,
            "avg_winner": gross_w / wins if wins else 0.0,
            "avg_loser": gross_l / losses if losses else 0.0,
            "avg_hold_seconds": statistics.mean(
                [t.hold_seconds for t in trades]
            ),
            "avg_management_efficiency": (
                statistics.mean(efficiencies) if efficiencies else 0.0
            ),
            "total_fees": sum(t.fees for t in trades),
            "consecutive_wins": self._current_streak(trades, True),
            "consecutive_losses": self._current_streak(trades, False),
        }

        window._cache = result
        window._dirty = False
        return result

    def get_all_strategy_ids(self) -> list[str]:
        """Return all strategy IDs that have rolling data."""
        return [
            sid
            for sid, w in self._windows.items()
            if len(w.trades) > 0
        ]

    def trade_count(self, strategy_id: str) -> int:
        """Number of trades in the rolling window for a strategy."""
        window = self._windows.get(strategy_id)
        return len(window.trades) if window else 0

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _current_streak(
        trades: list[_TradeSnapshot], count_wins: bool
    ) -> int:
        """Count current consecutive wins or losses from the end."""
        streak = 0
        for t in reversed(trades):
            if t.won == count_wins:
                streak += 1
            else:
                break
        return streak
