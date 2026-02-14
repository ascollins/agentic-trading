"""Time-of-day and session performance analysis.

Breaks down trading performance by hour, session (Asia/London/NY),
and day-of-week to reveal temporal patterns.  Answers questions like
"Am I better in the London session?" or "Should I avoid Mondays?"

Usage::

    analyser = SessionAnalyser()
    analyser.add_trade(trade_record)
    report = analyser.report("trend_following")
    print(report["by_session"]["london"]["win_rate"])
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .record import TradeOutcome, TradePhase, TradeRecord

logger = logging.getLogger(__name__)

# Session definitions (UTC hours, inclusive start, exclusive end)
SESSIONS = {
    "asia":   (0, 8),     # 00:00-08:00 UTC
    "london": (8, 16),    # 08:00-16:00 UTC
    "new_york": (13, 21), # 13:00-21:00 UTC (overlaps with London)
    "off_hours": None,    # Anything not in above sessions
}

DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


@dataclass
class _BucketStats:
    """Accumulator for a time bucket."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_r: float = 0.0
    total_hold_seconds: float = 0.0
    gross_wins: float = 0.0
    gross_losses: float = 0.0

    def record(self, trade: TradeRecord) -> None:
        pnl = float(trade.net_pnl)
        self.trades += 1
        self.total_pnl += pnl
        self.total_r += trade.r_multiple
        self.total_hold_seconds += trade.hold_duration_seconds
        if trade.outcome == TradeOutcome.WIN:
            self.wins += 1
            self.gross_wins += pnl
        elif trade.outcome == TradeOutcome.LOSS:
            self.losses += 1
            self.gross_losses += abs(pnl)

    def to_dict(self) -> dict[str, Any]:
        if self.trades == 0:
            return {"trades": 0}
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.wins / self.trades, 4),
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl": round(self.total_pnl / self.trades, 2),
            "avg_r": round(self.total_r / self.trades, 4),
            "profit_factor": round(
                self.gross_wins / self.gross_losses, 4
            ) if self.gross_losses > 0 else (float("inf") if self.gross_wins > 0 else 0.0),
            "avg_hold_minutes": round(self.total_hold_seconds / self.trades / 60, 1),
        }


class SessionAnalyser:
    """Time-of-day and session performance analyser.

    Parameters
    ----------
    max_trades : int
        Maximum trades per strategy to keep in memory.  Default 5000.
    """

    def __init__(self, *, max_trades: int = 5000) -> None:
        self._max_trades = max_trades
        # Store raw trade summaries for re-bucketing
        self._trades: dict[str, list[dict]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a closed trade to the analyser."""
        if trade.phase != TradePhase.CLOSED or trade.opened_at is None:
            return

        summary = {
            "hour": trade.opened_at.hour,
            "weekday": trade.opened_at.weekday(),  # 0=Monday
            "pnl": float(trade.net_pnl),
            "r_multiple": trade.r_multiple,
            "won": trade.outcome == TradeOutcome.WIN,
            "hold_seconds": trade.hold_duration_seconds,
            "outcome": trade.outcome,
            "gross_pnl": float(trade.gross_pnl),
            "_trade": trade,  # Keep reference for re-analysis
        }

        sid = trade.strategy_id
        self._trades[sid].append(summary)
        while len(self._trades[sid]) > self._max_trades:
            self._trades[sid].pop(0)

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def report(self, strategy_id: str) -> dict[str, Any]:
        """Generate session/time analysis report.

        Returns
        -------
        dict
            ``by_hour`` : dict[int, stats] — performance per hour (0-23)
            ``by_session`` : dict[str, stats] — performance per session
            ``by_day`` : dict[str, stats] — performance per day of week
            ``best_hour`` : int — highest win rate hour
            ``worst_hour`` : int — lowest win rate hour
            ``best_session`` : str — highest win rate session
            ``best_day`` : str — highest win rate day
            ``total_trades`` : int
        """
        trades = self._trades.get(strategy_id, [])
        if not trades:
            return {
                "strategy_id": strategy_id,
                "total_trades": 0,
                "by_hour": {},
                "by_session": {},
                "by_day": {},
                "best_hour": None,
                "worst_hour": None,
                "best_session": None,
                "best_day": None,
            }

        # Bucket by hour
        by_hour: dict[int, _BucketStats] = defaultdict(_BucketStats)
        by_session: dict[str, _BucketStats] = defaultdict(_BucketStats)
        by_day: dict[str, _BucketStats] = defaultdict(_BucketStats)

        for s in trades:
            trade = s["_trade"]
            hour = s["hour"]
            weekday = s["weekday"]

            by_hour[hour].record(trade)
            by_day[DAY_NAMES[weekday]].record(trade)

            # Determine session
            session_found = False
            for session_name, hours in SESSIONS.items():
                if hours is None:
                    continue
                start_h, end_h = hours
                if start_h <= hour < end_h:
                    by_session[session_name].record(trade)
                    session_found = True
            if not session_found:
                by_session["off_hours"].record(trade)

        # Convert to dicts
        hour_dicts = {h: stats.to_dict() for h, stats in sorted(by_hour.items())}
        session_dicts = {s: stats.to_dict() for s, stats in by_session.items()}
        day_dicts = {d: stats.to_dict() for d, stats in by_day.items()}

        # Find best/worst
        def _best_key(d, metric="win_rate"):
            valid = {k: v for k, v in d.items() if v.get("trades", 0) >= 3}
            if not valid:
                return None
            return max(valid, key=lambda k: valid[k].get(metric, 0))

        def _worst_key(d, metric="win_rate"):
            valid = {k: v for k, v in d.items() if v.get("trades", 0) >= 3}
            if not valid:
                return None
            return min(valid, key=lambda k: valid[k].get(metric, 0))

        return {
            "strategy_id": strategy_id,
            "total_trades": len(trades),
            "by_hour": hour_dicts,
            "by_session": session_dicts,
            "by_day": day_dicts,
            "best_hour": _best_key(hour_dicts),
            "worst_hour": _worst_key(hour_dicts),
            "best_session": _best_key(session_dicts),
            "best_day": _best_key(day_dicts),
        }

    def get_all_strategy_ids(self) -> list[str]:
        """Return all strategy IDs with session data."""
        return [sid for sid, t in self._trades.items() if t]
