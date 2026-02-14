"""Trade export — CSV/JSON output and periodic report generation.

Exports closed trade records in standard formats for external analysis,
reporting, and archival.  Supports filtering by strategy, symbol, and
date range.

Usage::

    exporter = TradeExporter()
    csv_str = exporter.to_csv(trades)
    json_str = exporter.to_json(trades)
    report = exporter.periodic_report(trades, period="daily")
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .record import TradeOutcome, TradePhase, TradeRecord

logger = logging.getLogger(__name__)

# Default CSV columns
_CSV_COLUMNS = [
    "trade_id",
    "trace_id",
    "strategy_id",
    "symbol",
    "direction",
    "phase",
    "outcome",
    "entry_price",
    "exit_price",
    "qty",
    "gross_pnl",
    "total_fees",
    "net_pnl",
    "r_multiple",
    "mae",
    "mfe",
    "mae_r",
    "mfe_r",
    "management_efficiency",
    "hold_duration_seconds",
    "signal_confidence",
    "opened_at",
    "closed_at",
    "entry_fills",
    "exit_fills",
    "tags",
    "mistakes",
]


class TradeExporter:
    """Export trades to CSV/JSON and generate periodic reports.

    Parameters
    ----------
    decimal_places : int
        Rounding precision for numeric fields.  Default 4.
    """

    def __init__(self, *, decimal_places: int = 4) -> None:
        self._dp = decimal_places

    # ------------------------------------------------------------------ #
    # CSV Export                                                           #
    # ------------------------------------------------------------------ #

    def to_csv(
        self,
        trades: list[TradeRecord],
        *,
        columns: list[str] | None = None,
    ) -> str:
        """Export trades as a CSV string.

        Parameters
        ----------
        trades : list[TradeRecord]
            Trades to export.
        columns : list[str] | None
            Column selection.  Defaults to ``_CSV_COLUMNS``.

        Returns
        -------
        str
            CSV-formatted string with header row.
        """
        cols = columns or _CSV_COLUMNS
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()

        for trade in trades:
            row = self._trade_to_row(trade)
            writer.writerow({c: row.get(c, "") for c in cols})

        return buf.getvalue()

    # ------------------------------------------------------------------ #
    # JSON Export                                                          #
    # ------------------------------------------------------------------ #

    def to_json(
        self,
        trades: list[TradeRecord],
        *,
        indent: int = 2,
    ) -> str:
        """Export trades as a JSON string.

        Parameters
        ----------
        trades : list[TradeRecord]
            Trades to export.
        indent : int
            JSON indentation level.

        Returns
        -------
        str
            JSON-formatted string (list of trade objects).
        """
        rows = [self._trade_to_row(t) for t in trades]
        return json.dumps(rows, indent=indent, default=str)

    # ------------------------------------------------------------------ #
    # Periodic Report                                                      #
    # ------------------------------------------------------------------ #

    def periodic_report(
        self,
        trades: list[TradeRecord],
        *,
        period: str = "daily",
    ) -> dict[str, Any]:
        """Generate a periodic performance summary.

        Parameters
        ----------
        trades : list[TradeRecord]
            Closed trades to analyse.
        period : str
            Grouping period: ``"daily"``, ``"weekly"``, or ``"monthly"``.

        Returns
        -------
        dict
            ``period`` : str
            ``buckets`` : list of dicts with per-period stats
            ``totals`` : overall summary across all periods
        """
        closed = [t for t in trades if t.phase == TradePhase.CLOSED]
        if not closed:
            return {"period": period, "buckets": [], "totals": self._empty_totals()}

        # Bucket trades by period
        buckets: dict[str, list[TradeRecord]] = defaultdict(list)
        for trade in closed:
            key = self._period_key(trade, period)
            if key:
                buckets[key].append(trade)

        # Build bucket summaries
        bucket_summaries = []
        for key in sorted(buckets.keys()):
            group = buckets[key]
            bucket_summaries.append(self._group_stats(key, group))

        # Build totals
        totals = self._group_stats("all", closed)
        totals.pop("period_key", None)

        return {
            "period": period,
            "buckets": bucket_summaries,
            "totals": totals,
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _trade_to_row(self, trade: TradeRecord) -> dict[str, Any]:
        """Convert a TradeRecord to a flat dict for export."""
        dp = self._dp
        return {
            "trade_id": trade.trade_id,
            "trace_id": trade.trace_id,
            "strategy_id": trade.strategy_id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "phase": trade.phase.value,
            "outcome": trade.outcome.value if trade.phase == TradePhase.CLOSED else None,
            "entry_price": round(float(trade.avg_entry_price), dp) if trade.entry_fills else None,
            "exit_price": round(float(trade.avg_exit_price), dp) if trade.exit_fills else None,
            "qty": round(float(trade.entry_qty), dp),
            "gross_pnl": round(float(trade.gross_pnl), dp),
            "total_fees": round(float(trade.total_fees), dp),
            "net_pnl": round(float(trade.net_pnl), dp),
            "r_multiple": round(trade.r_multiple, dp),
            "mae": round(float(trade.mae), dp),
            "mfe": round(float(trade.mfe), dp),
            "mae_r": round(trade.mae_r, dp),
            "mfe_r": round(trade.mfe_r, dp),
            "management_efficiency": round(trade.management_efficiency, dp),
            "hold_duration_seconds": trade.hold_duration_seconds,
            "signal_confidence": round(trade.signal_confidence, dp),
            "opened_at": trade.opened_at.isoformat() if trade.opened_at else None,
            "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
            "entry_fills": len(trade.entry_fills),
            "exit_fills": len(trade.exit_fills),
            "tags": ",".join(trade.tags) if trade.tags else "",
            "mistakes": ",".join(trade.mistakes) if trade.mistakes else "",
        }

    def _period_key(self, trade: TradeRecord, period: str) -> str | None:
        """Get the period bucket key for a trade."""
        ts = trade.closed_at or trade.opened_at
        if ts is None:
            return None

        if period == "daily":
            return ts.strftime("%Y-%m-%d")
        elif period == "weekly":
            # ISO week
            return f"{ts.isocalendar()[0]}-W{ts.isocalendar()[1]:02d}"
        elif period == "monthly":
            return ts.strftime("%Y-%m")
        return ts.strftime("%Y-%m-%d")

    def _group_stats(self, key: str, group: list[TradeRecord]) -> dict[str, Any]:
        """Compute aggregate statistics for a group of trades."""
        dp = self._dp
        wins = [t for t in group if t.outcome == TradeOutcome.WIN]
        losses = [t for t in group if t.outcome == TradeOutcome.LOSS]
        total_pnl = sum(float(t.net_pnl) for t in group)
        gross_wins = sum(float(t.net_pnl) for t in wins)
        gross_losses = sum(abs(float(t.net_pnl)) for t in losses)

        pf = (
            round(gross_wins / gross_losses, dp) if gross_losses > 0
            else (float("inf") if gross_wins > 0 else 0.0)
        )

        return {
            "period_key": key,
            "trades": len(group),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(group), dp) if group else 0.0,
            "total_pnl": round(total_pnl, dp),
            "avg_pnl": round(total_pnl / len(group), dp) if group else 0.0,
            "avg_r": round(
                sum(t.r_multiple for t in group) / len(group), dp
            ) if group else 0.0,
            "profit_factor": pf,
            "best_trade": round(max(float(t.net_pnl) for t in group), dp) if group else 0.0,
            "worst_trade": round(min(float(t.net_pnl) for t in group), dp) if group else 0.0,
        }

    def _empty_totals(self) -> dict[str, Any]:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }
