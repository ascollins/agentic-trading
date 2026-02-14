"""Trade replay — structured mark-to-market history for visualization.

Converts a TradeRecord with its fill legs and mark samples into a
time-indexed replay format suitable for rendering as charts or
exporting to external visualization tools.

Usage::

    replayer = TradeReplayer()
    replay = replayer.build_replay(trade_record)
    print(replay["timeline"])  # [{ts, price, equity, event_type, ...}, ...]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .record import TradePhase, TradeRecord

logger = logging.getLogger(__name__)


class TradeReplayer:
    """Build structured replay data from a TradeRecord."""

    def build_replay(self, trade: TradeRecord) -> dict[str, Any]:
        """Build a replay timeline for a single trade.

        Returns
        -------
        dict
            ``trade_id`` : str
            ``summary`` : dict of trade-level metrics
            ``timeline`` : list of time-ordered events
            ``zones`` : dict with entry_zone, stop_zone, target_zone
            ``excursion_chart`` : list of {ts, pnl, mark_price} for MAE/MFE viz
        """
        timeline: list[dict[str, Any]] = []

        # Entry fills
        for fill in trade.entry_fills:
            timeline.append({
                "timestamp": fill.timestamp.isoformat() if fill.timestamp else None,
                "event_type": "entry_fill",
                "price": float(fill.price),
                "qty": float(fill.qty),
                "side": fill.side,
                "fee": float(fill.fee),
                "cumulative_qty": 0.0,  # Computed below
            })

        # Mark samples
        for mark in trade.mark_samples:
            timeline.append({
                "timestamp": mark.timestamp.isoformat() if mark.timestamp else None,
                "event_type": "mark",
                "price": float(mark.mark_price),
                "unrealized_pnl": float(mark.unrealized_pnl),
            })

        # Exit fills
        for fill in trade.exit_fills:
            timeline.append({
                "timestamp": fill.timestamp.isoformat() if fill.timestamp else None,
                "event_type": "exit_fill",
                "price": float(fill.price),
                "qty": float(fill.qty),
                "side": fill.side,
                "fee": float(fill.fee),
            })

        # Sort by timestamp
        timeline.sort(key=lambda e: e.get("timestamp") or "")

        # Compute cumulative quantity for entries
        cum_qty = Decimal("0")
        for event in timeline:
            if event["event_type"] == "entry_fill":
                cum_qty += Decimal(str(event["qty"]))
                event["cumulative_qty"] = float(cum_qty)

        # Build excursion chart (mark samples only)
        excursion = []
        mae_point = None
        mfe_point = None
        for mark in trade.mark_samples:
            pnl = float(mark.unrealized_pnl)
            point = {
                "timestamp": mark.timestamp.isoformat() if mark.timestamp else None,
                "mark_price": float(mark.mark_price),
                "unrealized_pnl": pnl,
            }
            excursion.append(point)
            if mae_point is None or pnl < mae_point["unrealized_pnl"]:
                mae_point = point
            if mfe_point is None or pnl > mfe_point["unrealized_pnl"]:
                mfe_point = point

        # Build zones
        zones = {
            "entry_price": float(trade.avg_entry_price) if trade.entry_fills else None,
            "exit_price": float(trade.avg_exit_price) if trade.exit_fills else None,
            "stop_price": float(trade.initial_risk_price) if trade.initial_risk_price else None,
            "target_price": float(trade.planned_target_price) if trade.planned_target_price else None,
            "mae_price": float(trade.mae_price) if float(trade.mae_price) != 0 else None,
            "mfe_price": float(trade.mfe_price) if float(trade.mfe_price) != 0 else None,
        }

        # Summary
        summary = {
            "trade_id": trade.trade_id,
            "trace_id": trade.trace_id,
            "strategy_id": trade.strategy_id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "phase": trade.phase.value,
            "outcome": trade.outcome.value if trade.phase == TradePhase.CLOSED else None,
            "net_pnl": float(trade.net_pnl),
            "r_multiple": trade.r_multiple,
            "management_efficiency": trade.management_efficiency,
            "hold_duration_s": trade.hold_duration_seconds,
            "entry_fills": len(trade.entry_fills),
            "exit_fills": len(trade.exit_fills),
            "mark_samples": len(trade.mark_samples),
            "mistakes": trade.mistakes,
            "tags": trade.tags,
        }

        return {
            "trade_id": trade.trade_id,
            "summary": summary,
            "timeline": timeline,
            "zones": zones,
            "excursion_chart": excursion,
            "mae_point": mae_point,
            "mfe_point": mfe_point,
        }

    def build_batch_replay(
        self, trades: list[TradeRecord]
    ) -> list[dict[str, Any]]:
        """Build replays for multiple trades."""
        return [self.build_replay(t) for t in trades]
