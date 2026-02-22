"""Trade outcome recorder — converts closed trades into episodic memory entries.

When a trade closes, this component extracts the key dimensions (outcome,
R-multiple, hold duration, signal context) and stores them as a
:data:`MemoryEntryType.TRADE_OUTCOME` entry in the memory store.

These episodic memories enable downstream components (the reflection agent
and BM25 similarity search) to learn from past trades without an LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.context.memory_store import MemoryEntry
from agentic_trading.core.enums import MemoryEntryType

logger = logging.getLogger(__name__)

# Default TTL for trade outcome memories: 7 days
_DEFAULT_TTL_HOURS = 168.0


class TradeOutcomeRecorder:
    """Converts closed :class:`TradeRecord` instances into memory entries.

    Parameters
    ----------
    memory_store:
        Any object implementing the ``IMemoryStore`` protocol (``store()``).
    ttl_hours:
        Time-to-live for outcome entries (default 168 = 7 days).
    """

    def __init__(
        self,
        memory_store: Any,
        ttl_hours: float = _DEFAULT_TTL_HOURS,
    ) -> None:
        self._store = memory_store
        self._ttl = ttl_hours
        self._recorded_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_trade_closed(self, trade: Any) -> None:
        """Record a closed trade as a TRADE_OUTCOME memory entry.

        Parameters
        ----------
        trade:
            A ``TradeRecord`` (from the journal) with computed properties
            like ``net_pnl``, ``r_multiple``, ``outcome``, ``mae``, ``mfe``.
        """
        try:
            entry = self._build_entry(trade)
            self._store.store(entry)
            self._recorded_count += 1
            logger.debug(
                "Trade outcome recorded: %s %s %s (R=%.2f)",
                trade.strategy_id,
                trade.symbol,
                trade.outcome.value,
                trade.r_multiple,
            )
        except Exception:
            logger.warning("Failed to record trade outcome", exc_info=True)

    @property
    def recorded_count(self) -> int:
        """Number of trade outcomes recorded since init."""
        return self._recorded_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_entry(self, trade: Any) -> MemoryEntry:
        """Build a MemoryEntry from a TradeRecord."""
        outcome_str = trade.outcome.value  # "win", "loss", "breakeven"
        direction = trade.direction  # "long" or "short"
        r_mult = trade.r_multiple
        net_pnl = float(trade.net_pnl)

        # Compute hold duration in hours
        hold_hours = 0.0
        if trade.opened_at and trade.closed_at:
            hold_hours = (
                trade.closed_at - trade.opened_at
            ).total_seconds() / 3600.0

        # Compute return percentage
        entry_notional = self._entry_notional(trade)
        return_pct = (net_pnl / entry_notional * 100) if entry_notional > 0 else 0.0

        # Build content dict
        content = {
            "trade_id": trade.trade_id,
            "trace_id": trade.trace_id,
            "strategy_id": trade.strategy_id,
            "symbol": trade.symbol,
            "direction": direction,
            "outcome": outcome_str,
            "net_pnl": net_pnl,
            "return_pct": round(return_pct, 2),
            "r_multiple": round(r_mult, 2),
            "hold_hours": round(hold_hours, 1),
            "signal_confidence": trade.signal_confidence,
            "signal_rationale": trade.signal_rationale or "",
            "mae": float(trade.mae),
            "mfe": float(trade.mfe),
        }

        # Optional fields
        if trade.signal_features:
            content["signal_features"] = {
                k: v for k, v in trade.signal_features.items()
                if isinstance(v, (int, float, str, bool))
            }
        if trade.initial_risk_price is not None:
            content["planned_stop"] = float(trade.initial_risk_price)
        if trade.planned_target_price is not None:
            content["planned_target"] = float(trade.planned_target_price)

        # Tags for fast filtering
        tags = [
            outcome_str,
            direction,
            trade.strategy_id,
        ]
        if r_mult >= 2.0:
            tags.append("big_winner")
        elif r_mult <= -2.0:
            tags.append("big_loser")

        # Summary for BM25 search
        summary = (
            f"{outcome_str.upper()}: {direction} {trade.symbol} "
            f"R={r_mult:+.2f} ({return_pct:+.1f}%) "
            f"via {trade.strategy_id}"
        )

        return MemoryEntry(
            entry_type=MemoryEntryType.TRADE_OUTCOME,
            symbol=trade.symbol,
            strategy_id=trade.strategy_id,
            tags=tags,
            content=content,
            summary=summary,
            relevance_score=1.0,
            ttl_hours=self._ttl,
        )

    @staticmethod
    def _entry_notional(trade: Any) -> float:
        """Estimate entry notional from fills."""
        total = 0.0
        for fill in getattr(trade, "entry_fills", []):
            price = float(getattr(fill, "price", 0))
            qty = float(getattr(fill, "qty", 0))
            total += price * qty
        return total
