"""Decision audit trail.

Logs the complete decision chain:
features → signal → risk decision → order intent → exchange ack → fill → PnL attribution.

Every decision is tagged with a trace_id for end-to-end correlation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .logger import get_trace_id

logger = logging.getLogger(__name__)


class DecisionAudit:
    """Logs and stores the complete decision chain for audit."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._decisions: list[dict[str, Any]] = []

    def log_features(
        self, symbol: str, timeframe: str, features: dict[str, float]
    ) -> None:
        """Log computed features."""
        if not self._enabled:
            return
        self._log_step("features", {
            "symbol": symbol,
            "timeframe": timeframe,
            "feature_count": len(features),
            "features": {k: round(v, 6) for k, v in list(features.items())[:20]},
        })

    def log_signal(
        self,
        strategy_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        rationale: str,
        features_used: dict[str, float],
    ) -> None:
        """Log strategy signal."""
        if not self._enabled:
            return
        self._log_step("signal", {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "rationale": rationale,
            "features_used": {k: round(v, 6) for k, v in features_used.items()},
        })

    def log_risk_check(
        self, check_name: str, passed: bool, reason: str = ""
    ) -> None:
        """Log risk check result."""
        if not self._enabled:
            return
        self._log_step("risk_check", {
            "check_name": check_name,
            "passed": passed,
            "reason": reason,
        })

    def log_order_intent(
        self, dedupe_key: str, symbol: str, side: str, qty: str, price: str | None
    ) -> None:
        """Log order intent creation."""
        if not self._enabled:
            return
        self._log_step("order_intent", {
            "dedupe_key": dedupe_key,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
        })

    def log_order_ack(
        self, order_id: str, status: str, message: str = ""
    ) -> None:
        """Log exchange acknowledgment."""
        if not self._enabled:
            return
        self._log_step("order_ack", {
            "order_id": order_id,
            "status": status,
            "message": message,
        })

    def log_fill(
        self,
        fill_id: str,
        order_id: str,
        price: str,
        qty: str,
        fee: str,
    ) -> None:
        """Log fill/execution."""
        if not self._enabled:
            return
        self._log_step("fill", {
            "fill_id": fill_id,
            "order_id": order_id,
            "price": price,
            "qty": qty,
            "fee": fee,
        })

    def log_pnl(
        self, symbol: str, realized_pnl: float, unrealized_pnl: float
    ) -> None:
        """Log PnL attribution."""
        if not self._enabled:
            return
        self._log_step("pnl", {
            "symbol": symbol,
            "realized_pnl": round(realized_pnl, 6),
            "unrealized_pnl": round(unrealized_pnl, 6),
        })

    def _log_step(self, step: str, data: dict[str, Any]) -> None:
        """Log a single decision step."""
        entry = {
            "trace_id": get_trace_id(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            **data,
        }
        self._decisions.append(entry)
        logger.info("AUDIT %s: %s", step, json.dumps(data, default=str))

    def get_decisions(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        """Get audit trail, optionally filtered by trace_id."""
        if trace_id is None:
            return list(self._decisions)
        return [d for d in self._decisions if d.get("trace_id") == trace_id]

    def clear(self) -> None:
        """Clear stored decisions."""
        self._decisions.clear()
