"""Daily Effectiveness Scorecard — 4 scores, 0-10 scale.

Aggregates data from:
  - TradeJournal (edge quality, management efficiency)
  - ExecutionQualityTracker (slippage, fill rate, adverse selection)
  - RiskManager / CircuitBreakers (drawdown, trip count, overrides)
  - AgentRegistry / EventBus (health, freshness, DLQ size)

Each score is weighted to produce a single daily effectiveness number.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Score weights
WEIGHTS: dict[str, float] = {
    "edge_quality": 0.30,
    "execution_quality": 0.25,
    "risk_discipline": 0.25,
    "operational_integrity": 0.20,
}


class DailyEffectivenessScorecard:
    """Computes and caches the 4 effectiveness scores.

    Call ``compute()`` periodically (e.g., on trade close + hourly).
    """

    def __init__(
        self,
        journal: Any = None,
        quality_tracker: Any = None,
        risk_manager: Any = None,
        agent_registry: Any = None,
        event_bus: Any = None,
    ) -> None:
        self._journal = journal
        self._quality_tracker = quality_tracker
        self._risk_manager = risk_manager
        self._agent_registry = agent_registry
        self._event_bus = event_bus
        self._last_scores: dict[str, float] | None = None

    def compute(self) -> dict[str, float]:
        """Compute all 4 scores and the weighted total.

        Returns dict with keys:
            edge_quality, execution_quality, risk_discipline,
            operational_integrity, total
        """
        edge = self._compute_edge_quality()
        execution = self._compute_execution_quality()
        risk = self._compute_risk_discipline()
        ops = self._compute_operational_integrity()

        total = (
            edge * WEIGHTS["edge_quality"]
            + execution * WEIGHTS["execution_quality"]
            + risk * WEIGHTS["risk_discipline"]
            + ops * WEIGHTS["operational_integrity"]
        )

        self._last_scores = {
            "edge_quality": round(edge, 1),
            "execution_quality": round(execution, 1),
            "risk_discipline": round(risk, 1),
            "operational_integrity": round(ops, 1),
            "total": round(total, 1),
        }

        # Emit Prometheus
        self._emit_metrics()

        return self._last_scores

    @property
    def last_scores(self) -> dict[str, float] | None:
        """Return the most recently computed scores, or None."""
        return self._last_scores

    # ------------------------------------------------------------------
    # Score 1: Edge Quality (0-10)
    # ------------------------------------------------------------------

    def _compute_edge_quality(self) -> float:
        """Win rate vs target, profit factor, avg R, confidence calibration."""
        if self._journal is None:
            return 5.0  # Neutral default

        try:
            # Get aggregate stats from journal
            stats = self._journal.get_aggregate_stats()
            if not stats or stats.get("total_trades", 0) < 5:
                return 5.0  # Not enough data

            # Win rate vs target (assume 0.50 target)
            wr_score = self._scale(stats.get("win_rate", 0) / 0.50, 0.6, 1.2)
            # Profit factor
            pf_score = self._scale(stats.get("profit_factor", 1.0), 1.0, 2.0)
            # Avg R-multiple
            r_score = self._scale(stats.get("avg_r_multiple", 0), 0.0, 0.5)
            # Confidence calibration (lower brier = better)
            brier = stats.get("brier_score", 0.25)
            cal_score = self._scale(1.0 - brier / 0.30, 0.0, 1.0)

            return (wr_score + pf_score + r_score + cal_score) / 4 * 10
        except Exception:
            logger.debug("Edge quality computation failed", exc_info=True)
            return 5.0

    # ------------------------------------------------------------------
    # Score 2: Execution Quality (0-10)
    # ------------------------------------------------------------------

    def _compute_execution_quality(self) -> float:
        """Slippage, fill rate, adverse selection, management efficiency."""
        if self._quality_tracker is None:
            return 5.0

        try:
            slip = self._quality_tracker.avg_slippage_bps
            fill = self._quality_tracker.fill_rate

            slip_score = self._scale(1.0 - slip / 15.0, 0.0, 1.0)
            fill_score = self._scale(fill, 0.85, 1.0)

            return (slip_score + fill_score) / 2 * 10
        except Exception:
            return 5.0

    # ------------------------------------------------------------------
    # Score 3: Risk Discipline (0-10)
    # ------------------------------------------------------------------

    def _compute_risk_discipline(self) -> float:
        """Drawdown vs limit, sizing adherence, CB trips, override rate."""
        # Start at 10.0, deduct for violations
        score = 10.0

        try:
            if self._risk_manager:
                # Drawdown deduction
                dd_pct = getattr(self._risk_manager, "current_drawdown_pct", 0.0)
                dd_limit = getattr(self._risk_manager, "max_drawdown_pct", 0.15)
                if dd_limit > 0:
                    dd_ratio = dd_pct / dd_limit
                    if dd_ratio > 0.75:
                        score -= 4.0
                    elif dd_ratio > 0.50:
                        score -= 2.0
                    elif dd_ratio > 0.25:
                        score -= 1.0

                # Circuit breaker trips
                trips = getattr(self._risk_manager, "circuit_breaker_trips_today", 0)
                score -= min(trips * 2.0, 6.0)
        except Exception:
            pass

        return max(0.0, min(10.0, score))

    # ------------------------------------------------------------------
    # Score 4: Operational Integrity (0-10)
    # ------------------------------------------------------------------

    def _compute_operational_integrity(self) -> float:
        """Agent health, data freshness, DLQ size, reconciliation drift."""
        score = 10.0

        try:
            # Agent health
            if self._agent_registry:
                health = self._agent_registry.health_check_all()
                total = len(health)
                healthy = sum(1 for h in health.values() if h.healthy)
                if total > 0:
                    health_ratio = healthy / total
                    if health_ratio < 0.67:
                        score -= 4.0
                    elif health_ratio < 0.83:
                        score -= 2.0
                    elif health_ratio < 1.0:
                        score -= 1.0

            # Event bus DLQ
            if self._event_bus:
                dlq_size = len(getattr(self._event_bus, "dead_letters", []))
                if dlq_size > 20:
                    score -= 4.0
                elif dlq_size > 5:
                    score -= 2.0
                elif dlq_size > 0:
                    score -= 1.0
        except Exception:
            pass

        return max(0.0, min(10.0, score))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scale(value: float, low: float, high: float) -> float:
        """Scale a value to 0.0-1.0 range between low and high."""
        if high <= low:
            return 0.5
        clamped = max(low, min(high, value))
        return (clamped - low) / (high - low)

    def _emit_metrics(self) -> None:
        """Push scores to Prometheus."""
        if self._last_scores is None:
            return
        try:
            from agentic_trading.observability.metrics import (
                update_effectiveness_score,
            )

            for key, val in self._last_scores.items():
                update_effectiveness_score(key, val)
        except Exception:
            pass
