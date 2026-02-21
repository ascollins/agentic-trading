"""Daily Effectiveness Scorecard — 4 scores, 0-10 scale (spec §10).

Aggregates data from:
  - TradeJournal (edge quality: information ratio, hit rate, Sharpe)
  - ExecutionQualityTracker (slippage, participation, latency)
  - RiskManager / CircuitBreakers (utilisation, breaches, VaR coverage)
  - AgentRegistry / EventBus / Recon (health, breaks, incidents, canary)

Each score is weighted to produce a single daily effectiveness number.

Formulas aligned with the institutional design specification §10:

  edge_quality = 0.5 * clamp(information_ratio / 2, 0, 10)
               + 0.3 * clamp(hit_rate * 10, 0, 10)
               + 0.2 * clamp(sharpe_ratio, 0, 10)

  execution_quality = (slippage_score + participation_score + latency_score) / 3

  risk_discipline = (utilisation_score + breach_penalty + var_score) / 3

  operational_integrity = (break_score + data_quality_score
                           + incident_response_score + canary_health_score) / 4
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

    Parameters
    ----------
    journal:
        TradeJournal with ``get_aggregate_stats()`` returning a dict
        with keys: ``win_rate``, ``sharpe_ratio``, ``information_ratio``,
        ``total_trades``.
    quality_tracker:
        :class:`ExecutionQualityTracker` with score properties.
    risk_manager:
        RiskManager with exposure/drawdown attributes.
    agent_registry:
        AgentRegistry with ``health_check_all()``.
    event_bus:
        Event bus with ``dead_letters`` attribute.
    recon_provider:
        Optional callable returning ``{"break_count": int}``.
    incident_provider:
        Optional callable returning
        ``{"avg_resolution_hours": float, "open_count": int}``.
    canary_provider:
        Optional callable returning ``{"healthy": bool}``.
    """

    def __init__(
        self,
        journal: Any = None,
        quality_tracker: Any = None,
        risk_manager: Any = None,
        agent_registry: Any = None,
        event_bus: Any = None,
        recon_provider: Any = None,
        incident_provider: Any = None,
        canary_provider: Any = None,
    ) -> None:
        self._journal = journal
        self._quality_tracker = quality_tracker
        self._risk_manager = risk_manager
        self._agent_registry = agent_registry
        self._event_bus = event_bus
        self._recon_provider = recon_provider
        self._incident_provider = incident_provider
        self._canary_provider = canary_provider
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
    # Score 1: Edge Quality (0-10)  — spec §10.1
    # ------------------------------------------------------------------

    def _compute_edge_quality(self) -> float:
        """Information ratio, hit rate, and Sharpe ratio.

        Formula:
            0.5 * clamp(information_ratio / 2, 0, 10)
          + 0.3 * clamp(hit_rate * 10, 0, 10)
          + 0.2 * clamp(sharpe_ratio, 0, 10)
        """
        if self._journal is None:
            return 5.0  # Neutral default

        try:
            stats = self._journal.get_aggregate_stats()
            if not stats or stats.get("total_trades", 0) < 5:
                return 5.0  # Not enough data

            # Information ratio
            ir = stats.get("information_ratio", 0.0)
            if ir == 0.0:
                # Fall back: compute from mean_return / std_return if available
                mean_ret = stats.get("mean_return", 0.0)
                std_ret = stats.get("std_return", 0.0)
                if std_ret > 0:
                    ir = mean_ret / std_ret

            ir_score = self._clamp(ir / 2.0, 0.0, 10.0)

            # Hit rate
            hit_rate = stats.get("win_rate", 0.0)
            hr_score = self._clamp(hit_rate * 10.0, 0.0, 10.0)

            # Sharpe ratio
            sharpe = stats.get("sharpe_ratio", stats.get("sharpe", 0.0))
            sharpe_score = self._clamp(sharpe, 0.0, 10.0)

            return 0.5 * ir_score + 0.3 * hr_score + 0.2 * sharpe_score

        except Exception:
            logger.debug("Edge quality computation failed", exc_info=True)
            return 5.0

    # ------------------------------------------------------------------
    # Score 2: Execution Quality (0-10)  — spec §10.2
    # ------------------------------------------------------------------

    def _compute_execution_quality(self) -> float:
        """Slippage, participation rate, and latency scores.

        Formula:
            (slippage_score + participation_score + latency_score) / 3

        Falls back to slippage + fill_rate (legacy) if the quality
        tracker provides those simpler properties.
        """
        if self._quality_tracker is None:
            return 5.0

        try:
            # Try the new spec-aligned score properties first
            if hasattr(self._quality_tracker, "slippage_score"):
                slip_score = self._quality_tracker.slippage_score
                part_score = self._quality_tracker.participation_score
                lat_score = self._quality_tracker.latency_score
                return (slip_score + part_score + lat_score) / 3.0

            # Legacy fallback
            slip = self._quality_tracker.avg_slippage_bps
            fill = self._quality_tracker.fill_rate

            slip_score = self._scale(1.0 - slip / 15.0, 0.0, 1.0) * 10.0
            fill_score = self._scale(fill, 0.85, 1.0) * 10.0

            return (slip_score + fill_score) / 2.0
        except Exception:
            return 5.0

    # ------------------------------------------------------------------
    # Score 3: Risk Discipline (0-10)  — spec §10.3
    # ------------------------------------------------------------------

    def _compute_risk_discipline(self) -> float:
        """Utilisation ratio, breach count penalty, and VaR coverage.

        Formula:
            utilisation_score = 10 * (1 - current_exposure / max_exposure)
            breach_penalty    = max(0, 10 - breach_count * penalty_per_breach)
            var_score         = 10 * (1 - max(0, realised_loss - var_limit) / var_limit)
            risk_discipline   = (utilisation_score + breach_penalty + var_score) / 3
        """
        if self._risk_manager is None:
            return 5.0

        try:
            # Utilisation score
            current_exposure = getattr(self._risk_manager, "current_exposure", 0.0)
            max_exposure = getattr(self._risk_manager, "max_exposure", 0.0)
            if max_exposure > 0:
                utilisation = current_exposure / max_exposure
                utilisation_score = 10.0 * max(0.0, 1.0 - utilisation)
            else:
                # Fall back to drawdown-based scoring
                dd_pct = getattr(self._risk_manager, "current_drawdown_pct", 0.0)
                dd_limit = getattr(self._risk_manager, "max_drawdown_pct", 0.15)
                if dd_limit > 0:
                    utilisation_score = 10.0 * max(0.0, 1.0 - dd_pct / dd_limit)
                else:
                    utilisation_score = 10.0

            # Breach penalty (2 points per breach, max deduction = 10)
            breach_count = getattr(
                self._risk_manager, "circuit_breaker_trips_today", 0
            )
            breach_penalty = max(0.0, 10.0 - breach_count * 2.0)

            # VaR coverage score
            var_limit = getattr(self._risk_manager, "var_limit", 0.0)
            realised_loss = getattr(self._risk_manager, "realised_loss_today", 0.0)
            if var_limit > 0 and realised_loss > 0:
                excess = max(0.0, realised_loss - var_limit) / var_limit
                var_score = 10.0 * max(0.0, 1.0 - excess)
            else:
                var_score = 10.0  # No breach

            return (utilisation_score + breach_penalty + var_score) / 3.0

        except Exception:
            logger.debug("Risk discipline computation failed", exc_info=True)
            return 5.0

    # ------------------------------------------------------------------
    # Score 4: Operational Integrity (0-10)  — spec §10.4
    # ------------------------------------------------------------------

    def _compute_operational_integrity(self) -> float:
        """Recon breaks, data quality incidents, incident response, canary.

        Formula:
            break_score     = 10 - min(10, break_count)
            data_quality_score = 10 - min(10, data_incident_count)
            incident_response_score = 10 - min(10, avg_resolution_hours)
            canary_health_score     = 10 if canary healthy else 0
            operational_integrity   = (break_score + data_quality_score
                                       + incident_response_score
                                       + canary_health_score) / 4
        """
        # 1. Reconciliation break score
        break_count = 0
        if self._recon_provider is not None:
            try:
                recon = self._recon_provider()
                break_count = recon.get("break_count", 0)
            except Exception:
                pass
        break_score = 10.0 - min(10.0, float(break_count))

        # 2. Data quality / DLQ score
        data_incidents = 0
        if self._event_bus is not None:
            try:
                data_incidents = len(getattr(self._event_bus, "dead_letters", []))
            except Exception:
                pass
        # Also count unhealthy agents as data quality incidents
        if self._agent_registry is not None:
            try:
                health = self._agent_registry.health_check_all()
                unhealthy = sum(1 for h in health.values() if not h.healthy)
                data_incidents += unhealthy
            except Exception:
                pass
        data_quality_score = 10.0 - min(10.0, float(data_incidents))

        # 3. Incident response time score
        avg_resolution_hours = 0.0
        if self._incident_provider is not None:
            try:
                incidents = self._incident_provider()
                avg_resolution_hours = incidents.get("avg_resolution_hours", 0.0)
            except Exception:
                pass
        incident_response_score = 10.0 - min(10.0, avg_resolution_hours)

        # 4. Canary health score
        canary_healthy = True
        if self._canary_provider is not None:
            try:
                canary = self._canary_provider()
                canary_healthy = canary.get("healthy", True)
            except Exception:
                pass
        canary_health_score = 10.0 if canary_healthy else 0.0

        return (
            break_score
            + data_quality_score
            + incident_response_score
            + canary_health_score
        ) / 4.0

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

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        """Clamp a value to [low, high]."""
        return max(low, min(high, value))

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
