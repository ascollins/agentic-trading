"""Strategy lifecycle state machine with evidence-gated promotion.

Manages strategy progression through:
  Candidate → Backtest → EvalPack → Paper → Limited → Scale

Demotion can happen at any live stage when automated triggers fire.
All transitions emit MaturityTransition events and require evidence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import (
    AgentType,
    MaturityLevel,
    StrategyStage,
)
from agentic_trading.core.events import (
    AgentCapabilities,
    MaturityTransition,
)
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


# Evidence thresholds (configurable via constructor)
DEFAULT_EVIDENCE_GATES: dict[StrategyStage, dict[str, float]] = {
    StrategyStage.BACKTEST: {
        # CANDIDATE → BACKTEST: backtest must be submitted (no metric gates)
    },
    StrategyStage.EVAL_PACK: {
        "min_trades": 50,
        "min_sharpe": 0.5,
        "min_profit_factor": 1.1,
        "max_drawdown_pct": 20.0,
    },
    StrategyStage.PAPER: {
        # Requires operator approval (L2_OPERATOR)
    },
    StrategyStage.LIMITED: {
        "min_trades": 100,
        "min_quality_score": 60,
        "max_sharpe_drift_sigma": 1.0,  # live Sharpe within 1σ of backtest
    },
    StrategyStage.SCALE: {
        "min_days_at_limited": 30,
        "min_quality_score": 70,
        "max_incidents": 0,
        # Requires operator approval (L3_RISK)
    },
}

# Demotion triggers (any one fires → demote)
DEFAULT_DEMOTION_TRIGGERS: dict[str, float] = {
    "max_drawdown_pct": 10.0,
    "max_loss_streak": 10,
    "min_quality_score": 50,
    "max_drift_pct": 30.0,
}

# Scorecard-based automation thresholds (spec §7.2).
# Each entry: metric name → (threshold, comparison, consecutive_days_required)
# "lt" = demote when metric < threshold for N consecutive days.
# "gt" = demote when metric > threshold for N consecutive days.
DEFAULT_SCORECARD_TRIGGERS: dict[str, tuple[float, str, int]] = {
    "edge_quality": (3.0, "lt", 5),
    "hit_rate": (0.35, "lt", 7),
    "sharpe_ratio": (0.0, "lt", 5),
    "profit_factor": (0.8, "lt", 5),
}

# Promotion thresholds: sustained good metrics can auto-promote
DEFAULT_SCORECARD_PROMOTION: dict[str, tuple[float, str, int]] = {
    "edge_quality": (7.0, "gt", 10),
    "hit_rate": (0.55, "gt", 10),
    "sharpe_ratio": (1.5, "gt", 10),
    "profit_factor": (1.5, "gt", 10),
}

# Mapping: StrategyStage → MaturityLevel
STAGE_TO_MATURITY: dict[StrategyStage, MaturityLevel] = {
    StrategyStage.CANDIDATE: MaturityLevel.L0_SHADOW,
    StrategyStage.BACKTEST: MaturityLevel.L0_SHADOW,
    StrategyStage.EVAL_PACK: MaturityLevel.L1_PAPER,
    StrategyStage.PAPER: MaturityLevel.L2_GATED,
    StrategyStage.LIMITED: MaturityLevel.L3_CONSTRAINED,
    StrategyStage.SCALE: MaturityLevel.L4_AUTONOMOUS,
    StrategyStage.DEMOTED: MaturityLevel.L1_PAPER,
}


class StrategyLifecycleManager(BaseAgent):
    """Manages strategy lifecycle with evidence-gated promotion.

    Periodic agent that:
      1. Subscribes to journal close events to collect evidence
      2. Checks promotion eligibility on each cycle
      3. Checks demotion triggers on each cycle
      4. Emits MaturityTransition events on state changes
    """

    def __init__(
        self,
        event_bus: IEventBus,
        journal: Any,
        governance_gate: Any,
        *,
        evidence_gates: dict | None = None,
        demotion_triggers: dict | None = None,
        scorecard_triggers: dict | None = None,
        scorecard_promotion: dict | None = None,
        scorecard: Any = None,
        interval: float = 60.0,  # Check every 60s
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, interval=interval)
        self._event_bus = event_bus
        self._journal = journal
        self._governance_gate = governance_gate
        self._evidence_gates = evidence_gates or DEFAULT_EVIDENCE_GATES
        self._demotion_triggers = demotion_triggers or DEFAULT_DEMOTION_TRIGGERS
        self._scorecard_triggers = scorecard_triggers or DEFAULT_SCORECARD_TRIGGERS
        self._scorecard_promotion = scorecard_promotion or DEFAULT_SCORECARD_PROMOTION
        self._scorecard = scorecard  # DailyEffectivenessScorecard instance

        # strategy_id → current stage
        self._stages: dict[str, StrategyStage] = {}
        # strategy_id → promotion history
        self._promotion_history: dict[str, list[dict]] = {}
        # Scorecard consecutive day counters:
        # strategy_id → metric_name → consecutive_days_triggered
        self._demotion_streak: dict[str, dict[str, int]] = {}
        self._promotion_streak: dict[str, dict[str, int]] = {}

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CUSTOM

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["execution"],
            publishes_to=["governance"],
            description="Strategy lifecycle state machine with evidence-gated promotion",
        )

    def register_strategy(
        self,
        strategy_id: str,
        initial_stage: StrategyStage = StrategyStage.CANDIDATE,
    ) -> None:
        """Register a strategy with its initial lifecycle stage."""
        self._stages[strategy_id] = initial_stage
        self._promotion_history.setdefault(strategy_id, [])
        logger.info(
            "Strategy %s registered at stage %s",
            strategy_id,
            initial_stage.value,
        )

    def get_stage(self, strategy_id: str) -> StrategyStage | None:
        """Return the current lifecycle stage for a strategy."""
        return self._stages.get(strategy_id)

    def get_all_stages(self) -> dict[str, str]:
        """Return all strategy stages as {strategy_id: stage_value}."""
        return {sid: stage.value for sid, stage in self._stages.items()}

    def get_promotion_history(self, strategy_id: str) -> list[dict]:
        """Return the promotion history for a strategy."""
        return self._promotion_history.get(strategy_id, [])

    # ------------------------------------------------------------------
    # Periodic work
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Check all strategies for promotion eligibility and demotion triggers."""
        for strategy_id, stage in list(self._stages.items()):
            # Skip non-live stages
            if stage in (StrategyStage.CANDIDATE, StrategyStage.BACKTEST):
                continue

            # Check demotion triggers for live strategies
            if stage in (
                StrategyStage.PAPER,
                StrategyStage.LIMITED,
                StrategyStage.SCALE,
            ):
                evidence = self._collect_evidence(strategy_id)
                if self._should_demote(strategy_id, evidence):
                    await self._transition(
                        strategy_id,
                        StrategyStage.DEMOTED,
                        reason="automated_demotion_trigger",
                        evidence=evidence,
                    )
                    continue

                # Scorecard-based automation (spec §7.2)
                scorecard_data = self._collect_scorecard(strategy_id)
                if scorecard_data:
                    # Check scorecard demotion
                    demotion_reason = self._check_scorecard_demotion(
                        strategy_id, scorecard_data,
                    )
                    if demotion_reason:
                        await self._transition(
                            strategy_id,
                            StrategyStage.DEMOTED,
                            reason=demotion_reason,
                            evidence={**evidence, **scorecard_data},
                        )
                        continue

                    # Check scorecard auto-promotion
                    promotion_reason = self._check_scorecard_promotion(
                        strategy_id, scorecard_data,
                    )
                    if promotion_reason:
                        next_stage = self._next_stage(stage)
                        if next_stage is not None:
                            await self._transition(
                                strategy_id,
                                next_stage,
                                reason=promotion_reason,
                                evidence={**evidence, **scorecard_data},
                            )

    # ------------------------------------------------------------------
    # Promotion (called externally or via API)
    # ------------------------------------------------------------------

    async def request_promotion(
        self,
        strategy_id: str,
        *,
        operator_id: str = "",
    ) -> dict[str, Any]:
        """Request promotion to the next stage.

        Returns dict with 'approved' bool and 'reason' string.
        """
        current = self._stages.get(strategy_id)
        if current is None:
            return {"approved": False, "reason": "strategy_not_registered"}

        next_stage = self._next_stage(current)
        if next_stage is None:
            return {"approved": False, "reason": "already_at_max_stage"}

        evidence = self._collect_evidence(strategy_id)
        gate = self._evidence_gates.get(next_stage, {})

        # Check evidence requirements
        for metric, threshold in gate.items():
            actual = evidence.get(metric)
            if actual is None:
                return {
                    "approved": False,
                    "reason": f"missing_evidence: {metric}",
                }
            if metric.startswith("min_") and actual < threshold:
                return {
                    "approved": False,
                    "reason": f"{metric}={actual} < {threshold}",
                }
            if metric.startswith("max_") and actual > threshold:
                return {
                    "approved": False,
                    "reason": f"{metric}={actual} > {threshold}",
                }

        await self._transition(
            strategy_id,
            next_stage,
            reason=f"promoted_by_{operator_id or 'system'}",
            evidence=evidence,
        )
        return {"approved": True, "reason": "all_evidence_gates_passed"}

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    def _collect_evidence(self, strategy_id: str) -> dict[str, float]:
        """Collect current metrics from journal and drift detector."""
        evidence: dict[str, float] = {}
        try:
            stats = self._journal.get_strategy_stats(strategy_id)
            if stats:
                evidence["min_trades"] = stats.get("total_trades", 0)
                evidence["min_quality_score"] = stats.get("quality_score", 0)
                evidence["min_sharpe"] = stats.get("sharpe_ratio", 0)
                evidence["min_profit_factor"] = stats.get("profit_factor", 0)
                evidence["max_drawdown_pct"] = stats.get("max_drawdown_pct", 0)
                evidence["max_loss_streak"] = stats.get("loss_streak", 0)
        except Exception:
            logger.debug("Failed to collect evidence for %s", strategy_id)

        # Drift from governance gate
        try:
            drift_status = self._governance_gate.drift.get_status(strategy_id)
            max_dev = 0.0
            for m in drift_status.get("metrics", {}).values():
                if m.get("deviation_pct") is not None:
                    max_dev = max(max_dev, m["deviation_pct"])
            evidence["max_drift_pct"] = max_dev
        except Exception:
            pass

        return evidence

    def _should_demote(
        self, strategy_id: str, evidence: dict[str, float]
    ) -> bool:
        """Check if any demotion trigger fires."""
        for trigger, threshold in self._demotion_triggers.items():
            actual = evidence.get(trigger)
            if actual is None:
                continue
            if trigger.startswith("max_") and actual > threshold:
                logger.warning(
                    "Demotion trigger: %s=%s > %s for %s",
                    trigger, actual, threshold, strategy_id,
                )
                return True
            if trigger.startswith("min_") and actual < threshold:
                logger.warning(
                    "Demotion trigger: %s=%s < %s for %s",
                    trigger, actual, threshold, strategy_id,
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Scorecard automation (spec §7.2)
    # ------------------------------------------------------------------

    def _collect_scorecard(self, strategy_id: str) -> dict[str, float]:
        """Collect current scorecard metrics for a strategy."""
        if self._scorecard is None:
            return {}
        try:
            if hasattr(self._scorecard, "compute"):
                result = self._scorecard.compute(strategy_id)
                if isinstance(result, dict):
                    return result
                # If it's a Pydantic model, dump it
                if hasattr(result, "model_dump"):
                    return {
                        k: float(v) for k, v in result.model_dump().items()
                        if isinstance(v, (int, float))
                    }
            return {}
        except Exception:
            logger.debug(
                "Failed to collect scorecard for %s", strategy_id,
                exc_info=True,
            )
            return {}

    def _check_scorecard_demotion(
        self, strategy_id: str, scorecard: dict[str, float],
    ) -> str:
        """Check if scorecard metrics warrant demotion.

        Returns a reason string if demotion should fire, else empty string.
        """
        streaks = self._demotion_streak.setdefault(strategy_id, {})

        for metric, (threshold, comparison, days_required) in self._scorecard_triggers.items():
            value = scorecard.get(metric)
            if value is None:
                streaks.pop(metric, None)
                continue

            triggered = (
                (comparison == "lt" and value < threshold)
                or (comparison == "gt" and value > threshold)
            )

            if triggered:
                streaks[metric] = streaks.get(metric, 0) + 1
                if streaks[metric] >= days_required:
                    return (
                        f"scorecard_demotion: {metric}={value:.3f} "
                        f"{comparison} {threshold} for {streaks[metric]} cycles"
                    )
            else:
                streaks[metric] = 0

        return ""

    def _check_scorecard_promotion(
        self, strategy_id: str, scorecard: dict[str, float],
    ) -> str:
        """Check if scorecard metrics warrant auto-promotion.

        Returns a reason string if promotion should fire, else empty string.
        All promotion thresholds must be satisfied simultaneously.
        """
        streaks = self._promotion_streak.setdefault(strategy_id, {})
        all_met = True
        min_days = 0

        for metric, (threshold, comparison, days_required) in self._scorecard_promotion.items():
            value = scorecard.get(metric)
            if value is None:
                all_met = False
                streaks.pop(metric, None)
                continue

            met = (
                (comparison == "gt" and value > threshold)
                or (comparison == "lt" and value < threshold)
            )

            if met:
                streaks[metric] = streaks.get(metric, 0) + 1
            else:
                streaks[metric] = 0
                all_met = False

            if streaks.get(metric, 0) < days_required:
                all_met = False
            min_days = max(min_days, days_required)

        if all_met:
            # Reset streaks after promotion
            for metric in self._scorecard_promotion:
                streaks[metric] = 0
            return (
                f"scorecard_auto_promotion: all metrics sustained "
                f"for {min_days}+ cycles"
            )
        return ""

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def _transition(
        self,
        strategy_id: str,
        new_stage: StrategyStage,
        reason: str,
        evidence: dict | None = None,
    ) -> None:
        old_stage = self._stages.get(strategy_id, StrategyStage.CANDIDATE)
        self._stages[strategy_id] = new_stage

        # Update maturity level in GovernanceGate
        new_maturity = STAGE_TO_MATURITY[new_stage]
        old_maturity = STAGE_TO_MATURITY[old_stage]
        if new_maturity != old_maturity:
            try:
                self._governance_gate.maturity.set_level(strategy_id, new_maturity)
            except Exception:
                logger.debug(
                    "Could not update maturity for %s", strategy_id,
                    exc_info=True,
                )

        # Record in history
        self._promotion_history.setdefault(strategy_id, []).append({
            "from": old_stage.value,
            "to": new_stage.value,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence": evidence or {},
        })

        # Emit event
        event = MaturityTransition(
            strategy_id=strategy_id,
            from_level=old_maturity,
            to_level=new_maturity,
            reason=f"{old_stage.value} → {new_stage.value}: {reason}",
            metrics_snapshot=evidence or {},
        )
        try:
            await self._event_bus.publish("governance", event)
        except Exception:
            logger.error("Failed to publish MaturityTransition", exc_info=True)

        logger.info(
            "Strategy %s transitioned: %s → %s (%s)",
            strategy_id,
            old_stage.value,
            new_stage.value,
            reason,
        )

    @staticmethod
    def _next_stage(current: StrategyStage) -> StrategyStage | None:
        """Return the next stage in the progression, or None if at max."""
        progression = [
            StrategyStage.CANDIDATE,
            StrategyStage.BACKTEST,
            StrategyStage.EVAL_PACK,
            StrategyStage.PAPER,
            StrategyStage.LIMITED,
            StrategyStage.SCALE,
        ]
        try:
            idx = progression.index(current)
            if idx + 1 < len(progression):
                return progression[idx + 1]
        except ValueError:
            pass
        return None
