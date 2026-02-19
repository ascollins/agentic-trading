"""Governance gate — single orchestrator for all pre-execution checks.

The :class:`GovernanceGate` is the sole entry point called by the
:class:`~agentic_trading.execution.engine.ExecutionEngine` before
submitting orders.  It composes maturity, health, impact, drift,
and token checks into a single :class:`GovernanceDecision`.

Inspired by the Soteria Policy Decision Service (C5) which acts as
the central arbiter before any agent action is permitted.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentic_trading.core.config import GovernanceConfig
from agentic_trading.core.enums import GovernanceAction, ImpactTier, MaturityLevel
from agentic_trading.core.events import GovernanceDecision

from .drift_detector import DriftDetector
from .health_score import HealthTracker
from .impact_classifier import ImpactClassifier
from .maturity import MaturityManager
from .tokens import TokenManager

logger = logging.getLogger(__name__)


class GovernanceGate:
    """Single entry point for all governance checks.

    Called by :class:`ExecutionEngine` between deduplication and
    pre-trade risk check.

    Usage::

        gate = GovernanceGate(config, maturity, health, impact, drift, tokens)
        decision = await gate.evaluate(
            strategy_id="trend_following",
            symbol="BTC/USDT",
            notional_usd=25_000,
            ...
        )
        if decision.action == GovernanceAction.ALLOW:
            # proceed with order
            pass
    """

    def __init__(
        self,
        config: GovernanceConfig,
        maturity: MaturityManager,
        health: HealthTracker,
        impact: ImpactClassifier,
        drift: DriftDetector,
        tokens: TokenManager | None = None,
        event_bus: Any = None,
        policy_engine: Any = None,
        approval_manager: Any = None,
    ) -> None:
        self._config = config
        self._maturity = maturity
        self._health = health
        self._impact = impact
        self._drift = drift
        self._tokens = tokens
        self._event_bus = event_bus
        self._policy_engine = policy_engine
        self._approval_manager = approval_manager

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        strategy_id: str,
        symbol: str,
        notional_usd: float = 0.0,
        portfolio_pct: float = 0.0,
        is_reduce_only: bool = False,
        leverage: int = 1,
        existing_positions: int = 0,
        trace_id: str = "",
        token_id: str | None = None,
    ) -> GovernanceDecision:
        """Run all governance checks and return a composite decision.

        Steps:
            1. If governance disabled → ALLOW (multiplier=1.0)
            2. Check maturity level → BLOCK if L0/L1
            3. Classify impact tier
            4. Get health score → sizing multiplier
            5. Check drift → may REDUCE_SIZE or PAUSE
            6. Validate token (if required) → BLOCK if missing/invalid
            7. Compose final decision

        Returns:
            :class:`GovernanceDecision` with action and sizing multiplier.
        """
        t0 = time.monotonic()

        # 1. Governance disabled → pass-through
        if not self._config.enabled:
            return GovernanceDecision(
                strategy_id=strategy_id,
                symbol=symbol,
                action=GovernanceAction.ALLOW,
                reason="governance_disabled",
                sizing_multiplier=1.0,
                trace_id=trace_id,
            )

        details: dict[str, Any] = {}

        # 2. Maturity check
        level = self._maturity.get_level(strategy_id)
        if not self._maturity.can_execute(strategy_id):
            decision = GovernanceDecision(
                strategy_id=strategy_id,
                symbol=symbol,
                action=GovernanceAction.BLOCK,
                reason=f"maturity_level_{level.value}_insufficient",
                sizing_multiplier=0.0,
                maturity_level=level,
                trace_id=trace_id,
            )
            await self._publish_and_log(decision, t0)
            return decision

        maturity_cap = self._maturity.get_sizing_cap(strategy_id)
        details["maturity_cap"] = maturity_cap

        # 3. Impact classification
        impact_tier = self._impact.classify(
            symbol=symbol,
            notional_usd=notional_usd,
            portfolio_pct=portfolio_pct,
            is_reduce_only=is_reduce_only,
            leverage=leverage,
            existing_positions=existing_positions,
        )
        details["impact_tier"] = impact_tier.value

        # Block CRITICAL impact at L2 (gated) maturity
        if impact_tier == ImpactTier.CRITICAL and level == MaturityLevel.L2_GATED:
            decision = GovernanceDecision(
                strategy_id=strategy_id,
                symbol=symbol,
                action=GovernanceAction.BLOCK,
                reason="critical_impact_at_gated_maturity",
                sizing_multiplier=0.0,
                maturity_level=level,
                impact_tier=impact_tier,
                trace_id=trace_id,
                details=details,
            )
            await self._publish_and_log(decision, t0)
            return decision

        # 3.5 Approval workflow check (if configured)
        if self._approval_manager is not None:
            try:
                approval_context = {
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "notional_usd": notional_usd,
                    "impact_tier": impact_tier.value,
                    "maturity_level": level.value,
                    "leverage": leverage,
                    "portfolio_pct": portfolio_pct,
                    "is_reduce_only": is_reduce_only,
                }
                matching_rule = self._approval_manager.check_approval_required(
                    approval_context,
                )
                if matching_rule is not None:
                    # Request approval and block until resolved
                    from .approval_models import ApprovalTrigger

                    request = await self._approval_manager.request_approval(
                        strategy_id=strategy_id,
                        symbol=symbol,
                        trigger=ApprovalTrigger(matching_rule.trigger.value),
                        escalation_level=matching_rule.escalation_level,
                        notional_usd=notional_usd,
                        impact_tier=impact_tier.value,
                        reason=f"Rule '{matching_rule.name}' requires approval",
                        ttl_seconds=matching_rule.ttl_seconds,
                    )
                    details["approval_request_id"] = request.request_id
                    details["approval_rule"] = matching_rule.rule_id

                    # If auto-approved (L1), continue; otherwise BLOCK
                    from .approval_models import ApprovalStatus

                    if request.status != ApprovalStatus.APPROVED:
                        decision = GovernanceDecision(
                            strategy_id=strategy_id,
                            symbol=symbol,
                            action=GovernanceAction.BLOCK,
                            reason=f"pending_approval: {matching_rule.name}",
                            sizing_multiplier=0.0,
                            maturity_level=level,
                            impact_tier=impact_tier,
                            trace_id=trace_id,
                            details=details,
                        )
                        await self._publish_and_log(decision, t0)
                        return decision
                    details["approval_auto_approved"] = True
            except Exception:
                # B6 FIX: Fail-closed. Approval subsystem error = BLOCK.
                logger.error(
                    "Approval manager check failed — BLOCKING (fail-closed)",
                    exc_info=True,
                )
                decision = GovernanceDecision(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    action=GovernanceAction.BLOCK,
                    reason="approval_manager_error",
                    sizing_multiplier=0.0,
                    maturity_level=level,
                    impact_tier=impact_tier,
                    trace_id=trace_id,
                    details={**details, "error": "approval_manager_unavailable"},
                )
                await self._publish_and_log(decision, t0)
                return decision

        # 4. Health score → sizing multiplier
        health_score = self._health.get_score(strategy_id)
        health_mult = self._health.get_sizing_multiplier(strategy_id)
        details["health_score"] = round(health_score, 4)
        details["health_multiplier"] = round(health_mult, 4)

        # 5. Drift detection
        drift_alerts = self._drift.check_drift(strategy_id)
        drift_action = GovernanceAction.ALLOW
        drift_mult = 1.0

        if drift_alerts:
            # Take the most severe action from all drift alerts
            for alert in drift_alerts:
                if alert.action_taken == GovernanceAction.PAUSE:
                    drift_action = GovernanceAction.PAUSE
                    break
                if alert.action_taken == GovernanceAction.REDUCE_SIZE:
                    drift_action = GovernanceAction.REDUCE_SIZE
                    drift_mult = 0.5  # Halve sizing on drift

            details["drift_alerts"] = [
                {
                    "metric": a.metric_name,
                    "deviation_pct": a.deviation_pct,
                    "action": a.action_taken.value,
                }
                for a in drift_alerts
            ]

        if drift_action == GovernanceAction.PAUSE:
            decision = GovernanceDecision(
                strategy_id=strategy_id,
                symbol=symbol,
                action=GovernanceAction.PAUSE,
                reason="drift_threshold_exceeded",
                sizing_multiplier=0.0,
                maturity_level=level,
                impact_tier=impact_tier,
                health_score=health_score,
                trace_id=trace_id,
                details=details,
            )
            await self._publish_and_log(decision, t0)
            return decision

        # 6. Token validation (if required)
        if self._tokens is not None and self._config.execution_tokens.require_tokens:
            if token_id is None or not self._tokens.validate(token_id):
                decision = GovernanceDecision(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    action=GovernanceAction.BLOCK,
                    reason="missing_or_invalid_execution_token",
                    sizing_multiplier=0.0,
                    maturity_level=level,
                    impact_tier=impact_tier,
                    health_score=health_score,
                    trace_id=trace_id,
                    details=details,
                )
                await self._publish_and_log(decision, t0)
                return decision
            # Consume the token
            self._tokens.consume(token_id)
            details["token_consumed"] = token_id

        # 6.5 Policy engine evaluation (if configured)
        policy_mult = 1.0
        if self._policy_engine is not None:
            try:
                policy_context = {
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "order_notional_usd": notional_usd,
                    "position_pct_of_equity": portfolio_pct,
                    "projected_leverage": leverage,
                    "projected_exposure_pct": leverage,
                    "is_reduce_only": is_reduce_only,
                    "existing_positions": existing_positions,
                }
                for set_id in self._policy_engine.registered_sets:
                    pd = self._policy_engine.evaluate(set_id, policy_context)
                    details[f"policy_{set_id}"] = {
                        "passed": pd.all_passed,
                        "action": pd.action.value,
                        "failed_count": len(pd.failed_rules),
                        "shadow_count": len(pd.shadow_violations),
                    }
                    if not pd.all_passed:
                        policy_mult = min(policy_mult, pd.sizing_multiplier)
                        if pd.action in (
                            GovernanceAction.BLOCK,
                            GovernanceAction.KILL,
                        ):
                            decision = GovernanceDecision(
                                strategy_id=strategy_id,
                                symbol=symbol,
                                action=pd.action,
                                reason=f"policy_violation: {pd.reason}",
                                sizing_multiplier=0.0,
                                maturity_level=level,
                                impact_tier=impact_tier,
                                health_score=round(health_score, 4),
                                trace_id=trace_id,
                                details=details,
                            )
                            await self._publish_and_log(decision, t0)
                            return decision
            except Exception:
                # B5 FIX: Fail-closed. Policy engine error = BLOCK.
                logger.error(
                    "Policy engine evaluation failed — BLOCKING (fail-closed)",
                    exc_info=True,
                )
                decision = GovernanceDecision(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    action=GovernanceAction.BLOCK,
                    reason="policy_engine_error",
                    sizing_multiplier=0.0,
                    maturity_level=level,
                    impact_tier=impact_tier,
                    health_score=round(health_score, 4),
                    trace_id=trace_id,
                    details={**details, "error": "policy_engine_unavailable"},
                )
                await self._publish_and_log(decision, t0)
                return decision

        # 7. Compose final sizing multiplier
        # Multiplicative: maturity_cap × health_mult × drift_mult × policy_mult
        final_mult = maturity_cap * health_mult * drift_mult * policy_mult
        final_mult = max(0.0, min(1.0, final_mult))

        if final_mult < 1.0 and drift_action == GovernanceAction.REDUCE_SIZE:
            action = GovernanceAction.REDUCE_SIZE
            reason = "sizing_reduced_by_governance"
        elif final_mult < 1.0:
            action = GovernanceAction.REDUCE_SIZE
            reason = "sizing_capped_by_maturity_or_health"
        else:
            action = GovernanceAction.ALLOW
            reason = "all_checks_passed"

        decision = GovernanceDecision(
            strategy_id=strategy_id,
            symbol=symbol,
            action=action,
            reason=reason,
            sizing_multiplier=round(final_mult, 4),
            maturity_level=level,
            impact_tier=impact_tier,
            health_score=round(health_score, 4),
            trace_id=trace_id,
            details=details,
        )
        await self._publish_and_log(decision, t0)
        return decision

    # ------------------------------------------------------------------
    # Post-trade callback
    # ------------------------------------------------------------------

    def record_trade_outcome(
        self,
        strategy_id: str,
        won: bool,
        r_multiple: float = 0.0,
    ) -> None:
        """Record a trade outcome to update health scores.

        Should be called after each fill is processed.
        """
        self._health.record_outcome(strategy_id, won, r_multiple)

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------

    @property
    def maturity(self) -> MaturityManager:
        return self._maturity

    @property
    def health(self) -> HealthTracker:
        return self._health

    @property
    def impact(self) -> ImpactClassifier:
        return self._impact

    @property
    def drift(self) -> DriftDetector:
        return self._drift

    @property
    def tokens(self) -> TokenManager | None:
        return self._tokens

    @property
    def policy_engine(self) -> Any:
        """Access the policy engine (may be None)."""
        return self._policy_engine

    @property
    def approval_manager(self) -> Any:
        """Access the approval manager (may be None)."""
        return self._approval_manager

    def get_sizing_multiplier(self, strategy_id: str) -> float:
        """Get the combined sizing multiplier for a strategy.

        Used by PortfolioManager for position sizing.
        """
        health_mult = self._health.get_sizing_multiplier(strategy_id)
        maturity_cap = self._maturity.get_sizing_cap(strategy_id)
        return max(0.0, min(1.0, health_mult * maturity_cap))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _publish_and_log(
        self, decision: GovernanceDecision, t0: float
    ) -> None:
        """Publish decision event and log latency."""
        elapsed = time.monotonic() - t0
        logger.info(
            "Governance decision: strategy=%s symbol=%s action=%s "
            "mult=%.2f reason=%s (%.1fms)",
            decision.strategy_id,
            decision.symbol,
            decision.action.value,
            decision.sizing_multiplier,
            decision.reason,
            elapsed * 1000,
        )

        # Emit Prometheus governance metrics
        try:
            from agentic_trading.observability.metrics import (
                record_governance_decision,
                record_governance_block,
                record_governance_latency,
                update_health_score,
                update_maturity_level,
            )
            record_governance_decision(
                decision.strategy_id, decision.action.value,
            )
            record_governance_latency(elapsed)

            if decision.action in (
                GovernanceAction.BLOCK,
                GovernanceAction.PAUSE,
                GovernanceAction.KILL,
            ):
                record_governance_block(
                    decision.strategy_id, decision.reason,
                )

            if decision.health_score is not None:
                update_health_score(
                    decision.strategy_id, decision.health_score,
                )
            if decision.maturity_level is not None:
                _maturity_to_int = {
                    "L0_shadow": 0, "L1_paper": 1, "L2_gated": 2,
                    "L3_constrained": 3, "L4_autonomous": 4,
                }
                mat_val = decision.maturity_level.value if hasattr(decision.maturity_level, "value") else str(decision.maturity_level)
                update_maturity_level(
                    decision.strategy_id,
                    _maturity_to_int.get(mat_val, 0),
                )
        except Exception:
            pass  # Metrics emission should never break governance

        if self._event_bus is not None:
            try:
                await self._event_bus.publish("governance", decision)
            except Exception:
                logger.error(
                    "Failed to publish governance decision", exc_info=True
                )
