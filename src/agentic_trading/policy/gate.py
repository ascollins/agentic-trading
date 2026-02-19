"""Unified policy gate — single-entry-point facade.

:class:`PolicyGate` composes the low-level governance components into a
high-level API that callers (execution engine, portfolio manager,
strategy runner) use to answer one question:

    "Should this action proceed, and at what size?"

It owns construction and wiring of:

- :class:`GovernanceGate` — maturity, health, impact, drift, token checks
- :class:`PolicyEngine` — declarative policy-as-code evaluation
- :class:`PolicyStore` — versioned policy management
- :class:`ApprovalManager` — multi-level approval workflows

Usage::

    from agentic_trading.policy.gate import PolicyGate

    gate = PolicyGate.from_config(config)
    decision = await gate.evaluate(
        strategy_id="trend_following",
        symbol="BTC/USDT",
        notional_usd=25_000,
    )
    if decision.action == GovernanceAction.ALLOW:
        # proceed with order
        pass
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.core.config import GovernanceConfig, RiskConfig
from agentic_trading.core.enums import GovernanceAction
from agentic_trading.core.events import GovernanceDecision

from .approval_manager import ApprovalManager
from .approval_models import ApprovalRule
from .default_policies import (
    build_post_trade_policies,
    build_pre_trade_policies,
    build_strategy_constraint_policies,
)
from .drift_detector import DriftDetector
from .engine import PolicyEngine
from .governance_gate import GovernanceGate
from .health_score import HealthTracker
from .impact_classifier import ImpactClassifier
from .maturity import MaturityManager
from .models import PolicyMode, PolicySet
from .store import PolicyStore
from .tokens import TokenManager

logger = logging.getLogger(__name__)


class PolicyGate:
    """High-level facade combining all governance and policy subsystems.

    Provides a single :meth:`evaluate` entry point that delegates to
    :class:`GovernanceGate` (which internally uses the policy engine,
    approval manager, and all other governance components).

    This is the **recommended** way to interact with governance from
    the execution engine and other pipeline stages.
    """

    def __init__(
        self,
        governance_gate: GovernanceGate,
        policy_engine: PolicyEngine,
        policy_store: PolicyStore,
        approval_manager: ApprovalManager | None = None,
    ) -> None:
        self._governance_gate = governance_gate
        self._policy_engine = policy_engine
        self._policy_store = policy_store
        self._approval_manager = approval_manager

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        governance_config: GovernanceConfig,
        risk_config: RiskConfig | None = None,
        *,
        event_bus: Any = None,
        approval_rules: list[ApprovalRule] | None = None,
        auto_approve_l1: bool = True,
        policy_persist_dir: str | None = None,
        policy_mode: PolicyMode = PolicyMode.ENFORCED,
    ) -> PolicyGate:
        """Construct a fully-wired PolicyGate from configuration.

        This is the preferred way to create a PolicyGate instance.
        It builds all sub-components and wires them together.

        Args:
            governance_config: Core governance configuration.
            risk_config: Optional risk config for default policy thresholds.
            event_bus: Optional event bus for publishing governance events.
            approval_rules: Optional approval rules for the approval manager.
            auto_approve_l1: Whether to auto-approve L1 escalation requests.
            policy_persist_dir: Optional directory for persisting policy sets.
            policy_mode: Enforcement mode for default policies.
        """
        # Build governance sub-components
        maturity = MaturityManager(governance_config.maturity)
        health = HealthTracker(governance_config.health_score)
        impact = ImpactClassifier(governance_config.impact_classifier)
        drift = DriftDetector(governance_config.drift_detector)
        tokens = TokenManager(governance_config.execution_tokens)

        # Build policy engine + store
        policy_engine = PolicyEngine()
        policy_store = PolicyStore(persist_dir=policy_persist_dir)

        # Register default policies
        pre_trade = build_pre_trade_policies(
            risk_config, mode=policy_mode,
        )
        post_trade = build_post_trade_policies(mode=policy_mode)
        strategy_constraints = build_strategy_constraint_policies(
            mode=policy_mode,
        )

        for ps in (pre_trade, post_trade, strategy_constraints):
            policy_engine.register(ps)
            policy_store.save(ps)

        # Load any persisted policies (may override defaults)
        if policy_persist_dir is not None:
            policy_store.load_from_dir()
            # Re-register active versions into the engine
            for set_id, ver in policy_store.active_sets.items():
                ps = policy_store.get_version(set_id, ver)
                if ps is not None:
                    policy_engine.register(ps)

        # Build approval manager
        approval_manager: ApprovalManager | None = None
        if approval_rules is not None:
            approval_manager = ApprovalManager(
                rules=approval_rules,
                event_bus=event_bus,
                auto_approve_l1=auto_approve_l1,
            )

        # Build governance gate (the core orchestrator)
        governance_gate = GovernanceGate(
            config=governance_config,
            maturity=maturity,
            health=health,
            impact=impact,
            drift=drift,
            tokens=tokens,
            event_bus=event_bus,
            policy_engine=policy_engine,
            approval_manager=approval_manager,
        )

        return cls(
            governance_gate=governance_gate,
            policy_engine=policy_engine,
            policy_store=policy_store,
            approval_manager=approval_manager,
        )

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
        """Evaluate all governance and policy checks.

        Delegates to :meth:`GovernanceGate.evaluate` which runs
        maturity, impact, approval, health, drift, token, and
        policy engine checks in sequence.

        Returns:
            :class:`GovernanceDecision` with action and sizing multiplier.
        """
        return await self._governance_gate.evaluate(
            strategy_id=strategy_id,
            symbol=symbol,
            notional_usd=notional_usd,
            portfolio_pct=portfolio_pct,
            is_reduce_only=is_reduce_only,
            leverage=leverage,
            existing_positions=existing_positions,
            trace_id=trace_id,
            token_id=token_id,
        )

    # ------------------------------------------------------------------
    # Post-trade
    # ------------------------------------------------------------------

    def record_trade_outcome(
        self,
        strategy_id: str,
        won: bool,
        r_multiple: float = 0.0,
    ) -> None:
        """Record a trade outcome to update health scores.

        Delegates to :meth:`GovernanceGate.record_trade_outcome`.
        """
        self._governance_gate.record_trade_outcome(strategy_id, won, r_multiple)

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def register_policy_set(
        self, policy_set: PolicySet, *, activate: bool = True
    ) -> None:
        """Register a policy set in both the engine and store."""
        self._policy_engine.register(policy_set)
        self._policy_store.save(policy_set, activate=activate)

    def set_policy_mode(self, set_id: str, mode: PolicyMode) -> bool:
        """Switch a policy set between shadow/enforced mode."""
        success = self._policy_store.set_mode(set_id, mode)
        if success:
            # Re-register in the engine with updated mode
            ps = self._policy_store.get_active(set_id)
            if ps is not None:
                self._policy_engine.register(ps)
        return success

    def rollback_policy(self, set_id: str) -> PolicySet | None:
        """Roll back a policy set to its previous version."""
        ps = self._policy_store.rollback(set_id)
        if ps is not None:
            self._policy_engine.register(ps)
        return ps

    # ------------------------------------------------------------------
    # Approval management
    # ------------------------------------------------------------------

    def add_approval_rule(self, rule: ApprovalRule) -> None:
        """Add an approval rule (only if approval manager is enabled)."""
        if self._approval_manager is not None:
            self._approval_manager.add_rule(rule)

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------

    @property
    def governance_gate(self) -> GovernanceGate:
        """Access the underlying governance gate."""
        return self._governance_gate

    @property
    def policy_engine(self) -> PolicyEngine:
        """Access the policy engine."""
        return self._policy_engine

    @property
    def policy_store(self) -> PolicyStore:
        """Access the policy store."""
        return self._policy_store

    @property
    def approval_manager(self) -> ApprovalManager | None:
        """Access the approval manager (may be None)."""
        return self._approval_manager

    @property
    def maturity(self) -> MaturityManager:
        """Shortcut to governance gate's maturity manager."""
        return self._governance_gate.maturity

    @property
    def health(self) -> HealthTracker:
        """Shortcut to governance gate's health tracker."""
        return self._governance_gate.health

    @property
    def drift(self) -> DriftDetector:
        """Shortcut to governance gate's drift detector."""
        return self._governance_gate.drift

    def get_sizing_multiplier(self, strategy_id: str) -> float:
        """Get the combined sizing multiplier for a strategy.

        Delegates to :meth:`GovernanceGate.get_sizing_multiplier`.
        """
        return self._governance_gate.get_sizing_multiplier(strategy_id)
