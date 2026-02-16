"""Risk Gate Agent.

Wraps the RiskManager and GovernanceGate into a single agent that
provides pre-trade risk checking and governance evaluation.

Unlike the Execution and MarketIntelligence agents which are event-driven,
the RiskGateAgent is a passive service agent (no periodic loop, no event
subscriptions). It exposes the RiskManager and GovernanceGate for use by
other agents and the orchestrator.

The RiskManager's circuit breakers and kill switch are monitored by the
GovernanceCanary agent separately.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentCapabilities
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


class RiskGateAgent(BaseAgent):
    """Pre-trade risk gating and governance evaluation.

    Orchestrates:
    - RiskManager (pre/post-trade checks, circuit breakers, kill switch)
    - GovernanceGate (maturity, health, impact, drift, tokens)

    Usage::

        agent = RiskGateAgent(
            event_bus=event_bus,
            risk_config=settings.risk,
            governance_config=settings.governance,
            instruments=ctx.instruments,
        )
        await agent.start()
        # Access components:
        risk_mgr = agent.risk_manager
        gov_gate = agent.governance_gate
    """

    def __init__(
        self,
        event_bus: IEventBus,
        risk_config: Any = None,
        governance_config: Any = None,
        instruments: dict[str, Any] | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, interval=0)
        self._event_bus = event_bus
        self._risk_config = risk_config
        self._governance_config = governance_config
        self._instruments = instruments or {}

        # Components -- created during start
        self._risk_manager: Any = None
        self._governance_gate: Any = None

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.RISK_GATE

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=[],
            publishes_to=["risk", "governance"],
            description="Pre-trade risk checks and governance evaluation",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        from agentic_trading.risk.manager import RiskManager

        self._risk_manager = RiskManager(
            config=self._risk_config,
            event_bus=self._event_bus,
            instruments=self._instruments,
        )
        logger.info("RiskGateAgent: RiskManager initialized")

        # Wire governance if enabled
        if self._governance_config is not None and self._governance_config.enabled:
            try:
                from agentic_trading.governance.drift_detector import DriftDetector
                from agentic_trading.governance.gate import GovernanceGate
                from agentic_trading.governance.health_score import HealthTracker
                from agentic_trading.governance.impact_classifier import ImpactClassifier
                from agentic_trading.governance.maturity import MaturityManager
                from agentic_trading.governance.tokens import TokenManager

                maturity_mgr = MaturityManager(
                    self._governance_config.maturity,
                )
                health_tracker = HealthTracker(
                    self._governance_config.health_score,
                )
                impact_clf = ImpactClassifier(
                    self._governance_config.impact_classifier,
                )
                drift_det = DriftDetector(
                    self._governance_config.drift_detector,
                )
                token_mgr = TokenManager(
                    self._governance_config.execution_tokens,
                )

                self._governance_gate = GovernanceGate(
                    config=self._governance_config,
                    maturity=maturity_mgr,
                    health=health_tracker,
                    impact=impact_clf,
                    drift=drift_det,
                    tokens=(
                        token_mgr
                        if self._governance_config.execution_tokens.require_tokens
                        else None
                    ),
                    event_bus=self._event_bus,
                )
                logger.info(
                    "RiskGateAgent: GovernanceGate initialized "
                    "(maturity, health, impact, drift, tokens)",
                )
            except Exception:
                logger.warning(
                    "RiskGateAgent: GovernanceGate init failed",
                    exc_info=True,
                )

    async def _on_stop(self) -> None:
        logger.info("RiskGateAgent stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def risk_manager(self) -> Any:
        """Access the underlying RiskManager."""
        return self._risk_manager

    @property
    def governance_gate(self) -> Any:
        """Access the underlying GovernanceGate (may be None)."""
        return self._governance_gate
