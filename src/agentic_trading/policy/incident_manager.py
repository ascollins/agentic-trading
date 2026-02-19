"""Incident lifecycle management with degraded mode transitions.

Subscribes to risk and system topics to detect incidents.
Manages degraded mode state: NORMAL → REDUCE_ONLY → NO_NEW_TRADES → KILLED.
Implements auto-triage rules and recovery criteria.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import (
    AgentType,
    DegradedMode,
    IncidentSeverity,
    IncidentStatus,
)
from agentic_trading.core.events import (
    AgentCapabilities,
    BaseEvent,
    CircuitBreakerEvent,
    KillSwitchEvent,
)
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


# Auto-triage rules: trigger_type → (severity, degraded_mode)
AUTO_TRIAGE_RULES: dict[str, tuple[IncidentSeverity, DegradedMode]] = {
    "circuit_breaker_single": (IncidentSeverity.LOW, DegradedMode.NORMAL),
    "circuit_breaker_daily_loss": (IncidentSeverity.HIGH, DegradedMode.NO_NEW_TRADES),
    "kill_switch": (IncidentSeverity.CRITICAL, DegradedMode.KILLED),
    "canary_unhealthy": (IncidentSeverity.MEDIUM, DegradedMode.REDUCE_ONLY),
    "exchange_disconnected": (IncidentSeverity.HIGH, DegradedMode.NO_NEW_TRADES),
    "reconciliation_drift": (IncidentSeverity.MEDIUM, DegradedMode.REDUCE_ONLY),
}

# Recovery criteria: severity → (cooldown_minutes, requires_approval)
RECOVERY_CRITERIA: dict[IncidentSeverity, tuple[int, bool]] = {
    IncidentSeverity.LOW: (0, False),        # Auto-resume
    IncidentSeverity.MEDIUM: (15, False),     # Auto-resume after cooldown
    IncidentSeverity.HIGH: (60, True),        # Operator approval
    IncidentSeverity.CRITICAL: (240, True),   # L3_RISK approval
}


class Incident:
    """Single incident record."""

    __slots__ = (
        "incident_id",
        "severity",
        "status",
        "trigger_type",
        "trigger_event_id",
        "description",
        "degraded_mode",
        "affected_strategies",
        "affected_symbols",
        "declared_at",
        "resolved_at",
        "auto_actions",
    )

    def __init__(
        self,
        incident_id: str,
        severity: IncidentSeverity,
        trigger_type: str,
        trigger_event_id: str,
        description: str,
    ) -> None:
        self.incident_id = incident_id
        self.severity = severity
        self.status = IncidentStatus.DETECTED
        self.trigger_type = trigger_type
        self.trigger_event_id = trigger_event_id
        self.description = description
        self.degraded_mode = DegradedMode.NORMAL
        self.affected_strategies: list[str] = []
        self.affected_symbols: list[str] = []
        self.declared_at = datetime.now(timezone.utc)
        self.resolved_at: datetime | None = None
        self.auto_actions: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API/UI."""
        return {
            "incident_id": self.incident_id,
            "severity": self.severity.value,
            "status": self.status.value,
            "trigger": self.trigger_type,
            "trigger_event_id": self.trigger_event_id,
            "description": self.description,
            "degraded_mode": self.degraded_mode.value,
            "affected_strategies": self.affected_strategies,
            "affected_symbols": self.affected_symbols,
            "declared_at": self.declared_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "auto_actions": self.auto_actions,
        }


class IncidentManager(BaseAgent):
    """Manages incident lifecycle and degraded mode transitions.

    Event-driven agent that:
      1. Subscribes to ``risk`` and ``system`` topics for triggers
      2. Auto-triages incidents by severity
      3. Applies degraded mode transitions
      4. Monitors recovery criteria
      5. Emits IncidentDeclared and DegradedModeEnabled events
    """

    def __init__(
        self,
        event_bus: IEventBus,
        *,
        agent_id: str | None = None,
        interval: float = 30.0,  # Check recovery every 30s
    ) -> None:
        super().__init__(agent_id=agent_id, interval=interval)
        self._event_bus = event_bus
        self._current_mode = DegradedMode.NORMAL
        self._active_incidents: dict[str, Incident] = {}
        self._resolved_incidents: list[Incident] = []
        self._incident_counter = 0

    @property
    def agent_type(self) -> AgentType:
        return AgentType.INCIDENT_RESPONSE

    @property
    def current_mode(self) -> DegradedMode:
        """Return the current system degraded mode."""
        return self._current_mode

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["risk", "system"],
            publishes_to=["governance"],
            description="Incident lifecycle management with degraded mode transitions",
        )

    async def _on_start(self) -> None:
        """Subscribe to risk and system event topics."""
        await self._event_bus.subscribe(
            topic="risk",
            group="incident_manager",
            handler=self._on_risk_event,
        )
        await self._event_bus.subscribe(
            topic="system",
            group="incident_manager",
            handler=self._on_system_event,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_risk_event(self, event: BaseEvent) -> None:
        """Handle risk topic events (circuit breakers, kill switch)."""
        if isinstance(event, CircuitBreakerEvent) and event.tripped:
            trigger = (
                "circuit_breaker_daily_loss"
                if "daily" in event.reason.lower()
                else "circuit_breaker_single"
            )
            await self.declare_incident(
                trigger_type=trigger,
                trigger_event_id=event.event_id,
                description=f"Circuit breaker tripped: {event.reason}",
                affected_symbols=[event.symbol] if event.symbol else [],
            )
        elif isinstance(event, KillSwitchEvent) and event.activated:
            await self.declare_incident(
                trigger_type="kill_switch",
                trigger_event_id=event.event_id,
                description=f"Kill switch activated: {event.reason}",
            )

    async def _on_system_event(self, event: BaseEvent) -> None:
        """Handle system topic events (canary alerts, health)."""
        from agentic_trading.core.events import CanaryAlert

        if isinstance(event, CanaryAlert) and not event.healthy:
            await self.declare_incident(
                trigger_type="canary_unhealthy",
                trigger_event_id=event.event_id,
                description=f"Canary alert: {event.component} - {event.message}",
            )

    # ------------------------------------------------------------------
    # Incident declaration
    # ------------------------------------------------------------------

    async def declare_incident(
        self,
        trigger_type: str,
        trigger_event_id: str,
        description: str,
        affected_strategies: list[str] | None = None,
        affected_symbols: list[str] | None = None,
    ) -> Incident:
        """Declare a new incident and auto-triage."""
        self._incident_counter += 1
        incident_id = f"inc-{self._incident_counter:04d}"

        severity, degraded_mode = AUTO_TRIAGE_RULES.get(
            trigger_type,
            (IncidentSeverity.MEDIUM, DegradedMode.REDUCE_ONLY),
        )

        incident = Incident(
            incident_id=incident_id,
            severity=severity,
            trigger_type=trigger_type,
            trigger_event_id=trigger_event_id,
            description=description,
        )
        incident.affected_strategies = affected_strategies or []
        incident.affected_symbols = affected_symbols or []
        incident.status = IncidentStatus.TRIAGED

        self._active_incidents[incident_id] = incident

        # Apply degraded mode (escalate only, never de-escalate automatically)
        if degraded_mode.rank > self._current_mode.rank:
            await self._set_degraded_mode(
                degraded_mode, f"Incident {incident_id}: {description}"
            )
            incident.degraded_mode = degraded_mode

        logger.warning(
            "Incident declared: %s severity=%s trigger=%s mode=%s",
            incident_id,
            severity.value,
            trigger_type,
            self._current_mode.value,
        )

        # Emit IncidentDeclared event
        await self._emit_incident_declared(incident)

        return incident

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Periodic check for auto-recovery conditions."""
        now = datetime.now(timezone.utc)
        resolved = []

        for incident_id, incident in self._active_incidents.items():
            cooldown_min, requires_approval = RECOVERY_CRITERIA.get(
                incident.severity, (60, True)
            )
            elapsed_min = (now - incident.declared_at).total_seconds() / 60

            if elapsed_min >= cooldown_min and not requires_approval:
                incident.status = IncidentStatus.RESOLVED
                incident.resolved_at = now
                resolved.append(incident_id)
                logger.info(
                    "Incident %s auto-resolved after %.1f min",
                    incident_id,
                    elapsed_min,
                )

        for incident_id in resolved:
            incident = self._active_incidents.pop(incident_id)
            self._resolved_incidents.append(incident)

        # If no active incidents remain, return to NORMAL
        if not self._active_incidents and self._current_mode != DegradedMode.NORMAL:
            await self._set_degraded_mode(DegradedMode.NORMAL, "All incidents resolved")

    async def resolve_incident(
        self,
        incident_id: str,
        *,
        resolved_by: str = "operator",
    ) -> bool:
        """Manually resolve an incident (for HIGH/CRITICAL requiring approval)."""
        incident = self._active_incidents.get(incident_id)
        if incident is None:
            return False

        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now(timezone.utc)
        self._resolved_incidents.append(
            self._active_incidents.pop(incident_id)
        )

        logger.info(
            "Incident %s manually resolved by %s",
            incident_id,
            resolved_by,
        )

        # Check if we can de-escalate
        if not self._active_incidents:
            await self._set_degraded_mode(
                DegradedMode.NORMAL,
                f"Incident {incident_id} resolved by {resolved_by}",
            )

        return True

    # ------------------------------------------------------------------
    # Degraded mode management
    # ------------------------------------------------------------------

    async def _set_degraded_mode(
        self, mode: DegradedMode, reason: str
    ) -> None:
        """Transition to a new degraded mode."""
        previous = self._current_mode
        self._current_mode = mode
        logger.warning(
            "Degraded mode transition: %s → %s (%s)",
            previous.value,
            mode.value,
            reason,
        )

        # Emit DegradedModeEnabled event
        from agentic_trading.core.events import DegradedModeEnabled

        event = DegradedModeEnabled(
            mode=mode.value,
            previous_mode=previous.value,
            reason=reason,
        )
        try:
            await self._event_bus.publish("governance", event)
        except Exception:
            logger.error("Failed to publish DegradedModeEnabled", exc_info=True)

    async def _emit_incident_declared(self, incident: Incident) -> None:
        """Publish an IncidentDeclared event to the governance topic."""
        from agentic_trading.core.events import IncidentDeclared

        event = IncidentDeclared(
            incident_id=incident.incident_id,
            severity=incident.severity.value,
            trigger=incident.trigger_type,
            trigger_event_id=incident.trigger_event_id,
            description=incident.description,
            affected_strategies=incident.affected_strategies,
            affected_symbols=incident.affected_symbols,
        )
        try:
            await self._event_bus.publish("governance", event)
        except Exception:
            logger.error("Failed to publish IncidentDeclared", exc_info=True)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_active_incidents(self) -> list[dict[str, Any]]:
        """Return all active incidents as serializable dicts."""
        return [i.to_dict() for i in self._active_incidents.values()]

    def get_resolved_incidents(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent resolved incidents."""
        recent = self._resolved_incidents[-limit:]
        return [i.to_dict() for i in reversed(recent)]

    def get_summary(self) -> dict[str, Any]:
        """Return summary for UI/API."""
        return {
            "current_mode": self._current_mode.value,
            "active_count": len(self._active_incidents),
            "resolved_count": len(self._resolved_incidents),
            "total_declared": self._incident_counter,
        }
