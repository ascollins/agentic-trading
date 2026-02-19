"""IncidentResponse Agent stub.

Consumes :class:`IncidentCreated` events, applies degraded modes,
and executes runbook steps for the fail-closed architecture.

Day 6 stub: subscribes to ``system`` topic, processes incident events,
escalates severity via degraded-mode transitions.  Full runbook
execution deferred to Week 2.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import (
    AgentCapabilities,
    BaseEvent,
    DegradedModeEnabled,
    IncidentCreated,
)
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


class IncidentResponseAgent(BaseAgent):
    """Consumes incident events and applies protective degraded modes.

    Day 6 stub implementation:
    - Subscribes to ``system`` topic
    - Processes ``IncidentCreated`` events
    - Tracks active incidents by component
    - Escalates to ``DegradedModeEnabled`` based on severity
    - ``critical`` -> ``RISK_OFF_ONLY`` mode
    - ``emergency`` -> ``FULL_STOP`` mode
    - ``warning`` -> logged, no mode change

    Full implementation (Week 2):
    - Runbook step execution
    - Incident correlation and deduplication
    - Auto-recovery and mode restoration
    - Notification integration (PagerDuty, Slack)
    """

    def __init__(
        self,
        event_bus: IEventBus,
        policy_evaluator: Any = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id or "incident-response", interval=5.0)
        self._event_bus = event_bus
        self._policy_evaluator = policy_evaluator
        self._active_incidents: dict[str, IncidentCreated] = {}  # id -> event
        self._current_mode: str = "normal"

    @property
    def agent_type(self) -> AgentType:
        return AgentType.INCIDENT_RESPONSE

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["system"],
            publishes_to=["system"],
            description="Incident response and degraded mode management",
        )

    async def _on_start(self) -> None:
        await self._event_bus.subscribe(
            topic="system",
            group="incident_response",
            handler=self._on_system_event,
        )
        logger.info("IncidentResponseAgent started")

    async def _on_system_event(self, event: BaseEvent) -> None:
        """Route system events to appropriate handler."""
        if isinstance(event, IncidentCreated):
            await self._handle_incident(event)

    async def _handle_incident(self, incident: IncidentCreated) -> None:
        """Process an incident and apply protective actions."""
        self._active_incidents[incident.incident_id] = incident
        logger.warning(
            "Incident received: id=%s severity=%s component=%s desc=%s",
            incident.incident_id,
            incident.severity,
            incident.component,
            incident.description,
        )

        # Determine degraded mode based on severity
        new_mode = self._severity_to_mode(incident.severity)
        if new_mode and self._mode_rank(new_mode) > self._mode_rank(self._current_mode):
            previous_mode = self._current_mode
            self._current_mode = new_mode

            # Update policy evaluator if available
            if self._policy_evaluator is not None:
                self._policy_evaluator.set_system_state(
                    "degraded_mode", new_mode,
                )

            logger.warning(
                "Degraded mode escalated: %s -> %s (incident=%s)",
                previous_mode,
                new_mode,
                incident.incident_id,
            )

            # Publish degraded mode event
            await self._event_bus.publish(
                "system",
                DegradedModeEnabled(
                    mode=new_mode,
                    previous_mode=previous_mode,
                    reason=incident.description,
                    triggered_by=f"incident:{incident.incident_id}",
                ),
            )

    async def _work(self) -> None:
        """Periodic work: check for incident resolution (stub).

        Week 2: implement auto-recovery, stale incident cleanup,
        and mode restoration after all incidents resolve.
        """
        pass  # Stub — full logic in Week 2

    @staticmethod
    def _severity_to_mode(severity: str) -> str | None:
        """Map incident severity to degraded mode.

        Returns None for severities that don't trigger mode changes.
        """
        mapping = {
            "emergency": "full_stop",
            "critical": "risk_off_only",
        }
        return mapping.get(severity)

    @staticmethod
    def _mode_rank(mode: str) -> int:
        """Rank degraded modes for escalation comparison.

        Higher rank = more restrictive.
        """
        ranks = {
            "normal": 0,
            "risk_off_only": 1,
            "read_only": 2,
            "full_stop": 3,
        }
        return ranks.get(mode, 0)

    @property
    def active_incidents(self) -> dict[str, IncidentCreated]:
        """Expose active incidents for introspection."""
        return dict(self._active_incidents)

    @property
    def current_mode(self) -> str:
        """Current degraded mode level."""
        return self._current_mode
