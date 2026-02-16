"""Governance canary — independent infrastructure safety watchdog.

Runs periodic health checks on critical infrastructure components
(event bus, Redis, kill switch) and auto-activates the kill switch
if failures exceed a configurable threshold.

Inspired by Soteria's Governance Canary (C12): an independent process
that verifies the governance infrastructure itself is operational.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.config import CanaryConfig
from agentic_trading.core.enums import AgentType, GovernanceAction
from agentic_trading.core.events import (
    AgentCapabilities,
    CanaryAlert,
    GovernanceCanaryCheck,
)

logger = logging.getLogger(__name__)


class GovernanceCanary(BaseAgent):
    """Independent infrastructure health verifier.

    Usage::

        canary = GovernanceCanary(config, kill_switch_fn, event_bus)
        canary.register_component("redis", lambda: redis.ping())
        await canary.start()
    """

    def __init__(
        self,
        config: CanaryConfig,
        kill_switch_fn: Callable[..., Any] | None = None,
        event_bus: Any | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            interval=config.check_interval_seconds,
        )
        self._config = config
        self._kill_switch_fn = kill_switch_fn
        self._event_bus = event_bus
        self._components: dict[str, Callable[[], bool]] = {}
        self._consecutive_failures: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.GOVERNANCE_CANARY

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=[],
            publishes_to=["governance"],
            description="Independent infrastructure health watchdog",
        )

    # ------------------------------------------------------------------
    # Component registration
    # ------------------------------------------------------------------

    def register_component(
        self, name: str, check_fn: Callable[[], bool]
    ) -> None:
        """Register a component health-check function.

        Args:
            name: Component identifier (e.g. ``"redis"``, ``"event_bus"``).
            check_fn: Callable returning ``True`` if healthy,
                ``False`` otherwise.  Should not raise.
        """
        self._components[name] = check_fn
        logger.debug("Canary registered component: %s", name)

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    async def run_checks(self) -> GovernanceCanaryCheck:
        """Run all registered component health checks.

        Returns a :class:`GovernanceCanaryCheck` event summarising results.
        Triggers kill switch if any component fails beyond threshold.
        """
        failed: list[str] = []
        alerts: list[CanaryAlert] = []

        for name, check_fn in self._components.items():
            try:
                healthy = check_fn()
            except Exception as exc:
                logger.warning("Canary check exception for %s: %s", name, exc)
                healthy = False

            if healthy:
                self._consecutive_failures[name] = 0
            else:
                self._consecutive_failures[name] += 1
                failed.append(name)

                count = self._consecutive_failures[name]
                logger.warning(
                    "Canary: %s unhealthy (consecutive=%d/%d)",
                    name,
                    count,
                    self._config.failure_threshold,
                )

                if count >= self._config.failure_threshold:
                    action = self._resolve_action()
                    alert = CanaryAlert(
                        component=name,
                        healthy=False,
                        message=f"{name} failed {count} consecutive checks",
                        action_taken=action,
                    )
                    alerts.append(alert)
                    await self._execute_action(action, name, count)

        # Publish alerts
        if self._event_bus is not None:
            for alert in alerts:
                try:
                    await self._event_bus.publish("governance", alert)
                except Exception:
                    logger.error("Failed to publish canary alert", exc_info=True)

        check_event = GovernanceCanaryCheck(
            all_healthy=len(failed) == 0,
            components_checked=len(self._components),
            failed_components=failed,
        )

        # Emit canary Prometheus metrics
        try:
            from agentic_trading.observability.metrics import update_canary_status
            for name in self._components:
                update_canary_status(name, name not in failed)
        except Exception:
            pass

        if self._event_bus is not None:
            try:
                await self._event_bus.publish("governance", check_event)
            except Exception:
                logger.error("Failed to publish canary check event", exc_info=True)

        return check_event

    # ------------------------------------------------------------------
    # BaseAgent periodic work
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Run health checks as the periodic work unit."""
        await self.run_checks()

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    async def start_periodic(
        self, interval: int | None = None
    ) -> None:
        """Start the canary. Backward-compatible alias for ``start()``.

        If ``interval`` is provided, it overrides the config value.
        """
        if interval is not None:
            self._interval = interval
        await self.start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_action(self) -> GovernanceAction:
        """Map config action string to enum."""
        mapping = {
            "kill": GovernanceAction.KILL,
            "pause": GovernanceAction.PAUSE,
            "alert": GovernanceAction.ALLOW,
        }
        return mapping.get(
            self._config.action_on_failure, GovernanceAction.KILL
        )

    async def _execute_action(
        self, action: GovernanceAction, component: str, count: int
    ) -> None:
        """Execute the configured action on threshold breach."""
        if action == GovernanceAction.KILL and self._kill_switch_fn is not None:
            logger.critical(
                "Canary activating kill switch: %s failed %d times",
                component,
                count,
            )
            try:
                result = self._kill_switch_fn(
                    reason=f"Canary: {component} failed {count} checks",
                    triggered_by="governance_canary",
                )
                # Handle async kill switch functions
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.error(
                    "Failed to activate kill switch from canary",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def registered_components(self) -> list[str]:
        return list(self._components.keys())

    def get_failure_count(self, component: str) -> int:
        """Return consecutive failure count for a component."""
        return self._consecutive_failures.get(component, 0)
