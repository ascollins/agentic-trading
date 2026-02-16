"""Agent registry for lifecycle management and discovery.

The AgentRegistry is the central point for managing all agents in the
platform.  It handles registration, coordinated startup/shutdown,
health monitoring, and agent discovery by type.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentHealthReport
from agentic_trading.core.interfaces import IAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Manages agent lifecycle, discovery, and health monitoring.

    Usage::

        registry = AgentRegistry()
        registry.register(my_agent)
        await registry.start_all()
        # ... trading runs ...
        await registry.stop_all()
    """

    def __init__(self) -> None:
        self._agents: dict[str, IAgent] = {}
        self._start_order: list[str] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, agent: IAgent) -> None:
        """Register an agent. Raises ValueError if agent_id already registered."""
        if agent.agent_id in self._agents:
            raise ValueError(
                f"Agent already registered: {agent.agent_id}"
            )
        self._agents[agent.agent_id] = agent
        self._start_order.append(agent.agent_id)
        logger.info(
            "Registered agent: %s (type=%s, id=%s)",
            type(agent).__name__,
            agent.agent_type.value,
            agent.agent_id[:8],
        )

    def unregister(self, agent_id: str) -> IAgent | None:
        """Remove an agent from the registry. Returns the agent or None."""
        agent = self._agents.pop(agent_id, None)
        if agent is not None:
            self._start_order = [
                aid for aid in self._start_order if aid != agent_id
            ]
            logger.info(
                "Unregistered agent: %s (id=%s)",
                type(agent).__name__,
                agent_id[:8],
            )
        return agent

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_all(self) -> None:
        """Start all registered agents in registration order."""
        logger.info("Starting %d agents...", len(self._agents))
        for agent_id in self._start_order:
            agent = self._agents.get(agent_id)
            if agent is None:
                continue
            try:
                await agent.start()
            except Exception:
                logger.exception(
                    "Failed to start agent %s (id=%s)",
                    type(agent).__name__,
                    agent_id[:8],
                )
        logger.info("All agents started")

    async def stop_all(self) -> None:
        """Stop all registered agents in reverse registration order."""
        logger.info("Stopping %d agents...", len(self._agents))
        for agent_id in reversed(self._start_order):
            agent = self._agents.get(agent_id)
            if agent is None:
                continue
            try:
                await agent.stop()
            except Exception:
                logger.exception(
                    "Failed to stop agent %s (id=%s)",
                    type(agent).__name__,
                    agent_id[:8],
                )
        logger.info("All agents stopped")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check_all(self) -> dict[str, AgentHealthReport]:
        """Run health checks on all registered agents."""
        results: dict[str, AgentHealthReport] = {}
        for agent_id, agent in self._agents.items():
            try:
                results[agent_id] = agent.health_check()
            except Exception as exc:
                results[agent_id] = AgentHealthReport(
                    healthy=False,
                    message=f"Health check failed: {exc}",
                )
        return results

    def all_healthy(self) -> bool:
        """Return True if all agents report healthy."""
        reports = self.health_check_all()
        return all(r.healthy for r in reports.values())

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> IAgent | None:
        """Look up an agent by ID."""
        return self._agents.get(agent_id)

    def get_agents_by_type(self, agent_type: AgentType) -> list[IAgent]:
        """Find all agents of a given type."""
        return [
            a for a in self._agents.values()
            if a.agent_type == agent_type
        ]

    @property
    def agents(self) -> dict[str, IAgent]:
        """All registered agents (read-only view)."""
        return dict(self._agents)

    @property
    def count(self) -> int:
        return len(self._agents)

    def summary(self) -> list[dict[str, Any]]:
        """Return a summary of all registered agents for logging/UI."""
        return [
            {
                "agent_id": a.agent_id[:8],
                "type": a.agent_type.value,
                "name": type(a).__name__,
                "running": a.is_running,
            }
            for a in self._agents.values()
        ]
