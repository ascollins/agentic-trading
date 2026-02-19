"""Base agent ABC providing the shared lifecycle pattern.

All agents in the platform extend BaseAgent, which provides:

- Unique agent identity (``agent_id``, ``agent_type``)
- Lifecycle management (``start`` / ``stop`` with graceful shutdown)
- Optional periodic background loop (override ``_work``)
- Health reporting with error tracking
- Capability declarations

Subclasses must implement:
- ``agent_type`` property
- ``capabilities()`` method
- ``_work()`` coroutine (for periodic agents) or override ``start``
  for event-driven agents
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from agentic_trading.core.enums import AgentStatus, AgentType
from agentic_trading.core.events import AgentCapabilities, AgentHealthReport

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Abstract base for all platform agents.

    Parameters
    ----------
    agent_id:
        Unique identifier for this agent instance. Auto-generated if omitted.
    interval:
        Seconds between ``_work()`` invocations for periodic agents.
        Set to ``0`` to disable the periodic loop (event-driven mode).
    """

    def __init__(
        self,
        *,
        agent_id: str | None = None,
        interval: float = 0,
        context_manager: Any | None = None,
    ) -> None:
        self._agent_id = agent_id or str(uuid.uuid4())
        self._interval = interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._status = AgentStatus.CREATED
        self._error_count = 0
        self._last_work_at: datetime | None = None
        self._last_error: str | None = None
        self._context_manager = context_manager
        self._reasoning_builder: Any | None = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    @abc.abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        ...

    @property
    def agent_name(self) -> str:
        """Human-readable name (defaults to class name)."""
        return self.__class__.__name__

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def status(self) -> AgentStatus:
        return self._status

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the agent. Creates a background loop if interval > 0."""
        if self._running:
            logger.warning("%s is already running", self.agent_name)
            return

        self._status = AgentStatus.STARTING
        self._running = True

        await self._on_start()

        if self._interval > 0:
            self._task = asyncio.create_task(
                self._loop(),
                name=f"agent-{self.agent_name}-{self._agent_id[:8]}",
            )

        self._status = AgentStatus.RUNNING
        logger.info(
            "%s started (id=%s, type=%s)",
            self.agent_name,
            self._agent_id[:8],
            self.agent_type.value,
        )

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self._status = AgentStatus.STOPPING
        self._running = False

        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        await self._on_stop()

        self._status = AgentStatus.STOPPED
        logger.info("%s stopped (id=%s)", self.agent_name, self._agent_id[:8])

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        """Called during start before the loop begins. Override for setup."""

    async def _on_stop(self) -> None:
        """Called during stop after the loop ends. Override for cleanup."""

    async def _work(self) -> None:
        """Single unit of periodic work. Override in subclasses.

        Only called when ``interval > 0``.  For event-driven agents,
        subscribe to topics in ``_on_start`` instead.
        """

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Internal periodic loop."""
        while self._running:
            try:
                await self._work()
                self._last_work_at = datetime.now(timezone.utc)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._error_count += 1
                self._last_error = logger.name
                logger.exception(
                    "%s work cycle failed (errors=%d)",
                    self.agent_name,
                    self._error_count,
                )
            await asyncio.sleep(self._interval)

    # ------------------------------------------------------------------
    # Health & capabilities
    # ------------------------------------------------------------------

    def health_check(self) -> AgentHealthReport:
        """Return current health status."""
        healthy = self._running and self._status == AgentStatus.RUNNING
        message = ""
        if not self._running:
            message = "Agent is not running"
        elif self._error_count > 0:
            message = f"Last error count: {self._error_count}"

        return AgentHealthReport(
            healthy=healthy,
            message=message,
            last_work_at=self._last_work_at,
            error_count=self._error_count,
            details={
                "agent_id": self._agent_id,
                "agent_type": self.agent_type.value,
                "status": self._status.value,
            },
        )

    @abc.abstractmethod
    def capabilities(self) -> AgentCapabilities:
        """Declare this agent's event subscriptions and publications."""
        ...

    # ------------------------------------------------------------------
    # Context & Reasoning (optional, backward-compatible)
    # ------------------------------------------------------------------

    def set_context_manager(self, cm: Any) -> None:
        """Inject context manager after construction."""
        self._context_manager = cm

    def _read_context(
        self, symbol: str | None = None
    ) -> Any | None:
        """Read context before reasoning. Returns None if no CM."""
        if self._context_manager is None:
            return None
        return self._context_manager.read_context(symbol=symbol)

    def _write_analysis(
        self,
        entry_type: Any,
        content: dict[str, Any],
        **kwargs: Any,
    ) -> str | None:
        """Write analysis to memory store. Returns entry_id or None."""
        if self._context_manager is None:
            return None
        return self._context_manager.write_analysis(
            entry_type=entry_type, content=content, **kwargs
        )

    def _start_reasoning(
        self, symbol: str = "", pipeline_id: str = ""
    ) -> Any:
        """Create a new reasoning trace for this agent."""
        if self._reasoning_builder is None:
            from agentic_trading.reasoning.builder import ReasoningBuilder

            self._reasoning_builder = ReasoningBuilder(
                self._agent_id, self.agent_type.value
            )
        return self._reasoning_builder.start(
            symbol=symbol, pipeline_id=pipeline_id
        )
