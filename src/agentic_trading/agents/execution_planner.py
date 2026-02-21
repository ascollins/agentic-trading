"""Execution Planner Agent (spec §5.1).

Sits between the PortfolioManager (which produces ``OrderIntent`` events)
and the ExecutionEngine (which submits orders).  Constructs an
:class:`~agentic_trading.execution.plan.ExecutionPlan` for each intent,
selecting venue, order type, slicing strategy, and contingencies.

**Initial implementation**: single-slice pass-through that preserves
current behaviour.  Multi-slice and multi-venue logic can be added later
without changing the interface.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import (
    AgentCapabilities,
    BaseEvent,
    ExecutionPlanCreated,
    OrderIntent,
)
from agentic_trading.core.interfaces import IEventBus
from agentic_trading.execution.plan import (
    Contingency,
    ExecutionPlan,
    OrderSlice,
)

logger = logging.getLogger(__name__)


class ExecutionPlannerAgent(BaseAgent):
    """Creates execution plans from order intents.

    Subscribes to ``execution`` for :class:`OrderIntent` events,
    creates an :class:`ExecutionPlan`, publishes the plan on
    ``execution.plan``, and re-publishes the original intent for
    the ExecutionEngine to consume.

    Parameters
    ----------
    event_bus:
        Event bus for subscriptions and publishing.
    default_max_participation:
        Default max participation rate for order slices (0 = unlimited).
    default_max_execution_seconds:
        Default max execution time per plan.
    agent_id:
        Optional agent identifier.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        *,
        default_max_participation: float = 0.0,
        default_max_execution_seconds: float = 300.0,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id or "execution-planner", interval=0)
        self._event_bus = event_bus
        self._default_max_participation = default_max_participation
        self._default_max_execution_seconds = default_max_execution_seconds
        self._plans_created: int = 0

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.EXECUTION_PLANNER

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["execution"],
            publishes_to=["execution.plan"],
            description="Converts order intents into structured execution plans",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        logger.info("ExecutionPlannerAgent starting")

    async def _on_stop(self) -> None:
        logger.info(
            "ExecutionPlannerAgent stopped (plans_created=%d)",
            self._plans_created,
        )

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------

    def create_plan(self, intent: OrderIntent) -> ExecutionPlan:
        """Create an execution plan from an order intent.

        Current implementation: single-slice pass-through.
        Future: multi-slice TWAP/VWAP, multi-venue routing, urgency-based
        order type selection.

        Parameters
        ----------
        intent:
            The order intent to plan execution for.

        Returns
        -------
        ExecutionPlan
            A plan with one or more order slices.
        """
        # Single-slice plan: the entire intent qty in one order
        slice_0 = OrderSlice(
            sequence=0,
            symbol=intent.symbol,
            exchange=intent.exchange,
            side=intent.side,
            order_type=intent.order_type,
            time_in_force=intent.time_in_force,
            qty=intent.qty,
            price=intent.price,
            reduce_only=intent.reduce_only,
            max_participation_rate=self._default_max_participation,
            urgency=0.5,
        )

        # Default contingencies
        contingencies = [
            Contingency(
                trigger="rejection",
                action="retry",
                params={"max_retries": 3},
            ),
            Contingency(
                trigger="timeout",
                action="cancel_remaining",
                params={"timeout_seconds": self._default_max_execution_seconds},
            ),
        ]

        plan = ExecutionPlan(
            intent_dedupe_key=intent.dedupe_key,
            trace_id=intent.trace_id,
            strategy_id=intent.strategy_id,
            symbol=intent.symbol,
            slices=[slice_0],
            venue=intent.exchange,
            venue_selection_reason="default_venue",
            contingencies=contingencies,
            max_execution_seconds=self._default_max_execution_seconds,
        )

        self._plans_created += 1
        logger.info(
            "Execution plan created: plan_id=%s symbol=%s slices=%d "
            "strategy=%s dedupe_key=%s",
            plan.plan_id,
            plan.symbol,
            plan.slice_count,
            plan.strategy_id,
            plan.intent_dedupe_key,
        )

        return plan

    async def create_and_publish_plan(self, intent: OrderIntent) -> ExecutionPlan:
        """Create a plan and publish the :class:`ExecutionPlanCreated` event.

        Parameters
        ----------
        intent:
            The order intent to plan.

        Returns
        -------
        ExecutionPlan
            The created plan.
        """
        plan = self.create_plan(intent)

        event = ExecutionPlanCreated(
            plan_id=plan.plan_id,
            intent_dedupe_key=plan.intent_dedupe_key,
            strategy_id=plan.strategy_id,
            symbol=plan.symbol,
            slice_count=plan.slice_count,
            expected_slippage_bps=plan.expected_slippage_bps,
            venue=plan.venue.value,
            trace_id=plan.trace_id,
        )
        await self._event_bus.publish("execution.plan", event)

        return plan

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def plans_created(self) -> int:
        return self._plans_created
