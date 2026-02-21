"""Execution plan models (spec §5.1).

An :class:`ExecutionPlan` describes how to execute a trading intent:
which venue, what order type, how to slice, and what contingencies apply.

The :class:`ExecutionPlannerAgent` creates plans from :class:`OrderIntent`
events.  The :class:`ExecutionEngine` then executes each slice.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.enums import Exchange, OrderType, Side, TimeInForce
from agentic_trading.core.ids import new_id, utc_now


class OrderSlice(BaseModel):
    """A single order slice within an execution plan."""

    slice_id: str = Field(default_factory=new_id)
    sequence: int = 0  # Execution order (0-based)
    symbol: str
    exchange: Exchange
    side: Side
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.GTC
    qty: Decimal
    price: Decimal | None = None
    reduce_only: bool = False

    # Execution constraints
    max_participation_rate: float = 0.0  # 0 = no limit
    urgency: float = 0.5  # 0=passive, 1=aggressive


class Contingency(BaseModel):
    """A contingency action if a slice fails or times out."""

    trigger: str  # "timeout", "partial_fill", "rejection"
    action: str  # "retry", "cancel_remaining", "reduce_size", "skip"
    params: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """A plan describing how to execute a trading intent.

    Created by :class:`ExecutionPlannerAgent` from an ``OrderIntent``.
    Consumed by :class:`ExecutionEngine`` which executes each slice.
    """

    plan_id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # Linkage to the originating intent
    intent_dedupe_key: str
    trace_id: str = ""
    strategy_id: str = ""
    symbol: str = ""

    # Slices
    slices: list[OrderSlice] = Field(default_factory=list)

    # Venue selection
    venue: Exchange = Exchange.BYBIT
    venue_selection_reason: str = ""

    # Cost estimates
    expected_slippage_bps: float = 0.0
    expected_fee_bps: float = 0.0

    # Contingencies
    contingencies: list[Contingency] = Field(default_factory=list)

    # Timing
    max_execution_seconds: float = 300.0  # 5 min default

    # Metadata
    planner_version: str = "1.0"

    @property
    def total_qty(self) -> Decimal:
        return sum((s.qty for s in self.slices), Decimal("0"))

    @property
    def slice_count(self) -> int:
        return len(self.slices)

    @property
    def is_single_slice(self) -> bool:
        return len(self.slices) == 1
