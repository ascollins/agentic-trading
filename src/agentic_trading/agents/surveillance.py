"""SurveillanceAgent — market abuse detection (design spec §7).

Subscribes to execution topics (intents, acks, fills) and detects
potential market abuse patterns:

    - **Wash trading**: same-symbol buy + sell from same strategy within
      a configurable time window.
    - **Spoofing / layering**: large order submitted and then cancelled
      within a short window.

Each detection produces a :class:`SurveillanceCaseEvent` published on
the ``surveillance`` topic, and optionally creates a
:class:`ComplianceCase` via the case manager.

This is the initial (Phase 1) implementation.  Future phases add:
cross-strategy correlation, pattern learning, regulatory reporting.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType, OrderStatus, Side
from agentic_trading.core.events import (
    AgentCapabilities,
    BaseEvent,
    FillEvent,
    OrderAck,
    OrderIntent,
    OrderUpdate,
    SurveillanceCaseEvent,
)
from agentic_trading.core.ids import new_id
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal tracking structures
# ---------------------------------------------------------------------------


@dataclass
class _FillRecord:
    """Lightweight record of a fill for wash-trade detection."""

    fill_id: str
    symbol: str
    strategy_id: str
    side: str
    price: float
    qty: float
    timestamp: float  # monotonic


@dataclass
class _OrderRecord:
    """Lightweight record of an order for spoofing detection."""

    dedupe_key: str
    order_id: str
    symbol: str
    strategy_id: str
    side: str
    qty: float
    submitted_at: float  # monotonic
    cancelled_at: float | None = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SurveillanceAgent(BaseAgent):
    """Detects potential market abuse patterns from execution events.

    Parameters
    ----------
    event_bus:
        Event bus for subscriptions and publishing.
    case_manager:
        Optional :class:`CaseManager` for persisting compliance cases.
    wash_trade_window_sec:
        Time window (seconds) for wash-trade detection.  A buy+sell on
        the same symbol from the same strategy within this window is
        flagged.
    spoof_cancel_window_sec:
        Time window (seconds) for spoofing detection.  An order that is
        submitted then cancelled within this window is flagged.
    spoof_min_qty:
        Minimum order quantity (in base units) to consider for spoofing
        detection.  Small orders are ignored.
    max_history:
        Maximum number of fill/order records to retain per symbol.
    agent_id:
        Optional agent identifier.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        case_manager: Any = None,
        *,
        wash_trade_window_sec: float = 30.0,
        spoof_cancel_window_sec: float = 5.0,
        spoof_min_qty: float = 0.0,
        max_history: int = 500,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id or "surveillance", interval=0)
        self._event_bus = event_bus
        self._case_manager = case_manager

        self._wash_window = wash_trade_window_sec
        self._spoof_window = spoof_cancel_window_sec
        self._spoof_min_qty = spoof_min_qty
        self._max_history = max_history

        # Per-symbol fill history for wash-trade detection
        self._fills: dict[str, deque[_FillRecord]] = defaultdict(
            lambda: deque(maxlen=self._max_history)
        )

        # Pending orders for spoofing detection (dedupe_key → record)
        self._pending_orders: dict[str, _OrderRecord] = {}

        # Counters
        self._cases_created: int = 0

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SURVEILLANCE

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["execution.intent", "execution.ack",
                           "execution.fill", "execution.update"],
            publishes_to=["surveillance"],
            description=(
                "Monitors execution events for wash trading, "
                "spoofing, and other market abuse patterns"
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        await self._event_bus.subscribe(
            topic="execution.intent",
            group="surveillance",
            handler=self._on_execution_event,
        )
        await self._event_bus.subscribe(
            topic="execution.fill",
            group="surveillance",
            handler=self._on_execution_event,
        )
        await self._event_bus.subscribe(
            topic="execution.ack",
            group="surveillance",
            handler=self._on_execution_event,
        )
        await self._event_bus.subscribe(
            topic="execution.update",
            group="surveillance",
            handler=self._on_execution_event,
        )
        logger.info("SurveillanceAgent started")

    async def _on_stop(self) -> None:
        logger.info(
            "SurveillanceAgent stopped (cases_created=%d)",
            self._cases_created,
        )

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    async def _on_execution_event(self, event: BaseEvent) -> None:
        """Route execution events to appropriate detector."""
        if isinstance(event, FillEvent):
            await self._on_fill(event)
        elif isinstance(event, OrderIntent):
            self._on_intent(event)
        elif isinstance(event, OrderAck):
            self._on_ack(event)
        elif isinstance(event, OrderUpdate):
            await self._on_order_update(event)

    # ------------------------------------------------------------------
    # Wash-trade detection
    # ------------------------------------------------------------------

    async def _on_fill(self, fill: FillEvent) -> None:
        """Check fills for wash-trade patterns."""
        now = time.monotonic()
        record = _FillRecord(
            fill_id=fill.fill_id,
            symbol=fill.symbol,
            strategy_id=fill.strategy_id,
            side=fill.side.value if hasattr(fill.side, "value") else str(fill.side),
            price=float(fill.price),
            qty=float(fill.qty),
            timestamp=now,
        )

        fills = self._fills[fill.symbol]

        # Check for opposite-side fill from same strategy within window
        for prev in fills:
            if prev.strategy_id and prev.strategy_id == record.strategy_id:
                if prev.side != record.side:
                    elapsed = now - prev.timestamp
                    if elapsed <= self._wash_window:
                        await self._raise_case(
                            case_type="wash_trade",
                            severity="high",
                            symbol=fill.symbol,
                            strategy_id=record.strategy_id,
                            description=(
                                f"Potential wash trade: {prev.side} + {record.side} "
                                f"on {fill.symbol} within {elapsed:.1f}s "
                                f"from strategy {record.strategy_id}"
                            ),
                            evidence=[
                                {
                                    "type": "fill",
                                    "fill_id": prev.fill_id,
                                    "side": prev.side,
                                    "price": prev.price,
                                    "qty": prev.qty,
                                    "seconds_ago": elapsed,
                                },
                                {
                                    "type": "fill",
                                    "fill_id": record.fill_id,
                                    "side": record.side,
                                    "price": record.price,
                                    "qty": record.qty,
                                    "seconds_ago": 0.0,
                                },
                            ],
                        )
                        break  # One case per fill

        fills.append(record)

    # ------------------------------------------------------------------
    # Spoofing / layering detection
    # ------------------------------------------------------------------

    def _on_intent(self, intent: OrderIntent) -> None:
        """Track new order intents for spoofing detection."""
        now = time.monotonic()
        self._pending_orders[intent.dedupe_key] = _OrderRecord(
            dedupe_key=intent.dedupe_key,
            order_id="",
            symbol=intent.symbol,
            strategy_id=intent.strategy_id,
            side=intent.side.value if hasattr(intent.side, "value") else str(intent.side),
            qty=float(intent.qty),
            submitted_at=now,
        )

    def _on_ack(self, ack: OrderAck) -> None:
        """Update tracked orders with exchange order_id."""
        # Try to find by client_order_id (which maps to dedupe_key)
        record = self._pending_orders.get(ack.client_order_id)
        if record is not None:
            record.order_id = ack.order_id

        # If the ack is already a terminal state (filled/rejected), clean up
        if ack.status in (OrderStatus.FILLED, OrderStatus.REJECTED):
            self._pending_orders.pop(ack.client_order_id, None)

    async def _on_order_update(self, update: OrderUpdate) -> None:
        """Detect cancellations that may indicate spoofing."""
        if update.status != OrderStatus.CANCELLED:
            # If filled or other terminal, just clean up
            if update.status in (OrderStatus.FILLED, OrderStatus.REJECTED):
                self._remove_by_order_id(update.order_id)
            return

        now = time.monotonic()

        # Find the matching pending order
        record = self._find_by_order_id(update.order_id)
        if record is None:
            return

        record.cancelled_at = now
        elapsed = now - record.submitted_at

        # Check spoofing criteria: cancelled quickly + large size
        if elapsed <= self._spoof_window and record.qty >= self._spoof_min_qty:
            await self._raise_case(
                case_type="spoofing",
                severity="medium",
                symbol=record.symbol,
                strategy_id=record.strategy_id,
                description=(
                    f"Potential spoofing: {record.side} order for "
                    f"{record.qty} {record.symbol} cancelled after "
                    f"{elapsed:.2f}s (threshold: {self._spoof_window}s)"
                ),
                evidence=[
                    {
                        "type": "order",
                        "dedupe_key": record.dedupe_key,
                        "order_id": record.order_id,
                        "side": record.side,
                        "qty": record.qty,
                        "submitted_at_elapsed": 0.0,
                        "cancelled_after_sec": elapsed,
                    },
                ],
            )

        # Clean up
        self._pending_orders.pop(record.dedupe_key, None)

    # ------------------------------------------------------------------
    # Case creation
    # ------------------------------------------------------------------

    async def _raise_case(
        self,
        case_type: str,
        severity: str,
        symbol: str,
        strategy_id: str,
        description: str,
        evidence: list[dict[str, Any]],
    ) -> None:
        """Create and publish a surveillance case."""
        case_id = new_id()
        self._cases_created += 1

        logger.warning(
            "Surveillance case: type=%s severity=%s symbol=%s "
            "strategy=%s case_id=%s — %s",
            case_type, severity, symbol, strategy_id, case_id, description,
        )

        event = SurveillanceCaseEvent(
            case_id=case_id,
            case_type=case_type,
            severity=severity,
            symbol=symbol,
            strategy_id=strategy_id,
            description=description,
            evidence=evidence,
        )

        await self._event_bus.publish("surveillance", event)

        # Persist via case manager if available
        if self._case_manager is not None:
            try:
                self._case_manager.open_case(
                    case_id=case_id,
                    case_type=case_type,
                    severity=severity,
                    symbol=symbol,
                    strategy_id=strategy_id,
                    description=description,
                    evidence=evidence,
                )
            except Exception:
                logger.debug("Failed to persist case %s", case_id, exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_by_order_id(self, order_id: str) -> _OrderRecord | None:
        for record in self._pending_orders.values():
            if record.order_id == order_id:
                return record
        return None

    def _remove_by_order_id(self, order_id: str) -> None:
        to_remove = [
            dk for dk, r in self._pending_orders.items()
            if r.order_id == order_id
        ]
        for dk in to_remove:
            del self._pending_orders[dk]

    # ------------------------------------------------------------------
    # Periodic cleanup
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Periodic cleanup of stale pending orders."""
        now = time.monotonic()
        stale_cutoff = now - max(self._wash_window, self._spoof_window) * 10
        stale_keys = [
            dk for dk, r in self._pending_orders.items()
            if r.submitted_at < stale_cutoff
        ]
        for dk in stale_keys:
            del self._pending_orders[dk]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cases_created(self) -> int:
        return self._cases_created

    @property
    def pending_orders_count(self) -> int:
        return len(self._pending_orders)
