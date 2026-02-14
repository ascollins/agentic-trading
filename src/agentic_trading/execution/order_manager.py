"""Order state machine, deduplication, and retry logic.

Tracks all active orders in memory, enforces valid state transitions, and
provides deduplication via a seen-keys set so the same ``dedupe_key`` is
never submitted to the exchange more than once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Set

from agentic_trading.core.enums import OrderStatus
from agentic_trading.core.errors import DuplicateOrderError, OrderRejectedError
from agentic_trading.core.events import OrderIntent, OrderUpdate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid state transitions
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING: frozenset(
        {OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.CANCELLED}
    ),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,  # additional partial fills
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        }
    ),
    # Terminal states -- no further transitions allowed.
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}


# ---------------------------------------------------------------------------
# Internal order state container
# ---------------------------------------------------------------------------

@dataclass
class _TrackedOrder:
    """Internal mutable state for a tracked order."""

    dedupe_key: str
    order_id: str = ""
    symbol: str = ""
    strategy_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    total_qty: Decimal = Decimal("0")
    filled_qty: Decimal = Decimal("0")
    remaining_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    attempt_count: int = 0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    trace_id: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """In-memory order state machine with deduplication and retry tracking.

    Parameters
    ----------
    max_retries:
        Maximum number of submission attempts per order (default 3).
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        # dedupe_key -> _TrackedOrder
        self._orders: Dict[str, _TrackedOrder] = {}
        # Set of all seen dedupe keys (including terminal orders)
        self._seen_keys: Set[str] = set()

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def dedupe_check(self, dedupe_key: str) -> bool:
        """Return ``True`` if the dedupe_key has already been seen.

        This is the primary idempotency gate.  Once a key is registered
        (via ``register_intent``) it is never accepted again.
        """
        return dedupe_key in self._seen_keys

    # ------------------------------------------------------------------
    # Intent registration
    # ------------------------------------------------------------------

    def register_intent(self, intent: OrderIntent) -> _TrackedOrder:
        """Register a new ``OrderIntent`` and start tracking it.

        Raises ``DuplicateOrderError`` if the key was already registered.
        """
        if intent.dedupe_key in self._seen_keys:
            raise DuplicateOrderError(
                f"Duplicate dedupe_key: {intent.dedupe_key}"
            )
        self._seen_keys.add(intent.dedupe_key)
        tracked = _TrackedOrder(
            dedupe_key=intent.dedupe_key,
            symbol=intent.symbol,
            strategy_id=intent.strategy_id,
            total_qty=intent.qty,
            remaining_qty=intent.qty,
            trace_id=intent.trace_id,
        )
        self._orders[intent.dedupe_key] = tracked
        logger.debug(
            "Intent registered: dedupe_key=%s symbol=%s qty=%s",
            intent.dedupe_key,
            intent.symbol,
            intent.qty,
        )
        return tracked

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def update_order(self, update: OrderUpdate) -> _TrackedOrder:
        """Transition an order to a new status.

        Enforces the valid-transition map.  Raises ``OrderRejectedError``
        if the transition is invalid.

        Parameters
        ----------
        update:
            ``OrderUpdate`` event containing the new status and quantities.

        Returns
        -------
        The updated ``_TrackedOrder``.
        """
        tracked = self._orders.get(update.client_order_id)
        if tracked is None:
            logger.warning(
                "Received update for unknown client_order_id=%s, "
                "creating stub entry",
                update.client_order_id,
            )
            tracked = _TrackedOrder(
                dedupe_key=update.client_order_id,
                order_id=update.order_id,
                symbol=update.symbol,
            )
            self._orders[update.client_order_id] = tracked
            self._seen_keys.add(update.client_order_id)

        old_status = tracked.status
        new_status = update.status

        # Validate transition
        allowed = _VALID_TRANSITIONS.get(old_status, frozenset())
        if new_status not in allowed and old_status != new_status:
            msg = (
                f"Invalid order transition: {old_status.value} -> "
                f"{new_status.value} for dedupe_key={tracked.dedupe_key}"
            )
            logger.error(msg)
            raise OrderRejectedError(msg)

        # Apply update
        tracked.status = new_status
        if update.order_id:
            tracked.order_id = update.order_id
        if update.filled_qty:
            tracked.filled_qty = update.filled_qty
        if update.remaining_qty is not None:
            tracked.remaining_qty = update.remaining_qty
        if update.avg_fill_price is not None:
            tracked.avg_fill_price = update.avg_fill_price
        tracked.updated_at = datetime.now(timezone.utc)
        if update.trace_id:
            tracked.trace_id = update.trace_id

        logger.debug(
            "Order state transition: dedupe_key=%s %s -> %s "
            "(filled=%s remaining=%s)",
            tracked.dedupe_key,
            old_status.value,
            new_status.value,
            tracked.filled_qty,
            tracked.remaining_qty,
        )
        return tracked

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def should_retry(self, dedupe_key: str) -> bool:
        """Return ``True`` if the order has remaining retry attempts."""
        tracked = self._orders.get(dedupe_key)
        if tracked is None:
            # Not yet registered -- first attempt is allowed
            return True
        if tracked.is_terminal:
            return False
        return tracked.attempt_count < self._max_retries

    def record_attempt(self, dedupe_key: str) -> int:
        """Increment and return the attempt counter for an order."""
        tracked = self._orders.get(dedupe_key)
        if tracked is None:
            return 0
        tracked.attempt_count += 1
        logger.debug(
            "Attempt %d/%d for dedupe_key=%s",
            tracked.attempt_count,
            self._max_retries,
            dedupe_key,
        )
        return tracked.attempt_count

    def get_attempt_count(self, dedupe_key: str) -> int:
        """Return the current attempt count for an order."""
        tracked = self._orders.get(dedupe_key)
        if tracked is None:
            return 0
        return tracked.attempt_count

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_order(self, dedupe_key: str) -> _TrackedOrder | None:
        """Look up a tracked order by dedupe key."""
        return self._orders.get(dedupe_key)

    def get_active_orders(self) -> list[_TrackedOrder]:
        """Return all non-terminal orders."""
        return [
            o for o in self._orders.values() if not o.is_terminal
        ]

    def get_all_orders(self) -> dict[str, _TrackedOrder]:
        """Return the full order map (read-only view)."""
        return dict(self._orders)

    def get_orders_by_status(
        self, status: OrderStatus
    ) -> list[_TrackedOrder]:
        """Return all orders in a given status."""
        return [
            o for o in self._orders.values() if o.status == status
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def purge_terminal(self, keep_last_n: int = 1000) -> int:
        """Remove the oldest terminal orders beyond ``keep_last_n``.

        Returns the number of purged entries.  The ``_seen_keys`` set is
        *not* purged so that deduplication still works even after the
        order record has been evicted.
        """
        terminal = sorted(
            (
                o
                for o in self._orders.values()
                if o.is_terminal
            ),
            key=lambda o: o.updated_at,
        )
        to_remove = terminal[: max(0, len(terminal) - keep_last_n)]
        for o in to_remove:
            del self._orders[o.dedupe_key]
        if to_remove:
            logger.info(
                "Purged %d terminal orders (kept %d)",
                len(to_remove),
                keep_last_n,
            )
        return len(to_remove)

    @property
    def active_count(self) -> int:
        """Number of non-terminal orders."""
        return sum(1 for o in self._orders.values() if not o.is_terminal)

    @property
    def total_count(self) -> int:
        """Total tracked orders (including terminal)."""
        return len(self._orders)
