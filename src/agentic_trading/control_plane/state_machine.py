"""Order lifecycle state machine for the institutional control plane.

Every order follows a deterministic lifecycle:

    INTENT_RECEIVED -> PREFLIGHT_POLICY -> [AWAITING_APPROVAL|SUBMITTING|BLOCKED]
    AWAITING_APPROVAL -> [SUBMITTING|APPROVAL_DENIED|APPROVAL_EXPIRED]
    SUBMITTING -> [SUBMITTED|SUBMIT_FAILED]
    SUBMITTED -> MONITORING
    MONITORING -> [PARTIALLY_FILLED|COMPLETE|ABORT]
    PARTIALLY_FILLED -> [MONITORING|COMPLETE|ABORT]
    COMPLETE -> POST_TRADE -> TERMINAL
    ABORT -> POST_TRADE -> TERMINAL

Invalid transitions raise ValueError. Terminal states cannot be exited.
All transitions are recorded for audit trail.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


class OrderState(str, Enum):
    """Lifecycle states for a single order."""

    INTENT_RECEIVED = "intent_received"
    PREFLIGHT_POLICY = "preflight_policy"
    AWAITING_APPROVAL = "awaiting_approval"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    MONITORING = "monitoring"
    PARTIALLY_FILLED = "partially_filled"
    COMPLETE = "complete"
    ABORT = "abort"
    POST_TRADE = "post_trade"
    TERMINAL = "terminal"
    # Error terminals
    BLOCKED = "blocked"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_EXPIRED = "approval_expired"
    SUBMIT_FAILED = "submit_failed"


TERMINAL_STATES: frozenset[OrderState] = frozenset({
    OrderState.TERMINAL,
    OrderState.BLOCKED,
    OrderState.APPROVAL_DENIED,
    OrderState.APPROVAL_EXPIRED,
    OrderState.SUBMIT_FAILED,
})

# Valid transitions: from -> set of valid targets
TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.INTENT_RECEIVED: frozenset({OrderState.PREFLIGHT_POLICY}),
    OrderState.PREFLIGHT_POLICY: frozenset({
        OrderState.AWAITING_APPROVAL, OrderState.SUBMITTING, OrderState.BLOCKED,
    }),
    OrderState.AWAITING_APPROVAL: frozenset({
        OrderState.SUBMITTING, OrderState.APPROVAL_DENIED, OrderState.APPROVAL_EXPIRED,
    }),
    OrderState.SUBMITTING: frozenset({
        OrderState.SUBMITTED, OrderState.SUBMIT_FAILED,
    }),
    OrderState.SUBMITTED: frozenset({OrderState.MONITORING}),
    OrderState.MONITORING: frozenset({
        OrderState.PARTIALLY_FILLED, OrderState.COMPLETE, OrderState.ABORT,
    }),
    OrderState.PARTIALLY_FILLED: frozenset({
        OrderState.MONITORING, OrderState.COMPLETE, OrderState.ABORT,
    }),
    OrderState.COMPLETE: frozenset({OrderState.POST_TRADE}),
    OrderState.ABORT: frozenset({OrderState.POST_TRADE}),
    OrderState.POST_TRADE: frozenset({OrderState.TERMINAL}),
}

# Default timeouts per state (seconds)
DEFAULT_TIMEOUTS: dict[OrderState, float] = {
    OrderState.PREFLIGHT_POLICY: 5.0,       # 5s for policy eval
    OrderState.AWAITING_APPROVAL: 300.0,     # 5min for human approval
    OrderState.SUBMITTING: 30.0,             # 30s for exchange submission
    OrderState.MONITORING: 3600.0,           # 1h for order to fill
}


# ---------------------------------------------------------------------------
# OrderLifecycle
# ---------------------------------------------------------------------------


class OrderLifecycle:
    """State machine for a single order's lifecycle.

    Enforces valid transitions, timeouts, and records all transitions
    for audit.

    Args:
        action_id: The ProposedAction.action_id that initiated this order.
        correlation_id: Correlation ID for tracing.
        timeouts: Per-state timeout overrides (seconds).
    """

    def __init__(
        self,
        action_id: str,
        correlation_id: str,
        timeouts: dict[OrderState, float] | None = None,
    ) -> None:
        self.action_id = action_id
        self.correlation_id = correlation_id
        self.state = OrderState.INTENT_RECEIVED
        self.history: list[tuple[OrderState, OrderState, float]] = []
        self.created_at = time.monotonic()
        self._timeouts = dict(DEFAULT_TIMEOUTS)
        if timeouts:
            self._timeouts.update(timeouts)
        self._state_entered_at: float = self.created_at

        # Context accumulated during lifecycle
        self.policy_decision: Any = None
        self.approval_decision: Any = None
        self.tool_result: Any = None
        self.fills: list[Any] = []
        self.error: str | None = None

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition(self, new_state: OrderState) -> None:
        """Transition to a new state.

        Raises:
            ValueError: If the transition is not valid from the current state.
        """
        if self.state in TERMINAL_STATES:
            raise ValueError(
                f"Cannot transition from terminal state {self.state.value}"
            )
        valid = TRANSITIONS.get(self.state, frozenset())
        if new_state not in valid:
            raise ValueError(
                f"Invalid transition: {self.state.value} -> {new_state.value}. "
                f"Valid: {[s.value for s in valid]}"
            )
        now = time.monotonic()
        self.history.append((self.state, new_state, now))
        logger.debug(
            "OrderLifecycle %s: %s -> %s",
            self.action_id[:8], self.state.value, new_state.value,
        )
        self.state = new_state
        self._state_entered_at = now

    # ------------------------------------------------------------------
    # Timeout management
    # ------------------------------------------------------------------

    def is_timed_out(self) -> bool:
        """Check if the current state has exceeded its timeout."""
        timeout = self._timeouts.get(self.state)
        if timeout is None:
            return False
        return (time.monotonic() - self._state_entered_at) > timeout

    def time_in_state(self) -> float:
        """Seconds spent in current state."""
        return time.monotonic() - self._state_entered_at

    def total_time(self) -> float:
        """Total seconds since creation."""
        return time.monotonic() - self.created_at

    def timeout_for_state(self, state: OrderState | None = None) -> float | None:
        """Get the timeout for a state (current state if None)."""
        return self._timeouts.get(state or self.state)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Whether the order has reached a final state."""
        return self.state in TERMINAL_STATES

    @property
    def transition_count(self) -> int:
        """Number of transitions that have occurred."""
        return len(self.history)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_audit_dict(self) -> dict[str, Any]:
        """Serialize for audit logging."""
        return {
            "action_id": self.action_id,
            "correlation_id": self.correlation_id,
            "state": self.state.value,
            "history": [
                {"from": f.value, "to": t.value, "at": ts}
                for f, t, ts in self.history
            ],
            "total_time_s": round(self.total_time(), 3),
            "fill_count": len(self.fills),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# OrderLifecycleManager
# ---------------------------------------------------------------------------


class OrderLifecycleManager:
    """Manages a collection of OrderLifecycle instances.

    Provides creation, lookup, cleanup of timed-out orders, and
    summary statistics.
    """

    def __init__(self) -> None:
        self._lifecycles: dict[str, OrderLifecycle] = {}  # action_id -> lifecycle

    def create(
        self,
        action_id: str,
        correlation_id: str,
        timeouts: dict[OrderState, float] | None = None,
    ) -> OrderLifecycle:
        """Create a new OrderLifecycle for an action."""
        if action_id in self._lifecycles:
            raise ValueError(f"Lifecycle already exists for action_id: {action_id}")
        lifecycle = OrderLifecycle(
            action_id=action_id,
            correlation_id=correlation_id,
            timeouts=timeouts,
        )
        self._lifecycles[action_id] = lifecycle
        return lifecycle

    def get(self, action_id: str) -> OrderLifecycle | None:
        """Look up a lifecycle by action_id."""
        return self._lifecycles.get(action_id)

    def remove(self, action_id: str) -> OrderLifecycle | None:
        """Remove a lifecycle (e.g., after reaching terminal state)."""
        return self._lifecycles.pop(action_id, None)

    def get_by_state(self, state: OrderState) -> list[OrderLifecycle]:
        """Get all lifecycles currently in a given state."""
        return [lc for lc in self._lifecycles.values() if lc.state == state]

    def get_timed_out(self) -> list[OrderLifecycle]:
        """Get all lifecycles that have exceeded their current state timeout."""
        return [lc for lc in self._lifecycles.values() if lc.is_timed_out()]

    def get_active(self) -> list[OrderLifecycle]:
        """Get all non-terminal lifecycles."""
        return [lc for lc in self._lifecycles.values() if not lc.is_terminal]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_terminal(self, max_age_seconds: float = 300.0) -> int:
        """Remove terminal lifecycles older than max_age_seconds.

        Returns the number of lifecycles removed.
        """
        to_remove: list[str] = []
        for action_id, lifecycle in self._lifecycles.items():
            if lifecycle.is_terminal and lifecycle.total_time() > max_age_seconds:
                to_remove.append(action_id)

        for action_id in to_remove:
            del self._lifecycles[action_id]

        return len(to_remove)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Total number of tracked lifecycles."""
        return len(self._lifecycles)

    @property
    def active_count(self) -> int:
        """Number of non-terminal lifecycles."""
        return sum(1 for lc in self._lifecycles.values() if not lc.is_terminal)

    def summary(self) -> dict[str, int]:
        """Count of lifecycles in each state."""
        counts: dict[str, int] = {}
        for lc in self._lifecycles.values():
            counts[lc.state.value] = counts.get(lc.state.value, 0) + 1
        return counts
