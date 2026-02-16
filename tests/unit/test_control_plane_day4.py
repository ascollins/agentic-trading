"""Day 4 tests: OrderStateMachine + OrderLifecycleManager.

Tests:
    - Valid transitions through all happy paths
    - Invalid transitions raise ValueError
    - Terminal states cannot be exited
    - Timeout detection
    - Audit serialization
    - OrderLifecycleManager: create, get, cleanup, summary
    - Integration: lifecycle through full order flow
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agentic_trading.control_plane.state_machine import (
    DEFAULT_TIMEOUTS,
    OrderLifecycle,
    OrderLifecycleManager,
    OrderState,
    TERMINAL_STATES,
    TRANSITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lifecycle(
    action_id: str = "a1",
    correlation_id: str = "c1",
    timeouts: dict | None = None,
) -> OrderLifecycle:
    return OrderLifecycle(
        action_id=action_id,
        correlation_id=correlation_id,
        timeouts=timeouts,
    )


def _advance_to(lifecycle: OrderLifecycle, *states: OrderState) -> None:
    """Advance a lifecycle through a sequence of states."""
    for state in states:
        lifecycle.transition(state)


# ===========================================================================
# OrderState + Transitions
# ===========================================================================


class TestOrderState:
    def test_all_states_defined(self):
        """All expected states exist."""
        assert len(OrderState) == 15

    def test_terminal_states_are_subset(self):
        """All terminal states are valid OrderState members."""
        for state in TERMINAL_STATES:
            assert state in OrderState

    def test_terminal_states_have_no_outgoing_transitions(self):
        """Terminal states should not appear in TRANSITIONS keys."""
        for state in TERMINAL_STATES:
            assert state not in TRANSITIONS


# ===========================================================================
# Happy path transitions
# ===========================================================================


class TestHappyPaths:
    def test_full_happy_path_to_terminal(self):
        """INTENT -> PREFLIGHT -> SUBMITTING -> SUBMITTED -> MONITORING -> COMPLETE -> POST_TRADE -> TERMINAL."""
        lc = _lifecycle()
        assert lc.state == OrderState.INTENT_RECEIVED

        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
            OrderState.MONITORING,
            OrderState.COMPLETE,
            OrderState.POST_TRADE,
            OrderState.TERMINAL,
        )

        assert lc.is_terminal
        assert lc.transition_count == 7

    def test_approval_path(self):
        """INTENT -> PREFLIGHT -> AWAITING_APPROVAL -> SUBMITTING -> ... -> TERMINAL."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.AWAITING_APPROVAL,
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
            OrderState.MONITORING,
            OrderState.COMPLETE,
            OrderState.POST_TRADE,
            OrderState.TERMINAL,
        )
        assert lc.is_terminal
        assert lc.transition_count == 8

    def test_partial_fill_path(self):
        """Order goes through partial fill before completing."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
            OrderState.MONITORING,
            OrderState.PARTIALLY_FILLED,
            OrderState.MONITORING,  # back to monitoring
            OrderState.COMPLETE,
            OrderState.POST_TRADE,
            OrderState.TERMINAL,
        )
        assert lc.is_terminal

    def test_abort_path(self):
        """Order is aborted during monitoring."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
            OrderState.MONITORING,
            OrderState.ABORT,
            OrderState.POST_TRADE,
            OrderState.TERMINAL,
        )
        assert lc.is_terminal

    def test_blocked_path(self):
        """Policy blocks the order."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.BLOCKED,
        )
        assert lc.is_terminal

    def test_approval_denied_path(self):
        """Approval is denied."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.AWAITING_APPROVAL,
            OrderState.APPROVAL_DENIED,
        )
        assert lc.is_terminal

    def test_approval_expired_path(self):
        """Approval request expires."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.AWAITING_APPROVAL,
            OrderState.APPROVAL_EXPIRED,
        )
        assert lc.is_terminal

    def test_submit_failed_path(self):
        """Exchange submission fails."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.SUBMITTING,
            OrderState.SUBMIT_FAILED,
        )
        assert lc.is_terminal


# ===========================================================================
# Invalid transitions
# ===========================================================================


class TestInvalidTransitions:
    def test_cannot_skip_preflight(self):
        """Cannot go directly from INTENT_RECEIVED to SUBMITTING."""
        lc = _lifecycle()
        with pytest.raises(ValueError, match="Invalid transition"):
            lc.transition(OrderState.SUBMITTING)

    def test_cannot_go_backwards(self):
        """Cannot transition backwards."""
        lc = _lifecycle()
        lc.transition(OrderState.PREFLIGHT_POLICY)
        with pytest.raises(ValueError, match="Invalid transition"):
            lc.transition(OrderState.INTENT_RECEIVED)

    def test_cannot_exit_terminal(self):
        """Terminal states cannot be exited."""
        lc = _lifecycle()
        _advance_to(lc, OrderState.PREFLIGHT_POLICY, OrderState.BLOCKED)
        assert lc.is_terminal

        with pytest.raises(ValueError, match="terminal state"):
            lc.transition(OrderState.SUBMITTING)

    def test_cannot_go_from_monitoring_to_submitting(self):
        """MONITORING can only go to PARTIALLY_FILLED, COMPLETE, or ABORT."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
            OrderState.MONITORING,
        )
        with pytest.raises(ValueError, match="Invalid transition"):
            lc.transition(OrderState.SUBMITTING)

    def test_cannot_go_from_complete_to_monitoring(self):
        """COMPLETE can only go to POST_TRADE."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
            OrderState.MONITORING,
            OrderState.COMPLETE,
        )
        with pytest.raises(ValueError, match="Invalid transition"):
            lc.transition(OrderState.MONITORING)


# ===========================================================================
# History tracking
# ===========================================================================


class TestHistory:
    def test_history_records_transitions(self):
        lc = _lifecycle()
        _advance_to(lc, OrderState.PREFLIGHT_POLICY, OrderState.BLOCKED)

        assert len(lc.history) == 2
        assert lc.history[0][0] == OrderState.INTENT_RECEIVED
        assert lc.history[0][1] == OrderState.PREFLIGHT_POLICY
        assert lc.history[1][0] == OrderState.PREFLIGHT_POLICY
        assert lc.history[1][1] == OrderState.BLOCKED

    def test_history_timestamps_monotonic(self):
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
        )
        timestamps = [t for _, _, t in lc.history]
        assert timestamps == sorted(timestamps)


# ===========================================================================
# Timeout detection
# ===========================================================================


class TestTimeouts:
    def test_not_timed_out_immediately(self):
        lc = _lifecycle()
        lc.transition(OrderState.PREFLIGHT_POLICY)
        assert not lc.is_timed_out()

    def test_timed_out_after_threshold(self):
        lc = _lifecycle(timeouts={OrderState.PREFLIGHT_POLICY: 0.001})
        lc.transition(OrderState.PREFLIGHT_POLICY)
        time.sleep(0.01)
        assert lc.is_timed_out()

    def test_no_timeout_for_unlisted_state(self):
        lc = _lifecycle()
        # INTENT_RECEIVED has no timeout in defaults
        assert not lc.is_timed_out()

    def test_time_in_state(self):
        lc = _lifecycle()
        time.sleep(0.01)
        assert lc.time_in_state() > 0

    def test_total_time(self):
        lc = _lifecycle()
        time.sleep(0.01)
        assert lc.total_time() > 0

    def test_custom_timeout_overrides_default(self):
        lc = _lifecycle(timeouts={OrderState.PREFLIGHT_POLICY: 99.0})
        assert lc.timeout_for_state(OrderState.PREFLIGHT_POLICY) == 99.0


# ===========================================================================
# Audit serialization
# ===========================================================================


class TestAuditSerialization:
    def test_to_audit_dict(self):
        lc = _lifecycle(action_id="abc123", correlation_id="corr456")
        _advance_to(lc, OrderState.PREFLIGHT_POLICY, OrderState.BLOCKED)
        lc.error = "policy_blocked"

        audit = lc.to_audit_dict()
        assert audit["action_id"] == "abc123"
        assert audit["correlation_id"] == "corr456"
        assert audit["state"] == "blocked"
        assert len(audit["history"]) == 2
        assert audit["error"] == "policy_blocked"
        assert audit["total_time_s"] >= 0

    def test_fills_tracked(self):
        lc = _lifecycle()
        lc.fills.append({"fill_id": "f1", "qty": "0.5"})
        audit = lc.to_audit_dict()
        assert audit["fill_count"] == 1


# ===========================================================================
# Context accumulation
# ===========================================================================


class TestContext:
    def test_policy_decision_stored(self):
        lc = _lifecycle()
        lc.policy_decision = {"allowed": True, "tier": "T0"}
        assert lc.policy_decision["allowed"] is True

    def test_tool_result_stored(self):
        lc = _lifecycle()
        lc.tool_result = {"success": True, "order_id": "o1"}
        assert lc.tool_result["success"] is True


# ===========================================================================
# OrderLifecycleManager
# ===========================================================================


class TestOrderLifecycleManager:
    def test_create_and_get(self):
        mgr = OrderLifecycleManager()
        lc = mgr.create("a1", "c1")
        assert lc.action_id == "a1"
        assert mgr.get("a1") is lc
        assert mgr.count == 1

    def test_duplicate_create_raises(self):
        mgr = OrderLifecycleManager()
        mgr.create("a1", "c1")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create("a1", "c1")

    def test_get_nonexistent_returns_none(self):
        mgr = OrderLifecycleManager()
        assert mgr.get("nope") is None

    def test_remove(self):
        mgr = OrderLifecycleManager()
        mgr.create("a1", "c1")
        removed = mgr.remove("a1")
        assert removed is not None
        assert mgr.count == 0
        assert mgr.remove("a1") is None

    def test_get_by_state(self):
        mgr = OrderLifecycleManager()
        lc1 = mgr.create("a1", "c1")
        lc2 = mgr.create("a2", "c2")
        lc1.transition(OrderState.PREFLIGHT_POLICY)

        result = mgr.get_by_state(OrderState.PREFLIGHT_POLICY)
        assert len(result) == 1
        assert result[0] is lc1

    def test_get_active(self):
        mgr = OrderLifecycleManager()
        lc1 = mgr.create("a1", "c1")
        lc2 = mgr.create("a2", "c2")
        _advance_to(lc2, OrderState.PREFLIGHT_POLICY, OrderState.BLOCKED)

        active = mgr.get_active()
        assert len(active) == 1
        assert active[0] is lc1

    def test_active_count(self):
        mgr = OrderLifecycleManager()
        mgr.create("a1", "c1")
        lc2 = mgr.create("a2", "c2")
        _advance_to(lc2, OrderState.PREFLIGHT_POLICY, OrderState.BLOCKED)

        assert mgr.active_count == 1
        assert mgr.count == 2

    def test_get_timed_out(self):
        mgr = OrderLifecycleManager()
        lc = mgr.create("a1", "c1", timeouts={OrderState.PREFLIGHT_POLICY: 0.001})
        lc.transition(OrderState.PREFLIGHT_POLICY)
        time.sleep(0.01)

        timed_out = mgr.get_timed_out()
        assert len(timed_out) == 1
        assert timed_out[0] is lc

    def test_cleanup_terminal(self):
        mgr = OrderLifecycleManager()
        lc = mgr.create("a1", "c1")
        _advance_to(lc, OrderState.PREFLIGHT_POLICY, OrderState.BLOCKED)

        # Not old enough yet
        removed = mgr.cleanup_terminal(max_age_seconds=9999)
        assert removed == 0

        # Force age by using very short max_age
        removed = mgr.cleanup_terminal(max_age_seconds=0.0)
        assert removed == 1
        assert mgr.count == 0

    def test_summary(self):
        mgr = OrderLifecycleManager()
        mgr.create("a1", "c1")
        lc2 = mgr.create("a2", "c2")
        lc3 = mgr.create("a3", "c3")
        _advance_to(lc2, OrderState.PREFLIGHT_POLICY, OrderState.BLOCKED)
        lc3.transition(OrderState.PREFLIGHT_POLICY)

        summary = mgr.summary()
        assert summary["intent_received"] == 1
        assert summary["blocked"] == 1
        assert summary["preflight_policy"] == 1


# ===========================================================================
# Integration: full lifecycle flow
# ===========================================================================


class TestLifecycleIntegration:
    def test_full_order_lifecycle_with_manager(self):
        """End-to-end: create lifecycle, advance through states, cleanup."""
        mgr = OrderLifecycleManager()

        # Order 1: happy path
        lc1 = mgr.create("a1", "c1")
        _advance_to(
            lc1,
            OrderState.PREFLIGHT_POLICY,
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
            OrderState.MONITORING,
            OrderState.COMPLETE,
            OrderState.POST_TRADE,
            OrderState.TERMINAL,
        )
        lc1.fills.append({"fill_id": "f1"})

        # Order 2: blocked by policy
        lc2 = mgr.create("a2", "c2")
        _advance_to(lc2, OrderState.PREFLIGHT_POLICY, OrderState.BLOCKED)
        lc2.error = "policy_blocked: max_notional_exceeded"

        # Order 3: still in progress
        lc3 = mgr.create("a3", "c3")
        lc3.transition(OrderState.PREFLIGHT_POLICY)

        assert mgr.count == 3
        assert mgr.active_count == 1  # only lc3

        # Cleanup old terminals
        removed = mgr.cleanup_terminal(max_age_seconds=0.0)
        assert removed == 2
        assert mgr.count == 1
        assert mgr.get("a3") is lc3

    def test_partial_fill_loop(self):
        """Order goes through multiple partial fills."""
        lc = _lifecycle()
        _advance_to(
            lc,
            OrderState.PREFLIGHT_POLICY,
            OrderState.SUBMITTING,
            OrderState.SUBMITTED,
            OrderState.MONITORING,
        )

        # Simulate 3 partial fills
        for i in range(3):
            lc.transition(OrderState.PARTIALLY_FILLED)
            lc.fills.append({"fill_id": f"f{i}"})
            lc.transition(OrderState.MONITORING)

        # Final complete
        lc.transition(OrderState.COMPLETE)
        lc.transition(OrderState.POST_TRADE)
        lc.transition(OrderState.TERMINAL)

        assert lc.is_terminal
        assert len(lc.fills) == 3
        assert lc.transition_count == 13  # 4 + 3*2 + 3
