"""Property test: OrderManager state machine invariants.

Uses hypothesis to generate random sequences of state transitions and verify
that OrderManager never enters an invalid state.
"""

import pytest
from decimal import Decimal

from hypothesis import given, settings, strategies as st, assume

from agentic_trading.core.enums import Exchange, OrderStatus, OrderType, Side, TimeInForce
from agentic_trading.core.errors import DuplicateOrderError, OrderRejectedError
from agentic_trading.core.events import OrderIntent, OrderUpdate
from agentic_trading.execution.order_manager import OrderManager, _VALID_TRANSITIONS


# All possible statuses
ALL_STATUSES = list(OrderStatus)

# Terminal statuses
TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
}

# Non-terminal statuses
ACTIVE_STATUSES = {
    OrderStatus.PENDING,
    OrderStatus.SUBMITTED,
    OrderStatus.PARTIALLY_FILLED,
}


def _make_intent(key: str) -> OrderIntent:
    """Create a test OrderIntent."""
    return OrderIntent(
        dedupe_key=key,
        strategy_id="prop_test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        qty=Decimal("1.0"),
        price=Decimal("50000"),
    )


def _make_update(key: str, status: OrderStatus) -> OrderUpdate:
    """Create a test OrderUpdate."""
    return OrderUpdate(
        order_id=f"order-{key}",
        client_order_id=key,
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        status=status,
        filled_qty=Decimal("0.5") if status in (
            OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED
        ) else Decimal("0"),
        remaining_qty=Decimal("0.5") if status == OrderStatus.PARTIALLY_FILLED else Decimal("0"),
    )


@given(
    transitions=st.lists(
        st.sampled_from(ALL_STATUSES),
        min_size=1,
        max_size=10,
    ),
)
def test_random_transitions_never_corrupt_state(transitions):
    """Apply random state transitions and verify invariants hold.

    The OrderManager should either:
    - Accept a valid transition and update the state, or
    - Reject an invalid transition with OrderRejectedError.

    It should never silently enter an inconsistent state.
    """
    mgr = OrderManager()
    intent = _make_intent("prop-key")
    mgr.register_intent(intent)

    current_status = OrderStatus.PENDING

    for new_status in transitions:
        allowed = _VALID_TRANSITIONS.get(current_status, frozenset())

        if new_status in allowed or new_status == current_status:
            # Valid transition or same-status (no-op): should succeed
            tracked = mgr.update_order(_make_update("prop-key", new_status))
            if new_status != current_status:
                current_status = new_status
            assert tracked.status == current_status
        else:
            # Invalid transition: should raise
            with pytest.raises(OrderRejectedError):
                mgr.update_order(_make_update("prop-key", new_status))

    # Invariant: after all transitions, the tracked order status matches
    tracked = mgr.get_order("prop-key")
    assert tracked is not None
    assert tracked.status == current_status

    # Invariant: terminal orders cannot be active
    if current_status in TERMINAL_STATUSES:
        assert tracked.is_terminal
        active = mgr.get_active_orders()
        assert tracked not in active


@given(
    n_orders=st.integers(min_value=1, max_value=20),
)
def test_multiple_orders_independent(n_orders):
    """Multiple orders tracked simultaneously maintain independent state."""
    mgr = OrderManager()

    for i in range(n_orders):
        key = f"multi-{i}"
        mgr.register_intent(_make_intent(key))

    assert mgr.total_count == n_orders
    assert mgr.active_count == n_orders

    # Transition first order to FILLED
    if n_orders > 0:
        mgr.update_order(_make_update("multi-0", OrderStatus.SUBMITTED))
        mgr.update_order(_make_update("multi-0", OrderStatus.FILLED))
        assert mgr.active_count == n_orders - 1

    # Other orders are still PENDING
    for i in range(1, n_orders):
        tracked = mgr.get_order(f"multi-{i}")
        assert tracked is not None
        assert tracked.status == OrderStatus.PENDING


@given(
    keys=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=20,
        ),
        min_size=2,
        max_size=10,
        unique=True,
    ),
)
def test_dedupe_keys_are_unique(keys):
    """Each unique dedupe_key can only be registered once."""
    mgr = OrderManager()

    for key in keys:
        mgr.register_intent(_make_intent(key))

    # All keys should be seen
    for key in keys:
        assert mgr.dedupe_check(key)

    # Re-registering any key should fail
    for key in keys:
        with pytest.raises(DuplicateOrderError):
            mgr.register_intent(_make_intent(key))


def test_valid_transitions_map_is_consistent():
    """Verify the transition map itself is well-formed.

    - Every status appears as a key.
    - Terminal states have no outgoing transitions.
    - Non-terminal states have at least one outgoing transition.
    """
    for status in OrderStatus:
        assert status in _VALID_TRANSITIONS, f"{status} missing from transition map"

    for status in TERMINAL_STATUSES:
        assert len(_VALID_TRANSITIONS[status]) == 0, (
            f"Terminal status {status} should have no outgoing transitions"
        )

    for status in ACTIVE_STATUSES:
        assert len(_VALID_TRANSITIONS[status]) > 0, (
            f"Active status {status} should have outgoing transitions"
        )
