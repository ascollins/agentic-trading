"""Test OrderManager dedupe_check, state machine transitions, retry tracking."""

from decimal import Decimal

import pytest

from agentic_trading.core.enums import Exchange, OrderStatus, Side
from agentic_trading.core.errors import DuplicateOrderError, OrderRejectedError
from agentic_trading.core.events import OrderIntent, OrderUpdate
from agentic_trading.execution.order_manager import OrderManager


def _make_intent(dedupe_key: str = "key-001") -> OrderIntent:
    return OrderIntent(
        dedupe_key=dedupe_key,
        strategy_id="test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        qty=Decimal("0.1"),
    )


def _make_update(
    client_order_id: str,
    status: OrderStatus,
    order_id: str = "exch-001",
) -> OrderUpdate:
    return OrderUpdate(
        order_id=order_id,
        client_order_id=client_order_id,
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        status=status,
    )


class TestDedupeCheck:
    def test_unseen_key_returns_false(self):
        mgr = OrderManager()
        assert mgr.dedupe_check("new-key") is False

    def test_seen_key_returns_true(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        assert mgr.dedupe_check("key-001") is True

    def test_different_key_still_unseen(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        assert mgr.dedupe_check("key-002") is False


class TestRegisterIntent:
    def test_register_new_intent(self):
        mgr = OrderManager()
        tracked = mgr.register_intent(_make_intent("key-001"))
        assert tracked.dedupe_key == "key-001"
        assert tracked.status == OrderStatus.PENDING

    def test_duplicate_raises(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        with pytest.raises(DuplicateOrderError, match="Duplicate"):
            mgr.register_intent(_make_intent("key-001"))


class TestStateTransitions:
    def test_pending_to_submitted(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        update = _make_update("key-001", OrderStatus.SUBMITTED)
        tracked = mgr.update_order(update)
        assert tracked.status == OrderStatus.SUBMITTED

    def test_submitted_to_filled(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        mgr.update_order(_make_update("key-001", OrderStatus.SUBMITTED))
        tracked = mgr.update_order(_make_update("key-001", OrderStatus.FILLED))
        assert tracked.status == OrderStatus.FILLED
        assert tracked.is_terminal is True

    def test_submitted_to_partially_filled(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        mgr.update_order(_make_update("key-001", OrderStatus.SUBMITTED))
        tracked = mgr.update_order(_make_update("key-001", OrderStatus.PARTIALLY_FILLED))
        assert tracked.status == OrderStatus.PARTIALLY_FILLED
        assert tracked.is_terminal is False

    def test_partially_filled_to_filled(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        mgr.update_order(_make_update("key-001", OrderStatus.SUBMITTED))
        mgr.update_order(_make_update("key-001", OrderStatus.PARTIALLY_FILLED))
        tracked = mgr.update_order(_make_update("key-001", OrderStatus.FILLED))
        assert tracked.status == OrderStatus.FILLED

    def test_invalid_transition_raises(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        # PENDING -> FILLED is not a valid transition (must go through SUBMITTED)
        with pytest.raises(OrderRejectedError, match="Invalid order transition"):
            mgr.update_order(_make_update("key-001", OrderStatus.FILLED))

    def test_pending_to_rejected(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        tracked = mgr.update_order(_make_update("key-001", OrderStatus.REJECTED))
        assert tracked.status == OrderStatus.REJECTED
        assert tracked.is_terminal is True


class TestRetryTracking:
    def test_should_retry_new_key(self):
        mgr = OrderManager(max_retries=3)
        assert mgr.should_retry("new-key") is True

    def test_should_retry_within_limit(self):
        mgr = OrderManager(max_retries=3)
        mgr.register_intent(_make_intent("key-001"))
        mgr.record_attempt("key-001")
        mgr.record_attempt("key-001")
        assert mgr.should_retry("key-001") is True

    def test_should_not_retry_at_limit(self):
        mgr = OrderManager(max_retries=3)
        mgr.register_intent(_make_intent("key-001"))
        mgr.record_attempt("key-001")
        mgr.record_attempt("key-001")
        mgr.record_attempt("key-001")
        assert mgr.should_retry("key-001") is False

    def test_should_not_retry_terminal(self):
        mgr = OrderManager(max_retries=3)
        mgr.register_intent(_make_intent("key-001"))
        mgr.update_order(_make_update("key-001", OrderStatus.REJECTED))
        assert mgr.should_retry("key-001") is False

    def test_record_attempt_increments(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        assert mgr.get_attempt_count("key-001") == 0
        mgr.record_attempt("key-001")
        assert mgr.get_attempt_count("key-001") == 1


class TestOrderManagerQueries:
    def test_get_active_orders(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        mgr.register_intent(_make_intent("key-002"))
        # Must go through SUBMITTED before FILLED
        mgr.update_order(_make_update("key-001", OrderStatus.SUBMITTED))
        mgr.update_order(_make_update("key-001", OrderStatus.FILLED))
        active = mgr.get_active_orders()
        assert len(active) == 1
        assert active[0].dedupe_key == "key-002"

    def test_active_count(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        mgr.register_intent(_make_intent("key-002"))
        assert mgr.active_count == 2
        mgr.update_order(_make_update("key-001", OrderStatus.REJECTED))
        assert mgr.active_count == 1

    def test_total_count(self):
        mgr = OrderManager()
        mgr.register_intent(_make_intent("key-001"))
        mgr.register_intent(_make_intent("key-002"))
        assert mgr.total_count == 2
