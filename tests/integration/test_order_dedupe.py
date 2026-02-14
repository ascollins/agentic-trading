"""Integration test: OrderManager deduplication.

Verifies that OrderManager prevents duplicate submissions with the same
dedupe_key, both through the dedupe_check gate and the register_intent path.
"""

import pytest
from decimal import Decimal

from agentic_trading.core.enums import Exchange, OrderType, Side, TimeInForce
from agentic_trading.core.errors import DuplicateOrderError
from agentic_trading.core.events import OrderIntent
from agentic_trading.execution.order_manager import OrderManager


def _make_intent(dedupe_key: str = "test-key-001") -> OrderIntent:
    """Create a test OrderIntent with a given dedupe key."""
    return OrderIntent(
        dedupe_key=dedupe_key,
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        qty=Decimal("0.1"),
        price=Decimal("65000"),
    )


class TestOrderDedupe:
    """Verify that OrderManager enforces deduplication."""

    def test_first_intent_accepted(self):
        """The first registration of a dedupe_key succeeds."""
        mgr = OrderManager()
        intent = _make_intent("key-1")

        assert not mgr.dedupe_check("key-1"), "Key should not exist yet"
        tracked = mgr.register_intent(intent)

        assert tracked.dedupe_key == "key-1"
        assert tracked.symbol == "BTC/USDT"

    def test_duplicate_raises_error(self):
        """Registering the same dedupe_key twice raises DuplicateOrderError."""
        mgr = OrderManager()
        intent = _make_intent("dup-key")

        mgr.register_intent(intent)

        with pytest.raises(DuplicateOrderError):
            mgr.register_intent(intent)

    def test_dedupe_check_returns_true_after_registration(self):
        """After registration, dedupe_check returns True for the key."""
        mgr = OrderManager()
        intent = _make_intent("check-key")

        assert not mgr.dedupe_check("check-key")
        mgr.register_intent(intent)
        assert mgr.dedupe_check("check-key")

    def test_different_keys_are_independent(self):
        """Different dedupe_keys do not interfere with each other."""
        mgr = OrderManager()

        intent_a = _make_intent("key-a")
        intent_b = _make_intent("key-b")

        mgr.register_intent(intent_a)
        mgr.register_intent(intent_b)

        assert mgr.dedupe_check("key-a")
        assert mgr.dedupe_check("key-b")
        assert not mgr.dedupe_check("key-c")

    def test_seen_keys_persist_after_purge(self):
        """After purging terminal orders, dedupe keys are still remembered."""
        mgr = OrderManager()
        intent = _make_intent("purge-key")
        mgr.register_intent(intent)

        # Transition to terminal state
        from agentic_trading.core.enums import OrderStatus
        from agentic_trading.core.events import OrderUpdate

        mgr.update_order(OrderUpdate(
            order_id="order-123",
            client_order_id="purge-key",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            status=OrderStatus.REJECTED,
        ))

        # Purge terminal orders
        mgr.purge_terminal(keep_last_n=0)

        # The key is still in _seen_keys, so re-registration fails
        assert mgr.dedupe_check("purge-key")
        with pytest.raises(DuplicateOrderError):
            mgr.register_intent(_make_intent("purge-key"))

    def test_order_count_tracking(self):
        """Active and total counts are tracked correctly."""
        mgr = OrderManager()

        intent_1 = _make_intent("count-1")
        intent_2 = _make_intent("count-2")

        mgr.register_intent(intent_1)
        mgr.register_intent(intent_2)

        assert mgr.total_count == 2
        assert mgr.active_count == 2
