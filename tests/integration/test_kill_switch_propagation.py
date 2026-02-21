"""Integration test: kill switch propagation to ExecutionEngine.

Verifies that when the kill switch is activated (via the event bus),
the ExecutionEngine rejects new OrderIntents with a REJECTED status.
"""

import pytest
from decimal import Decimal

from agentic_trading.core.enums import Exchange, OrderStatus, OrderType, Side, TimeInForce
from agentic_trading.core.events import (
    KillSwitchEvent,
    OrderAck,
    OrderIntent,
    RiskCheckResult,
)
from agentic_trading.core.interfaces import PortfolioState
from agentic_trading.core.models import Fill
from agentic_trading.event_bus.memory_bus import MemoryEventBus
from agentic_trading.execution.engine import ExecutionEngine


class _AlwaysPassRiskChecker:
    """Minimal risk checker that always passes pre-trade and post-trade checks."""

    def pre_trade_check(
        self, intent: OrderIntent, portfolio_state: PortfolioState
    ) -> RiskCheckResult:
        return RiskCheckResult(
            check_name="always_pass",
            passed=True,
            order_intent_id=intent.event_id,
        )

    def post_trade_check(
        self, fill: Fill, portfolio_state: PortfolioState
    ) -> RiskCheckResult:
        return RiskCheckResult(
            check_name="always_pass",
            passed=True,
        )


class _DummyAdapter:
    """Minimal exchange adapter that records submissions."""

    def __init__(self):
        self.submitted = []
        self._fill_prices: dict[str, Decimal] = {}

    async def submit_order(self, intent: OrderIntent) -> OrderAck:
        self.submitted.append(intent)
        # Store fill price so _resolve_fill_price can find it
        self._fill_prices["fake-order-id"] = Decimal("50000")
        return OrderAck(
            order_id="fake-order-id",
            client_order_id=intent.dedupe_key,
            symbol=intent.symbol,
            exchange=intent.exchange,
            status=OrderStatus.FILLED,
            message="Filled",
        )


def _make_intent(dedupe_key: str) -> OrderIntent:
    """Create a test OrderIntent."""
    return OrderIntent(
        dedupe_key=dedupe_key,
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=Decimal("0.01"),
    )


@pytest.mark.asyncio
async def test_kill_switch_rejects_new_orders():
    """After kill switch activation, new OrderIntents are rejected."""
    bus = MemoryEventBus()
    adapter = _DummyAdapter()
    risk_checker = _AlwaysPassRiskChecker()

    engine = ExecutionEngine(
        adapter=adapter,
        event_bus=bus,
        risk_manager=risk_checker,
    )
    await engine.start()

    # Submit an order before kill switch -- should succeed
    intent_before = _make_intent("before-kill")
    ack_before = await engine.handle_intent(intent_before)
    assert ack_before is not None
    assert ack_before.status == OrderStatus.FILLED
    assert len(adapter.submitted) == 1

    # Activate kill switch via the event bus
    kill_event = KillSwitchEvent(
        activated=True,
        reason="test: drawdown limit hit",
        triggered_by="test",
    )
    await bus.publish("system", kill_event)

    # Submit an order after kill switch -- should be rejected
    intent_after = _make_intent("after-kill")
    ack_after = await engine.handle_intent(intent_after)
    assert ack_after is not None
    assert ack_after.status == OrderStatus.REJECTED
    assert "Kill switch" in ack_after.message

    # The adapter should not have received the second order
    assert len(adapter.submitted) == 1

    await engine.stop()


@pytest.mark.asyncio
async def test_kill_switch_deactivation_allows_orders():
    """After deactivating the kill switch, orders flow through again."""
    bus = MemoryEventBus()
    adapter = _DummyAdapter()
    risk_checker = _AlwaysPassRiskChecker()

    engine = ExecutionEngine(
        adapter=adapter,
        event_bus=bus,
        risk_manager=risk_checker,
    )
    await engine.start()

    # Activate kill switch
    await bus.publish("system", KillSwitchEvent(
        activated=True, reason="test", triggered_by="test",
    ))

    # Deactivate kill switch
    await bus.publish("system", KillSwitchEvent(
        activated=False, reason="test deactivated", triggered_by="test",
    ))

    # Submit an order -- should succeed now
    intent = _make_intent("after-deactivate")
    ack = await engine.handle_intent(intent)
    assert ack is not None
    assert ack.status == OrderStatus.FILLED

    await engine.stop()


@pytest.mark.asyncio
async def test_kill_switch_callable_mode():
    """ExecutionEngine also checks a callable kill switch."""
    bus = MemoryEventBus()
    adapter = _DummyAdapter()
    risk_checker = _AlwaysPassRiskChecker()

    kill_active = False

    def kill_switch_fn():
        return kill_active

    engine = ExecutionEngine(
        adapter=adapter,
        event_bus=bus,
        risk_manager=risk_checker,
        kill_switch=kill_switch_fn,
    )
    await engine.start()

    # Submit before activation -- succeeds
    ack1 = await engine.handle_intent(_make_intent("callable-before"))
    assert ack1.status == OrderStatus.FILLED

    # Activate via callable
    kill_active = True

    # Submit after activation -- rejected
    ack2 = await engine.handle_intent(_make_intent("callable-after"))
    assert ack2.status == OrderStatus.REJECTED

    await engine.stop()
