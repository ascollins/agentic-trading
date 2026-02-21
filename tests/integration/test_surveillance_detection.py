"""Integration test: SurveillanceAgent wash trade and spoofing detection.

Verifies that the SurveillanceAgent subscribes to execution events via
MemoryEventBus, detects abuse patterns, publishes SurveillanceCaseEvent,
and creates cases in CaseManager.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from agentic_trading.agents.surveillance import SurveillanceAgent
from agentic_trading.compliance.case_manager import CaseManager
from agentic_trading.core.enums import Exchange, OrderStatus, Side
from agentic_trading.core.events import (
    FillEvent,
    OrderAck,
    OrderIntent,
    OrderUpdate,
    SurveillanceCaseEvent,
)
from agentic_trading.core.ids import new_id
from agentic_trading.event_bus.memory_bus import MemoryEventBus


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_fill(
    *,
    symbol: str = "BTC/USDT",
    side: Side = Side.BUY,
    strategy_id: str = "strat_a",
    price: Decimal = Decimal("50000"),
    qty: Decimal = Decimal("0.1"),
) -> FillEvent:
    return FillEvent(
        fill_id=new_id(),
        order_id=new_id(),
        client_order_id="",
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        price=price,
        qty=qty,
        fee=Decimal("0"),
        fee_currency="USDT",
        strategy_id=strategy_id,
    )


def _make_intent(
    *,
    symbol: str = "BTC/USDT",
    side: Side = Side.BUY,
    strategy_id: str = "strat_a",
    dedupe_key: str = "",
    qty: Decimal = Decimal("1.0"),
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        order_type="limit",
        qty=qty,
        price=Decimal("50000"),
        time_in_force="GTC",
        dedupe_key=dedupe_key or new_id(),
        strategy_id=strategy_id,
    )


def _make_ack(
    *,
    order_id: str = "",
    client_order_id: str = "",
    symbol: str = "BTC/USDT",
    status: OrderStatus = OrderStatus.SUBMITTED,
) -> OrderAck:
    return OrderAck(
        order_id=order_id or new_id(),
        client_order_id=client_order_id,
        symbol=symbol,
        exchange=Exchange.BYBIT,
        status=status,
    )


def _make_update(
    *,
    order_id: str = "",
    client_order_id: str = "",
    symbol: str = "BTC/USDT",
    status: OrderStatus = OrderStatus.CANCELLED,
) -> OrderUpdate:
    return OrderUpdate(
        order_id=order_id or new_id(),
        client_order_id=client_order_id,
        symbol=symbol,
        exchange=Exchange.BYBIT,
        status=status,
    )


async def _settle() -> None:
    """Give handlers time to process events."""
    await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSurveillanceDetection:

    @pytest.mark.asyncio
    async def test_wash_trade_detected_buy_then_sell(self):
        """Buy fill followed by sell fill on same symbol/strategy within window."""
        bus = MemoryEventBus()
        cm = CaseManager()
        captured: list[SurveillanceCaseEvent] = []

        async def capture(event):
            captured.append(event)

        await bus.subscribe("surveillance", "test", capture)

        agent = SurveillanceAgent(
            event_bus=bus,
            case_manager=cm,
            wash_trade_window_sec=30.0,
        )
        await agent.start()

        # Publish buy fill then sell fill on same symbol/strategy
        await bus.publish("execution.fill", _make_fill(side=Side.BUY))
        await _settle()
        await bus.publish("execution.fill", _make_fill(side=Side.SELL))
        await _settle()

        assert agent.cases_created >= 1
        assert len(captured) >= 1
        assert captured[0].case_type == "wash_trade"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_wash_trade_detected_sell_then_buy(self):
        """Sell fill followed by buy fill on same symbol/strategy within window."""
        bus = MemoryEventBus()
        cm = CaseManager()

        agent = SurveillanceAgent(
            event_bus=bus,
            case_manager=cm,
            wash_trade_window_sec=30.0,
        )
        await agent.start()

        await bus.publish("execution.fill", _make_fill(side=Side.SELL))
        await _settle()
        await bus.publish("execution.fill", _make_fill(side=Side.BUY))
        await _settle()

        assert agent.cases_created >= 1
        await agent.stop()

    @pytest.mark.asyncio
    async def test_wash_trade_not_raised_across_strategies(self):
        """Buy and sell on same symbol but different strategies is not wash trade."""
        bus = MemoryEventBus()
        cm = CaseManager()

        agent = SurveillanceAgent(
            event_bus=bus,
            case_manager=cm,
            wash_trade_window_sec=30.0,
        )
        await agent.start()

        await bus.publish("execution.fill", _make_fill(
            side=Side.BUY, strategy_id="strat_a",
        ))
        await _settle()
        await bus.publish("execution.fill", _make_fill(
            side=Side.SELL, strategy_id="strat_b",
        ))
        await _settle()

        assert agent.cases_created == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_wash_trade_not_raised_different_symbol(self):
        """Opposite fills on different symbols are not wash trades."""
        bus = MemoryEventBus()
        cm = CaseManager()

        agent = SurveillanceAgent(
            event_bus=bus,
            case_manager=cm,
            wash_trade_window_sec=30.0,
        )
        await agent.start()

        await bus.publish("execution.fill", _make_fill(
            side=Side.BUY, symbol="BTC/USDT",
        ))
        await _settle()
        await bus.publish("execution.fill", _make_fill(
            side=Side.SELL, symbol="ETH/USDT",
        ))
        await _settle()

        assert agent.cases_created == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_spoofing_detected_quick_cancel(self):
        """Order submitted then cancelled within spoof_cancel_window triggers case."""
        bus = MemoryEventBus()
        cm = CaseManager()
        captured: list[SurveillanceCaseEvent] = []

        async def capture(event):
            captured.append(event)

        await bus.subscribe("surveillance", "test", capture)

        agent = SurveillanceAgent(
            event_bus=bus,
            case_manager=cm,
            spoof_cancel_window_sec=5.0,
            spoof_min_qty=0.0,
        )
        await agent.start()

        dedupe = "spoof-dedupe-001"
        order_id = "spoof-order-001"

        # Submit intent
        await bus.publish("execution.intent", _make_intent(dedupe_key=dedupe))
        await _settle()

        # Ack with order_id
        await bus.publish("execution.ack", _make_ack(
            order_id=order_id, client_order_id=dedupe,
        ))
        await _settle()

        # Cancel within window
        await bus.publish("execution.update", _make_update(
            order_id=order_id, status=OrderStatus.CANCELLED,
        ))
        await _settle()

        assert agent.cases_created >= 1
        spoofing_cases = [c for c in captured if c.case_type == "spoofing"]
        assert len(spoofing_cases) >= 1

        await agent.stop()

    @pytest.mark.asyncio
    async def test_spoofing_not_raised_for_filled_order(self):
        """An order that fills (not cancelled) is not flagged as spoofing."""
        bus = MemoryEventBus()
        cm = CaseManager()

        agent = SurveillanceAgent(
            event_bus=bus,
            case_manager=cm,
            spoof_cancel_window_sec=5.0,
        )
        await agent.start()

        dedupe = "no-spoof-001"
        order_id = "filled-order-001"

        await bus.publish("execution.intent", _make_intent(dedupe_key=dedupe))
        await _settle()

        # Ack as FILLED — removes from tracking
        await bus.publish("execution.ack", _make_ack(
            order_id=order_id, client_order_id=dedupe, status=OrderStatus.FILLED,
        ))
        await _settle()

        # No spoofing cases
        assert agent.cases_created == 0
        await agent.stop()

    @pytest.mark.asyncio
    async def test_surveillance_case_persisted_in_case_manager(self):
        """Detected case is stored in CaseManager."""
        bus = MemoryEventBus()
        cm = CaseManager()

        agent = SurveillanceAgent(
            event_bus=bus,
            case_manager=cm,
            wash_trade_window_sec=30.0,
        )
        await agent.start()

        await bus.publish("execution.fill", _make_fill(side=Side.BUY))
        await _settle()
        await bus.publish("execution.fill", _make_fill(side=Side.SELL))
        await _settle()

        assert cm.total_cases >= 1
        # Verify case attributes
        wash_cases = [c for c in cm.list_open() if c.case_type == "wash_trade"]
        assert len(wash_cases) >= 1
        assert wash_cases[0].severity == "high"
        assert wash_cases[0].symbol == "BTC/USDT"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_surveillance_event_published_on_bus(self):
        """SurveillanceCaseEvent is published on 'surveillance' topic."""
        bus = MemoryEventBus()
        captured: list = []

        async def capture(event):
            captured.append(event)

        await bus.subscribe("surveillance", "test", capture)

        agent = SurveillanceAgent(event_bus=bus, wash_trade_window_sec=30.0)
        await agent.start()

        await bus.publish("execution.fill", _make_fill(side=Side.BUY))
        await _settle()
        await bus.publish("execution.fill", _make_fill(side=Side.SELL))
        await _settle()

        assert len(captured) >= 1
        assert isinstance(captured[0], SurveillanceCaseEvent)
        assert captured[0].case_type == "wash_trade"
        assert len(captured[0].evidence) >= 2

        await agent.stop()

    @pytest.mark.asyncio
    async def test_surveillance_agent_start_stop_lifecycle(self):
        """Agent starts, processes events, and stops cleanly."""
        bus = MemoryEventBus()
        agent = SurveillanceAgent(event_bus=bus, agent_id="surv-lifecycle")

        assert not agent.is_running
        await agent.start()
        assert agent.is_running

        # Publish a fill to verify event handling works
        await bus.publish("execution.fill", _make_fill())
        await _settle()

        await agent.stop()
        assert not agent.is_running
