"""Property test: position open/close math.

Uses hypothesis to verify that if you buy X and sell X, the net position
quantity is zero. Tests the PaperAdapter's position management logic.
"""

import pytest
from decimal import Decimal

from hypothesis import given, settings, strategies as st, assume

from agentic_trading.core.enums import (
    Exchange,
    OrderType,
    Side,
    TimeInForce,
)
from agentic_trading.core.events import OrderIntent
from agentic_trading.execution.adapters.paper import PaperAdapter


@given(
    qty=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("100"), places=6),
    price=st.decimals(min_value=Decimal("1"), max_value=Decimal("50000"), places=2),
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_buy_then_sell_same_qty_nets_to_zero(qty, price):
    """Buying X then selling X results in a zero-quantity position."""
    assume(qty * price < Decimal("5000000"))  # Stay within balance
    adapter = PaperAdapter(
        exchange=Exchange.BINANCE,
        initial_balances={"USDT": Decimal("10000000")},
    )
    adapter.set_market_price("BTC/USDT", price)

    # Buy
    buy_intent = OrderIntent(
        dedupe_key="buy-001",
        strategy_id="test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=qty,
    )
    await adapter.submit_order(buy_intent)

    # Sell the same quantity
    sell_intent = OrderIntent(
        dedupe_key="sell-001",
        strategy_id="test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.SELL,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=qty,
    )
    await adapter.submit_order(sell_intent)

    # Position should be flat
    positions = await adapter.get_positions("BTC/USDT")
    # If qty is zero the position is not returned by get_positions
    if positions:
        assert positions[0].qty == Decimal("0"), (
            f"Expected zero position after buy+sell of {qty}, "
            f"got {positions[0].qty}"
        )


@given(
    qty=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("1000"), places=6),
    price=st.decimals(min_value=Decimal("1"), max_value=Decimal("100000"), places=2),
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_sell_then_buy_same_qty_nets_to_zero(qty, price):
    """Selling X (going short) then buying X back results in zero position."""
    adapter = PaperAdapter(
        exchange=Exchange.BINANCE,
        initial_balances={"USDT": Decimal("10000000")},
    )
    adapter.set_market_price("BTC/USDT", price)

    # Sell (open short)
    sell_intent = OrderIntent(
        dedupe_key="sell-first",
        strategy_id="test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.SELL,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=qty,
    )
    await adapter.submit_order(sell_intent)

    # Buy to close
    buy_intent = OrderIntent(
        dedupe_key="buy-close",
        strategy_id="test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=qty,
    )
    await adapter.submit_order(buy_intent)

    positions = await adapter.get_positions("BTC/USDT")
    if positions:
        assert positions[0].qty == Decimal("0"), (
            f"Expected zero position after sell+buy of {qty}, "
            f"got {positions[0].qty}"
        )


@given(
    buy_qty=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("100"), places=6),
    sell_qty=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("100"), places=6),
    price=st.decimals(min_value=Decimal("10"), max_value=Decimal("100000"), places=2),
)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_partial_close_leaves_remainder(buy_qty, sell_qty, price):
    """Buying X and selling Y leaves position of X - Y."""
    assume(buy_qty != sell_qty)
    # Ensure notional + fees fits within the initial balance
    assume(buy_qty * price < Decimal("5000000"))

    adapter = PaperAdapter(
        exchange=Exchange.BINANCE,
        initial_balances={"USDT": Decimal("10000000")},
    )
    adapter.set_market_price("BTC/USDT", price)

    # Buy
    buy_intent = OrderIntent(
        dedupe_key="buy-partial",
        strategy_id="test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=buy_qty,
    )
    await adapter.submit_order(buy_intent)

    # Sell partial or more
    sell_intent = OrderIntent(
        dedupe_key="sell-partial",
        strategy_id="test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.SELL,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=sell_qty,
    )
    await adapter.submit_order(sell_intent)

    # Check the resulting position.  The internal _positions dict always
    # has the entry, but get_positions filters out zero-qty.
    expected_qty = buy_qty - sell_qty  # Can be negative (short)

    # Access internal state directly for assertion
    pos = adapter._positions.get("BTC/USDT")
    assert pos is not None
    assert pos.qty == expected_qty, (
        f"Expected position qty={expected_qty} after buying {buy_qty} "
        f"and selling {sell_qty}, got {pos.qty}"
    )


@given(
    n_trades=st.integers(min_value=2, max_value=10),
    per_trade_qty=st.decimals(
        min_value=Decimal("0.01"), max_value=Decimal("10"), places=4,
    ),
    price=st.decimals(min_value=Decimal("100"), max_value=Decimal("50000"), places=2),
)
@settings(max_examples=50)
@pytest.mark.asyncio
async def test_multiple_buys_then_full_sell_nets_to_zero(n_trades, per_trade_qty, price):
    """Buy in N increments then sell the total in one shot."""
    adapter = PaperAdapter(
        exchange=Exchange.BINANCE,
        initial_balances={"USDT": Decimal("10000000000")},
    )
    adapter.set_market_price("BTC/USDT", price)

    total_qty = per_trade_qty * n_trades

    # Buy in increments
    for i in range(n_trades):
        intent = OrderIntent(
            dedupe_key=f"inc-buy-{i}",
            strategy_id="test",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            qty=per_trade_qty,
        )
        await adapter.submit_order(intent)

    # Verify accumulated position
    pos_before = adapter._positions.get("BTC/USDT")
    assert pos_before is not None
    assert pos_before.qty == total_qty

    # Sell everything
    sell_intent = OrderIntent(
        dedupe_key="sell-all",
        strategy_id="test",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.SELL,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=total_qty,
    )
    await adapter.submit_order(sell_intent)

    pos_after = adapter._positions.get("BTC/USDT")
    assert pos_after is not None
    assert pos_after.qty == Decimal("0"), (
        f"Expected zero after buying {n_trades}x{per_trade_qty} "
        f"and selling {total_qty}, got {pos_after.qty}"
    )
