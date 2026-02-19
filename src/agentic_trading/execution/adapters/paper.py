"""Paper trading adapter: simulated execution with no real exchange calls.

Maintains in-memory positions, balances, and orders.  Fills are simulated
immediately at the current market price (or limit price) with configurable
slippage and maker/taker fees.  Thread-safe via asyncio locks.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import (
    Exchange,
    InstrumentType,
    MarginMode,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
)
from agentic_trading.core.errors import (
    ExchangeError,
    InsufficientBalanceError,
)
from agentic_trading.core.events import OrderAck, OrderIntent
from agentic_trading.core.models import Balance, Instrument, Order, Position

from .base import FeeSchedule, SlippageConfig

logger = logging.getLogger(__name__)


def _uid() -> str:
    return str(uuid.uuid4())


class _TradingStop:
    """Stores active TP/SL/trailing levels for a symbol."""

    __slots__ = ("take_profit", "stop_loss", "trailing_stop", "trailing_ref", "active_price")

    def __init__(
        self,
        take_profit: Decimal | None = None,
        stop_loss: Decimal | None = None,
        trailing_stop: Decimal | None = None,
        active_price: Decimal | None = None,
    ) -> None:
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        #: Best price seen since trailing was set (used for trailing logic)
        self.trailing_ref: Decimal | None = None
        #: Trailing stop only arms when price reaches this level (breakeven activation).
        #: Once reached, set to None so normal trailing logic takes over.
        self.active_price: Decimal | None = active_price


class PaperAdapter:
    """Simulated exchange adapter for paper trading.

    All state is kept in memory.  Orders are filled immediately at the
    current market price (as set via ``set_market_price``) adjusted for
    slippage and fees.

    Parameters
    ----------
    exchange:
        Which exchange this paper adapter pretends to be.
    initial_balances:
        Starting balances, e.g. ``{"USDT": Decimal("10000")}``.
    fees:
        Maker/taker fee schedule.
    slippage:
        Slippage configuration.
    instruments:
        Pre-loaded instrument metadata (symbol -> Instrument).
    """

    def __init__(
        self,
        exchange: Exchange = Exchange.BINANCE,
        initial_balances: dict[str, Decimal] | None = None,
        fees: FeeSchedule | None = None,
        slippage: SlippageConfig | None = None,
        instruments: dict[str, Instrument] | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self._exchange = exchange
        self._fees = fees or FeeSchedule()
        self._slippage = slippage or SlippageConfig()
        self._instruments: dict[str, Instrument] = instruments or {}

        # State stores
        self._orders: dict[str, Order] = {}  # order_id -> Order
        self._positions: dict[str, Position] = {}  # symbol -> Position
        self._balances: dict[str, Balance] = {}  # currency -> Balance

        # Current market prices (must be fed externally)
        self._market_prices: dict[str, Decimal] = {}  # symbol -> price

        # TP/SL simulation
        self._trading_stops: dict[str, _TradingStop] = {}  # symbol -> stops
        self._event_bus: Any | None = event_bus

        # Asyncio lock for thread safety
        self._lock = asyncio.Lock()

        # Initialise balances
        for currency, amount in (initial_balances or {}).items():
            self._balances[currency] = Balance(
                currency=currency,
                exchange=self._exchange,
                total=amount,
                free=amount,
                used=Decimal("0"),
                updated_at=datetime.now(timezone.utc),
            )

        logger.info(
            "PaperAdapter initialised (exchange=%s, balances=%s)",
            exchange.value,
            {k: str(v.total) for k, v in self._balances.items()},
        )

    # ------------------------------------------------------------------
    # External price feed
    # ------------------------------------------------------------------

    def set_market_price(self, symbol: str, price: Decimal) -> None:
        """Update the simulated market price for a symbol.

        Also checks whether the new price triggers any active TP/SL
        levels.  When triggered, a synthetic close fill is created
        and the stop is cleared.
        """
        self._market_prices[symbol] = price
        self._check_trading_stops(symbol, price)

    def get_market_price(self, symbol: str) -> Decimal | None:
        """Return the current simulated market price."""
        return self._market_prices.get(symbol)

    # ------------------------------------------------------------------
    # IExchangeAdapter: submit_order
    # ------------------------------------------------------------------

    async def submit_order(self, intent: OrderIntent) -> OrderAck:
        """Simulate order submission with immediate fill."""
        async with self._lock:
            market_price = self._market_prices.get(intent.symbol)
            if market_price is None:
                raise ExchangeError(
                    f"No market price set for {intent.symbol}. "
                    f"Call set_market_price() first."
                )

            # Determine fill price
            if intent.order_type == OrderType.MARKET:
                fill_price = self._slippage.apply(
                    market_price, intent.side.value
                )
            elif intent.price is not None:
                # For limit orders in paper mode we fill immediately at
                # the limit price (optimistic simulation).
                fill_price = intent.price
            else:
                fill_price = self._slippage.apply(
                    market_price, intent.side.value
                )

            # Determine fee
            is_maker = intent.order_type == OrderType.LIMIT
            fee_rate = self._fees.fee_for(is_maker)
            notional = fill_price * intent.qty
            fee_amount = notional * fee_rate

            # Check balance (simplified: check quote currency for buys)
            instrument = self._instruments.get(intent.symbol)
            quote_currency = "USDT"
            if instrument is not None:
                quote_currency = instrument.quote

            if intent.side == Side.BUY:
                required = notional + fee_amount
                bal = self._balances.get(quote_currency)
                if bal is None or bal.free < required:
                    raise InsufficientBalanceError(
                        f"Insufficient {quote_currency} balance: "
                        f"need {required}, have "
                        f"{bal.free if bal else Decimal('0')}"
                    )
                # Debit quote currency
                self._balances[quote_currency] = bal.model_copy(
                    update={
                        "free": bal.free - required,
                        "used": bal.used + required,
                        "total": bal.total - fee_amount,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
            else:
                # For sells, credit quote currency
                credit = notional - fee_amount
                bal = self._balances.get(quote_currency)
                if bal is not None:
                    self._balances[quote_currency] = bal.model_copy(
                        update={
                            "free": bal.free + credit,
                            "total": bal.total + credit - fee_amount,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    )

            # Create the order record
            order_id = _uid()
            now = datetime.now(timezone.utc)
            order = Order(
                order_id=order_id,
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                side=intent.side,
                order_type=intent.order_type,
                price=fill_price,
                qty=intent.qty,
                filled_qty=intent.qty,
                remaining_qty=Decimal("0"),
                avg_fill_price=fill_price,
                status=OrderStatus.FILLED,
                reduce_only=intent.reduce_only,
                post_only=intent.post_only,
                leverage=intent.leverage,
                created_at=now,
                updated_at=now,
                strategy_id=intent.strategy_id,
                trace_id=intent.trace_id,
            )
            self._orders[order_id] = order

            # Update position
            self._update_position(
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                fill_price=fill_price,
                leverage=intent.leverage,
            )

            logger.info(
                "Paper fill: order_id=%s symbol=%s side=%s qty=%s "
                "price=%s fee=%s",
                order_id,
                intent.symbol,
                intent.side.value,
                intent.qty,
                fill_price,
                fee_amount,
            )

            return OrderAck(
                order_id=order_id,
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                status=OrderStatus.FILLED,
                message="Paper fill",
                trace_id=intent.trace_id,
            )

    # ------------------------------------------------------------------
    # IExchangeAdapter: cancel_order
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str, symbol: str) -> OrderAck:
        """Cancel a paper order (only works if not already filled)."""
        async with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                raise ExchangeError(f"Unknown order_id: {order_id}")
            if order.is_terminal:
                raise ExchangeError(
                    f"Order {order_id} already terminal: {order.status.value}"
                )
            updated = order.model_copy(
                update={
                    "status": OrderStatus.CANCELLED,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._orders[order_id] = updated
            return OrderAck(
                order_id=order_id,
                client_order_id=order.client_order_id,
                symbol=symbol,
                exchange=self._exchange,
                status=OrderStatus.CANCELLED,
            )

    # ------------------------------------------------------------------
    # IExchangeAdapter: cancel_all_orders
    # ------------------------------------------------------------------

    async def cancel_all_orders(
        self, symbol: str | None = None
    ) -> list[OrderAck]:
        """Cancel all open paper orders."""
        async with self._lock:
            acks: list[OrderAck] = []
            for oid, order in list(self._orders.items()):
                if order.is_terminal:
                    continue
                if symbol is not None and order.symbol != symbol:
                    continue
                updated = order.model_copy(
                    update={
                        "status": OrderStatus.CANCELLED,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                self._orders[oid] = updated
                acks.append(
                    OrderAck(
                        order_id=oid,
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        exchange=self._exchange,
                        status=OrderStatus.CANCELLED,
                    )
                )
            return acks

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_open_orders
    # ------------------------------------------------------------------

    async def get_open_orders(
        self, symbol: str | None = None
    ) -> list[Order]:
        """Return all non-terminal orders."""
        async with self._lock:
            result: list[Order] = []
            for order in self._orders.values():
                if order.is_terminal:
                    continue
                if symbol is not None and order.symbol != symbol:
                    continue
                result.append(order)
            return result

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_positions
    # ------------------------------------------------------------------

    async def get_positions(
        self, symbol: str | None = None
    ) -> list[Position]:
        """Return open positions."""
        async with self._lock:
            positions: list[Position] = []
            for sym, pos in self._positions.items():
                if symbol is not None and sym != symbol:
                    continue
                if pos.qty != Decimal("0"):
                    # Update mark price and PnL
                    mp = self._market_prices.get(sym)
                    if mp is not None:
                        pnl = self._calculate_unrealized_pnl(pos, mp)
                        pos = pos.model_copy(
                            update={
                                "mark_price": mp,
                                "unrealized_pnl": pnl,
                                "notional": abs(pos.qty * mp),
                            }
                        )
                    positions.append(pos)
            return positions

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_balances
    # ------------------------------------------------------------------

    async def get_balances(self) -> list[Balance]:
        """Return all non-zero balances."""
        async with self._lock:
            return [
                b for b in self._balances.values()
                if b.total != Decimal("0")
            ]

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_instrument
    # ------------------------------------------------------------------

    async def get_instrument(self, symbol: str) -> Instrument:
        """Return instrument metadata for a symbol."""
        inst = self._instruments.get(symbol)
        if inst is None:
            raise ExchangeError(
                f"Instrument not loaded for symbol: {symbol}"
            )
        return inst

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_funding_rate
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Decimal:
        """Return a simulated funding rate (always zero for paper)."""
        return Decimal("0")

    # ------------------------------------------------------------------
    # IExchangeAdapter: set_trading_stop
    # ------------------------------------------------------------------

    async def set_trading_stop(
        self,
        symbol: str,
        *,
        take_profit: Decimal | None = None,
        stop_loss: Decimal | None = None,
        trailing_stop: Decimal | None = None,
        active_price: Decimal | None = None,
    ) -> dict[str, Any]:
        """Store TP/SL levels for paper-mode simulation.

        On each subsequent ``set_market_price`` call, the adapter checks
        whether the new price crosses any active TP/SL level.  When
        triggered, a synthetic close fill is generated.

        Parameters
        ----------
        active_price:
            When set alongside ``trailing_stop``, the trailing stop does not
            arm (track the high-water mark or trigger) until the market price
            first reaches this level.  Mirrors Bybit V5's ``activePrice``
            parameter for breakeven activation.
        """
        existing = self._trading_stops.get(symbol)
        if existing is not None:
            # Merge: only update non-None values
            if take_profit is not None:
                existing.take_profit = take_profit
            if stop_loss is not None:
                existing.stop_loss = stop_loss
            if trailing_stop is not None:
                existing.trailing_stop = trailing_stop
                existing.active_price = active_price  # update activation gate
                # Only reset trailing_ref immediately if there is no activation gate
                if active_price is None:
                    existing.trailing_ref = self._market_prices.get(symbol)
                else:
                    existing.trailing_ref = None  # will be set when active_price is reached
            elif active_price is not None:
                existing.active_price = active_price
        else:
            stop = _TradingStop(
                take_profit=take_profit,
                stop_loss=stop_loss,
                trailing_stop=trailing_stop,
                active_price=active_price,
            )
            if trailing_stop is not None and active_price is None:
                # No activation gate: arm trailing immediately
                stop.trailing_ref = self._market_prices.get(symbol)
            self._trading_stops[symbol] = stop

        logger.info(
            "PaperAdapter TP/SL stored: %s tp=%s sl=%s trail=%s active_price=%s",
            symbol, take_profit, stop_loss, trailing_stop, active_price,
        )
        return {"result": "paper_stop_stored"}

    # ------------------------------------------------------------------
    # TP/SL simulation
    # ------------------------------------------------------------------

    def _check_trading_stops(self, symbol: str, price: Decimal) -> None:
        """Check if *price* triggers any active TP/SL for *symbol*.

        When triggered:
        1. Closes the position at the trigger price.
        2. Publishes a synthetic ``FillEvent`` on the event bus.
        3. Clears the stop for that symbol.

        Called synchronously from ``set_market_price``.  If there is
        an event bus, fill publication is scheduled as a fire-and-forget
        task.
        """
        stop = self._trading_stops.get(symbol)
        if stop is None:
            return

        pos = self._positions.get(symbol)
        if pos is None or pos.qty == Decimal("0"):
            # No position → clear orphan stop
            self._trading_stops.pop(symbol, None)
            return

        is_long = pos.qty > Decimal("0")
        triggered = False
        trigger_reason = ""
        trigger_price = price

        # --- Trailing stop: update reference price ---
        if stop.trailing_stop is not None:
            # Breakeven activation gate: trailing stop does not arm until
            # the market price reaches active_price (1× SL distance in profit).
            if stop.active_price is not None:
                _activated = (
                    (is_long and price >= stop.active_price)
                    or (not is_long and price <= stop.active_price)
                )
                if _activated:
                    # Price has reached breakeven — arm the trailing stop now.
                    stop.active_price = None
                    stop.trailing_ref = price
                # else: not yet activated, skip all trailing logic this tick
            else:
                # No activation gate (or already activated) — normal tracking.
                if stop.trailing_ref is None:
                    stop.trailing_ref = price
                elif is_long and price > stop.trailing_ref:
                    stop.trailing_ref = price
                elif not is_long and price < stop.trailing_ref:
                    stop.trailing_ref = price

            # Check trailing trigger (only when armed, i.e. trailing_ref is set)
            if is_long and stop.trailing_ref is not None:
                trail_sl = stop.trailing_ref - stop.trailing_stop
                if price <= trail_sl:
                    triggered = True
                    trigger_reason = "trailing_stop"
                    trigger_price = trail_sl
            elif not is_long and stop.trailing_ref is not None:
                trail_sl = stop.trailing_ref + stop.trailing_stop
                if price >= trail_sl:
                    triggered = True
                    trigger_reason = "trailing_stop"
                    trigger_price = trail_sl

        # --- Take profit ---
        if not triggered and stop.take_profit is not None:
            if is_long and price >= stop.take_profit:
                triggered = True
                trigger_reason = "take_profit"
                trigger_price = stop.take_profit
            elif not is_long and price <= stop.take_profit:
                triggered = True
                trigger_reason = "take_profit"
                trigger_price = stop.take_profit

        # --- Stop loss ---
        if not triggered and stop.stop_loss is not None:
            if is_long and price <= stop.stop_loss:
                triggered = True
                trigger_reason = "stop_loss"
                trigger_price = stop.stop_loss
            elif not is_long and price >= stop.stop_loss:
                triggered = True
                trigger_reason = "stop_loss"
                trigger_price = stop.stop_loss

        if not triggered:
            return

        # ---- Execute the close ----
        close_qty = abs(pos.qty)
        close_side = Side.SELL if is_long else Side.BUY

        logger.info(
            "PaperAdapter %s triggered for %s at %s (pos=%s qty=%s): "
            "closing at %s",
            trigger_reason,
            symbol,
            price,
            "long" if is_long else "short",
            close_qty,
            trigger_price,
        )

        self._update_position(
            symbol=symbol,
            side=close_side,
            qty=close_qty,
            fill_price=trigger_price,
        )

        # Clear the stop
        self._trading_stops.pop(symbol, None)

        # Publish synthetic fill on event bus
        if self._event_bus is not None:
            from agentic_trading.core.events import FillEvent

            fill = FillEvent(
                fill_id=_uid(),
                order_id=f"paper_tpsl_{_uid()[:8]}",
                client_order_id=f"tpsl_{trigger_reason}",
                symbol=symbol,
                exchange=self._exchange,
                side=close_side,
                price=trigger_price,
                qty=close_qty,
                fee=Decimal("0"),
                fee_currency="USDT",
                is_maker=False,
                trace_id=f"tpsl_{trigger_reason}_{symbol}",
            )

            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._event_bus.publish("execution.fill", fill),
                    name=f"paper-tpsl-{symbol}",
                )
            except RuntimeError:
                # No running event loop (e.g. in sync tests)
                logger.debug(
                    "No event loop for TP/SL fill publish: %s", symbol,
                )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _update_position(
        self,
        symbol: str,
        side: Side,
        qty: Decimal,
        fill_price: Decimal,
        leverage: int | None = None,
    ) -> None:
        """Update the simulated position after a fill."""
        existing = self._positions.get(symbol)
        now = datetime.now(timezone.utc)

        if existing is None or existing.qty == Decimal("0"):
            # New position
            pos_side = (
                PositionSide.LONG if side == Side.BUY else PositionSide.SHORT
            )
            signed_qty = qty if side == Side.BUY else -qty
            self._positions[symbol] = Position(
                symbol=symbol,
                exchange=self._exchange,
                side=pos_side,
                qty=signed_qty,
                entry_price=fill_price,
                mark_price=fill_price,
                leverage=leverage or 1,
                notional=abs(signed_qty * fill_price),
                updated_at=now,
            )
        else:
            # Modify existing position
            old_qty = existing.qty
            if side == Side.BUY:
                new_qty = old_qty + qty
            else:
                new_qty = old_qty - qty

            # Calculate new entry price (weighted average for adds,
            # unchanged for reduces)
            if (side == Side.BUY and old_qty >= 0) or (
                side == Side.SELL and old_qty <= 0
            ):
                # Adding to position -- weighted average
                if new_qty != Decimal("0"):
                    new_entry = (
                        existing.entry_price * abs(old_qty)
                        + fill_price * qty
                    ) / abs(new_qty)
                else:
                    new_entry = Decimal("0")
            else:
                # Reducing or flipping -- keep existing entry for the
                # remaining portion
                new_entry = existing.entry_price

            # Determine side
            if new_qty > 0:
                pos_side = PositionSide.LONG
            elif new_qty < 0:
                pos_side = PositionSide.SHORT
            else:
                pos_side = existing.side

            realized = Decimal("0")
            if (side == Side.SELL and old_qty > 0) or (
                side == Side.BUY and old_qty < 0
            ):
                # Closing portion generates realized PnL
                close_qty = min(qty, abs(old_qty))
                if old_qty > 0:
                    realized = (fill_price - existing.entry_price) * close_qty
                else:
                    realized = (existing.entry_price - fill_price) * close_qty

            self._positions[symbol] = Position(
                symbol=symbol,
                exchange=self._exchange,
                side=pos_side,
                qty=new_qty,
                entry_price=new_entry,
                mark_price=fill_price,
                realized_pnl=existing.realized_pnl + realized,
                leverage=leverage or existing.leverage,
                margin_mode=existing.margin_mode,
                notional=abs(new_qty * fill_price),
                updated_at=now,
            )

    @staticmethod
    def _calculate_unrealized_pnl(
        position: Position, mark_price: Decimal
    ) -> Decimal:
        """Calculate unrealized PnL for a position at a given mark price."""
        if position.qty == Decimal("0"):
            return Decimal("0")
        if position.qty > 0:
            return (mark_price - position.entry_price) * position.qty
        else:
            return (position.entry_price - mark_price) * abs(position.qty)

    # ------------------------------------------------------------------
    # Instrument management
    # ------------------------------------------------------------------

    def load_instrument(self, instrument: Instrument) -> None:
        """Register an instrument for paper trading."""
        self._instruments[instrument.symbol] = instrument

    # ------------------------------------------------------------------
    # State reset (useful for tests)
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Clear all state: orders, positions, balances, prices, stops."""
        async with self._lock:
            self._orders.clear()
            self._positions.clear()
            self._balances.clear()
            self._market_prices.clear()
            self._trading_stops.clear()
            logger.info("PaperAdapter state reset")
