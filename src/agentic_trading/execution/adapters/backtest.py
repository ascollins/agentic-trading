"""Backtest adapter: deterministic simulated execution against historical data.

Used by the backtest engine to simulate fills against candle data.
All randomness is seeded for reproducibility.  Supports partial fills
based on candle volume, configurable slippage and fee models, and
funding rate simulation for perpetual futures.
"""

from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

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
from agentic_trading.core.models import Balance, Candle, Instrument, Order, Position

from .base import FeeSchedule, SlippageConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

SlippageModel = Callable[[Decimal, str, float], Decimal]
"""Callable(price, side, random_factor) -> slipped_price."""

FeeModel = Callable[[Decimal, Decimal, bool], Decimal]
"""Callable(price, qty, is_maker) -> fee_amount."""


class BacktestAdapter:
    """Deterministic exchange adapter for backtesting.

    Fills orders against candle data with configurable slippage, fee,
    and partial-fill models.  All random state is derived from a seeded
    RNG so backtests are reproducible.

    Parameters
    ----------
    exchange:
        The simulated exchange.
    initial_balances:
        Starting balances, e.g. ``{"USDT": Decimal("10000")}``.
    fees:
        Fee schedule.  Overridden by ``fee_model`` if provided.
    slippage:
        Slippage config.  Overridden by ``slippage_model`` if provided.
    slippage_model:
        Custom callable for computing slipped price.
    fee_model:
        Custom callable for computing fee amount.
    fill_ratio_of_volume:
        Maximum fraction of candle volume the adapter will fill in a
        single candle (default 0.10 = 10 %).  Orders exceeding this
        receive a partial fill.
    seed:
        RNG seed for determinism (default 42).
    instruments:
        Pre-loaded instrument metadata (symbol -> Instrument).
    funding_rates:
        Static funding rates per symbol for simulation.
    """

    def __init__(
        self,
        exchange: Exchange = Exchange.BINANCE,
        initial_balances: dict[str, Decimal] | None = None,
        fees: FeeSchedule | None = None,
        slippage: SlippageConfig | None = None,
        slippage_model: SlippageModel | None = None,
        fee_model: FeeModel | None = None,
        fill_ratio_of_volume: float = 0.10,
        seed: int = 42,
        instruments: dict[str, Instrument] | None = None,
        funding_rates: dict[str, Decimal] | None = None,
    ) -> None:
        self._exchange = exchange
        self._fees = fees or FeeSchedule()
        self._slippage_cfg = slippage or SlippageConfig()
        self._slippage_model = slippage_model
        self._fee_model = fee_model
        self._fill_ratio = Decimal(str(fill_ratio_of_volume))
        self._rng = random.Random(seed)
        self._instruments: dict[str, Instrument] = instruments or {}
        self._funding_rates: dict[str, Decimal] = funding_rates or {}

        # State
        self._orders: dict[str, Order] = {}  # order_id -> Order
        self._positions: dict[str, Position] = {}  # symbol -> Position
        self._balances: dict[str, Balance] = {}  # currency -> Balance
        self._current_candles: dict[str, Candle] = {}  # symbol -> last candle

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
            "BacktestAdapter initialised (seed=%d, fill_ratio=%.2f, "
            "exchange=%s)",
            seed,
            fill_ratio_of_volume,
            exchange.value,
        )

    # ------------------------------------------------------------------
    # Candle feed
    # ------------------------------------------------------------------

    def set_candle(self, candle: Candle) -> None:
        """Feed the latest candle for a symbol.

        Must be called before ``submit_order`` or ``fill_order`` for the
        symbol.
        """
        self._current_candles[candle.symbol] = candle

    # ------------------------------------------------------------------
    # IExchangeAdapter: submit_order
    # ------------------------------------------------------------------

    async def submit_order(self, intent: OrderIntent) -> OrderAck:
        """Submit a backtest order.

        If a current candle is available the order is filled immediately
        (possibly partially).  Otherwise it is queued as SUBMITTED.
        """
        candle = self._current_candles.get(intent.symbol)
        if candle is not None:
            return self._fill_against_candle(intent, candle)
        else:
            # No candle yet -- queue the order
            order_id = self._make_order_id()
            now = datetime.now(timezone.utc)
            order = Order(
                order_id=order_id,
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                side=intent.side,
                order_type=intent.order_type,
                price=intent.price,
                stop_price=intent.stop_price,
                qty=intent.qty,
                filled_qty=Decimal("0"),
                remaining_qty=intent.qty,
                status=OrderStatus.SUBMITTED,
                reduce_only=intent.reduce_only,
                post_only=intent.post_only,
                leverage=intent.leverage,
                created_at=now,
                updated_at=now,
                strategy_id=intent.strategy_id,
                trace_id=intent.trace_id,
            )
            self._orders[order_id] = order
            return OrderAck(
                order_id=order_id,
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                status=OrderStatus.SUBMITTED,
                trace_id=intent.trace_id,
            )

    # ------------------------------------------------------------------
    # Core fill logic
    # ------------------------------------------------------------------

    def fill_order(
        self, intent: OrderIntent, candle: Candle
    ) -> OrderAck:
        """Simulate a fill against a specific candle.

        This is the primary entry point used by the backtest engine.
        Handles partial fills based on candle volume and the configured
        ``fill_ratio_of_volume``.

        Parameters
        ----------
        intent:
            The order intent to fill.
        candle:
            The candle data to fill against.

        Returns
        -------
        ``OrderAck`` with the resulting status (FILLED or
        PARTIALLY_FILLED).
        """
        return self._fill_against_candle(intent, candle)

    def _fill_against_candle(
        self, intent: OrderIntent, candle: Candle
    ) -> OrderAck:
        """Internal fill logic.

        Determines:
        - Whether the order would have been triggered during the candle.
        - The fill price (with slippage).
        - The fill quantity (subject to volume cap).
        - Fee computation.
        - Position and balance updates.
        """
        order_id = self._make_order_id()
        now = datetime.now(timezone.utc)

        # --- 1. Price determination ---
        base_price = self._determine_fill_price(intent, candle)
        if base_price is None:
            # Order could not be triggered during this candle
            order = Order(
                order_id=order_id,
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                side=intent.side,
                order_type=intent.order_type,
                price=intent.price,
                qty=intent.qty,
                filled_qty=Decimal("0"),
                remaining_qty=intent.qty,
                status=OrderStatus.SUBMITTED,
                created_at=now,
                updated_at=now,
                strategy_id=intent.strategy_id,
                trace_id=intent.trace_id,
            )
            self._orders[order_id] = order
            return OrderAck(
                order_id=order_id,
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                status=OrderStatus.SUBMITTED,
                message="Order queued (not triggered during candle)",
                trace_id=intent.trace_id,
            )

        # Apply slippage
        random_factor = self._rng.random()
        if self._slippage_model is not None:
            fill_price = self._slippage_model(
                base_price, intent.side.value, random_factor
            )
        else:
            fill_price = self._slippage_cfg.apply(
                base_price, intent.side.value, random_factor
            )

        # --- 2. Volume-based partial fill ---
        max_fill_qty = Decimal(str(candle.volume)) * self._fill_ratio
        fill_qty = min(intent.qty, max_fill_qty)
        remaining_qty = intent.qty - fill_qty

        if fill_qty <= Decimal("0"):
            # Zero volume candle -- cannot fill
            order = Order(
                order_id=order_id,
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                side=intent.side,
                order_type=intent.order_type,
                price=intent.price,
                qty=intent.qty,
                filled_qty=Decimal("0"),
                remaining_qty=intent.qty,
                status=OrderStatus.SUBMITTED,
                created_at=now,
                updated_at=now,
                strategy_id=intent.strategy_id,
                trace_id=intent.trace_id,
            )
            self._orders[order_id] = order
            return OrderAck(
                order_id=order_id,
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=self._exchange,
                status=OrderStatus.SUBMITTED,
                message="Zero volume; fill deferred",
                trace_id=intent.trace_id,
            )

        # --- 3. Fee computation ---
        is_maker = intent.order_type == OrderType.LIMIT
        if self._fee_model is not None:
            fee_amount = self._fee_model(fill_price, fill_qty, is_maker)
        else:
            fee_rate = self._fees.fee_for(is_maker)
            fee_amount = fill_price * fill_qty * fee_rate

        # --- 4. Balance check and update ---
        instrument = self._instruments.get(intent.symbol)
        quote_currency = "USDT"
        if instrument is not None:
            quote_currency = instrument.quote

        notional = fill_price * fill_qty
        if intent.side == Side.BUY:
            required = notional + fee_amount
            bal = self._balances.get(quote_currency)
            if bal is None or bal.free < required:
                raise InsufficientBalanceError(
                    f"Backtest: insufficient {quote_currency}. "
                    f"Need {required}, have "
                    f"{bal.free if bal else Decimal('0')}"
                )
            self._balances[quote_currency] = bal.model_copy(
                update={
                    "free": bal.free - required,
                    "total": bal.total - fee_amount,
                    "updated_at": now,
                }
            )
        else:
            credit = notional - fee_amount
            bal = self._balances.get(quote_currency)
            if bal is not None:
                self._balances[quote_currency] = bal.model_copy(
                    update={
                        "free": bal.free + credit,
                        "total": bal.total + credit - fee_amount,
                        "updated_at": now,
                    }
                )

        # --- 5. Position update ---
        self._update_position(
            symbol=intent.symbol,
            side=intent.side,
            qty=fill_qty,
            fill_price=fill_price,
            leverage=intent.leverage,
        )

        # --- 6. Record order ---
        status = (
            OrderStatus.FILLED
            if remaining_qty <= Decimal("0")
            else OrderStatus.PARTIALLY_FILLED
        )
        order = Order(
            order_id=order_id,
            client_order_id=intent.dedupe_key,
            symbol=intent.symbol,
            exchange=self._exchange,
            side=intent.side,
            order_type=intent.order_type,
            price=fill_price,
            qty=intent.qty,
            filled_qty=fill_qty,
            remaining_qty=max(remaining_qty, Decimal("0")),
            avg_fill_price=fill_price,
            status=status,
            reduce_only=intent.reduce_only,
            post_only=intent.post_only,
            leverage=intent.leverage,
            created_at=now,
            updated_at=now,
            strategy_id=intent.strategy_id,
            trace_id=intent.trace_id,
            metadata={
                "candle_close": candle.close,
                "candle_volume": candle.volume,
                "slippage_factor": random_factor,
                "fee": str(fee_amount),
            },
        )
        self._orders[order_id] = order

        logger.debug(
            "Backtest fill: order_id=%s symbol=%s side=%s "
            "fill_qty=%s/%s price=%s fee=%s status=%s",
            order_id,
            intent.symbol,
            intent.side.value,
            fill_qty,
            intent.qty,
            fill_price,
            fee_amount,
            status.value,
        )

        return OrderAck(
            order_id=order_id,
            client_order_id=intent.dedupe_key,
            symbol=intent.symbol,
            exchange=self._exchange,
            status=status,
            message=(
                f"Filled {fill_qty}/{intent.qty} @ {fill_price}"
                if status == OrderStatus.PARTIALLY_FILLED
                else f"Filled @ {fill_price}"
            ),
            trace_id=intent.trace_id,
        )

    # ------------------------------------------------------------------
    # Price determination
    # ------------------------------------------------------------------

    def _determine_fill_price(
        self, intent: OrderIntent, candle: Candle
    ) -> Decimal | None:
        """Determine the base fill price (before slippage) for a candle.

        For market orders, uses the candle's open price.  For limit orders,
        checks whether the limit price was reached during the candle's
        high/low range.  Returns ``None`` if the order cannot be triggered.
        """
        candle_open = Decimal(str(candle.open))
        candle_high = Decimal(str(candle.high))
        candle_low = Decimal(str(candle.low))

        if intent.order_type == OrderType.MARKET:
            return candle_open

        if intent.order_type == OrderType.LIMIT:
            if intent.price is None:
                return candle_open
            limit_price = intent.price
            if intent.side == Side.BUY:
                # Buy limit triggers if candle low <= limit price
                if candle_low <= limit_price:
                    return limit_price
                return None
            else:
                # Sell limit triggers if candle high >= limit price
                if candle_high >= limit_price:
                    return limit_price
                return None

        if intent.order_type in (
            OrderType.STOP_MARKET,
            OrderType.TAKE_PROFIT_MARKET,
        ):
            stop = intent.stop_price or intent.price
            if stop is None:
                return candle_open
            if intent.side == Side.BUY:
                # Buy stop triggers when price rises to stop
                if candle_high >= stop:
                    return stop
                return None
            else:
                # Sell stop triggers when price falls to stop
                if candle_low <= stop:
                    return stop
                return None

        if intent.order_type in (
            OrderType.STOP_LIMIT,
            OrderType.TAKE_PROFIT_LIMIT,
        ):
            stop = intent.stop_price
            limit = intent.price
            if stop is None or limit is None:
                return candle_open
            if intent.side == Side.BUY:
                if candle_high >= stop and candle_low <= limit:
                    return limit
                return None
            else:
                if candle_low <= stop and candle_high >= limit:
                    return limit
                return None

        # Fallback
        return candle_open

    # ------------------------------------------------------------------
    # IExchangeAdapter: cancel_order
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str, symbol: str) -> OrderAck:
        """Cancel a backtest order."""
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
        """Cancel all non-terminal backtest orders."""
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
        """Return non-terminal orders."""
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
        """Return open simulated positions."""
        positions: list[Position] = []
        for sym, pos in self._positions.items():
            if symbol is not None and sym != symbol:
                continue
            if pos.qty != Decimal("0"):
                # Update mark price from latest candle
                candle = self._current_candles.get(sym)
                if candle is not None:
                    mp = Decimal(str(candle.close))
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
        """Return non-zero balances."""
        return [
            b for b in self._balances.values()
            if b.total != Decimal("0")
        ]

    # ------------------------------------------------------------------
    # IExchangeAdapter: get_instrument
    # ------------------------------------------------------------------

    async def get_instrument(self, symbol: str) -> Instrument:
        """Return instrument metadata."""
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
        """Return the simulated funding rate for a symbol."""
        return self._funding_rates.get(symbol, Decimal("0"))

    # ------------------------------------------------------------------
    # Funding rate simulation
    # ------------------------------------------------------------------

    def apply_funding(self, symbol: str, timestamp: datetime | None = None) -> Decimal:
        """Apply a funding payment to the position for ``symbol``.

        Returns the funding payment amount (positive = received,
        negative = paid).  Debits/credits the quote currency balance.
        """
        rate = self._funding_rates.get(symbol, Decimal("0"))
        pos = self._positions.get(symbol)
        if pos is None or pos.qty == Decimal("0") or rate == Decimal("0"):
            return Decimal("0")

        # Payment = -position_value * funding_rate
        # Long pays positive rate, short receives.
        position_value = pos.qty * pos.mark_price
        payment = -position_value * rate

        instrument = self._instruments.get(symbol)
        quote_currency = "USDT"
        if instrument is not None:
            quote_currency = instrument.quote

        bal = self._balances.get(quote_currency)
        if bal is not None:
            self._balances[quote_currency] = bal.model_copy(
                update={
                    "total": bal.total + payment,
                    "free": bal.free + payment,
                    "updated_at": timestamp or datetime.now(timezone.utc),
                }
            )

        logger.debug(
            "Funding applied: symbol=%s rate=%s payment=%s",
            symbol,
            rate,
            payment,
        )
        return payment

    def set_funding_rate(self, symbol: str, rate: Decimal) -> None:
        """Set the simulated funding rate for a symbol."""
        self._funding_rates[symbol] = rate

    # ------------------------------------------------------------------
    # Position management (shared with paper adapter)
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
            old_qty = existing.qty
            new_qty = old_qty + qty if side == Side.BUY else old_qty - qty

            # Weighted average entry for additions
            if (side == Side.BUY and old_qty >= 0) or (
                side == Side.SELL and old_qty <= 0
            ):
                if new_qty != Decimal("0"):
                    new_entry = (
                        existing.entry_price * abs(old_qty)
                        + fill_price * qty
                    ) / abs(new_qty)
                else:
                    new_entry = Decimal("0")
            else:
                new_entry = existing.entry_price

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
        """Calculate unrealized PnL."""
        if position.qty == Decimal("0"):
            return Decimal("0")
        if position.qty > 0:
            return (mark_price - position.entry_price) * position.qty
        else:
            return (position.entry_price - mark_price) * abs(position.qty)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_order_id(self) -> str:
        """Generate a deterministic order ID using the seeded RNG."""
        return f"bt-{self._rng.randint(100_000_000, 999_999_999)}"

    # ------------------------------------------------------------------
    # Instrument management
    # ------------------------------------------------------------------

    def load_instrument(self, instrument: Instrument) -> None:
        """Register an instrument for backtesting."""
        self._instruments[instrument.symbol] = instrument

    # ------------------------------------------------------------------
    # State accessors (useful for backtest reporting)
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def balances(self) -> dict[str, Balance]:
        return dict(self._balances)

    @property
    def orders(self) -> dict[str, Order]:
        return dict(self._orders)

    @property
    def filled_orders(self) -> list[Order]:
        """Return all fully or partially filled orders."""
        return [
            o
            for o in self._orders.values()
            if o.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
        ]

    @property
    def total_fees(self) -> Decimal:
        """Sum of all fees paid across all orders."""
        total = Decimal("0")
        for o in self._orders.values():
            fee_str = o.metadata.get("fee")
            if fee_str is not None:
                total += Decimal(fee_str)
        return total

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reset all state and optionally re-seed the RNG."""
        self._orders.clear()
        self._positions.clear()
        self._balances.clear()
        self._current_candles.clear()
        if seed is not None:
            self._rng = random.Random(seed)
        logger.info("BacktestAdapter state reset")
