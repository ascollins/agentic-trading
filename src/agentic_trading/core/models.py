"""Core domain models used across the trading platform.

These are the canonical "truth models" for the system.
All modules use these same types — no exchange-specific variants leak out.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from .enums import (
    Exchange,
    InstrumentType,
    MarginMode,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
    Timeframe,
    TimeInForce,
)


# ---------------------------------------------------------------------------
# Instrument metadata
# ---------------------------------------------------------------------------

class Instrument(BaseModel):
    """Exchange-normalized instrument specification."""

    symbol: str  # Unified symbol, e.g. "BTC/USDT"
    exchange: Exchange
    instrument_type: InstrumentType
    base: str  # e.g. "BTC"
    quote: str  # e.g. "USDT"
    settle: str | None = None  # Settlement currency for perps

    # Precision
    price_precision: int = 2  # Decimal places for price
    qty_precision: int = 6  # Decimal places for quantity
    tick_size: Decimal = Decimal("0.01")
    step_size: Decimal = Decimal("0.000001")

    # Limits
    min_qty: Decimal = Decimal("0")
    max_qty: Decimal = Decimal("999999999")
    min_notional: Decimal = Decimal("0")

    # Perp-specific
    margin_mode: MarginMode | None = None
    max_leverage: int = 1
    funding_interval_hours: int = 8

    # Fees (default maker/taker)
    maker_fee: Decimal = Decimal("0.0002")
    taker_fee: Decimal = Decimal("0.0004")

    # Filters
    is_active: bool = True

    def round_price(self, price: float | Decimal) -> Decimal:
        """Round price to tick size."""
        d = Decimal(str(price))
        return (d / self.tick_size).quantize(Decimal("1")) * self.tick_size

    def round_qty(self, qty: float | Decimal) -> Decimal:
        """Round quantity to step size."""
        d = Decimal(str(qty))
        return (d / self.step_size).quantize(Decimal("1")) * self.step_size


# ---------------------------------------------------------------------------
# OHLCV Candle
# ---------------------------------------------------------------------------

class Candle(BaseModel):
    """OHLCV candle with volume and metadata."""

    symbol: str
    exchange: Exchange
    timeframe: Timeframe
    timestamp: datetime  # Candle open time
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    is_closed: bool = True


# ---------------------------------------------------------------------------
# Order lifecycle
# ---------------------------------------------------------------------------

class Order(BaseModel):
    """Canonical order representation."""

    order_id: str  # Exchange order ID
    client_order_id: str  # Our dedupe key → clientOrderId
    symbol: str
    exchange: Exchange
    side: Side
    order_type: OrderType
    time_in_force: TimeInForce = TimeInForce.GTC
    price: Decimal | None = None  # Limit price
    stop_price: Decimal | None = None
    qty: Decimal
    filled_qty: Decimal = Decimal("0")
    remaining_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    status: OrderStatus = OrderStatus.PENDING
    reduce_only: bool = False
    post_only: bool = False
    leverage: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    strategy_id: str | None = None
    trace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


# ---------------------------------------------------------------------------
# Fill / trade
# ---------------------------------------------------------------------------

class Fill(BaseModel):
    """A single fill (execution) of an order."""

    fill_id: str
    order_id: str
    client_order_id: str
    symbol: str
    exchange: Exchange
    side: Side
    price: Decimal
    qty: Decimal
    fee: Decimal
    fee_currency: str
    is_maker: bool = False
    timestamp: datetime
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class Position(BaseModel):
    """Canonical position state."""

    symbol: str
    exchange: Exchange
    side: PositionSide
    qty: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    mark_price: Decimal = Decimal("0")
    liquidation_price: Decimal | None = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.CROSS
    notional: Decimal = Decimal("0")
    updated_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.qty != Decimal("0")


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------

class Balance(BaseModel):
    """Account balance for a single currency."""

    currency: str
    exchange: Exchange
    total: Decimal = Decimal("0")
    free: Decimal = Decimal("0")
    used: Decimal = Decimal("0")  # In open orders / margin
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Funding payment
# ---------------------------------------------------------------------------

class FundingPayment(BaseModel):
    """A funding rate payment for perpetual futures."""

    symbol: str
    exchange: Exchange
    funding_rate: Decimal
    payment: Decimal  # Positive = received, negative = paid
    position_qty: Decimal
    timestamp: datetime
