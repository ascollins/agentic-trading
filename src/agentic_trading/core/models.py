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
    AssetClass,
    Exchange,
    InstrumentType,
    MarginMode,
    OrderStatus,
    OrderType,
    PositionSide,
    QtyUnit,
    Side,
    Timeframe,
    TimeInForce,
)


# ---------------------------------------------------------------------------
# Trading session (FX market hours)
# ---------------------------------------------------------------------------

class TradingSession(BaseModel):
    """A named trading session window (e.g., London, New York)."""

    name: str  # "london", "new_york", "tokyo", "sydney"
    open_utc: str  # "08:00" (HH:MM)
    close_utc: str  # "17:00"
    days: list[int] = Field(  # ISO weekday: 1=Mon .. 5=Fri
        default_factory=lambda: [1, 2, 3, 4, 5]
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

    # --- Asset classification ---
    asset_class: AssetClass = AssetClass.CRYPTO

    # --- FX-specific ---
    pip_size: Decimal | None = None  # 0.0001 for EUR/USD, 0.01 for USD/JPY
    pip_value_per_lot: Decimal | None = None  # value of 1 pip per standard lot
    lot_size: Decimal = Decimal("100000")  # standard FX lot; ignored for crypto
    venue_symbol: str = ""  # broker-native format ("EUR_USD" for OANDA)

    # --- Session / trading hours ---
    trading_sessions: list[TradingSession] = Field(default_factory=list)
    weekend_close: bool = False  # True for FX (closed Sat-Sun)

    # --- Rollover / swap ---
    rollover_enabled: bool = False
    long_swap_rate: Decimal = Decimal("0")  # daily rate
    short_swap_rate: Decimal = Decimal("0")

    # --- Versioned metadata hash ---
    instrument_hash: str = ""  # SHA256[:16] of all spec fields

    def round_price(self, price: float | Decimal) -> Decimal:
        """Round price to tick size."""
        d = Decimal(str(price))
        return (d / self.tick_size).quantize(Decimal("1")) * self.tick_size

    def round_qty(self, qty: float | Decimal) -> Decimal:
        """Round quantity to step size."""
        d = Decimal(str(qty))
        return (d / self.step_size).quantize(Decimal("1")) * self.step_size

    def price_to_pips(self, price_diff: Decimal) -> Decimal:
        """Convert a price difference to pips."""
        if self.pip_size is None or self.pip_size == Decimal("0"):
            return Decimal("0")
        return Decimal(str(price_diff)) / self.pip_size

    def pips_to_price(self, pips: Decimal) -> Decimal:
        """Convert pips to a price difference."""
        if self.pip_size is None:
            return Decimal("0")
        return Decimal(str(pips)) * self.pip_size

    def notional_value(self, qty: Decimal, price: Decimal) -> Decimal:
        """Compute notional value in quote currency.

        Crypto: qty * price.
        FX lots: qty * lot_size * price.
        """
        q = Decimal(str(qty))
        p = Decimal(str(price))
        if self.asset_class == AssetClass.FX:
            return q * self.lot_size * p
        return q * p

    def normalize_qty(
        self, qty: Decimal, unit: QtyUnit = QtyUnit.BASE
    ) -> Decimal:
        """Normalize quantity to base units.

        If *unit* is LOTS and asset_class is FX, multiply by lot_size.
        """
        q = Decimal(str(qty))
        if unit == QtyUnit.LOTS and self.asset_class == AssetClass.FX:
            return q * self.lot_size
        return q

    def compute_hash(self) -> str:
        """Compute a deterministic hash of all specification fields."""
        from .ids import instrument_hash as _ih
        return _ih(self.model_dump(mode="json"))

    def model_post_init(self, __context: Any) -> None:
        if not self.instrument_hash:
            object.__setattr__(self, "instrument_hash", self.compute_hash())


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
