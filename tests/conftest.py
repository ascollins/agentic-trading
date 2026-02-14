"""Shared fixtures for the agentic-trading test suite."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from agentic_trading.core.clock import SimClock, WallClock
from agentic_trading.core.enums import (
    Exchange,
    InstrumentType,
    MarginMode,
    Mode,
    OrderStatus,
    OrderType,
    PositionSide,
    Side,
    SignalDirection,
    Timeframe,
    TimeInForce,
)
from agentic_trading.core.events import (
    BaseEvent,
    CandleEvent,
    OrderIntent,
    Signal,
)
from agentic_trading.core.models import (
    Balance,
    Candle,
    Fill,
    Instrument,
    Order,
    Position,
)
from agentic_trading.event_bus.memory_bus import MemoryEventBus


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_instrument() -> Instrument:
    """Return a mock BTC/USDT instrument."""
    return Instrument(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        instrument_type=InstrumentType.PERP,
        base="BTC",
        quote="USDT",
        settle="USDT",
        price_precision=2,
        qty_precision=3,
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("1000"),
        min_notional=Decimal("10"),
        margin_mode=MarginMode.CROSS,
        max_leverage=125,
        funding_interval_hours=8,
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0004"),
        is_active=True,
    )


# ---------------------------------------------------------------------------
# Candle helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_candle() -> Candle:
    """Return a single BTC/USDT 1-minute candle."""
    return Candle(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        open=67000.0,
        high=67150.0,
        low=66900.0,
        close=67100.0,
        volume=12.5,
        quote_volume=838_750.0,
        trades=350,
        is_closed=True,
    )


@pytest.fixture
def sample_candles():
    """Factory that returns a list of n candles with realistic price data."""

    def _make(n: int = 100) -> list[Candle]:
        rng = random.Random(42)
        candles: list[Candle] = []
        price = 67000.0
        base_time = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

        for i in range(n):
            change_pct = rng.gauss(0, 0.003)  # ~0.3% std per bar
            open_price = price
            close_price = open_price * (1 + change_pct)
            high_price = max(open_price, close_price) * (1 + abs(rng.gauss(0, 0.001)))
            low_price = min(open_price, close_price) * (1 - abs(rng.gauss(0, 0.001)))
            volume = max(0.1, rng.gauss(10.0, 3.0))

            candles.append(
                Candle(
                    symbol="BTC/USDT",
                    exchange=Exchange.BINANCE,
                    timeframe=Timeframe.M1,
                    timestamp=base_time + timedelta(minutes=i),
                    open=round(open_price, 2),
                    high=round(high_price, 2),
                    low=round(low_price, 2),
                    close=round(close_price, 2),
                    volume=round(volume, 4),
                    quote_volume=round(close_price * volume, 2),
                    trades=rng.randint(50, 500),
                    is_closed=True,
                )
            )
            price = close_price
        return candles

    return _make


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_clock() -> SimClock:
    """Return a SimClock starting at 2024-06-01 00:00 UTC."""
    return SimClock(start=datetime(2024, 6, 1, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_bus() -> MemoryEventBus:
    """Return a fresh MemoryEventBus instance."""
    return MemoryEventBus()


# ---------------------------------------------------------------------------
# Order intent
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_order_intent() -> OrderIntent:
    """Return an OrderIntent with a dedupe key."""
    return OrderIntent(
        dedupe_key="tf-btcusdt-12345-001",
        strategy_id="trend_following",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        qty=Decimal("0.01"),
        price=Decimal("67000.00"),
    )


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fill() -> Fill:
    """Return a sample Fill."""
    return Fill(
        fill_id="fill-001",
        order_id="order-001",
        client_order_id="tf-btcusdt-12345-001",
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=Side.BUY,
        price=Decimal("67005.50"),
        qty=Decimal("0.01"),
        fee=Decimal("0.2680"),
        fee_currency="USDT",
        is_maker=False,
        timestamp=datetime(2024, 6, 1, 12, 0, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_position() -> Position:
    """Return a sample open Position."""
    return Position(
        symbol="BTC/USDT",
        exchange=Exchange.BINANCE,
        side=PositionSide.LONG,
        qty=Decimal("0.5"),
        entry_price=Decimal("67000.00"),
        mark_price=Decimal("67500.00"),
        unrealized_pnl=Decimal("250.00"),
        realized_pnl=Decimal("0"),
        leverage=10,
        margin_mode=MarginMode.CROSS,
        notional=Decimal("33750.00"),
    )


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_balance() -> Balance:
    """Return a sample Balance."""
    return Balance(
        currency="USDT",
        exchange=Exchange.BINANCE,
        total=Decimal("100000.00"),
        free=Decimal("65000.00"),
        used=Decimal("35000.00"),
    )
