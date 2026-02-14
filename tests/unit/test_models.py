"""Test core domain models."""

from datetime import datetime, timezone
from decimal import Decimal

from agentic_trading.core.enums import (
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
from agentic_trading.core.models import (
    Balance,
    Candle,
    Fill,
    FundingPayment,
    Instrument,
    Order,
    Position,
)


class TestInstrument:
    def test_round_price_to_tick_size(self, sample_instrument):
        result = sample_instrument.round_price(67123.456)
        assert result == Decimal("67123.46")

    def test_round_price_exact(self, sample_instrument):
        result = sample_instrument.round_price(67000.00)
        assert result == Decimal("67000.00")

    def test_round_qty_to_step_size(self, sample_instrument):
        result = sample_instrument.round_qty(1.23456789)
        # quantize uses ROUND_HALF_EVEN: 1234.56789 rounds to 1235
        assert result == Decimal("1.235")

    def test_round_qty_small(self, sample_instrument):
        result = sample_instrument.round_qty(0.0015)
        # quantize uses ROUND_HALF_EVEN: 1.5 rounds to 2
        assert result == Decimal("0.002")

    def test_round_price_with_decimal_input(self, sample_instrument):
        result = sample_instrument.round_price(Decimal("67123.456"))
        assert result == Decimal("67123.46")

    def test_round_qty_with_decimal_input(self, sample_instrument):
        result = sample_instrument.round_qty(Decimal("0.0019"))
        # quantize uses ROUND_HALF_EVEN: 1.9 rounds to 2
        assert result == Decimal("0.002")

    def test_instrument_fields(self, sample_instrument):
        assert sample_instrument.symbol == "BTC/USDT"
        assert sample_instrument.exchange == Exchange.BINANCE
        assert sample_instrument.instrument_type == InstrumentType.PERP
        assert sample_instrument.base == "BTC"
        assert sample_instrument.quote == "USDT"
        assert sample_instrument.max_leverage == 125
        assert sample_instrument.is_active is True


class TestCandle:
    def test_candle_creation(self, sample_candle):
        assert sample_candle.symbol == "BTC/USDT"
        assert sample_candle.exchange == Exchange.BINANCE
        assert sample_candle.timeframe == Timeframe.M1
        assert sample_candle.open == 67000.0
        assert sample_candle.high == 67150.0
        assert sample_candle.low == 66900.0
        assert sample_candle.close == 67100.0
        assert sample_candle.volume == 12.5
        assert sample_candle.is_closed is True

    def test_candle_from_dict(self):
        candle = Candle(
            symbol="ETH/USDT",
            exchange=Exchange.BINANCE,
            timeframe=Timeframe.M5,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=2000.0,
            high=2050.0,
            low=1990.0,
            close=2030.0,
            volume=100.0,
        )
        assert candle.symbol == "ETH/USDT"
        assert candle.quote_volume == 0.0
        assert candle.trades == 0


class TestOrder:
    def test_is_terminal_filled(self):
        order = Order(
            order_id="o1",
            client_order_id="c1",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("0.1"),
            status=OrderStatus.FILLED,
        )
        assert order.is_terminal is True

    def test_is_terminal_cancelled(self):
        order = Order(
            order_id="o2",
            client_order_id="c2",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            qty=Decimal("0.1"),
            status=OrderStatus.CANCELLED,
        )
        assert order.is_terminal is True

    def test_is_terminal_rejected(self):
        order = Order(
            order_id="o3",
            client_order_id="c3",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1"),
            status=OrderStatus.REJECTED,
        )
        assert order.is_terminal is True

    def test_is_terminal_expired(self):
        order = Order(
            order_id="o4",
            client_order_id="c4",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1"),
            status=OrderStatus.EXPIRED,
        )
        assert order.is_terminal is True

    def test_is_not_terminal_pending(self):
        order = Order(
            order_id="o5",
            client_order_id="c5",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1"),
            status=OrderStatus.PENDING,
        )
        assert order.is_terminal is False

    def test_is_not_terminal_submitted(self):
        order = Order(
            order_id="o6",
            client_order_id="c6",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1"),
            status=OrderStatus.SUBMITTED,
        )
        assert order.is_terminal is False

    def test_is_not_terminal_partially_filled(self):
        order = Order(
            order_id="o7",
            client_order_id="c7",
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1"),
            status=OrderStatus.PARTIALLY_FILLED,
        )
        assert order.is_terminal is False


class TestPosition:
    def test_is_open_with_qty(self, sample_position):
        assert sample_position.is_open is True

    def test_is_not_open_with_zero_qty(self):
        pos = Position(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            side=PositionSide.LONG,
            qty=Decimal("0"),
        )
        assert pos.is_open is False

    def test_position_fields(self, sample_position):
        assert sample_position.symbol == "BTC/USDT"
        assert sample_position.side == PositionSide.LONG
        assert sample_position.leverage == 10


class TestBalance:
    def test_balance_fields(self, sample_balance):
        assert sample_balance.currency == "USDT"
        assert sample_balance.total == Decimal("100000.00")
        assert sample_balance.free == Decimal("65000.00")
        assert sample_balance.used == Decimal("35000.00")


class TestFill:
    def test_fill_fields(self, sample_fill):
        assert sample_fill.fill_id == "fill-001"
        assert sample_fill.order_id == "order-001"
        assert sample_fill.side == Side.BUY
        assert sample_fill.price == Decimal("67005.50")
        assert sample_fill.fee == Decimal("0.2680")
        assert sample_fill.is_maker is False


class TestFundingPayment:
    def test_creation(self):
        fp = FundingPayment(
            symbol="BTC/USDT",
            exchange=Exchange.BINANCE,
            funding_rate=Decimal("0.0001"),
            payment=Decimal("-6.70"),
            position_qty=Decimal("1.0"),
            timestamp=datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc),
        )
        assert fp.funding_rate == Decimal("0.0001")
        assert fp.payment == Decimal("-6.70")
