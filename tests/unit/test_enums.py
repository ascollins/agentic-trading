"""Test all enums have expected members."""

from agentic_trading.core.enums import (
    CircuitBreakerType,
    Exchange,
    InstrumentType,
    MarginMode,
    Mode,
    OrderStatus,
    OrderType,
    PositionSide,
    RegimeType,
    RiskAlertSeverity,
    Side,
    SignalDirection,
    Timeframe,
    TimeInForce,
    VolatilityRegime,
    LiquidityRegime,
)


class TestMode:
    def test_members(self):
        assert set(Mode) == {Mode.BACKTEST, Mode.PAPER, Mode.LIVE}

    def test_values(self):
        assert Mode.BACKTEST.value == "backtest"
        assert Mode.PAPER.value == "paper"
        assert Mode.LIVE.value == "live"


class TestSide:
    def test_members(self):
        assert set(Side) == {Side.BUY, Side.SELL}

    def test_values(self):
        assert Side.BUY.value == "buy"
        assert Side.SELL.value == "sell"


class TestOrderType:
    def test_members(self):
        expected = {
            OrderType.MARKET,
            OrderType.LIMIT,
            OrderType.STOP_MARKET,
            OrderType.STOP_LIMIT,
            OrderType.TAKE_PROFIT_MARKET,
            OrderType.TAKE_PROFIT_LIMIT,
        }
        assert set(OrderType) == expected


class TestOrderStatus:
    def test_members(self):
        expected = {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        assert set(OrderStatus) == expected


class TestTimeInForce:
    def test_members(self):
        assert set(TimeInForce) == {
            TimeInForce.GTC,
            TimeInForce.IOC,
            TimeInForce.FOK,
            TimeInForce.GTD,
        }


class TestPositionSide:
    def test_members(self):
        assert set(PositionSide) == {
            PositionSide.LONG,
            PositionSide.SHORT,
            PositionSide.BOTH,
        }


class TestMarginMode:
    def test_members(self):
        assert set(MarginMode) == {MarginMode.CROSS, MarginMode.ISOLATED}


class TestInstrumentType:
    def test_members(self):
        assert set(InstrumentType) == {
            InstrumentType.SPOT,
            InstrumentType.PERP,
            InstrumentType.FUTURE,
        }


class TestRegimeType:
    def test_members(self):
        assert set(RegimeType) == {
            RegimeType.TREND,
            RegimeType.RANGE,
            RegimeType.UNKNOWN,
        }


class TestVolatilityRegime:
    def test_members(self):
        assert set(VolatilityRegime) == {
            VolatilityRegime.HIGH,
            VolatilityRegime.LOW,
            VolatilityRegime.UNKNOWN,
        }


class TestLiquidityRegime:
    def test_members(self):
        assert set(LiquidityRegime) == {
            LiquidityRegime.HIGH,
            LiquidityRegime.LOW,
            LiquidityRegime.UNKNOWN,
        }


class TestSignalDirection:
    def test_members(self):
        assert set(SignalDirection) == {
            SignalDirection.LONG,
            SignalDirection.SHORT,
            SignalDirection.FLAT,
        }


class TestTimeframe:
    def test_members(self):
        expected = {
            Timeframe.M1,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.H1,
            Timeframe.H4,
            Timeframe.D1,
        }
        assert set(Timeframe) == expected

    def test_minutes_property(self):
        assert Timeframe.M1.minutes == 1
        assert Timeframe.M5.minutes == 5
        assert Timeframe.M15.minutes == 15
        assert Timeframe.H1.minutes == 60
        assert Timeframe.H4.minutes == 240
        assert Timeframe.D1.minutes == 1440

    def test_seconds_property(self):
        assert Timeframe.M1.seconds == 60
        assert Timeframe.H1.seconds == 3600


class TestCircuitBreakerType:
    def test_members(self):
        expected = {
            CircuitBreakerType.VOLATILITY,
            CircuitBreakerType.SPREAD,
            CircuitBreakerType.LIQUIDITY,
            CircuitBreakerType.STALENESS,
            CircuitBreakerType.ERROR_RATE,
            CircuitBreakerType.CLOCK_SKEW,
        }
        assert set(CircuitBreakerType) == expected


class TestRiskAlertSeverity:
    def test_members(self):
        assert set(RiskAlertSeverity) == {
            RiskAlertSeverity.INFO,
            RiskAlertSeverity.WARNING,
            RiskAlertSeverity.CRITICAL,
            RiskAlertSeverity.EMERGENCY,
        }


class TestExchange:
    def test_members(self):
        assert set(Exchange) == {Exchange.BINANCE, Exchange.BYBIT}

    def test_values(self):
        assert Exchange.BINANCE.value == "binance"
        assert Exchange.BYBIT.value == "bybit"
