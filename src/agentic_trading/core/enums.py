"""Enumerations used across the trading platform."""

from enum import Enum, auto


class Mode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # One-way mode


class MarginMode(str, Enum):
    CROSS = "cross"
    ISOLATED = "isolated"


class InstrumentType(str, Enum):
    SPOT = "spot"
    PERP = "perp"  # Perpetual futures
    FUTURE = "future"  # Dated futures


class RegimeType(str, Enum):
    TREND = "trend"
    RANGE = "range"
    UNKNOWN = "unknown"


class VolatilityRegime(str, Enum):
    HIGH = "high"
    LOW = "low"
    UNKNOWN = "unknown"


class LiquidityRegime(str, Enum):
    HIGH = "high"
    LOW = "low"
    UNKNOWN = "unknown"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        mapping = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        return mapping[self.value]

    @property
    def seconds(self) -> int:
        return self.minutes * 60


class CircuitBreakerType(str, Enum):
    VOLATILITY = "volatility"
    SPREAD = "spread"
    LIQUIDITY = "liquidity"
    STALENESS = "staleness"
    ERROR_RATE = "error_rate"
    CLOCK_SKEW = "clock_skew"


class RiskAlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class Exchange(str, Enum):
    BINANCE = "binance"
    BYBIT = "bybit"


class ConvictionLevel(str, Enum):
    """Trade plan conviction level."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INVALIDATED = "invalidated"


class SetupGrade(str, Enum):
    """Qualitative setup assessment grade."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class MarketStructureBias(str, Enum):
    """Higher-timeframe market structure bias."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"
