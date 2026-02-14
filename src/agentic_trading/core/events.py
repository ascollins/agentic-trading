"""Event schemas for the event-driven architecture.

All events inherit from BaseEvent and are Pydantic models.
24 event types across 7 topics.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from .enums import (
    CircuitBreakerType,
    Exchange,
    OrderStatus,
    OrderType,
    RiskAlertSeverity,
    RegimeType,
    Side,
    SignalDirection,
    Timeframe,
    TimeInForce,
    VolatilityRegime,
    LiquidityRegime,
)


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class BaseEvent(BaseModel):
    """Base for all events. Provides identity, time, and tracing."""

    event_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    trace_id: str = Field(default_factory=_uuid)
    source_module: str = ""


# ===========================================================================
# Topic: market
# ===========================================================================

class TickEvent(BaseEvent):
    source_module: str = "data"
    symbol: str
    exchange: Exchange
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    last: float = 0.0


class TradeEvent(BaseEvent):
    source_module: str = "data"
    symbol: str
    exchange: Exchange
    price: float
    qty: float
    side: Side
    trade_id: str = ""


class OrderBookSnapshot(BaseEvent):
    source_module: str = "data"
    symbol: str
    exchange: Exchange
    bids: list[list[float]] = Field(default_factory=list)  # [[price, qty], ...]
    asks: list[list[float]] = Field(default_factory=list)
    depth: int = 0


class CandleEvent(BaseEvent):
    source_module: str = "data"
    symbol: str
    exchange: Exchange
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    is_closed: bool = True


# ===========================================================================
# Topic: feature
# ===========================================================================

class FeatureVector(BaseEvent):
    source_module: str = "features"
    symbol: str
    timeframe: Timeframe
    features: dict[str, float] = Field(default_factory=dict)


class NewsEvent(BaseEvent):
    """Scaffold: parsed news with sentiment scoring."""

    source_module: str = "features.sentiment"
    headline: str = ""
    source: str = ""
    symbols: list[str] = Field(default_factory=list)
    sentiment: float = 0.0  # -1.0 to 1.0
    urgency: float = 0.0  # 0.0 to 1.0
    entities: list[str] = Field(default_factory=list)
    decay_seconds: int = 3600


class WhaleEvent(BaseEvent):
    """Scaffold: whale / large transaction detection."""

    source_module: str = "features.whale_monitor"
    symbol: str = ""
    direction: str = ""  # "inflow" / "outflow" / "transfer"
    amount_usd: float = 0.0
    wallet: str = ""
    exchange_name: str = ""


# ===========================================================================
# Topic: strategy
# ===========================================================================

class Signal(BaseEvent):
    source_module: str = "strategy"
    strategy_id: str
    symbol: str
    direction: SignalDirection
    confidence: float = 0.0  # 0.0 to 1.0
    rationale: str = ""
    features_used: dict[str, float] = Field(default_factory=dict)
    timeframe: Timeframe = Timeframe.M1
    risk_constraints: dict[str, Any] = Field(default_factory=dict)


class RegimeState(BaseEvent):
    source_module: str = "strategy.regime"
    symbol: str
    regime: RegimeType = RegimeType.UNKNOWN
    volatility: VolatilityRegime = VolatilityRegime.UNKNOWN
    liquidity: LiquidityRegime = LiquidityRegime.UNKNOWN
    confidence: float = 0.0
    consecutive_count: int = 0  # How many consecutive same-regime signals
    switches_today: int = 0


class TargetPosition(BaseEvent):
    source_module: str = "portfolio"
    strategy_id: str
    symbol: str
    target_qty: Decimal
    side: Side
    reason: str = ""
    urgency: float = 0.5  # 0=passive, 1=aggressive


# ===========================================================================
# Topic: execution
# ===========================================================================

class OrderIntent(BaseEvent):
    """Idempotent order request. The dedupe_key prevents duplicates."""

    source_module: str = "execution"
    dedupe_key: str  # hash(strategy_id, symbol, signal_id, ts_bucket)
    strategy_id: str
    symbol: str
    exchange: Exchange
    side: Side
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC
    qty: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    reduce_only: bool = False
    post_only: bool = False
    leverage: int | None = None


class OrderAck(BaseEvent):
    source_module: str = "execution"
    order_id: str
    client_order_id: str
    symbol: str
    exchange: Exchange
    status: OrderStatus
    message: str = ""


class OrderUpdate(BaseEvent):
    source_module: str = "execution"
    order_id: str
    client_order_id: str
    symbol: str
    exchange: Exchange
    status: OrderStatus
    filled_qty: Decimal = Decimal("0")
    remaining_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None


class FillEvent(BaseEvent):
    source_module: str = "execution"
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


# ===========================================================================
# Topic: state
# ===========================================================================

class PositionUpdate(BaseEvent):
    source_module: str = "state"
    symbol: str
    exchange: Exchange
    qty: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    leverage: int = 1


class BalanceUpdate(BaseEvent):
    source_module: str = "state"
    currency: str
    exchange: Exchange
    total: Decimal
    free: Decimal
    used: Decimal


class FundingPaymentEvent(BaseEvent):
    source_module: str = "state"
    symbol: str
    exchange: Exchange
    funding_rate: Decimal
    payment: Decimal
    position_qty: Decimal


# ===========================================================================
# Topic: risk
# ===========================================================================

class RiskCheckResult(BaseEvent):
    source_module: str = "risk"
    check_name: str
    passed: bool
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    order_intent_id: str = ""  # Links to OrderIntent.event_id


class RiskAlert(BaseEvent):
    source_module: str = "risk"
    severity: RiskAlertSeverity
    alert_type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class CircuitBreakerEvent(BaseEvent):
    source_module: str = "risk"
    breaker_type: CircuitBreakerType
    tripped: bool  # True = tripped, False = reset
    symbol: str = ""
    reason: str = ""
    threshold: float = 0.0
    current_value: float = 0.0


# ===========================================================================
# Topic: system
# ===========================================================================

class SystemHealth(BaseEvent):
    source_module: str = "system"
    component: str
    healthy: bool
    message: str = ""
    latency_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class KillSwitchEvent(BaseEvent):
    source_module: str = "risk"
    activated: bool
    reason: str
    triggered_by: str = ""  # "risk_engine" / "cli" / "config"


class ReconciliationResult(BaseEvent):
    source_module: str = "execution"
    exchange: Exchange
    discrepancies: list[dict[str, Any]] = Field(default_factory=list)
    orders_synced: int = 0
    positions_synced: int = 0
    balances_synced: int = 0
    repairs_applied: int = 0
