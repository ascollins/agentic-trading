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
    AgentStatus,
    AgentType,
    CircuitBreakerType,
    Exchange,
    GovernanceAction,
    ImpactTier,
    MaturityLevel,
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
    features: dict[str, float | None] = Field(default_factory=dict)


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

    # Explicit server-side TP/SL levels (set by strategies)
    take_profit: Decimal | None = None
    stop_loss: Decimal | None = None
    trailing_stop: Decimal | None = None  # Trailing stop distance (not price)


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
    price_estimate: float = 0.0  # Latest price for notional calculation


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


class OpenInterestEvent(BaseEvent):
    """Snapshot of open interest for a perpetual futures contract."""

    source_module: str = "state"
    symbol: str
    exchange: Exchange
    open_interest: float  # contracts / base-asset units
    open_interest_value: float = 0.0  # notional value in quote currency
    timestamp_exchange: datetime | None = None  # exchange-reported time


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


# ===========================================================================
# Topic: governance
# ===========================================================================

class GovernanceDecision(BaseEvent):
    """Result of a governance gate check."""

    source_module: str = "governance"
    strategy_id: str
    symbol: str
    action: GovernanceAction
    reason: str = ""
    sizing_multiplier: float = 1.0
    maturity_level: MaturityLevel = MaturityLevel.L0_SHADOW
    impact_tier: ImpactTier = ImpactTier.LOW
    health_score: float = 1.0
    details: dict[str, Any] = Field(default_factory=dict)


class MaturityTransition(BaseEvent):
    """Strategy maturity level change."""

    source_module: str = "governance.maturity"
    strategy_id: str
    from_level: MaturityLevel
    to_level: MaturityLevel
    reason: str = ""
    metrics_snapshot: dict[str, float] = Field(default_factory=dict)


class HealthScoreUpdate(BaseEvent):
    """Strategy health score change."""

    source_module: str = "governance.health_score"
    strategy_id: str
    score: float
    debt: float = 0.0
    credit: float = 0.0
    sizing_multiplier: float = 1.0
    window_trades: int = 0


class CanaryAlert(BaseEvent):
    """Governance canary (safety watchdog) alert."""

    source_module: str = "governance.canary"
    component: str
    healthy: bool
    message: str = ""
    action_taken: GovernanceAction = GovernanceAction.ALLOW


class DriftAlert(BaseEvent):
    """Live-vs-backtest drift detection alert."""

    source_module: str = "governance.drift"
    strategy_id: str
    metric_name: str
    baseline_value: float = 0.0
    live_value: float = 0.0
    deviation_pct: float = 0.0
    action_taken: GovernanceAction = GovernanceAction.ALLOW


class TokenEvent(BaseEvent):
    """Execution token lifecycle event."""

    source_module: str = "governance.tokens"
    token_id: str
    strategy_id: str
    action: str = ""  # "issued", "used", "expired", "revoked"
    scope: str = ""
    ttl_seconds: int = 0


class GovernanceCanaryCheck(BaseEvent):
    """Periodic canary health-check result."""

    source_module: str = "governance.canary"
    all_healthy: bool
    components_checked: int = 0
    failed_components: list[str] = Field(default_factory=list)


class ApprovalRequested(BaseEvent):
    """Published when a high-impact action is held for approval."""

    source_module: str = "governance.approval"
    request_id: str
    strategy_id: str
    symbol: str
    action_type: str = "order"
    trigger: str = ""
    escalation_level: str = ""
    notional_usd: float = 0.0
    impact_tier: str = "low"
    reason: str = ""
    ttl_seconds: int = 300


class ApprovalResolved(BaseEvent):
    """Published when an approval request is approved, rejected, or expired."""

    source_module: str = "governance.approval"
    request_id: str
    strategy_id: str
    symbol: str
    status: str = ""  # "approved", "rejected", "expired", "escalated"
    decided_by: str = ""
    reason: str = ""
    decision_time_seconds: float = 0.0


# ===========================================================================
# Topic: agent
# ===========================================================================


class AgentHealthReport(BaseModel):
    """Health status report from an agent."""

    healthy: bool = True
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    last_work_at: datetime | None = None
    error_count: int = 0


class AgentCapabilities(BaseModel):
    """Declares what an agent can do and what it needs."""

    subscribes_to: list[str] = Field(default_factory=list)
    publishes_to: list[str] = Field(default_factory=list)
    description: str = ""


class AgentStarted(BaseEvent):
    """Published when an agent starts."""

    source_module: str = "agents"
    agent_id: str
    agent_type: AgentType
    agent_name: str = ""


class AgentStopped(BaseEvent):
    """Published when an agent stops."""

    source_module: str = "agents"
    agent_id: str
    agent_type: AgentType
    agent_name: str = ""
    reason: str = ""


class AgentHealthChanged(BaseEvent):
    """Published when an agent's health status changes."""

    source_module: str = "agents"
    agent_id: str
    agent_type: AgentType
    healthy: bool
    message: str = ""


# ===========================================================================
# Topic: governance (incident management)
# ===========================================================================


class IncidentDeclared(BaseEvent):
    """Published when an incident is declared by the IncidentManager."""

    source_module: str = "governance.incident"
    incident_id: str
    severity: str  # IncidentSeverity.value
    trigger: str
    trigger_event_id: str = ""
    description: str = ""
    affected_strategies: list[str] = Field(default_factory=list)
    affected_symbols: list[str] = Field(default_factory=list)


class DegradedModeEnabled(BaseEvent):
    """Published when the system degraded mode changes."""

    source_module: str = "control_plane"
    mode: str  # DegradedMode.value
    previous_mode: str = ""
    reason: str = ""
    triggered_by: str = ""
    blocked_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)


class IncidentCreated(BaseEvent):
    """Published when an incident is detected by the control plane."""

    source_module: str = "control_plane"
    incident_id: str = Field(default_factory=_uuid)
    severity: str  # "warning", "critical", "emergency"
    component: str
    description: str
    auto_action: str = ""  # "degraded_mode", "kill_switch", "none"
    affected_symbols: list[str] = Field(default_factory=list)


class ToolCallRecorded(BaseEvent):
    """Published after every ToolGateway call (success or failure)."""

    source_module: str = "control_plane.tool_gateway"
    action_id: str
    tool_name: str
    success: bool
    request_hash: str = ""
    response_hash: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    was_idempotent_replay: bool = False


# ===========================================================================
# Topic: intelligence.cmt
# ===========================================================================


class CMTAssessment(BaseEvent):
    """Structured 9-layer CMT analysis produced by CMTAnalystAgent."""

    source_module: str = "intelligence.cmt"
    symbol: str
    timeframes_analyzed: list[str] = Field(default_factory=list)
    layers: dict[str, Any] = Field(default_factory=dict)
    confluence_score: dict[str, Any] = Field(default_factory=dict)
    trade_plan: dict[str, Any] | None = None
    thesis: str = ""
    system_health: str = "green"  # green / amber / red
    raw_llm_response: str = ""


# ===========================================================================
# Topic: optimizer.result
# ===========================================================================


class OptimizationCompleted(BaseEvent):
    """Published when a full optimization cycle finishes."""

    source_module: str = "optimizer"
    run_number: int
    strategies_optimized: int = 0
    strategies_skipped: int = 0
    strategies_failed: int = 0
    duration_seconds: float = 0.0
    recommendations: dict[str, Any] = Field(default_factory=dict)


class StrategyOptimizationResult(BaseEvent):
    """Published per strategy after optimization."""

    source_module: str = "optimizer"
    strategy_id: str
    recommendation: str = ""  # OptimizationRecommendation.value
    current_composite_score: float = 0.0
    optimized_composite_score: float = 0.0
    improvement_pct: float = 0.0
    current_params: dict[str, Any] = Field(default_factory=dict)
    optimized_params: dict[str, Any] = Field(default_factory=dict)
    is_overfit: bool = False
    walk_forward_passed: bool = False
    auto_applied: bool = False
    rationale: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


class ParameterChangeApplied(BaseEvent):
    """Published when optimized parameters are auto-applied to a running strategy."""

    source_module: str = "optimizer"
    strategy_id: str
    old_params: dict[str, Any] = Field(default_factory=dict)
    new_params: dict[str, Any] = Field(default_factory=dict)
    improvement_pct: float = 0.0
    approval_required: bool = False
    approval_id: str = ""
