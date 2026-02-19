"""Topic → schema registry.

Maps event bus topic names to their Pydantic event models.
Used for serialization/deserialization and validation.
"""

from __future__ import annotations

from agentic_trading.core.events import (
    ApprovalRequested,
    ApprovalResolved,
    BalanceUpdate,
    BaseEvent,
    CandleEvent,
    CircuitBreakerEvent,
    CMTAssessment,
    DegradedModeEnabled,
    FeatureVector,
    FillEvent,
    FundingPaymentEvent,
    IncidentCreated,
    KillSwitchEvent,
    NewsEvent,
    OpenInterestEvent,
    OptimizationCompleted,
    OrderAck,
    OrderBookSnapshot,
    OrderIntent,
    OrderUpdate,
    ParameterChangeApplied,
    PositionUpdate,
    ReconciliationResult,
    RegimeState,
    RiskAlert,
    RiskCheckResult,
    Signal,
    StrategyOptimizationResult,
    SystemHealth,
    TargetPosition,
    TickEvent,
    ToolCallRecorded,
    TradeEvent,
    WhaleEvent,
)

# Topic name → list of event types that can appear on that topic
TOPIC_SCHEMAS: dict[str, list[type[BaseEvent]]] = {
    "market.tick": [TickEvent],
    "market.trade": [TradeEvent],
    "market.orderbook": [OrderBookSnapshot],
    "market.candle": [CandleEvent],
    "feature.vector": [FeatureVector],
    "feature.news": [NewsEvent],
    "feature.whale": [WhaleEvent],
    "strategy.signal": [Signal],
    "strategy.regime": [RegimeState],
    "strategy.target": [TargetPosition],
    "execution.intent": [OrderIntent],
    "execution.ack": [OrderAck],
    "execution.update": [OrderUpdate],
    "execution.fill": [FillEvent],
    "state.position": [PositionUpdate],
    "state.balance": [BalanceUpdate],
    "state.funding": [FundingPaymentEvent],
    "state.open_interest": [OpenInterestEvent],
    "risk.check": [RiskCheckResult],
    "risk.alert": [RiskAlert],
    "risk.circuit_breaker": [CircuitBreakerEvent],
    "system.health": [SystemHealth],
    "system.kill_switch": [KillSwitchEvent],
    "system.reconciliation": [ReconciliationResult],
    "system.degraded_mode": [DegradedModeEnabled],
    "system.incident": [IncidentCreated],
    "governance.approval": [ApprovalRequested, ApprovalResolved],
    "control_plane.tool_call": [ToolCallRecorded],
    "intelligence.cmt": [CMTAssessment],
    "optimizer.result": [
        OptimizationCompleted, StrategyOptimizationResult, ParameterChangeApplied,
    ],
}

# Flat map: event class name → event class (for deserialization)
EVENT_TYPE_MAP: dict[str, type[BaseEvent]] = {}
for _schemas in TOPIC_SCHEMAS.values():
    for _cls in _schemas:
        EVENT_TYPE_MAP[_cls.__name__] = _cls


def get_event_class(event_type_name: str) -> type[BaseEvent] | None:
    """Look up event class by name."""
    return EVENT_TYPE_MAP.get(event_type_name)


def get_topic_for_event(event: BaseEvent) -> str | None:
    """Find the topic a given event should be published on."""
    cls_name = type(event).__name__
    for topic, schemas in TOPIC_SCHEMAS.items():
        for schema_cls in schemas:
            if schema_cls.__name__ == cls_name:
                return topic
    return None
