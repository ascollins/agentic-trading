"""Unit tests for BaseEvent -> SpineEvent mapper."""

from __future__ import annotations

from datetime import datetime, timezone

from agentic_trading.core.enums import (
    CircuitBreakerType,
    Exchange,
    GovernanceAction,
    ImpactTier,
    MaturityLevel,
    Side,
    SignalDirection,
    Timeframe,
)
from agentic_trading.core.events import (
    BaseEvent,
    CandleEvent,
    CircuitBreakerEvent,
    FillEvent,
    GovernanceDecision,
    KillSwitchEvent,
    ReconciliationResult,
    RiskCheckResult,
    Signal,
    ToolCallRecorded,
)
from agentic_trading.telemetry.mapper import map_base_event_to_spine
from agentic_trading.telemetry.models import (
    AgentStepCompletedEvent,
    AgentStepStartedEvent,
    DecisionEvent,
    EventTaxonomy,
    ExceptionEvent,
    RetrievalEvent,
    ToolCallEvent,
    ValidationEvent,
)


class TestToolCallMapping:
    """ToolCallRecorded -> TOOL_CALL."""

    def test_maps_to_tool_call(self):
        event = ToolCallRecorded(
            action_id="act-123",
            tool_name="submit_order",
            success=True,
            request_hash="abc123def456",
            response_hash="789xyz",
            latency_ms=45.2,
        )
        spine = map_base_event_to_spine(event, topic="system")

        assert isinstance(spine, ToolCallEvent)
        assert spine.event_type == EventTaxonomy.TOOL_CALL
        assert spine.input_hash == "abc123def456"
        assert spine.output_hash == "789xyz"
        assert spine.latency_ms == 45.2

    def test_preserves_trace_id(self):
        event = ToolCallRecorded(
            trace_id="my-trace",
            action_id="act-123",
            tool_name="submit_order",
            success=True,
        )
        spine = map_base_event_to_spine(event)
        assert spine.trace_id == "my-trace"

    def test_preserves_causation_id(self):
        event = ToolCallRecorded(
            causation_id="parent-event",
            action_id="act-123",
            tool_name="submit_order",
            success=True,
        )
        spine = map_base_event_to_spine(event)
        assert spine.causation_id == "parent-event"


class TestDecisionMapping:
    """RiskCheckResult and GovernanceDecision -> DECISION."""

    def test_risk_check_maps_to_decision(self):
        event = RiskCheckResult(
            check_name="max_position_size",
            passed=True,
            reason="",
            details={"current_exposure": 0.15},
        )
        spine = map_base_event_to_spine(event, topic="risk")

        assert isinstance(spine, DecisionEvent)
        assert spine.event_type == EventTaxonomy.DECISION
        assert spine.payload["passed"] is True
        assert spine.payload["check_name"] == "max_position_size"

    def test_risk_check_rejected(self):
        event = RiskCheckResult(
            check_name="max_leverage",
            passed=False,
            reason="Leverage 20x exceeds limit 10x",
        )
        spine = map_base_event_to_spine(event)

        assert spine.event_type == EventTaxonomy.DECISION
        assert spine.payload["passed"] is False
        assert "exceeds limit" in spine.payload["reason"]

    def test_governance_decision_maps_to_decision(self):
        event = GovernanceDecision(
            strategy_id="smc_btcusdt",
            symbol="BTCUSDT",
            action=GovernanceAction.BLOCK,
            reason="Shadow mode",
            maturity_level=MaturityLevel.L0_SHADOW,
            impact_tier=ImpactTier.MEDIUM,
        )
        spine = map_base_event_to_spine(event, topic="governance")

        assert isinstance(spine, DecisionEvent)
        assert spine.event_type == EventTaxonomy.DECISION
        assert spine.payload["action"] == "block"
        assert spine.payload["strategy_id"] == "smc_btcusdt"


class TestAgentStepCompletedMapping:
    """Signal, FillEvent, FeatureVector -> AGENT_STEP_COMPLETED."""

    def test_signal_maps_to_agent_step_completed(self):
        event = Signal(
            strategy_id="smc_btcusdt",
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            confidence=0.85,
        )
        spine = map_base_event_to_spine(event, topic="strategy.signal")

        assert isinstance(spine, AgentStepCompletedEvent)
        assert spine.event_type == EventTaxonomy.AGENT_STEP_COMPLETED
        assert spine.payload["strategy_id"] == "smc_btcusdt"
        assert spine.payload["direction"] == "long"

    def test_fill_event_maps_to_agent_step_completed(self):
        from decimal import Decimal

        event = FillEvent(
            fill_id="fill-1",
            order_id="ord-1",
            client_order_id="client-1",
            symbol="BTCUSDT",
            exchange=Exchange.BYBIT,
            side=Side.BUY,
            price=Decimal("42500.0"),
            qty=Decimal("0.1"),
            fee=Decimal("0.005"),
            fee_currency="USDT",
        )
        spine = map_base_event_to_spine(event)

        assert isinstance(spine, AgentStepCompletedEvent)
        assert spine.event_type == EventTaxonomy.AGENT_STEP_COMPLETED


class TestRetrievalMapping:
    """CandleEvent -> RETRIEVAL."""

    def test_candle_event_maps_to_retrieval(self):
        event = CandleEvent(
            symbol="BTCUSDT",
            exchange=Exchange.BYBIT,
            timeframe=Timeframe.M1,
            open=42500.0,
            high=42550.0,
            low=42480.0,
            close=42530.0,
            volume=125.5,
        )
        spine = map_base_event_to_spine(event, topic="market.candle")

        assert isinstance(spine, RetrievalEvent)
        assert spine.event_type == EventTaxonomy.RETRIEVAL
        assert spine.payload["symbol"] == "BTCUSDT"


class TestExceptionMapping:
    """CircuitBreakerEvent, KillSwitchEvent -> EXCEPTION."""

    def test_circuit_breaker_maps_to_exception(self):
        event = CircuitBreakerEvent(
            breaker_type=CircuitBreakerType.VOLATILITY,
            tripped=True,
            symbol="BTCUSDT",
            reason="Volatility exceeded threshold",
        )
        spine = map_base_event_to_spine(event, topic="risk")

        assert isinstance(spine, ExceptionEvent)
        assert spine.event_type == EventTaxonomy.EXCEPTION
        assert spine.payload["tripped"] is True

    def test_kill_switch_maps_to_exception(self):
        event = KillSwitchEvent(
            activated=True,
            reason="Emergency stop",
        )
        spine = map_base_event_to_spine(event)

        assert isinstance(spine, ExceptionEvent)
        assert spine.event_type == EventTaxonomy.EXCEPTION


class TestValidationMapping:
    """ReconciliationResult -> VALIDATION."""

    def test_reconciliation_maps_to_validation(self):
        event = ReconciliationResult(
            exchange=Exchange.BYBIT,
            orders_synced=5,
            positions_synced=2,
        )
        spine = map_base_event_to_spine(event, topic="system")

        assert isinstance(spine, ValidationEvent)
        assert spine.event_type == EventTaxonomy.VALIDATION


class TestDefaultMapping:
    """Unknown BaseEvent types -> AGENT_STEP_STARTED (safe default)."""

    def test_unknown_event_maps_to_agent_step_started(self):
        event = BaseEvent(source_module="unknown_module")
        spine = map_base_event_to_spine(event)

        assert isinstance(spine, AgentStepStartedEvent)
        assert spine.event_type == EventTaxonomy.AGENT_STEP_STARTED


class TestFieldPreservation:
    """Verify trace_id, causation_id, timestamp preserved from source."""

    def test_trace_id_preserved(self):
        event = Signal(
            trace_id="trace-abc",
            strategy_id="s1",
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
        )
        spine = map_base_event_to_spine(event)
        assert spine.trace_id == "trace-abc"

    def test_causation_id_preserved(self):
        event = Signal(
            causation_id="cause-xyz",
            strategy_id="s1",
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
        )
        spine = map_base_event_to_spine(event)
        assert spine.causation_id == "cause-xyz"

    def test_timestamp_preserved(self):
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)  # noqa: UP017
        event = Signal(
            timestamp=ts,
            strategy_id="s1",
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
        )
        spine = map_base_event_to_spine(event)
        assert spine.timestamp == ts

    def test_tenant_id_propagated(self):
        event = BaseEvent()
        spine = map_base_event_to_spine(event, tenant_id="acme_corp")
        assert spine.tenant_id == "acme_corp"

    def test_component_set_from_source_module(self):
        event = RiskCheckResult(
            check_name="test",
            passed=True,
        )
        spine = map_base_event_to_spine(event)
        assert spine.component == "risk"
        assert spine.actor == "risk"

    def test_span_id_is_source_event_id(self):
        event = Signal(
            strategy_id="s1",
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
        )
        spine = map_base_event_to_spine(event)
        # span_id = event.event_id (each event is its own span)
        assert spine.span_id == event.event_id

    def test_topic_included_in_payload(self):
        event = BaseEvent()
        spine = map_base_event_to_spine(event, topic="risk")
        assert spine.payload["topic"] == "risk"

    def test_source_event_type_in_payload(self):
        event = RiskCheckResult(
            check_name="test",
            passed=True,
        )
        spine = map_base_event_to_spine(event)
        assert spine.payload["source_event_type"] == "RiskCheckResult"
