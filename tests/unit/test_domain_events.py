"""Tests for canonical domain events (``domain/events.py``).

Covers:
- All 17 event types instantiate correctly.
- Every event is immutable (frozen dataclass).
- ``event_id`` is unique across instances.
- ``WRITE_OWNERSHIP`` maps every event type to a source string.
- Serialization round-trip via ``dataclasses.asdict`` preserves values.
- ``FeatureComputed.features_dict()`` convenience works.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from agentic_trading.domain.events import (
    ALL_DOMAIN_EVENTS,
    AuditLogged,
    CandleReceived,
    DecisionApproved,
    DecisionPending,
    DecisionProposed,
    DecisionRejected,
    DomainEvent,
    FeatureComputed,
    FillReceived,
    IncidentRaised,
    OrderAccepted,
    OrderPlanned,
    OrderRejected,
    OrderSubmitted,
    PnLUpdated,
    PositionUpdated,
    SignalCreated,
    TradingHalted,
    WRITE_OWNERSHIP,
)


class TestDomainEventBase:
    """Tests for the ``DomainEvent`` base class."""

    def test_default_fields_populated(self):
        e = DomainEvent()
        assert isinstance(e.event_id, str)
        assert len(e.event_id) == 36  # UUID4 format
        assert isinstance(e.timestamp, datetime)
        assert e.correlation_id == ""
        assert e.causation_id == ""
        assert e.source == ""

    def test_event_id_unique(self):
        ids = {DomainEvent().event_id for _ in range(100)}
        assert len(ids) == 100

    def test_frozen(self):
        e = DomainEvent()
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.source = "oops"  # type: ignore[misc]

    def test_custom_fields(self):
        e = DomainEvent(
            event_id="custom-id",
            correlation_id="corr-1",
            causation_id="cause-1",
            source="test",
        )
        assert e.event_id == "custom-id"
        assert e.correlation_id == "corr-1"
        assert e.causation_id == "cause-1"
        assert e.source == "test"


class TestAllCanonicalEvents:
    """Every canonical event type instantiates and is frozen."""

    @pytest.mark.parametrize("cls", ALL_DOMAIN_EVENTS)
    def test_instantiates_with_defaults(self, cls: type[DomainEvent]):
        event = cls(source=WRITE_OWNERSHIP.get(cls, "test"))
        assert isinstance(event, DomainEvent)
        assert isinstance(event.event_id, str)
        assert len(event.event_id) == 36

    @pytest.mark.parametrize("cls", ALL_DOMAIN_EVENTS)
    def test_frozen(self, cls: type[DomainEvent]):
        event = cls(source=WRITE_OWNERSHIP.get(cls, "test"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.source = "mutated"  # type: ignore[misc]


class TestWriteOwnership:
    """``WRITE_OWNERSHIP`` covers every canonical event type."""

    def test_all_events_have_owner(self):
        for cls in ALL_DOMAIN_EVENTS:
            assert cls in WRITE_OWNERSHIP, f"{cls.__name__} missing from WRITE_OWNERSHIP"

    def test_owners_are_non_empty_strings(self):
        for cls, owner in WRITE_OWNERSHIP.items():
            assert isinstance(owner, str)
            assert len(owner) > 0, f"{cls.__name__} has empty owner"

    def test_event_count(self):
        assert len(ALL_DOMAIN_EVENTS) == 17


class TestCandleReceived:
    def test_fields(self):
        e = CandleReceived(
            source="intelligence",
            symbol="BTCUSDT",
            exchange="bybit",
            timeframe="1h",
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("1234.5"),
            is_closed=True,
        )
        assert e.symbol == "BTCUSDT"
        assert e.close == Decimal("50500")
        assert e.is_closed is True


class TestFeatureComputed:
    def test_features_dict(self):
        e = FeatureComputed(
            source="intelligence",
            symbol="ETHUSDT",
            timeframe="5m",
            features=(("rsi", 65.3), ("ema_12", 2500.0), ("null_feat", None)),
        )
        d = e.features_dict()
        assert d == {"rsi": 65.3, "ema_12": 2500.0, "null_feat": None}
        # Returned dict is mutable (copy), doesn't affect original
        d["new"] = 42.0
        assert len(e.features) == 3

    def test_empty_features(self):
        e = FeatureComputed(source="intelligence")
        assert e.features_dict() == {}


class TestSignalCreated:
    def test_fields(self):
        e = SignalCreated(
            source="signal",
            strategy_id="trend_v1",
            symbol="BTCUSDT",
            direction="LONG",
            confidence=0.85,
            rationale="EMA crossover with high volume",
            take_profit=Decimal("52000"),
            stop_loss=Decimal("48000"),
        )
        assert e.direction == "LONG"
        assert e.confidence == 0.85
        assert e.take_profit == Decimal("52000")


class TestDecisionProposed:
    def test_back_reference(self):
        sig = SignalCreated(source="signal", strategy_id="s1", symbol="X")
        dec = DecisionProposed(
            source="signal",
            strategy_id="s1",
            symbol="X",
            side="buy",
            qty=Decimal("0.5"),
            signal_event_id=sig.event_id,
        )
        assert dec.signal_event_id == sig.event_id


class TestPolicyEvents:
    def test_approved(self):
        e = DecisionApproved(
            source="policy_gate",
            decision_event_id="dec-1",
            sizing_multiplier=0.8,
            maturity_level="L2_gated",
            impact_tier="medium",
            checks_passed=("kill_switch", "circuit_breaker", "position_limit"),
        )
        assert e.sizing_multiplier == 0.8
        assert len(e.checks_passed) == 3

    def test_rejected(self):
        e = DecisionRejected(
            source="policy_gate",
            decision_event_id="dec-1",
            reason="Max drawdown exceeded",
            failed_checks=("drawdown_limit",),
            action="BLOCK",
        )
        assert e.action == "BLOCK"

    def test_pending(self):
        e = DecisionPending(
            source="policy_gate",
            decision_event_id="dec-1",
            approval_request_id="apr-1",
            escalation_level="L2_OPERATOR",
            ttl_seconds=300,
        )
        assert e.ttl_seconds == 300


class TestExecutionEvents:
    def test_order_lifecycle(self):
        planned = OrderPlanned(
            source="execution",
            decision_event_id="dec-1",
            order_id="ord-1",
            client_order_id="coid-1",
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            qty=Decimal("0.1"),
        )
        submitted = OrderSubmitted(
            source="execution",
            order_id="ord-1",
            client_order_id="coid-1",
            symbol="BTCUSDT",
            exchange="bybit",
            causation_id=planned.event_id,
        )
        accepted = OrderAccepted(
            source="execution",
            order_id="ord-1",
            client_order_id="coid-1",
            exchange_order_id="exch-123",
            symbol="BTCUSDT",
            status="submitted",
            causation_id=submitted.event_id,
        )
        assert accepted.causation_id == submitted.event_id
        assert submitted.causation_id == planned.event_id

    def test_rejected(self):
        e = OrderRejected(
            source="execution",
            order_id="ord-1",
            client_order_id="coid-1",
            symbol="BTCUSDT",
            reason="Insufficient margin",
        )
        assert e.reason == "Insufficient margin"


class TestReconciliationEvents:
    def test_fill_received(self):
        e = FillReceived(
            source="reconciliation",
            fill_id="fill-1",
            order_id="ord-1",
            client_order_id="coid-1",
            symbol="BTCUSDT",
            exchange="bybit",
            side="buy",
            price=Decimal("50000"),
            qty=Decimal("0.1"),
            fee=Decimal("0.005"),
            fee_currency="USDT",
            is_maker=False,
        )
        assert e.price == Decimal("50000")
        assert e.is_maker is False

    def test_position_updated(self):
        e = PositionUpdated(
            source="reconciliation",
            symbol="BTCUSDT",
            exchange="bybit",
            qty=Decimal("0.1"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50500"),
            unrealized_pnl=Decimal("50"),
            leverage=10,
        )
        assert e.leverage == 10

    def test_pnl_updated(self):
        e = PnLUpdated(
            source="reconciliation",
            total_equity=Decimal("100050"),
            gross_exposure=Decimal("5000"),
            net_exposure=Decimal("5000"),
            daily_pnl=Decimal("50"),
            drawdown_pct=0.0,
        )
        assert e.total_equity == Decimal("100050")


class TestIncidentEvents:
    def test_incident_raised(self):
        e = IncidentRaised(
            source="incident",
            incident_id="inc-1",
            severity="CRITICAL",
            trigger="circuit_breaker",
            description="Volatility breaker tripped",
            affected_symbols=("BTCUSDT",),
        )
        assert e.severity == "CRITICAL"
        assert "BTCUSDT" in e.affected_symbols

    def test_trading_halted(self):
        e = TradingHalted(
            source="incident",
            reason="Daily loss limit exceeded",
            triggered_by="risk_manager",
            mode="FULL_STOP",
        )
        assert e.mode == "FULL_STOP"


class TestAuditLogged:
    def test_fields(self):
        e = AuditLogged(
            source="execution.gateway",
            action="submit_order",
            tool_name="SUBMIT_ORDER",
            params_hash="abc123",
            result_hash="def456",
            latency_ms=12.5,
            success=True,
        )
        assert e.success is True
        assert e.latency_ms == 12.5

    def test_failure(self):
        e = AuditLogged(
            source="execution.gateway",
            action="submit_order",
            tool_name="SUBMIT_ORDER",
            params_hash="abc123",
            result_hash="",
            latency_ms=5000.0,
            success=False,
            error="Connection timeout",
        )
        assert e.success is False
        assert e.error == "Connection timeout"


class TestSerialization:
    """``dataclasses.asdict`` round-trip preserves data."""

    def test_roundtrip_fill_received(self):
        original = FillReceived(
            source="reconciliation",
            fill_id="f1",
            order_id="o1",
            client_order_id="co1",
            symbol="BTCUSDT",
            exchange="bybit",
            side="buy",
            price=Decimal("50000.50"),
            qty=Decimal("0.001"),
            fee=Decimal("0.00001"),
            fee_currency="BTC",
            is_maker=True,
            correlation_id="corr-1",
            causation_id="cause-1",
        )
        d = dataclasses.asdict(original)
        restored = FillReceived(**d)
        assert restored == original
        assert restored.price == Decimal("50000.50")
        assert restored.correlation_id == "corr-1"


class TestCausalityChain:
    """Events link via ``correlation_id`` and ``causation_id``."""

    def test_chain(self):
        corr = "trade-lifecycle-1"

        sig = SignalCreated(
            source="signal",
            correlation_id=corr,
            strategy_id="s1",
            symbol="X",
            direction="LONG",
        )
        dec = DecisionProposed(
            source="signal",
            correlation_id=corr,
            causation_id=sig.event_id,
            signal_event_id=sig.event_id,
            strategy_id="s1",
            symbol="X",
            side="buy",
            qty=Decimal("1"),
        )
        approved = DecisionApproved(
            source="policy_gate",
            correlation_id=corr,
            causation_id=dec.event_id,
            decision_event_id=dec.event_id,
        )

        # All share the same correlation
        assert sig.correlation_id == corr
        assert dec.correlation_id == corr
        assert approved.correlation_id == corr

        # Causation forms a chain
        assert dec.causation_id == sig.event_id
        assert approved.causation_id == dec.event_id
