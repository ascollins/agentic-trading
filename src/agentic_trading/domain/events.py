"""Canonical domain events for the agentic trading platform.

Design invariants
-----------------
1.  Every event is **immutable** (``frozen=True``).
2.  Every event type has exactly **one writer** — see ``WRITE_OWNERSHIP``.
3.  ``event_id`` is a UUID4 generated at creation time; it serves as the
    idempotency / dedup key in the event store.
4.  ``correlation_id`` links all events that originate from the *same
    external trigger* (e.g. a single candle close).
5.  ``causation_id`` points to the ``event_id`` that *directly caused*
    this event, forming a DAG of causality.

Existing code (``core/events.py``) continues to work unchanged.
These canonical events will gradually replace legacy events as each
pipeline layer is migrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from agentic_trading.core.ids import new_id as _uuid
from agentic_trading.core.ids import utc_now as _now

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainEvent:
    """Immutable base for every canonical domain event.

    Shared fields
    ~~~~~~~~~~~~~
    event_id        Unique identity (UUID4).  Idempotency key.
    timestamp       UTC creation time.
    correlation_id  Groups events from the same causal chain.
    causation_id    The ``event_id`` that directly caused this event.
    source          Writer agent / module that produced this event.
    """

    event_id: str = field(default_factory=_uuid)
    timestamp: datetime = field(default_factory=_now)
    correlation_id: str = ""
    causation_id: str = ""
    source: str = ""


# =========================================================================
# Intelligence Layer  (writer: intelligence)
# =========================================================================

@dataclass(frozen=True)
class CandleReceived(DomainEvent):
    """Raw OHLCV candle ingested from the exchange feed."""

    symbol: str = ""
    exchange: str = ""
    timeframe: str = ""
    open: Decimal = Decimal("0")
    high: Decimal = Decimal("0")
    low: Decimal = Decimal("0")
    close: Decimal = Decimal("0")
    volume: Decimal = Decimal("0")
    is_closed: bool = True


@dataclass(frozen=True)
class FeatureComputed(DomainEvent):
    """Feature vector ready for strategy consumption."""

    symbol: str = ""
    timeframe: str = ""
    features: tuple[tuple[str, float | None], ...] = ()

    # Convenience --------------------------------------------------------
    def features_dict(self) -> dict[str, float | None]:
        """Return a mutable dict copy for strategy consumption."""
        return dict(self.features)


# =========================================================================
# Signal Layer  (writer: signal)
# =========================================================================

@dataclass(frozen=True)
class SignalCreated(DomainEvent):
    """A strategy produced a trading signal."""

    strategy_id: str = ""
    symbol: str = ""
    direction: str = ""          # LONG | SHORT | FLAT
    confidence: float = 0.0
    rationale: str = ""
    take_profit: Decimal | None = None
    stop_loss: Decimal | None = None
    trailing_stop: Decimal | None = None
    features_used: tuple[tuple[str, float], ...] = ()
    timeframe: str = ""


@dataclass(frozen=True)
class DecisionProposed(DomainEvent):
    """Portfolio manager aggregated signals into a sized intent."""

    strategy_id: str = ""
    symbol: str = ""
    side: str = ""               # buy | sell
    qty: Decimal = Decimal("0")
    order_type: str = "market"
    dedupe_key: str = ""
    signal_event_id: str = ""    # back-ref to SignalCreated


# =========================================================================
# Policy Layer  (writer: policy_gate)
# =========================================================================

@dataclass(frozen=True)
class DecisionApproved(DomainEvent):
    """PolicyGate approved a ``DecisionProposed``."""

    decision_event_id: str = ""  # back-ref to DecisionProposed
    sizing_multiplier: float = 1.0
    maturity_level: str = ""
    impact_tier: str = ""
    checks_passed: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionRejected(DomainEvent):
    """PolicyGate rejected a ``DecisionProposed``."""

    decision_event_id: str = ""
    reason: str = ""
    failed_checks: tuple[str, ...] = ()
    action: str = ""             # BLOCK | PAUSE | DEMOTE | KILL


@dataclass(frozen=True)
class DecisionPending(DomainEvent):
    """PolicyGate requires human approval for a ``DecisionProposed``."""

    decision_event_id: str = ""
    approval_request_id: str = ""
    escalation_level: str = ""
    ttl_seconds: int = 300


# =========================================================================
# Execution Layer  (writer: execution)
# =========================================================================

@dataclass(frozen=True)
class OrderPlanned(DomainEvent):
    """Execution planner created a validated order for submission."""

    decision_event_id: str = ""
    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    qty: Decimal = Decimal("0")
    price: Decimal | None = None


@dataclass(frozen=True)
class OrderSubmitted(DomainEvent):
    """Order sent to the exchange via the gateway."""

    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    exchange: str = ""


@dataclass(frozen=True)
class OrderAccepted(DomainEvent):
    """Exchange acknowledged the order."""

    order_id: str = ""
    client_order_id: str = ""
    exchange_order_id: str = ""
    symbol: str = ""
    status: str = ""


@dataclass(frozen=True)
class OrderRejected(DomainEvent):
    """Exchange or gateway rejected the order."""

    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    reason: str = ""


# =========================================================================
# Reconciliation Layer  (writer: reconciliation)
# =========================================================================

@dataclass(frozen=True)
class FillReceived(DomainEvent):
    """Fill confirmed by exchange or paper adapter."""

    fill_id: str = ""
    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    exchange: str = ""
    side: str = ""               # buy | sell
    price: Decimal = Decimal("0")
    qty: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    fee_currency: str = ""
    is_maker: bool = False


@dataclass(frozen=True)
class PositionUpdated(DomainEvent):
    """Canonical position snapshot after reconciliation."""

    symbol: str = ""
    exchange: str = ""
    qty: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    mark_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    leverage: int = 1


@dataclass(frozen=True)
class PnLUpdated(DomainEvent):
    """Portfolio-level P&L snapshot after reconciliation."""

    total_equity: Decimal = Decimal("0")
    gross_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    drawdown_pct: float = 0.0


# =========================================================================
# Incident Layer  (writer: incident)
# =========================================================================

@dataclass(frozen=True)
class IncidentRaised(DomainEvent):
    """Something went wrong — incident lifecycle begins."""

    incident_id: str = ""
    severity: str = ""           # LOW | MEDIUM | HIGH | CRITICAL
    trigger: str = ""
    description: str = ""
    affected_strategies: tuple[str, ...] = ()
    affected_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class TradingHalted(DomainEvent):
    """Kill switch or degraded mode activated."""

    reason: str = ""
    triggered_by: str = ""
    mode: str = ""               # FULL_STOP | NO_NEW_TRADES | READ_ONLY


# =========================================================================
# Audit  (writer: execution.gateway)
# =========================================================================

@dataclass(frozen=True)
class AuditLogged(DomainEvent):
    """Immutable audit entry for any exchange mutation."""

    action: str = ""
    tool_name: str = ""
    params_hash: str = ""
    result_hash: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None


# =========================================================================
# Write-ownership registry
# =========================================================================

#: Maps each canonical event type to the *only* ``source`` value that is
#: allowed to produce it.  The event bus enforcement layer (PR 2) will
#: use this table to reject misattributed publishes.
WRITE_OWNERSHIP: dict[type[DomainEvent], str] = {
    # Intelligence
    CandleReceived: "intelligence",
    FeatureComputed: "intelligence",
    # Signal
    SignalCreated: "signal",
    DecisionProposed: "signal",
    # Policy
    DecisionApproved: "policy_gate",
    DecisionRejected: "policy_gate",
    DecisionPending: "policy_gate",
    # Execution
    OrderPlanned: "execution",
    OrderSubmitted: "execution",
    OrderAccepted: "execution",
    OrderRejected: "execution",
    # Reconciliation
    FillReceived: "reconciliation",
    PositionUpdated: "reconciliation",
    PnLUpdated: "reconciliation",
    # Incident
    IncidentRaised: "incident",
    TradingHalted: "incident",
    # Audit
    AuditLogged: "execution.gateway",
}


#: All canonical event types in a deterministic order.
ALL_DOMAIN_EVENTS: tuple[type[DomainEvent], ...] = tuple(WRITE_OWNERSHIP.keys())
