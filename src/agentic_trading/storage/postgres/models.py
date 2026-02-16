"""SQLAlchemy ORM models for the trading platform audit/trade database.

All tables use UUIDs for primary keys, UTC timestamps, and proper indexes
for the most common query patterns (by symbol, by trace_id, by time range).

Relationships:
    OrderRecord 1--* FillRecord  (order_id foreign key)
    DecisionAudit links the full chain: features -> signal -> risk -> intent -> fill
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# OrderRecord
# ---------------------------------------------------------------------------

class OrderRecord(Base):
    """Persisted order lifecycle record.

    Maps from :class:`agentic_trading.core.models.Order`.
    Every status transition results in an UPDATE to this row rather than
    a new INSERT, so the row always reflects the latest state.
    """

    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    order_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    client_order_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(32), nullable=False)
    time_in_force: Mapped[str] = mapped_column(String(8), default="GTC")
    price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    qty: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    filled_qty: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    remaining_qty: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    avg_fill_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    reduce_only: Mapped[bool] = mapped_column(Boolean, default=False)
    post_only: Mapped[bool] = mapped_column(Boolean, default=False)
    leverage: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strategy_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now(),
    )

    # Relationships
    fills: Mapped[list[FillRecord]] = relationship(
        "FillRecord",
        back_populates="order",
        foreign_keys="[FillRecord.order_id]",
        primaryjoin="OrderRecord.order_id == FillRecord.order_id",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_orders_symbol", "symbol"),
        Index("ix_orders_status", "status"),
        Index("ix_orders_strategy_id", "strategy_id"),
        Index("ix_orders_trace_id", "trace_id"),
        Index("ix_orders_created_at", "created_at"),
        Index("ix_orders_symbol_status", "symbol", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<OrderRecord(order_id={self.order_id!r}, symbol={self.symbol!r}, "
            f"side={self.side!r}, status={self.status!r})>"
        )


# ---------------------------------------------------------------------------
# FillRecord
# ---------------------------------------------------------------------------

class FillRecord(Base):
    """Individual fill (execution) belonging to an order."""

    __tablename__ = "fills"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    fill_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    order_id: Mapped[str] = mapped_column(
        String(128), nullable=False,
    )
    client_order_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    fee_currency: Mapped[str] = mapped_column(String(16), nullable=False)
    is_maker: Mapped[bool] = mapped_column(Boolean, default=False)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fill_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )

    # Relationships
    order: Mapped[OrderRecord | None] = relationship(
        "OrderRecord",
        back_populates="fills",
        foreign_keys=[order_id],
        primaryjoin="FillRecord.order_id == OrderRecord.order_id",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_fills_order_id", "order_id"),
        Index("ix_fills_symbol", "symbol"),
        Index("ix_fills_trace_id", "trace_id"),
        Index("ix_fills_fill_timestamp", "fill_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<FillRecord(fill_id={self.fill_id!r}, order_id={self.order_id!r}, "
            f"price={self.price}, qty={self.qty})>"
        )


# ---------------------------------------------------------------------------
# PositionSnapshot
# ---------------------------------------------------------------------------

class PositionSnapshot(Base):
    """Periodic position snapshot for historical tracking and reconciliation."""

    __tablename__ = "position_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    mark_price: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    liquidation_price: Mapped[Decimal | None] = mapped_column(
        Numeric(24, 8), nullable=True,
    )
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(24, 8), default=Decimal("0"),
    )
    realized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(24, 8), default=Decimal("0"),
    )
    leverage: Mapped[int] = mapped_column(Integer, default=1)
    margin_mode: Mapped[str] = mapped_column(String(16), default="cross")
    notional: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )

    __table_args__ = (
        Index("ix_possnap_symbol", "symbol"),
        Index("ix_possnap_snapshot_at", "snapshot_at"),
        Index("ix_possnap_symbol_snapshot", "symbol", "snapshot_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<PositionSnapshot(symbol={self.symbol!r}, qty={self.qty}, "
            f"snapshot_at={self.snapshot_at})>"
        )


# ---------------------------------------------------------------------------
# BalanceSnapshot
# ---------------------------------------------------------------------------

class BalanceSnapshot(Base):
    """Periodic balance snapshot for historical equity curve tracking."""

    __tablename__ = "balance_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    currency: Mapped[str] = mapped_column(String(16), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    total: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    free: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    used: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )

    __table_args__ = (
        Index("ix_balsnap_currency", "currency"),
        Index("ix_balsnap_snapshot_at", "snapshot_at"),
        Index("ix_balsnap_currency_exchange_snapshot", "currency", "exchange", "snapshot_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<BalanceSnapshot(currency={self.currency!r}, total={self.total}, "
            f"snapshot_at={self.snapshot_at})>"
        )


# ---------------------------------------------------------------------------
# DecisionAudit
# ---------------------------------------------------------------------------

class DecisionAudit(Base):
    """Full decision audit chain: features -> signal -> risk -> intent -> fill.

    Each row captures one complete decision cycle. The JSONB columns allow
    flexible storage of the nested event payloads while keeping the
    trace_id and strategy_id indexed for fast lookups.
    """

    __tablename__ = "decision_audits"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)

    # Decision chain payloads (JSONB for flexible querying)
    feature_vector: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    signal: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    risk_check: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    order_intent: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    order_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    fill_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Summary fields for quick filtering
    signal_direction: Mapped[str | None] = mapped_column(String(8), nullable=True)
    signal_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    risk_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    final_status: Mapped[str | None] = mapped_column(String(24), nullable=True)

    decision_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_decision_trace_id", "trace_id"),
        Index("ix_decision_strategy_id", "strategy_id"),
        Index("ix_decision_symbol", "symbol"),
        Index("ix_decision_decision_at", "decision_at"),
        Index("ix_decision_strategy_symbol", "strategy_id", "symbol"),
    )

    def __repr__(self) -> str:
        return (
            f"<DecisionAudit(trace_id={self.trace_id!r}, strategy={self.strategy_id!r}, "
            f"symbol={self.symbol!r}, direction={self.signal_direction!r})>"
        )


# ---------------------------------------------------------------------------
# ExperimentLog
# ---------------------------------------------------------------------------

class ExperimentLog(Base):
    """Backtest experiment configuration and results.

    Stores the full parameter set used, date ranges, and aggregate
    performance metrics so that experiments are reproducible and
    comparable.
    """

    __tablename__ = "experiment_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    experiment_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Configuration snapshot
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    symbols: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    timeframes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    start_date: Mapped[str] = mapped_column(String(32), nullable=False)
    end_date: Mapped[str] = mapped_column(String(32), nullable=False)
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(24, 8), nullable=False)

    # Result metrics
    final_equity: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    total_return_pct: Mapped[float | None] = mapped_column(Numeric(12, 6), nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    sortino_ratio: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    max_drawdown_pct: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Numeric(8, 6), nullable=True)
    profit_factor: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)
    total_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_trade_duration_minutes: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True,
    )

    # Full results payload
    results_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    equity_curve: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # Metadata
    git_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="running")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_experiment_experiment_id", "experiment_id"),
        Index("ix_experiment_strategy_id", "strategy_id"),
        Index("ix_experiment_status", "status"),
        Index("ix_experiment_started_at", "started_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<ExperimentLog(experiment_id={self.experiment_id!r}, "
            f"name={self.name!r}, status={self.status!r})>"
        )


# ---------------------------------------------------------------------------
# GovernanceLog
# ---------------------------------------------------------------------------

class GovernanceLog(Base):
    """Governance decision audit log (Soteria-inspired).

    Records every governance gate decision for full traceability
    of why trades were allowed, sized-down, or blocked.
    """

    __tablename__ = "governance_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    maturity_level: Mapped[str | None] = mapped_column(String(24), nullable=True)
    impact_tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    health_score: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 4), nullable=True,
    )
    sizing_multiplier: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 4), nullable=True,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    decision_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_govlog_trace_id", "trace_id"),
        Index("ix_govlog_strategy_id", "strategy_id"),
        Index("ix_govlog_symbol", "symbol"),
        Index("ix_govlog_decision_at", "decision_at"),
        Index("ix_govlog_action", "action"),
    )

    def __repr__(self) -> str:
        return (
            f"<GovernanceLog(trace_id={self.trace_id!r}, "
            f"strategy={self.strategy_id!r}, action={self.action!r})>"
        )
