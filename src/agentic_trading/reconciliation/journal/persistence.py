"""PostgreSQL persistence layer for the trade journal.

Provides ORM models and repository for persisting TradeRecord
objects to the database, enabling historical queries, export,
and dashboard analytics.

Usage::

    from agentic_trading.storage.postgres.connection import get_session
    async with get_session() as session:
        repo = JournalRepo(session)
        await repo.save_trade(trade_record)
        trades = await repo.get_trades_by_strategy("trend_following")
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Sequence

from sqlalchemy import DateTime, Index, Integer, Numeric, String, Text, func, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from agentic_trading.storage.postgres.models import Base, _new_uuid, _utcnow
from .record import TradeOutcome, TradePhase, TradeRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------

class TradeRecordDB(Base):
    """Persisted trade journal record.

    Stores the full lifecycle of a trade including fills, marks, and
    computed analytics for historical querying.
    """

    __tablename__ = "trade_journal"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    trade_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)

    # Phase & outcome
    phase: Mapped[str] = mapped_column(String(16), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Prices
    avg_entry_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    avg_exit_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    initial_risk_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    planned_target_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)

    # Quantities
    total_entry_qty: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    total_exit_qty: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))

    # PnL
    gross_pnl: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    total_fees: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    net_pnl: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))

    # Risk metrics
    initial_risk_amount: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    r_multiple: Mapped[float] = mapped_column(Numeric(10, 4), default=0.0)
    mae: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    mfe: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    mae_price: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    mfe_price: Mapped[Decimal] = mapped_column(Numeric(24, 8), default=Decimal("0"))
    mae_r: Mapped[float] = mapped_column(Numeric(10, 4), default=0.0)
    mfe_r: Mapped[float] = mapped_column(Numeric(10, 4), default=0.0)
    management_efficiency: Mapped[float] = mapped_column(Numeric(10, 4), default=0.0)

    # Signal & governance
    signal_confidence: Mapped[float] = mapped_column(Numeric(8, 4), default=0.0)
    health_score_at_entry: Mapped[float] = mapped_column(Numeric(8, 4), default=1.0)
    governance_sizing_multiplier: Mapped[float] = mapped_column(Numeric(8, 4), default=1.0)

    # Duration
    hold_duration_seconds: Mapped[float] = mapped_column(Numeric(16, 2), default=0.0)

    # Timestamps
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Fill & mark data (JSONB for full detail preservation)
    entry_fills_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    exit_fills_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    mark_samples_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # Classification
    tags: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    mistakes: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_tradejournal_trade_id", "trade_id"),
        Index("ix_tradejournal_trace_id", "trace_id"),
        Index("ix_tradejournal_strategy_id", "strategy_id"),
        Index("ix_tradejournal_symbol", "symbol"),
        Index("ix_tradejournal_phase", "phase"),
        Index("ix_tradejournal_outcome", "outcome"),
        Index("ix_tradejournal_opened_at", "opened_at"),
        Index("ix_tradejournal_closed_at", "closed_at"),
        Index("ix_tradejournal_strategy_symbol", "strategy_id", "symbol"),
    )

    def __repr__(self) -> str:
        return (
            f"<TradeRecordDB(trade_id={self.trade_id!r}, "
            f"strategy={self.strategy_id!r}, outcome={self.outcome!r})>"
        )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _fill_to_json(fill) -> dict[str, Any]:
    """Convert a FillLeg to a JSON-serialisable dict."""
    return {
        "fill_id": fill.fill_id,
        "order_id": fill.order_id,
        "side": fill.side,
        "price": str(fill.price),
        "qty": str(fill.qty),
        "fee": str(fill.fee),
        "fee_currency": fill.fee_currency,
        "is_maker": fill.is_maker,
        "timestamp": fill.timestamp.isoformat() if fill.timestamp else None,
    }


def _mark_to_json(mark) -> dict[str, Any]:
    """Convert a MarkSample to a JSON-serialisable dict."""
    return {
        "timestamp": mark.timestamp.isoformat() if mark.timestamp else None,
        "mark_price": str(mark.mark_price),
        "unrealized_pnl": str(mark.unrealized_pnl),
    }


def trade_to_db(trade: TradeRecord) -> TradeRecordDB:
    """Convert a TradeRecord to a TradeRecordDB ORM instance."""
    return TradeRecordDB(
        trade_id=trade.trade_id,
        trace_id=trade.trace_id,
        strategy_id=trade.strategy_id,
        symbol=trade.symbol,
        direction=trade.direction,
        phase=trade.phase.value,
        outcome=trade.outcome.value if trade.phase == TradePhase.CLOSED else None,
        avg_entry_price=trade.avg_entry_price if trade.entry_fills else None,
        avg_exit_price=trade.avg_exit_price if trade.exit_fills else None,
        initial_risk_price=trade.initial_risk_price,
        planned_target_price=trade.planned_target_price,
        total_entry_qty=trade.entry_qty,
        total_exit_qty=trade.exit_qty,
        gross_pnl=trade.gross_pnl,
        total_fees=trade.total_fees,
        net_pnl=trade.net_pnl,
        initial_risk_amount=trade.initial_risk_amount,
        r_multiple=trade.r_multiple,
        mae=trade.mae,
        mfe=trade.mfe,
        mae_price=trade.mae_price,
        mfe_price=trade.mfe_price,
        mae_r=trade.mae_r,
        mfe_r=trade.mfe_r,
        management_efficiency=trade.management_efficiency,
        signal_confidence=trade.signal_confidence,
        health_score_at_entry=trade.health_score_at_entry,
        governance_sizing_multiplier=trade.governance_sizing_multiplier,
        hold_duration_seconds=trade.hold_duration_seconds,
        opened_at=trade.opened_at,
        closed_at=trade.closed_at,
        entry_fills_json=[_fill_to_json(f) for f in trade.entry_fills],
        exit_fills_json=[_fill_to_json(f) for f in trade.exit_fills],
        mark_samples_json=[_mark_to_json(m) for m in trade.mark_samples],
        tags=trade.tags,
        mistakes=trade.mistakes,
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class JournalRepo:
    """Repository for trade journal persistence and retrieval."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_trade(self, trade: TradeRecord) -> TradeRecordDB:
        """Persist a trade record (insert or update).

        If a record with the same ``trade_id`` already exists, it is
        updated in-place.

        Args:
            trade: In-memory trade record to persist.

        Returns:
            The persisted :class:`TradeRecordDB`.
        """
        stmt = select(TradeRecordDB).where(
            TradeRecordDB.trade_id == trade.trade_id
        )
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            # Update all mutable fields
            existing.phase = trade.phase.value
            existing.outcome = trade.outcome.value if trade.phase == TradePhase.CLOSED else None
            existing.avg_entry_price = trade.avg_entry_price if trade.entry_fills else None
            existing.avg_exit_price = trade.avg_exit_price if trade.exit_fills else None
            existing.total_entry_qty = trade.entry_qty
            existing.total_exit_qty = trade.exit_qty
            existing.gross_pnl = trade.gross_pnl
            existing.total_fees = trade.total_fees
            existing.net_pnl = trade.net_pnl
            existing.r_multiple = trade.r_multiple
            existing.mae = trade.mae
            existing.mfe = trade.mfe
            existing.mae_price = trade.mae_price
            existing.mfe_price = trade.mfe_price
            existing.mae_r = trade.mae_r
            existing.mfe_r = trade.mfe_r
            existing.management_efficiency = trade.management_efficiency
            existing.hold_duration_seconds = trade.hold_duration_seconds
            existing.closed_at = trade.closed_at
            existing.entry_fills_json = [_fill_to_json(f) for f in trade.entry_fills]
            existing.exit_fills_json = [_fill_to_json(f) for f in trade.exit_fills]
            existing.mark_samples_json = [_mark_to_json(m) for m in trade.mark_samples]
            existing.tags = trade.tags
            existing.mistakes = trade.mistakes
            await self._session.flush()
            logger.debug("Updated journal trade %s", trade.trade_id)
            return existing

        record = trade_to_db(trade)
        self._session.add(record)
        await self._session.flush()
        logger.debug("Inserted journal trade %s", trade.trade_id)
        return record

    async def get_trade(self, trade_id: str) -> TradeRecordDB | None:
        """Retrieve a single trade by trade_id."""
        stmt = select(TradeRecordDB).where(TradeRecordDB.trade_id == trade_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_trades_by_strategy(
        self,
        strategy_id: str,
        *,
        symbol: str | None = None,
        phase: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[TradeRecordDB]:
        """Retrieve trades for a strategy, newest first."""
        stmt = (
            select(TradeRecordDB)
            .where(TradeRecordDB.strategy_id == strategy_id)
            .order_by(TradeRecordDB.opened_at.desc())
            .limit(limit)
        )
        if symbol is not None:
            stmt = stmt.where(TradeRecordDB.symbol == symbol)
        if phase is not None:
            stmt = stmt.where(TradeRecordDB.phase == phase)
        if since is not None:
            stmt = stmt.where(TradeRecordDB.opened_at >= since)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_trades_by_symbol(
        self,
        symbol: str,
        *,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[TradeRecordDB]:
        """Retrieve trades for a symbol, newest first."""
        stmt = (
            select(TradeRecordDB)
            .where(TradeRecordDB.symbol == symbol)
            .order_by(TradeRecordDB.opened_at.desc())
            .limit(limit)
        )
        if since is not None:
            stmt = stmt.where(TradeRecordDB.opened_at >= since)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_closed_trades(
        self,
        *,
        strategy_id: str | None = None,
        symbol: str | None = None,
        limit: int = 500,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[TradeRecordDB]:
        """Retrieve closed trades with optional filters."""
        stmt = (
            select(TradeRecordDB)
            .where(TradeRecordDB.phase == "closed")
            .order_by(TradeRecordDB.closed_at.desc())
            .limit(limit)
        )
        if strategy_id is not None:
            stmt = stmt.where(TradeRecordDB.strategy_id == strategy_id)
        if symbol is not None:
            stmt = stmt.where(TradeRecordDB.symbol == symbol)
        if since is not None:
            stmt = stmt.where(TradeRecordDB.closed_at >= since)
        if until is not None:
            stmt = stmt.where(TradeRecordDB.closed_at <= until)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_trades(
        self,
        *,
        strategy_id: str | None = None,
        phase: str | None = None,
    ) -> int:
        """Count trades with optional filters."""
        from sqlalchemy import func as sqlfunc

        stmt = select(sqlfunc.count()).select_from(TradeRecordDB)
        if strategy_id is not None:
            stmt = stmt.where(TradeRecordDB.strategy_id == strategy_id)
        if phase is not None:
            stmt = stmt.where(TradeRecordDB.phase == phase)

        result = await self._session.execute(stmt)
        return result.scalar() or 0

    async def get_strategy_summary(self, strategy_id: str) -> dict[str, Any]:
        """Get aggregate stats for a strategy."""
        from sqlalchemy import func as sqlfunc

        stmt = (
            select(
                sqlfunc.count().label("total"),
                sqlfunc.sum(TradeRecordDB.net_pnl).label("total_pnl"),
                sqlfunc.avg(TradeRecordDB.net_pnl).label("avg_pnl"),
                sqlfunc.avg(TradeRecordDB.r_multiple).label("avg_r"),
                sqlfunc.min(TradeRecordDB.net_pnl).label("worst_trade"),
                sqlfunc.max(TradeRecordDB.net_pnl).label("best_trade"),
            )
            .where(TradeRecordDB.strategy_id == strategy_id)
            .where(TradeRecordDB.phase == "closed")
        )
        result = await self._session.execute(stmt)
        row = result.one()

        wins_stmt = (
            select(sqlfunc.count())
            .select_from(TradeRecordDB)
            .where(TradeRecordDB.strategy_id == strategy_id)
            .where(TradeRecordDB.outcome == "win")
        )
        wins_result = await self._session.execute(wins_stmt)
        wins = wins_result.scalar() or 0

        total = row.total or 0
        return {
            "strategy_id": strategy_id,
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total, 4) if total > 0 else 0.0,
            "total_pnl": float(row.total_pnl or 0),
            "avg_pnl": float(row.avg_pnl or 0),
            "avg_r": float(row.avg_r or 0),
            "best_trade": float(row.best_trade or 0),
            "worst_trade": float(row.worst_trade or 0),
        }
