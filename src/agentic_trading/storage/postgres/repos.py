"""Repository pattern for async database operations.

Each repository encapsulates query logic for a single aggregate root.
All methods accept an :class:`AsyncSession` obtained from
:func:`agentic_trading.storage.postgres.connection.get_session`.

Conversion helpers translate between core domain models
(:mod:`agentic_trading.core.models`) and ORM records.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentic_trading.core.enums import (
    Exchange,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from agentic_trading.core.models import Fill, Order

from .models import (
    DecisionAudit,
    ExperimentLog,
    FillRecord,
    OrderRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _order_to_record(order: Order) -> OrderRecord:
    """Convert a core :class:`Order` to an ORM :class:`OrderRecord`."""
    return OrderRecord(
        order_id=order.order_id,
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        exchange=order.exchange.value,
        side=order.side.value,
        order_type=order.order_type.value,
        time_in_force=order.time_in_force.value,
        price=order.price,
        stop_price=order.stop_price,
        qty=order.qty,
        filled_qty=order.filled_qty,
        remaining_qty=order.remaining_qty,
        avg_fill_price=order.avg_fill_price,
        status=order.status.value,
        reduce_only=order.reduce_only,
        post_only=order.post_only,
        leverage=order.leverage,
        strategy_id=order.strategy_id,
        trace_id=order.trace_id,
        metadata_json=order.metadata if order.metadata else None,
        created_at=order.created_at,
        updated_at=order.updated_at,
    )


def _record_to_order(record: OrderRecord) -> Order:
    """Convert an ORM :class:`OrderRecord` back to a core :class:`Order`."""
    return Order(
        order_id=record.order_id,
        client_order_id=record.client_order_id,
        symbol=record.symbol,
        exchange=Exchange(record.exchange),
        side=Side(record.side),
        order_type=OrderType(record.order_type),
        time_in_force=TimeInForce(record.time_in_force),
        price=record.price,
        stop_price=record.stop_price,
        qty=record.qty,
        filled_qty=record.filled_qty,
        remaining_qty=record.remaining_qty,
        avg_fill_price=record.avg_fill_price,
        status=OrderStatus(record.status),
        reduce_only=record.reduce_only,
        post_only=record.post_only,
        leverage=record.leverage,
        strategy_id=record.strategy_id,
        trace_id=record.trace_id,
        metadata=record.metadata_json or {},
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _fill_to_record(fill: Fill) -> FillRecord:
    """Convert a core :class:`Fill` to an ORM :class:`FillRecord`."""
    return FillRecord(
        fill_id=fill.fill_id,
        order_id=fill.order_id,
        client_order_id=fill.client_order_id,
        symbol=fill.symbol,
        exchange=fill.exchange.value,
        side=fill.side.value,
        price=fill.price,
        qty=fill.qty,
        fee=fill.fee,
        fee_currency=fill.fee_currency,
        is_maker=fill.is_maker,
        trace_id=fill.trace_id,
        fill_timestamp=fill.timestamp,
    )


def _record_to_fill(record: FillRecord) -> Fill:
    """Convert an ORM :class:`FillRecord` back to a core :class:`Fill`."""
    return Fill(
        fill_id=record.fill_id,
        order_id=record.order_id,
        client_order_id=record.client_order_id,
        symbol=record.symbol,
        exchange=Exchange(record.exchange),
        side=Side(record.side),
        price=record.price,
        qty=record.qty,
        fee=record.fee,
        fee_currency=record.fee_currency,
        is_maker=record.is_maker,
        trace_id=record.trace_id,
        timestamp=record.fill_timestamp,
    )


# ---------------------------------------------------------------------------
# OrderRepo
# ---------------------------------------------------------------------------

class OrderRepo:
    """Repository for :class:`OrderRecord` persistence and retrieval."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_order(self, order: Order) -> OrderRecord:
        """Insert or update an order record.

        If an order with the same ``order_id`` already exists, the row is
        updated in-place (upsert semantics via merge).

        Args:
            order: Core domain order to persist.

        Returns:
            The persisted :class:`OrderRecord`.
        """
        # Check for existing record
        stmt = select(OrderRecord).where(OrderRecord.order_id == order.order_id)
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            # Update mutable fields
            existing.status = order.status.value
            existing.filled_qty = order.filled_qty
            existing.remaining_qty = order.remaining_qty
            existing.avg_fill_price = order.avg_fill_price
            existing.updated_at = order.updated_at
            existing.metadata_json = order.metadata if order.metadata else existing.metadata_json
            await self._session.flush()
            logger.debug("Updated order %s -> status=%s", order.order_id, order.status.value)
            return existing

        record = _order_to_record(order)
        self._session.add(record)
        await self._session.flush()
        logger.debug("Inserted order %s", order.order_id)
        return record

    async def get_order(self, order_id: str) -> Order | None:
        """Retrieve an order by its exchange order ID.

        Args:
            order_id: The exchange-assigned order identifier.

        Returns:
            Core :class:`Order` if found, otherwise ``None``.
        """
        stmt = select(OrderRecord).where(OrderRecord.order_id == order_id)
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _record_to_order(record)

    async def get_order_by_client_id(self, client_order_id: str) -> Order | None:
        """Retrieve an order by its client-assigned order ID.

        Args:
            client_order_id: The client-side dedupe key.

        Returns:
            Core :class:`Order` if found, otherwise ``None``.
        """
        stmt = select(OrderRecord).where(
            OrderRecord.client_order_id == client_order_id,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None
        return _record_to_order(record)

    async def get_orders_by_symbol(
        self,
        symbol: str,
        *,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Order]:
        """Retrieve orders for a given symbol, newest first.

        Args:
            symbol: Unified symbol string (e.g. ``"BTC/USDT"``).
            limit: Maximum number of rows to return.
            since: If provided, only return orders created after this time.

        Returns:
            List of core :class:`Order` objects.
        """
        stmt = (
            select(OrderRecord)
            .where(OrderRecord.symbol == symbol)
            .order_by(OrderRecord.created_at.desc())
            .limit(limit)
        )
        if since is not None:
            stmt = stmt.where(OrderRecord.created_at >= since)

        result = await self._session.execute(stmt)
        records: Sequence[OrderRecord] = result.scalars().all()
        return [_record_to_order(r) for r in records]

    async def get_open_orders(
        self,
        symbol: str | None = None,
    ) -> list[Order]:
        """Retrieve all non-terminal orders.

        Args:
            symbol: Optional filter by symbol. If ``None``, returns
                open orders across all symbols.

        Returns:
            List of core :class:`Order` objects in non-terminal states.
        """
        terminal_statuses = {
            OrderStatus.FILLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value,
        }
        stmt = (
            select(OrderRecord)
            .where(OrderRecord.status.notin_(terminal_statuses))
            .order_by(OrderRecord.created_at.desc())
        )
        if symbol is not None:
            stmt = stmt.where(OrderRecord.symbol == symbol)

        result = await self._session.execute(stmt)
        records: Sequence[OrderRecord] = result.scalars().all()
        return [_record_to_order(r) for r in records]


# ---------------------------------------------------------------------------
# FillRepo
# ---------------------------------------------------------------------------

class FillRepo:
    """Repository for :class:`FillRecord` persistence and retrieval."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_fill(self, fill: Fill) -> FillRecord:
        """Persist a fill record.

        Duplicate fill IDs are silently skipped (idempotent).

        Args:
            fill: Core domain fill to persist.

        Returns:
            The persisted (or existing) :class:`FillRecord`.
        """
        # Check for duplicate
        stmt = select(FillRecord).where(FillRecord.fill_id == fill.fill_id)
        result = await self._session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            logger.debug("Fill %s already exists, skipping.", fill.fill_id)
            return existing

        record = _fill_to_record(fill)
        self._session.add(record)
        await self._session.flush()
        logger.debug("Inserted fill %s for order %s", fill.fill_id, fill.order_id)
        return record

    async def get_fills_by_order(self, order_id: str) -> list[Fill]:
        """Retrieve all fills for a given order.

        Args:
            order_id: Exchange order ID.

        Returns:
            List of core :class:`Fill` objects, ordered by timestamp.
        """
        stmt = (
            select(FillRecord)
            .where(FillRecord.order_id == order_id)
            .order_by(FillRecord.fill_timestamp.asc())
        )
        result = await self._session.execute(stmt)
        records: Sequence[FillRecord] = result.scalars().all()
        return [_record_to_fill(r) for r in records]

    async def get_fills_by_symbol(
        self,
        symbol: str,
        *,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Fill]:
        """Retrieve fills for a symbol, newest first.

        Args:
            symbol: Unified symbol string.
            limit: Maximum rows.
            since: Optional lower-bound timestamp.

        Returns:
            List of core :class:`Fill` objects.
        """
        stmt = (
            select(FillRecord)
            .where(FillRecord.symbol == symbol)
            .order_by(FillRecord.fill_timestamp.desc())
            .limit(limit)
        )
        if since is not None:
            stmt = stmt.where(FillRecord.fill_timestamp >= since)

        result = await self._session.execute(stmt)
        records: Sequence[FillRecord] = result.scalars().all()
        return [_record_to_fill(r) for r in records]


# ---------------------------------------------------------------------------
# AuditRepo
# ---------------------------------------------------------------------------

class AuditRepo:
    """Repository for :class:`DecisionAudit` records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_decision(
        self,
        *,
        trace_id: str,
        strategy_id: str,
        symbol: str,
        exchange: str,
        feature_vector: dict[str, Any] | None = None,
        signal: dict[str, Any] | None = None,
        risk_check: dict[str, Any] | None = None,
        order_intent: dict[str, Any] | None = None,
        order_result: dict[str, Any] | None = None,
        fill_result: dict[str, Any] | None = None,
        signal_direction: str | None = None,
        signal_confidence: float | None = None,
        risk_passed: bool | None = None,
        final_status: str | None = None,
        decision_at: datetime | None = None,
    ) -> DecisionAudit:
        """Persist a full decision audit record.

        Args:
            trace_id: Distributed trace identifier linking the full chain.
            strategy_id: Strategy that produced the signal.
            symbol: Trading symbol.
            exchange: Exchange name.
            feature_vector: Serialised feature vector payload.
            signal: Serialised signal payload.
            risk_check: Serialised risk check result payload.
            order_intent: Serialised order intent payload.
            order_result: Serialised order acknowledgement payload.
            fill_result: Serialised fill payload.
            signal_direction: Quick-filter direction string.
            signal_confidence: Quick-filter confidence value.
            risk_passed: Whether the risk check passed.
            final_status: Final order status string.
            decision_at: When the decision was made.

        Returns:
            The persisted :class:`DecisionAudit`.
        """
        record = DecisionAudit(
            trace_id=trace_id,
            strategy_id=strategy_id,
            symbol=symbol,
            exchange=exchange,
            feature_vector=feature_vector,
            signal=signal,
            risk_check=risk_check,
            order_intent=order_intent,
            order_result=order_result,
            fill_result=fill_result,
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
            risk_passed=risk_passed,
            final_status=final_status,
            decision_at=decision_at or datetime.utcnow(),
        )
        self._session.add(record)
        await self._session.flush()
        logger.debug("Saved decision audit trace_id=%s", trace_id)
        return record

    async def get_decisions_by_trace_id(self, trace_id: str) -> list[DecisionAudit]:
        """Retrieve all decision audit records for a given trace.

        Args:
            trace_id: The trace identifier.

        Returns:
            List of :class:`DecisionAudit` records.
        """
        stmt = (
            select(DecisionAudit)
            .where(DecisionAudit.trace_id == trace_id)
            .order_by(DecisionAudit.decision_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_decisions_by_strategy(
        self,
        strategy_id: str,
        *,
        symbol: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[DecisionAudit]:
        """Retrieve decision audits for a strategy, newest first.

        Args:
            strategy_id: Strategy identifier.
            symbol: Optional symbol filter.
            limit: Maximum rows.
            since: Optional lower-bound timestamp.

        Returns:
            List of :class:`DecisionAudit` records.
        """
        stmt = (
            select(DecisionAudit)
            .where(DecisionAudit.strategy_id == strategy_id)
            .order_by(DecisionAudit.decision_at.desc())
            .limit(limit)
        )
        if symbol is not None:
            stmt = stmt.where(DecisionAudit.symbol == symbol)
        if since is not None:
            stmt = stmt.where(DecisionAudit.decision_at >= since)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# ExperimentRepo
# ---------------------------------------------------------------------------

class ExperimentRepo:
    """Repository for :class:`ExperimentLog` records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_experiment(
        self,
        *,
        experiment_id: str,
        name: str,
        strategy_id: str,
        config: dict[str, Any],
        symbols: list[str],
        timeframes: list[str],
        start_date: str,
        end_date: str,
        initial_capital: Decimal,
        description: str | None = None,
        git_sha: str | None = None,
        random_seed: int | None = None,
    ) -> ExperimentLog:
        """Create a new experiment log entry.

        The experiment is initially in ``"running"`` status. Call
        :meth:`update_experiment_results` when the backtest completes.

        Args:
            experiment_id: Unique experiment identifier.
            name: Human-readable experiment name.
            strategy_id: Strategy under test.
            config: Full configuration snapshot.
            symbols: List of symbols tested.
            timeframes: List of timeframes used.
            start_date: Backtest start date string.
            end_date: Backtest end date string.
            initial_capital: Starting capital.
            description: Optional description.
            git_sha: Git commit hash for reproducibility.
            random_seed: Random seed used.

        Returns:
            The persisted :class:`ExperimentLog`.
        """
        record = ExperimentLog(
            experiment_id=experiment_id,
            name=name,
            strategy_id=strategy_id,
            config=config,
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            description=description,
            git_sha=git_sha,
            random_seed=random_seed,
            status="running",
        )
        self._session.add(record)
        await self._session.flush()
        logger.info("Created experiment %s (%s)", experiment_id, name)
        return record

    async def update_experiment_results(
        self,
        experiment_id: str,
        *,
        final_equity: Decimal | None = None,
        total_return_pct: float | None = None,
        sharpe_ratio: float | None = None,
        sortino_ratio: float | None = None,
        max_drawdown_pct: float | None = None,
        win_rate: float | None = None,
        profit_factor: float | None = None,
        total_trades: int | None = None,
        avg_trade_duration_minutes: float | None = None,
        results_json: dict[str, Any] | None = None,
        equity_curve: list[dict[str, Any]] | None = None,
        status: str = "completed",
        error_message: str | None = None,
        completed_at: datetime | None = None,
    ) -> ExperimentLog | None:
        """Update an experiment with its results.

        Args:
            experiment_id: The experiment to update.
            final_equity: Final portfolio equity.
            total_return_pct: Total return percentage.
            sharpe_ratio: Annualised Sharpe ratio.
            sortino_ratio: Annualised Sortino ratio.
            max_drawdown_pct: Maximum drawdown percentage.
            win_rate: Win rate (0.0 - 1.0).
            profit_factor: Gross profit / gross loss.
            total_trades: Total number of trades.
            avg_trade_duration_minutes: Average trade duration.
            results_json: Full results payload.
            equity_curve: Equity curve data points.
            status: Final status (``"completed"`` or ``"failed"``).
            error_message: Error message if failed.
            completed_at: Completion timestamp.

        Returns:
            The updated :class:`ExperimentLog`, or ``None`` if not found.
        """
        stmt = select(ExperimentLog).where(
            ExperimentLog.experiment_id == experiment_id,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            logger.warning("Experiment %s not found for update.", experiment_id)
            return None

        record.final_equity = final_equity
        record.total_return_pct = total_return_pct
        record.sharpe_ratio = sharpe_ratio
        record.sortino_ratio = sortino_ratio
        record.max_drawdown_pct = max_drawdown_pct
        record.win_rate = win_rate
        record.profit_factor = profit_factor
        record.total_trades = total_trades
        record.avg_trade_duration_minutes = avg_trade_duration_minutes
        record.results_json = results_json
        record.equity_curve = equity_curve
        record.status = status
        record.error_message = error_message
        record.completed_at = completed_at or datetime.utcnow()

        await self._session.flush()
        logger.info("Updated experiment %s -> status=%s", experiment_id, status)
        return record

    async def get_experiment(self, experiment_id: str) -> ExperimentLog | None:
        """Retrieve an experiment by its ID.

        Args:
            experiment_id: The experiment identifier.

        Returns:
            The :class:`ExperimentLog` if found, otherwise ``None``.
        """
        stmt = select(ExperimentLog).where(
            ExperimentLog.experiment_id == experiment_id,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_experiments(
        self,
        *,
        strategy_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ExperimentLog]:
        """List experiments, newest first.

        Args:
            strategy_id: Optional filter by strategy.
            status: Optional filter by status.
            limit: Maximum rows.

        Returns:
            List of :class:`ExperimentLog` records.
        """
        stmt = (
            select(ExperimentLog)
            .order_by(ExperimentLog.started_at.desc())
            .limit(limit)
        )
        if strategy_id is not None:
            stmt = stmt.where(ExperimentLog.strategy_id == strategy_id)
        if status is not None:
            stmt = stmt.where(ExperimentLog.status == status)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())
