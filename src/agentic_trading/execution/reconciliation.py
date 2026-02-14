"""Reconciliation loop: periodically syncs local state with exchange state.

Compares open orders, positions, and balances against the exchange adapter
and emits ``ReconciliationResult`` events for any discrepancies detected.
Optionally auto-repairs local state to match the exchange source of truth.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from agentic_trading.core.enums import Exchange, OrderStatus
from agentic_trading.core.errors import ExchangeError, ReconciliationError
from agentic_trading.core.events import ReconciliationResult
from agentic_trading.core.interfaces import IEventBus, IExchangeAdapter
from agentic_trading.core.models import Balance, Order, Position

from .order_manager import OrderManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discrepancy type constants
# ---------------------------------------------------------------------------

MISSING_LOCAL_ORDER = "missing_local_order"
STALE_LOCAL_ORDER = "stale_local_order"
POSITION_MISMATCH = "position_mismatch"
BALANCE_MISMATCH = "balance_mismatch"


class ReconciliationLoop:
    """Periodic reconciliation between local state and exchange state.

    Runs an asyncio background loop every ``interval_seconds`` that:

    1. Fetches open orders, positions, and balances from the exchange adapter.
    2. Compares them against the local ``OrderManager`` state.
    3. For each discrepancy, records it and (if ``auto_repair`` is enabled)
       updates local state to match the exchange.
    4. Publishes a ``ReconciliationResult`` event summarising the run.

    Parameters
    ----------
    adapter:
        Exchange adapter implementing ``IExchangeAdapter``.
    event_bus:
        Event bus implementing ``IEventBus``.
    order_manager:
        Local ``OrderManager`` tracking order state.
    exchange:
        The exchange enum value this loop reconciles against.
    interval_seconds:
        How often to run reconciliation (default 30).
    auto_repair:
        Whether to automatically update local state when discrepancies
        are detected (default ``True``).
    local_positions:
        Optional dict of ``symbol -> Position`` representing the local
        position book.  If not provided, position reconciliation is
        skipped.
    local_balances:
        Optional dict of ``currency -> Balance`` representing the local
        balance book.  If not provided, balance reconciliation is skipped.
    """

    def __init__(
        self,
        adapter: IExchangeAdapter,
        event_bus: IEventBus,
        order_manager: OrderManager,
        exchange: Exchange = Exchange.BINANCE,
        interval_seconds: float = 30.0,
        auto_repair: bool = True,
        local_positions: dict[str, Position] | None = None,
        local_balances: dict[str, Balance] | None = None,
    ) -> None:
        self._adapter = adapter
        self._event_bus = event_bus
        self._order_manager = order_manager
        self._exchange = exchange
        self._interval = interval_seconds
        self._auto_repair = auto_repair
        self._local_positions: dict[str, Position] = (
            local_positions if local_positions is not None else {}
        )
        self._local_balances: dict[str, Balance] = (
            local_balances if local_balances is not None else {}
        )
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background reconciliation loop."""
        if self._running:
            logger.warning("ReconciliationLoop is already running")
            return
        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(), name="reconciliation-loop"
        )
        logger.info(
            "ReconciliationLoop started (interval=%ss, auto_repair=%s)",
            self._interval,
            self._auto_repair,
        )

    async def stop(self) -> None:
        """Stop the background reconciliation loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ReconciliationLoop stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Internal loop that runs reconciliation on a timer."""
        while self._running:
            try:
                await self.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Reconciliation cycle failed")
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                raise

    # ------------------------------------------------------------------
    # Single reconciliation pass
    # ------------------------------------------------------------------

    async def reconcile(self) -> ReconciliationResult:
        """Run a single reconciliation pass.

        Returns the ``ReconciliationResult`` event that was published.
        """
        discrepancies: list[dict[str, Any]] = []
        orders_synced = 0
        positions_synced = 0
        balances_synced = 0
        repairs_applied = 0

        # --- 1. Orders ------------------------------------------------
        try:
            exchange_orders: list[Order] = (
                await self._adapter.get_open_orders()
            )
            o_synced, o_repairs, o_discreps = self._reconcile_orders(
                exchange_orders
            )
            orders_synced += o_synced
            repairs_applied += o_repairs
            discrepancies.extend(o_discreps)
        except ExchangeError as exc:
            logger.error("Failed to fetch open orders for recon: %s", exc)
            discrepancies.append(
                {
                    "type": "fetch_error",
                    "entity": "orders",
                    "message": str(exc),
                }
            )

        # --- 2. Positions ---------------------------------------------
        try:
            exchange_positions: list[Position] = (
                await self._adapter.get_positions()
            )
            p_synced, p_repairs, p_discreps = self._reconcile_positions(
                exchange_positions
            )
            positions_synced += p_synced
            repairs_applied += p_repairs
            discrepancies.extend(p_discreps)
        except ExchangeError as exc:
            logger.error("Failed to fetch positions for recon: %s", exc)
            discrepancies.append(
                {
                    "type": "fetch_error",
                    "entity": "positions",
                    "message": str(exc),
                }
            )

        # --- 3. Balances ----------------------------------------------
        try:
            exchange_balances: list[Balance] = (
                await self._adapter.get_balances()
            )
            b_synced, b_repairs, b_discreps = self._reconcile_balances(
                exchange_balances
            )
            balances_synced += b_synced
            repairs_applied += b_repairs
            discrepancies.extend(b_discreps)
        except ExchangeError as exc:
            logger.error("Failed to fetch balances for recon: %s", exc)
            discrepancies.append(
                {
                    "type": "fetch_error",
                    "entity": "balances",
                    "message": str(exc),
                }
            )

        # --- Publish result -------------------------------------------
        result = ReconciliationResult(
            exchange=self._exchange,
            discrepancies=discrepancies,
            orders_synced=orders_synced,
            positions_synced=positions_synced,
            balances_synced=balances_synced,
            repairs_applied=repairs_applied,
        )

        if discrepancies:
            logger.warning(
                "Reconciliation found %d discrepancies "
                "(orders=%d positions=%d balances=%d repairs=%d)",
                len(discrepancies),
                orders_synced,
                positions_synced,
                balances_synced,
                repairs_applied,
            )
        else:
            logger.debug("Reconciliation clean (no discrepancies)")

        await self._event_bus.publish("execution", result)
        return result

    # ------------------------------------------------------------------
    # Order reconciliation
    # ------------------------------------------------------------------

    def _reconcile_orders(
        self, exchange_orders: list[Order]
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Compare exchange orders against local tracked orders.

        Returns ``(synced_count, repair_count, discrepancies)``.
        """
        synced = 0
        repairs = 0
        discreps: list[dict[str, Any]] = []

        # Build index of exchange orders by client_order_id
        exchange_by_client_id: dict[str, Order] = {
            o.client_order_id: o for o in exchange_orders
        }

        # 1. Orders on exchange but not tracked locally
        for client_id, exch_order in exchange_by_client_id.items():
            local = self._order_manager.get_order(client_id)
            if local is None:
                discreps.append(
                    {
                        "type": MISSING_LOCAL_ORDER,
                        "client_order_id": client_id,
                        "order_id": exch_order.order_id,
                        "symbol": exch_order.symbol,
                        "exchange_status": exch_order.status.value,
                    }
                )
                if self._auto_repair:
                    from agentic_trading.core.events import OrderUpdate

                    try:
                        self._order_manager.update_order(
                            OrderUpdate(
                                order_id=exch_order.order_id,
                                client_order_id=client_id,
                                symbol=exch_order.symbol,
                                exchange=exch_order.exchange,
                                status=exch_order.status,
                                filled_qty=exch_order.filled_qty,
                                remaining_qty=exch_order.remaining_qty,
                            )
                        )
                        repairs += 1
                    except Exception as exc:
                        logger.warning(
                            "Auto-repair failed for missing local order "
                            "%s: %s",
                            client_id,
                            exc,
                        )
            synced += 1

        # 2. Orders tracked locally but not on exchange (stale)
        active_local = self._order_manager.get_active_orders()
        for tracked in active_local:
            if tracked.dedupe_key not in exchange_by_client_id:
                # The exchange no longer reports it -- it may have been
                # filled or cancelled without us noticing.
                discreps.append(
                    {
                        "type": STALE_LOCAL_ORDER,
                        "client_order_id": tracked.dedupe_key,
                        "order_id": tracked.order_id,
                        "symbol": tracked.symbol,
                        "local_status": tracked.status.value,
                    }
                )
                if self._auto_repair:
                    from agentic_trading.core.events import OrderUpdate

                    try:
                        self._order_manager.update_order(
                            OrderUpdate(
                                order_id=tracked.order_id,
                                client_order_id=tracked.dedupe_key,
                                symbol=tracked.symbol,
                                exchange=self._exchange,
                                status=OrderStatus.CANCELLED,
                            )
                        )
                        repairs += 1
                    except Exception as exc:
                        logger.warning(
                            "Auto-repair failed for stale local order "
                            "%s: %s",
                            tracked.dedupe_key,
                            exc,
                        )

        return synced, repairs, discreps

    # ------------------------------------------------------------------
    # Position reconciliation
    # ------------------------------------------------------------------

    def _reconcile_positions(
        self, exchange_positions: list[Position]
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Compare exchange positions against local position book.

        Returns ``(synced_count, repair_count, discrepancies)``.
        """
        synced = 0
        repairs = 0
        discreps: list[dict[str, Any]] = []

        exchange_by_symbol: dict[str, Position] = {
            p.symbol: p for p in exchange_positions
        }

        # Positions on exchange
        for symbol, exch_pos in exchange_by_symbol.items():
            local_pos = self._local_positions.get(symbol)
            synced += 1
            if local_pos is None:
                if exch_pos.qty != Decimal("0"):
                    discreps.append(
                        {
                            "type": POSITION_MISMATCH,
                            "symbol": symbol,
                            "local_qty": "0",
                            "exchange_qty": str(exch_pos.qty),
                            "detail": "Position exists on exchange but "
                            "not locally",
                        }
                    )
                    if self._auto_repair:
                        self._local_positions[symbol] = exch_pos
                        repairs += 1
            else:
                if local_pos.qty != exch_pos.qty:
                    discreps.append(
                        {
                            "type": POSITION_MISMATCH,
                            "symbol": symbol,
                            "local_qty": str(local_pos.qty),
                            "exchange_qty": str(exch_pos.qty),
                            "detail": "Quantity mismatch",
                        }
                    )
                    if self._auto_repair:
                        self._local_positions[symbol] = exch_pos
                        repairs += 1

        # Positions tracked locally but not on exchange
        for symbol, local_pos in list(self._local_positions.items()):
            if symbol not in exchange_by_symbol and local_pos.qty != Decimal(
                "0"
            ):
                discreps.append(
                    {
                        "type": POSITION_MISMATCH,
                        "symbol": symbol,
                        "local_qty": str(local_pos.qty),
                        "exchange_qty": "0",
                        "detail": "Position exists locally but not "
                        "on exchange",
                    }
                )
                if self._auto_repair:
                    local_pos_copy = local_pos.model_copy(
                        update={"qty": Decimal("0")}
                    )
                    self._local_positions[symbol] = local_pos_copy
                    repairs += 1

        return synced, repairs, discreps

    # ------------------------------------------------------------------
    # Balance reconciliation
    # ------------------------------------------------------------------

    def _reconcile_balances(
        self, exchange_balances: list[Balance]
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Compare exchange balances against local balance book.

        Returns ``(synced_count, repair_count, discrepancies)``.
        """
        synced = 0
        repairs = 0
        discreps: list[dict[str, Any]] = []

        exchange_by_currency: dict[str, Balance] = {
            b.currency: b for b in exchange_balances
        }

        for currency, exch_bal in exchange_by_currency.items():
            local_bal = self._local_balances.get(currency)
            synced += 1
            if local_bal is None:
                if exch_bal.total != Decimal("0"):
                    discreps.append(
                        {
                            "type": BALANCE_MISMATCH,
                            "currency": currency,
                            "local_total": "0",
                            "exchange_total": str(exch_bal.total),
                            "detail": "Balance on exchange but not locally",
                        }
                    )
                    if self._auto_repair:
                        self._local_balances[currency] = exch_bal
                        repairs += 1
            else:
                # Allow a small tolerance for rounding
                diff = abs(local_bal.total - exch_bal.total)
                if diff > Decimal("0.01"):
                    discreps.append(
                        {
                            "type": BALANCE_MISMATCH,
                            "currency": currency,
                            "local_total": str(local_bal.total),
                            "exchange_total": str(exch_bal.total),
                            "detail": f"Difference of {diff}",
                        }
                    )
                    if self._auto_repair:
                        self._local_balances[currency] = exch_bal
                        repairs += 1

        return synced, repairs, discreps

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def local_positions(self) -> dict[str, Position]:
        """Return a reference to the local position book."""
        return self._local_positions

    @property
    def local_balances(self) -> dict[str, Balance]:
        """Return a reference to the local balance book."""
        return self._local_balances
