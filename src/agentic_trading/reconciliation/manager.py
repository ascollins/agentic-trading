"""Unified ReconciliationManager facade for the reconciliation layer.

Composes the :class:`TradeJournal`, analytics components, and the
:class:`ReconciliationLoop` into a single entry point that can be wired
into the platform bootstrap.

Usage::

    from agentic_trading.reconciliation.manager import ReconciliationManager

    mgr = ReconciliationManager.from_config(
        adapter=adapter,
        event_bus=event_bus,
        order_manager=order_manager,
    )
    await mgr.start()
    # ... platform runs ...
    await mgr.stop()

Fill handling::

    result = mgr.handle_fill(fill_event, signal_cache, exit_map)
    # result.is_exit, result.strategy_id, result.entry_trace_id, ...

Position reconciliation::

    mgr.reconcile_positions(exchange_positions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from agentic_trading.reconciliation.journal.journal import TradeJournal
from agentic_trading.reconciliation.journal.quality_scorecard import QualityScorecard
from agentic_trading.reconciliation.journal.rolling_tracker import RollingTracker
from agentic_trading.reconciliation.loop import ReconciliationLoop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fill handling result
# ---------------------------------------------------------------------------

@dataclass
class FillResult:
    """Structured result returned by :meth:`ReconciliationManager.handle_fill`.

    Carries enough context for callers (e.g. main.py) to perform follow-up
    actions such as narration, TP/SL placement, and metrics emission without
    re-deriving the entry-vs-exit classification.
    """

    is_exit: bool
    strategy_id: str
    entry_trace_id: str | None = None
    direction: str = ""
    """Trade direction: ``"long"`` or ``"short"``."""


class ReconciliationManager:
    """Unified facade for the reconciliation layer.

    Owns the trade journal, rolling analytics, quality scorecard,
    and the exchange reconciliation loop.  Provides a clean lifecycle
    (start / stop) and accessor properties for each sub-component.

    Parameters
    ----------
    journal:
        TradeJournal instance for trade lifecycle tracking.
    rolling_tracker:
        RollingTracker for sliding-window performance metrics.
    quality_scorecard:
        QualityScorecard for strategy grading.
    recon_loop:
        ReconciliationLoop for exchange-state reconciliation.
        May be ``None`` in backtest mode where no exchange exists.
    """

    def __init__(
        self,
        journal: TradeJournal,
        rolling_tracker: RollingTracker | None = None,
        quality_scorecard: QualityScorecard | None = None,
        recon_loop: ReconciliationLoop | None = None,
    ) -> None:
        self._journal = journal
        self._rolling_tracker = rolling_tracker
        self._quality_scorecard = quality_scorecard
        self._recon_loop = recon_loop

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        adapter: Any | None = None,
        event_bus: Any | None = None,
        order_manager: Any | None = None,
        health_tracker: Any | None = None,
        drift_detector: Any | None = None,
        on_trade_closed: Any = None,
        max_closed_trades: int = 10_000,
        rolling_window: int = 100,
        recon_interval: float = 30.0,
        auto_repair: bool = True,
        exchange: Any | None = None,
        local_positions: dict | None = None,
        local_balances: dict | None = None,
    ) -> ReconciliationManager:
        """Build a fully wired ReconciliationManager from configuration.

        Parameters
        ----------
        adapter:
            Exchange adapter (``IExchangeAdapter``).  Required for the
            reconciliation loop; may be ``None`` in backtest mode.
        event_bus:
            Event bus (``IEventBus``).  Required for the reconciliation
            loop; may be ``None`` in backtest mode.
        order_manager:
            Local order manager.  Required for the reconciliation loop.
        health_tracker:
            Optional governance health tracker fed on trade close.
        drift_detector:
            Optional governance drift detector fed on trade close.
        on_trade_closed:
            Optional callback invoked when a trade closes.
        max_closed_trades:
            Max closed trades to keep in-memory (default 10 000).
        rolling_window:
            Number of trades in the rolling performance window
            (default 100).
        recon_interval:
            Reconciliation loop interval in seconds (default 30).
        auto_repair:
            Whether the recon loop auto-repairs discrepancies
            (default ``True``).
        exchange:
            Exchange enum value for the reconciliation loop.
        local_positions:
            Optional local position book for recon.
        local_balances:
            Optional local balance book for recon.

        Returns
        -------
        ReconciliationManager
        """
        journal = TradeJournal(
            max_closed_trades=max_closed_trades,
            health_tracker=health_tracker,
            drift_detector=drift_detector,
            on_trade_closed=on_trade_closed,
        )

        rolling_tracker = RollingTracker(window_size=rolling_window)
        quality_scorecard = QualityScorecard()

        # Only create recon loop when we have the required components
        recon_loop: ReconciliationLoop | None = None
        if adapter is not None and event_bus is not None and order_manager is not None:
            from agentic_trading.core.enums import Exchange as ExchangeEnum

            recon_loop = ReconciliationLoop(
                adapter=adapter,
                event_bus=event_bus,
                order_manager=order_manager,
                exchange=exchange or ExchangeEnum.BINANCE,
                interval_seconds=recon_interval,
                auto_repair=auto_repair,
                local_positions=local_positions,
                local_balances=local_balances,
            )

        return cls(
            journal=journal,
            rolling_tracker=rolling_tracker,
            quality_scorecard=quality_scorecard,
            recon_loop=recon_loop,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background tasks (reconciliation loop)."""
        if self._recon_loop is not None:
            await self._recon_loop.start()
        logger.info("ReconciliationManager started")

    async def stop(self) -> None:
        """Stop background tasks."""
        if self._recon_loop is not None:
            await self._recon_loop.stop()
        logger.info("ReconciliationManager stopped")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def journal(self) -> TradeJournal:
        """The trade journal."""
        return self._journal

    @property
    def rolling_tracker(self) -> RollingTracker | None:
        """Rolling performance tracker."""
        return self._rolling_tracker

    @property
    def quality_scorecard(self) -> QualityScorecard | None:
        """Strategy quality scorecard."""
        return self._quality_scorecard

    @property
    def recon_loop(self) -> ReconciliationLoop | None:
        """Exchange reconciliation loop (``None`` in backtest mode)."""
        return self._recon_loop

    # ------------------------------------------------------------------
    # Delegated operations
    # ------------------------------------------------------------------

    def get_open_trades(self) -> list:
        """Delegate to journal.get_open_trades()."""
        return self._journal.get_open_trades()

    def get_closed_trades(self, **kwargs) -> list:
        """Delegate to journal.get_closed_trades()."""
        return self._journal.get_closed_trades(**kwargs)

    def get_trade(self, trace_id: str):
        """Delegate to journal.get_trade()."""
        return self._journal.get_trade(trace_id)

    async def reconcile(self):
        """Run a single reconciliation pass (if loop is available)."""
        if self._recon_loop is not None:
            return await self._recon_loop.reconcile()
        return None

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def handle_fill(
        self,
        fill_event: Any,
        signal_cache: dict[str, Any],
        exit_map: dict[str, str],
        *,
        fallback_strategy_ids: list[str] | None = None,
    ) -> FillResult:
        """Classify a fill as entry or exit and record it in the journal.

        This method encapsulates the core journal-recording logic that was
        previously inline in ``main._run_live_or_paper.on_execution_event``.
        It does **not** handle narration, TP/SL placement, or Prometheus
        metrics — those remain in the caller.

        Parameters
        ----------
        fill_event:
            A ``FillEvent`` from the execution topic.
        signal_cache:
            Maps ``trace_id → Signal`` for strategy-id resolution.
        exit_map:
            Maps ``exit_trace_id → entry_trace_id``.  Consumed (popped)
            when an exit fill is detected.
        fallback_strategy_ids:
            Strategy IDs to probe when the signal cache has no match and
            a position-level lookup is needed.

        Returns
        -------
        FillResult
            Structured result carrying classification and journal state.
        """
        trace_id = fill_event.trace_id
        cached_sig = signal_cache.get(trace_id)
        strategy_id = cached_sig.strategy_id if cached_sig else "unknown"
        is_exit = trace_id in exit_map
        entry_trace_id: str | None = None

        if is_exit:
            entry_trace_id = exit_map.pop(trace_id)
        else:
            # Detect exit fills by checking for an open trade with opposite
            # direction on the same symbol.
            open_trade = self._journal.get_trade_by_position(
                strategy_id, fill_event.symbol,
            )
            if open_trade is None and strategy_id == "unknown":
                for _sid in (fallback_strategy_ids or []):
                    open_trade = self._journal.get_trade_by_position(
                        _sid, fill_event.symbol,
                    )
                    if open_trade is not None:
                        strategy_id = _sid
                        break
            if open_trade is not None:
                fill_side = (
                    fill_event.side.value
                    if hasattr(fill_event.side, "value")
                    else str(fill_event.side)
                )
                open_direction = open_trade.direction
                is_opposing = (
                    (open_direction == "long" and fill_side == "sell")
                    or (open_direction == "short" and fill_side == "buy")
                )
                if is_opposing:
                    is_exit = True
                    entry_trace_id = open_trade.trace_id

        side_str = (
            fill_event.side.value
            if hasattr(fill_event.side, "value")
            else str(fill_event.side)
        )

        direction = ""

        if is_exit and entry_trace_id:
            # ---- EXIT FILL ----
            self._journal.record_exit_fill(
                trace_id=entry_trace_id,
                fill_id=fill_event.fill_id,
                order_id=fill_event.order_id,
                side=side_str,
                price=fill_event.price,
                qty=fill_event.qty,
                fee=fill_event.fee,
                fee_currency=fill_event.fee_currency,
                is_maker=fill_event.is_maker,
                timestamp=fill_event.timestamp,
            )

            logger.info(
                "Exit fill recorded: %s %s %s qty=%s price=%s (entry_trace=%s)",
                strategy_id, side_str, fill_event.symbol,
                fill_event.qty, fill_event.price, entry_trace_id[:8],
            )
        else:
            # ---- ENTRY FILL ----
            if cached_sig:
                direction = cached_sig.direction.value
            elif side_str == "sell":
                direction = "short"
            else:
                direction = "long"

            exchange_str = (
                fill_event.exchange.value
                if hasattr(fill_event.exchange, "value")
                else str(fill_event.exchange)
            )

            self._journal.open_trade(
                trace_id=trace_id,
                strategy_id=strategy_id,
                symbol=fill_event.symbol,
                direction=direction,
                exchange=exchange_str,
                signal_confidence=cached_sig.confidence if cached_sig else 0.0,
                signal_rationale=cached_sig.rationale if cached_sig else "",
            )

            self._journal.record_entry_fill(
                trace_id=trace_id,
                fill_id=fill_event.fill_id,
                order_id=fill_event.order_id,
                side=side_str,
                price=fill_event.price,
                qty=fill_event.qty,
                fee=fill_event.fee,
                fee_currency=fill_event.fee_currency,
                is_maker=fill_event.is_maker,
                timestamp=fill_event.timestamp,
            )

            logger.info(
                "Entry fill recorded: %s %s %s qty=%s price=%s (trace=%s)",
                strategy_id, side_str, fill_event.symbol,
                fill_event.qty, fill_event.price, trace_id[:8],
            )

        return FillResult(
            is_exit=is_exit and entry_trace_id is not None,
            strategy_id=strategy_id,
            entry_trace_id=entry_trace_id,
            direction=direction,
        )

    # ------------------------------------------------------------------
    # Position reconciliation
    # ------------------------------------------------------------------

    def reconcile_positions(
        self,
        exchange_positions: list,
    ) -> list[str]:
        """Force-close journal trades whose positions no longer exist.

        Compares journal open trades against actual exchange positions.
        If an open trade's symbol has no corresponding exchange position,
        the trade is force-closed at the last known mark price (or entry
        price as fallback).

        This handles cases where a position was closed externally (manual
        close, exchange stop-loss, liquidation) and the fill event never
        reached the journal.

        Parameters
        ----------
        exchange_positions:
            List of position objects from the exchange adapter.  Each
            must have a ``.symbol`` attribute.

        Returns
        -------
        list[str]
            Trace IDs of force-closed trades.
        """
        open_trades = self._journal.get_open_trades()
        if not open_trades:
            return []

        # Build set of symbols with active exchange positions.
        # Exchange positions use CCXT format (e.g. "BTC/USDT:USDT"),
        # journal trades use spot-style format (e.g. "BTC/USDT").
        # Normalise by stripping the settle suffix for comparison.
        active_symbols: set[str] = set()
        for p in exchange_positions:
            sym = p.symbol
            base_sym = sym.split(":")[0] if ":" in sym else sym
            active_symbols.add(base_sym)
            active_symbols.add(sym)

        closed_trace_ids: list[str] = []
        for trade in open_trades:
            if trade.symbol in active_symbols:
                continue

            # Determine close price
            close_price = Decimal("0")
            if trade.mark_samples:
                close_price = trade.mark_samples[-1].mark_price
            if close_price == Decimal("0"):
                close_price = trade.avg_entry_price
            if close_price == Decimal("0") and trade.entry_fills:
                close_price = trade.entry_fills[0].price

            logger.warning(
                "Journal reconciliation: force-closing orphaned trade "
                "%s %s %s (trace=%s) — position absent on exchange",
                trade.strategy_id,
                trade.direction.upper(),
                trade.symbol,
                trade.trace_id[:8],
            )
            self._journal.force_close(trade.trace_id, close_price)
            closed_trace_ids.append(trade.trace_id)

        return closed_trace_ids
