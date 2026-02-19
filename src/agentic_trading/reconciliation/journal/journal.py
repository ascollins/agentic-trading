"""Trade Journal — event subscriber that builds trade records.

Subscribes to execution, state, and strategy topics on the event bus.
Correlates events via ``trace_id`` and ``strategy_id + symbol`` to
construct :class:`TradeRecord` instances that capture the full
lifecycle of each trade.

On trade close, feeds results to the health tracker and drift detector,
and emits journal-specific events for downstream analytics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .record import FillLeg, MarkSample, TradePhase, TradeRecord

logger = logging.getLogger(__name__)


class TradeJournal:
    """Central trade journal that aggregates events into trade records.

    Usage::

        journal = TradeJournal()

        # Manual recording (for direct integration)
        journal.open_trade(trace_id, strategy_id, symbol, ...)
        journal.record_entry_fill(trace_id, fill_data)
        journal.record_mark(trace_id, mark_data)
        journal.record_exit_fill(trace_id, fill_data)
        # ^ auto-closes when fully exited

        # Query
        open_trades = journal.get_open_trades()
        closed = journal.get_closed_trades(strategy_id="trend_following")
        trade = journal.get_trade(trace_id)

    Parameters
    ----------
    max_closed_trades : int
        Maximum number of closed trades to keep in memory.
        Older trades are evicted FIFO.  Default 10 000.
    health_tracker : object | None
        Optional governance HealthTracker.  Called on trade close with
        ``record_outcome(strategy_id, won, r_multiple)``.
    drift_detector : object | None
        Optional governance DriftDetector.  Live metrics are pushed
        after each trade close.
    on_trade_closed : callable | None
        Optional callback ``fn(trade: TradeRecord)`` invoked when a
        trade transitions to CLOSED.
    """

    def __init__(
        self,
        *,
        max_closed_trades: int = 10_000,
        health_tracker: Any = None,
        drift_detector: Any = None,
        on_trade_closed: Any = None,
    ) -> None:
        self._max_closed = max_closed_trades
        self._health_tracker = health_tracker
        self._drift_detector = drift_detector
        self._on_trade_closed = on_trade_closed

        # Active trades indexed by trace_id
        self._open_trades: dict[str, TradeRecord] = {}

        # Also index by (strategy_id, symbol) for position-level lookups
        self._position_index: dict[tuple[str, str], str] = {}  # → trace_id

        # Closed trades (bounded deque-like list)
        self._closed_trades: list[TradeRecord] = []

        # Per-strategy counters for analytics
        self._strategy_stats: dict[str, _StrategyStats] = defaultdict(
            _StrategyStats
        )

    # ------------------------------------------------------------------ #
    # Trade lifecycle                                                      #
    # ------------------------------------------------------------------ #

    def open_trade(
        self,
        trace_id: str,
        strategy_id: str,
        symbol: str,
        direction: str,
        exchange: str = "",
        signal_confidence: float = 0.0,
        signal_rationale: str = "",
        signal_features: dict | None = None,
        signal_timestamp: datetime | None = None,
        initial_risk_price: Decimal | None = None,
        planned_target_price: Decimal | None = None,
        maturity_level: str = "",
        health_score: float = 1.0,
        governance_multiplier: float = 1.0,
    ) -> TradeRecord:
        """Create a new pending trade record.

        If a trade already exists for this ``trace_id``, returns the
        existing record (idempotent).
        """
        if trace_id in self._open_trades:
            return self._open_trades[trace_id]

        trade = TradeRecord(
            trace_id=trace_id,
            strategy_id=strategy_id,
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            signal_confidence=signal_confidence,
            signal_rationale=signal_rationale,
            signal_features=signal_features or {},
            signal_timestamp=signal_timestamp,
            initial_risk_price=initial_risk_price,
            planned_target_price=planned_target_price,
            maturity_level=maturity_level,
            health_score_at_entry=health_score,
            governance_sizing_multiplier=governance_multiplier,
        )

        self._open_trades[trace_id] = trade
        self._position_index[(strategy_id, symbol)] = trace_id

        logger.info(
            "Trade opened: %s %s %s (conf=%.2f) trace=%s",
            strategy_id,
            direction.upper(),
            symbol,
            signal_confidence,
            trace_id[:8],
        )
        return trade

    def record_entry_fill(
        self,
        trace_id: str,
        fill_id: str,
        order_id: str,
        side: str,
        price: Decimal,
        qty: Decimal,
        fee: Decimal = Decimal("0"),
        fee_currency: str = "USDT",
        is_maker: bool = False,
        timestamp: datetime | None = None,
    ) -> TradeRecord | None:
        """Record an entry fill on an open trade."""
        trade = self._open_trades.get(trace_id)
        if trade is None:
            logger.debug("Entry fill for unknown trace_id=%s", trace_id[:8])
            return None

        fill = FillLeg(
            fill_id=fill_id,
            order_id=order_id,
            side=side,
            price=price,
            qty=qty,
            fee=fee,
            fee_currency=fee_currency,
            is_maker=is_maker,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        trade.add_entry_fill(fill)

        # Compute initial risk if stop price was provided
        if trade.initial_risk_price is not None and trade.initial_risk_amount is None:
            trade.compute_initial_risk()

        logger.debug(
            "Entry fill: %s %s qty=%s @ %s trace=%s",
            trade.symbol,
            side,
            qty,
            price,
            trace_id[:8],
        )
        return trade

    def record_exit_fill(
        self,
        trace_id: str,
        fill_id: str,
        order_id: str,
        side: str,
        price: Decimal,
        qty: Decimal,
        fee: Decimal = Decimal("0"),
        fee_currency: str = "USDT",
        is_maker: bool = False,
        timestamp: datetime | None = None,
    ) -> TradeRecord | None:
        """Record an exit fill.  Closes the trade if fully exited."""
        trade = self._open_trades.get(trace_id)
        if trade is None:
            logger.debug("Exit fill for unknown trace_id=%s", trace_id[:8])
            return None

        fill = FillLeg(
            fill_id=fill_id,
            order_id=order_id,
            side=side,
            price=price,
            qty=qty,
            fee=fee,
            fee_currency=fee_currency,
            is_maker=is_maker,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        trade.add_exit_fill(fill)

        logger.debug(
            "Exit fill: %s %s qty=%s @ %s remaining=%s trace=%s",
            trade.symbol,
            side,
            qty,
            price,
            trade.remaining_qty,
            trace_id[:8],
        )

        if trade.phase == TradePhase.CLOSED:
            self._close_trade(trade)

        return trade

    def record_mark(
        self,
        trace_id: str,
        mark_price: Decimal,
        unrealized_pnl: Decimal,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a mark-to-market sample for MAE/MFE tracking."""
        trade = self._open_trades.get(trace_id)
        if trade is None:
            return

        sample = MarkSample(
            timestamp=timestamp or datetime.now(timezone.utc),
            mark_price=mark_price,
            unrealized_pnl=unrealized_pnl,
        )
        trade.add_mark_sample(sample)

    def cancel_trade(self, trace_id: str) -> None:
        """Mark a pending trade as cancelled (order rejected/cancelled)."""
        trade = self._open_trades.get(trace_id)
        if trade is None:
            return
        trade.cancel()
        del self._open_trades[trace_id]
        key = (trade.strategy_id, trade.symbol)
        self._position_index.pop(key, None)
        logger.info("Trade cancelled: %s trace=%s", trade.symbol, trace_id[:8])

    def force_close(
        self,
        trace_id: str,
        exit_price: Decimal,
        timestamp: datetime | None = None,
    ) -> TradeRecord | None:
        """Force-close a trade at a given price (e.g. liquidation, kill switch).

        Creates a synthetic exit fill for the remaining quantity.
        """
        trade = self._open_trades.get(trace_id)
        if trade is None:
            return None

        remaining = trade.remaining_qty
        if remaining <= Decimal("0"):
            return trade

        exit_side = "sell" if trade.direction == "long" else "buy"
        return self.record_exit_fill(
            trace_id=trace_id,
            fill_id=f"force_close_{trade.trade_id[:8]}",
            order_id="force_close",
            side=exit_side,
            price=exit_price,
            qty=remaining,
            fee=Decimal("0"),
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------ #
    # Internal: trade close processing                                     #
    # ------------------------------------------------------------------ #

    def _close_trade(self, trade: TradeRecord) -> None:
        """Process a newly closed trade: update stats, feed governance."""
        # Remove from open tracking
        self._open_trades.pop(trade.trace_id, None)
        key = (trade.strategy_id, trade.symbol)
        self._position_index.pop(key, None)

        # Add to closed list (bounded)
        self._closed_trades.append(trade)
        if len(self._closed_trades) > self._max_closed:
            self._closed_trades.pop(0)

        # Update per-strategy stats
        stats = self._strategy_stats[trade.strategy_id]
        stats.record(trade)

        # Feed health tracker
        if self._health_tracker is not None:
            try:
                won = trade.outcome.value == "win"
                r_mult = trade.r_multiple
                self._health_tracker.record_outcome(
                    trade.strategy_id, won, r_mult
                )
            except Exception:
                logger.warning(
                    "Failed to feed health tracker", exc_info=True
                )

        # Feed drift detector with updated live metrics
        if self._drift_detector is not None:
            try:
                live = stats.to_drift_metrics()
                for metric_name, value in live.items():
                    self._drift_detector.update_live_metric(
                        trade.strategy_id, metric_name, value
                    )
            except Exception:
                logger.warning(
                    "Failed to feed drift detector", exc_info=True
                )

        # Callback
        if self._on_trade_closed is not None:
            try:
                self._on_trade_closed(trade)
            except Exception:
                logger.warning("on_trade_closed callback failed", exc_info=True)

        logger.info(
            "Trade closed: %s %s %s pnl=%s R=%.2f eff=%.0f%% hold=%ds",
            trade.strategy_id,
            trade.direction.upper(),
            trade.symbol,
            trade.net_pnl,
            trade.r_multiple,
            trade.management_efficiency * 100,
            trade.hold_duration_seconds,
        )

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get_trade(self, trace_id: str) -> TradeRecord | None:
        """Look up a trade by trace_id (open or closed)."""
        trade = self._open_trades.get(trace_id)
        if trade is not None:
            return trade
        for t in reversed(self._closed_trades):
            if t.trace_id == trace_id:
                return t
        return None

    def get_trade_by_position(
        self, strategy_id: str, symbol: str
    ) -> TradeRecord | None:
        """Look up the open trade for a given strategy + symbol."""
        trace_id = self._position_index.get((strategy_id, symbol))
        if trace_id is None:
            return None
        return self._open_trades.get(trace_id)

    def get_open_trades(
        self, strategy_id: str | None = None
    ) -> list[TradeRecord]:
        """Return all open trades, optionally filtered by strategy."""
        trades = list(self._open_trades.values())
        if strategy_id is not None:
            trades = [t for t in trades if t.strategy_id == strategy_id]
        return trades

    def get_closed_trades(
        self,
        strategy_id: str | None = None,
        symbol: str | None = None,
        last_n: int | None = None,
    ) -> list[TradeRecord]:
        """Return closed trades with optional filters."""
        trades = self._closed_trades
        if strategy_id is not None:
            trades = [t for t in trades if t.strategy_id == strategy_id]
        if symbol is not None:
            trades = [t for t in trades if t.symbol == symbol]
        if last_n is not None:
            trades = trades[-last_n:]
        return trades

    def get_strategy_stats(self, strategy_id: str) -> dict[str, Any]:
        """Return aggregated statistics for a strategy."""
        stats = self._strategy_stats.get(strategy_id)
        if stats is None:
            return {}
        return stats.to_dict()

    def get_all_strategy_stats(self) -> dict[str, dict[str, Any]]:
        """Return stats for all strategies that have recorded trades."""
        return {
            sid: stats.to_dict()
            for sid, stats in self._strategy_stats.items()
        }

    @property
    def open_trade_count(self) -> int:
        return len(self._open_trades)

    @property
    def closed_trade_count(self) -> int:
        return len(self._closed_trades)

    @property
    def total_trade_count(self) -> int:
        return self.open_trade_count + self.closed_trade_count


class _StrategyStats:
    """Running statistics for a single strategy."""

    def __init__(self) -> None:
        self.total_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.breakevens: int = 0
        self.total_pnl: float = 0.0
        self.gross_wins: float = 0.0
        self.gross_losses: float = 0.0
        self.total_fees: float = 0.0
        self.r_multiples: list[float] = []
        self.pnl_history: list[float] = []
        self.current_streak: int = 0  # positive = wins, negative = losses
        self.max_win_streak: int = 0
        self.max_loss_streak: int = 0
        self.best_trade_pnl: float = 0.0
        self.worst_trade_pnl: float = 0.0
        self.total_hold_seconds: float = 0.0
        self.management_efficiencies: list[float] = []

    def record(self, trade: TradeRecord) -> None:
        pnl = float(trade.net_pnl)
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_fees += float(trade.total_fees)
        self.pnl_history.append(pnl)
        self.r_multiples.append(trade.r_multiple)
        self.total_hold_seconds += trade.hold_duration_seconds

        if trade.management_efficiency > 0:
            self.management_efficiencies.append(trade.management_efficiency)

        if pnl > self.best_trade_pnl:
            self.best_trade_pnl = pnl
        if pnl < self.worst_trade_pnl:
            self.worst_trade_pnl = pnl

        if trade.outcome.value == "win":
            self.wins += 1
            self.gross_wins += pnl
            if self.current_streak > 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.max_win_streak = max(self.max_win_streak, self.current_streak)
        elif trade.outcome.value == "loss":
            self.losses += 1
            self.gross_losses += abs(pnl)
            if self.current_streak < 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.max_loss_streak = max(
                self.max_loss_streak, abs(self.current_streak)
            )
        else:
            self.breakevens += 1
            self.current_streak = 0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def profit_factor(self) -> float:
        if self.gross_losses == 0:
            return float("inf") if self.gross_wins > 0 else 0.0
        return self.gross_wins / self.gross_losses

    @property
    def expectancy(self) -> float:
        """Average P&L per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def avg_r(self) -> float:
        """Average R-multiple per trade."""
        if not self.r_multiples:
            return 0.0
        return sum(self.r_multiples) / len(self.r_multiples)

    @property
    def avg_winner(self) -> float:
        if self.wins == 0:
            return 0.0
        return self.gross_wins / self.wins

    @property
    def avg_loser(self) -> float:
        if self.losses == 0:
            return 0.0
        return self.gross_losses / self.losses

    @property
    def sharpe(self) -> float:
        """Simplified Sharpe-like ratio from trade P&L series."""
        if len(self.pnl_history) < 2:
            return 0.0
        import statistics

        mean = statistics.mean(self.pnl_history)
        std = statistics.stdev(self.pnl_history)
        if std == 0:
            return 0.0
        return mean / std

    @property
    def avg_hold_seconds(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_hold_seconds / self.total_trades

    @property
    def avg_management_efficiency(self) -> float:
        if not self.management_efficiencies:
            return 0.0
        return sum(self.management_efficiencies) / len(
            self.management_efficiencies
        )

    def to_drift_metrics(self) -> dict[str, float]:
        """Return metrics suitable for the drift detector."""
        return {
            "win_rate": self.win_rate,
            "avg_rr": self.avg_r,
            "sharpe": self.sharpe,
            "profit_factor": self.profit_factor,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "breakevens": self.breakevens,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 2),
            "avg_r": round(self.avg_r, 4),
            "avg_winner": round(self.avg_winner, 2),
            "avg_loser": round(self.avg_loser, 2),
            "sharpe": round(self.sharpe, 4),
            "total_pnl": round(self.total_pnl, 2),
            "total_fees": round(self.total_fees, 2),
            "best_trade": round(self.best_trade_pnl, 2),
            "worst_trade": round(self.worst_trade_pnl, 2),
            "current_streak": self.current_streak,
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "avg_hold_seconds": round(self.avg_hold_seconds, 1),
            "avg_management_efficiency": round(
                self.avg_management_efficiency, 4
            ),
        }
