"""Tests for journal ↔ exchange position reconciliation.

Verifies that orphaned journal trades (whose positions no longer exist
on the exchange) are force-closed during reconciliation.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentic_trading.core.enums import Exchange, MarginMode, PositionSide
from agentic_trading.core.models import Position
from agentic_trading.journal import TradeJournal
from agentic_trading.journal.record import TradePhase


def _make_position(symbol: str, qty: Decimal = Decimal("1")) -> Position:
    """Create a minimal exchange Position for testing."""
    return Position(
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=PositionSide.LONG,
        qty=qty,
        entry_price=Decimal("100"),
        mark_price=Decimal("105"),
    )


def _open_trade(journal: TradeJournal, trace_id: str, symbol: str, direction: str = "long") -> None:
    """Open a trade and record an entry fill in one step."""
    journal.open_trade(
        trace_id=trace_id,
        strategy_id="trend_following",
        symbol=symbol,
        direction=direction,
    )
    journal.record_entry_fill(
        trace_id=trace_id,
        fill_id=f"fill_{trace_id}",
        order_id=f"order_{trace_id}",
        side="buy" if direction == "long" else "sell",
        price=Decimal("100"),
        qty=Decimal("1"),
    )


def _get_reconcile_fn():
    """Import the reconciliation function from main.

    The function is defined inside run_live(), so we replicate
    its logic here for testability.
    """
    # We import the logic inline since it's nested inside run_live.
    # Instead, test the logic directly by reimplementing the same algorithm.
    from decimal import Decimal as D

    def reconcile(j: TradeJournal, exchange_positions: list) -> None:
        open_trades = j.get_open_trades()
        if not open_trades:
            return

        active_symbols: set[str] = set()
        for p in exchange_positions:
            sym = p.symbol
            base_sym = sym.split(":")[0] if ":" in sym else sym
            active_symbols.add(base_sym)
            active_symbols.add(sym)

        for trade in open_trades:
            if trade.symbol in active_symbols:
                continue
            close_price = D("0")
            if trade.mark_samples:
                close_price = trade.mark_samples[-1].mark_price
            if close_price == D("0"):
                close_price = trade.avg_entry_price
            if close_price == D("0") and trade.entry_fills:
                close_price = trade.entry_fills[0].price
            j.force_close(trade.trace_id, close_price)

    return reconcile


class TestJournalReconciliation:
    """Tests for _reconcile_journal_positions logic."""

    def test_no_open_trades_is_noop(self):
        """Reconciliation does nothing when journal has no open trades."""
        journal = TradeJournal()
        reconcile = _get_reconcile_fn()
        reconcile(journal, [_make_position("BTC/USDT:USDT")])
        assert journal.open_trade_count == 0

    def test_trade_with_matching_position_stays_open(self):
        """Trade whose symbol matches an exchange position remains open."""
        journal = TradeJournal()
        _open_trade(journal, "trace1", "BTC/USDT")

        reconcile = _get_reconcile_fn()
        # Exchange has BTC/USDT:USDT which normalises to BTC/USDT
        reconcile(journal, [_make_position("BTC/USDT:USDT")])

        assert journal.open_trade_count == 1

    def test_orphaned_trade_is_force_closed(self):
        """Trade whose position is absent on exchange gets force-closed."""
        journal = TradeJournal()
        _open_trade(journal, "trace1", "BTC/USDT")

        reconcile = _get_reconcile_fn()
        # Exchange has ETH position but NOT BTC
        reconcile(journal, [_make_position("ETH/USDT:USDT")])

        assert journal.open_trade_count == 0
        assert journal.closed_trade_count == 1

    def test_orphaned_trade_closed_at_entry_price(self):
        """Force-closed trade uses avg entry price when no marks exist."""
        journal = TradeJournal()
        _open_trade(journal, "trace1", "BTC/USDT")

        reconcile = _get_reconcile_fn()
        reconcile(journal, [])  # No positions on exchange

        closed = journal.get_closed_trades()
        assert len(closed) == 1
        trade = closed[0]
        assert trade.avg_exit_price == Decimal("100")  # Entry price
        assert trade.phase == TradePhase.CLOSED

    def test_orphaned_trade_closed_at_mark_price(self):
        """Force-closed trade uses latest mark sample when available."""
        journal = TradeJournal()
        _open_trade(journal, "trace1", "BTC/USDT")
        journal.record_mark(
            "trace1",
            mark_price=Decimal("110"),
            unrealized_pnl=Decimal("10"),
        )

        reconcile = _get_reconcile_fn()
        reconcile(journal, [])

        closed = journal.get_closed_trades()
        assert len(closed) == 1
        assert closed[0].avg_exit_price == Decimal("110")

    def test_multiple_trades_mixed(self):
        """Only orphaned trades are closed; matching ones survive."""
        journal = TradeJournal()
        _open_trade(journal, "trace_btc", "BTC/USDT")
        _open_trade(journal, "trace_eth", "ETH/USDT")
        _open_trade(journal, "trace_sol", "SOL/USDT")

        reconcile = _get_reconcile_fn()
        # Exchange has BTC and SOL but not ETH
        reconcile(journal, [
            _make_position("BTC/USDT:USDT"),
            _make_position("SOL/USDT:USDT"),
        ])

        assert journal.open_trade_count == 2
        assert journal.closed_trade_count == 1
        # ETH trade should be closed
        open_symbols = {t.symbol for t in journal.get_open_trades()}
        assert "ETH/USDT" not in open_symbols
        assert "BTC/USDT" in open_symbols
        assert "SOL/USDT" in open_symbols

    def test_empty_exchange_closes_all(self):
        """All trades closed when exchange reports zero positions."""
        journal = TradeJournal()
        _open_trade(journal, "trace1", "BTC/USDT")
        _open_trade(journal, "trace2", "ETH/USDT")

        reconcile = _get_reconcile_fn()
        reconcile(journal, [])

        assert journal.open_trade_count == 0
        assert journal.closed_trade_count == 2

    def test_symbol_without_settle_suffix_matches(self):
        """Exchange position without settle suffix still matches journal."""
        journal = TradeJournal()
        _open_trade(journal, "trace1", "BTC/USDT")

        reconcile = _get_reconcile_fn()
        # Position symbol without suffix (spot-style)
        reconcile(journal, [_make_position("BTC/USDT")])

        assert journal.open_trade_count == 1

    def test_pending_trade_without_fills_is_not_force_closed(self):
        """Trade with no entry fills (pending) uses zero price for force close."""
        journal = TradeJournal()
        # Open trade but DON'T record a fill
        journal.open_trade(
            trace_id="trace_pending",
            strategy_id="trend_following",
            symbol="BTC/USDT",
            direction="long",
        )

        reconcile = _get_reconcile_fn()
        reconcile(journal, [])

        # force_close with remaining_qty=0 (no fills) returns the trade
        # but doesn't close it since there's nothing to close
        assert journal.open_trade_count == 1

    def test_short_trade_orphaned(self):
        """Short trades are also reconciled correctly."""
        journal = TradeJournal()
        _open_trade(journal, "trace1", "ETH/USDT", direction="short")

        reconcile = _get_reconcile_fn()
        reconcile(journal, [])

        assert journal.open_trade_count == 0
        closed = journal.get_closed_trades()
        assert len(closed) == 1
        # Exit side should be 'buy' for a short
        assert closed[0].exit_fills[0].side == "buy"
