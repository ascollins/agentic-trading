"""Tests for TradeJournal — the event aggregator."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from agentic_trading.journal.record import TradePhase, TradeOutcome, TradeRecord
from agentic_trading.journal.journal import TradeJournal


class TestTradeJournalLifecycle:
    """Test trade lifecycle through the journal."""

    def test_open_trade(self, journal):
        trade = journal.open_trade(
            trace_id="t1",
            strategy_id="trend",
            symbol="BTC/USDT",
            direction="long",
            signal_confidence=0.85,
        )
        assert trade.phase == TradePhase.PENDING
        assert trade.strategy_id == "trend"
        assert journal.open_trade_count == 1

    def test_open_trade_idempotent(self, journal):
        t1 = journal.open_trade("t1", "trend", "BTC/USDT", "long")
        t2 = journal.open_trade("t1", "trend", "BTC/USDT", "long")
        assert t1 is t2
        assert journal.open_trade_count == 1

    def test_record_entry_fill(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        trade = journal.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("1"),
        )
        assert trade is not None
        assert trade.phase == TradePhase.OPEN
        assert trade.entry_qty == Decimal("1")

    def test_record_entry_fill_unknown_trace(self, journal):
        result = journal.record_entry_fill(
            "unknown", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("1"),
        )
        assert result is None

    def test_record_exit_fill_closes_trade(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        journal.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("1"),
        )
        trade = journal.record_exit_fill(
            "t1", fill_id="f2", order_id="o2",
            side="sell", price=Decimal("110"), qty=Decimal("1"),
        )
        assert trade is not None
        assert trade.phase == TradePhase.CLOSED
        assert journal.open_trade_count == 0
        assert journal.closed_trade_count == 1

    def test_partial_exit_stays_open(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        journal.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("10"),
        )
        trade = journal.record_exit_fill(
            "t1", fill_id="f2", order_id="o2",
            side="sell", price=Decimal("110"), qty=Decimal("5"),
        )
        assert trade.phase == TradePhase.OPEN
        assert journal.open_trade_count == 1

    def test_cancel_trade(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        journal.cancel_trade("t1")
        assert journal.open_trade_count == 0
        assert journal.closed_trade_count == 0

    def test_force_close(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        journal.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("5"),
        )
        trade = journal.force_close("t1", exit_price=Decimal("95"))
        assert trade.phase == TradePhase.CLOSED
        assert trade.exit_qty == Decimal("5")

    def test_record_mark(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        journal.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("1"),
        )
        journal.record_mark("t1", mark_price=Decimal("105"), unrealized_pnl=Decimal("5"))
        trade = journal.get_trade("t1")
        assert len(trade.mark_samples) == 1


class TestTradeJournalQueries:
    """Test query methods."""

    def test_get_trade_open(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        assert journal.get_trade("t1") is not None
        assert journal.get_trade("unknown") is None

    def test_get_trade_closed(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        journal.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("1"),
        )
        journal.record_exit_fill(
            "t1", fill_id="f2", order_id="o2",
            side="sell", price=Decimal("110"), qty=Decimal("1"),
        )
        # Now closed — should still be findable
        trade = journal.get_trade("t1")
        assert trade is not None
        assert trade.phase == TradePhase.CLOSED

    def test_get_trade_by_position(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        trade = journal.get_trade_by_position("trend", "BTC/USDT")
        assert trade is not None
        assert trade.trace_id == "t1"

    def test_get_open_trades_all(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        journal.open_trade("t2", "mean_rev", "ETH/USDT", "short")
        assert len(journal.get_open_trades()) == 2

    def test_get_open_trades_filtered(self, journal):
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        journal.open_trade("t2", "mean_rev", "ETH/USDT", "short")
        assert len(journal.get_open_trades(strategy_id="trend")) == 1

    def test_get_closed_trades_filtered(self, journal):
        # Create and close 2 trades
        for i, sid in enumerate(["trend", "mean_rev"]):
            tid = f"t{i}"
            journal.open_trade(tid, sid, "BTC/USDT", "long")
            journal.record_entry_fill(
                tid, fill_id=f"f{i}", order_id=f"o{i}",
                side="buy", price=Decimal("100"), qty=Decimal("1"),
            )
            journal.record_exit_fill(
                tid, fill_id=f"e{i}", order_id=f"eo{i}",
                side="sell", price=Decimal("110"), qty=Decimal("1"),
            )
        assert len(journal.get_closed_trades(strategy_id="trend")) == 1
        assert len(journal.get_closed_trades()) == 2

    def test_get_closed_trades_last_n(self, journal):
        for i in range(5):
            tid = f"t{i}"
            journal.open_trade(tid, "trend", "BTC/USDT", "long")
            journal.record_entry_fill(
                tid, fill_id=f"f{i}", order_id=f"o{i}",
                side="buy", price=Decimal("100"), qty=Decimal("1"),
            )
            journal.record_exit_fill(
                tid, fill_id=f"e{i}", order_id=f"eo{i}",
                side="sell", price=Decimal("110"), qty=Decimal("1"),
            )
        assert len(journal.get_closed_trades(last_n=3)) == 3

    def test_max_closed_trades_eviction(self):
        j = TradeJournal(max_closed_trades=5)
        for i in range(10):
            tid = f"t{i}"
            j.open_trade(tid, "trend", "BTC/USDT", "long")
            j.record_entry_fill(
                tid, fill_id=f"f{i}", order_id=f"o{i}",
                side="buy", price=Decimal("100"), qty=Decimal("1"),
            )
            j.record_exit_fill(
                tid, fill_id=f"e{i}", order_id=f"eo{i}",
                side="sell", price=Decimal("110"), qty=Decimal("1"),
            )
        assert j.closed_trade_count == 5  # Evicted oldest


class TestTradeJournalStats:
    """Test per-strategy statistics."""

    def test_strategy_stats_after_trades(self, journal):
        # Record a winning trade
        journal.open_trade("t1", "trend", "BTC/USDT", "long")
        journal.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("1"), fee=Decimal("0"),
        )
        journal.record_exit_fill(
            "t1", fill_id="e1", order_id="eo1",
            side="sell", price=Decimal("110"), qty=Decimal("1"), fee=Decimal("0"),
        )
        stats = journal.get_strategy_stats("trend")
        assert stats["total_trades"] == 1
        assert stats["wins"] == 1
        assert stats["win_rate"] == 1.0

    def test_all_strategy_stats(self, journal):
        for sid in ["trend", "mean_rev"]:
            journal.open_trade(f"t_{sid}", sid, "BTC/USDT", "long")
            journal.record_entry_fill(
                f"t_{sid}", fill_id=f"f_{sid}", order_id=f"o_{sid}",
                side="buy", price=Decimal("100"), qty=Decimal("1"),
            )
            journal.record_exit_fill(
                f"t_{sid}", fill_id=f"e_{sid}", order_id=f"eo_{sid}",
                side="sell", price=Decimal("110"), qty=Decimal("1"),
            )
        all_stats = journal.get_all_strategy_stats()
        assert "trend" in all_stats
        assert "mean_rev" in all_stats


class TestTradeJournalCallbacks:
    """Test callback integration."""

    def test_on_trade_closed_callback(self):
        callback = MagicMock()
        j = TradeJournal(on_trade_closed=callback)
        j.open_trade("t1", "trend", "BTC/USDT", "long")
        j.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("1"),
        )
        j.record_exit_fill(
            "t1", fill_id="e1", order_id="eo1",
            side="sell", price=Decimal("110"), qty=Decimal("1"),
        )
        callback.assert_called_once()
        trade_arg = callback.call_args[0][0]
        assert trade_arg.phase == TradePhase.CLOSED

    def test_health_tracker_integration(self):
        tracker = MagicMock()
        j = TradeJournal(health_tracker=tracker)
        j.open_trade("t1", "trend", "BTC/USDT", "long")
        j.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("1"), fee=Decimal("0"),
        )
        j.record_exit_fill(
            "t1", fill_id="e1", order_id="eo1",
            side="sell", price=Decimal("110"), qty=Decimal("1"), fee=Decimal("0"),
        )
        tracker.record_outcome.assert_called_once()
        args = tracker.record_outcome.call_args
        assert args[0][0] == "trend"  # strategy_id
        assert args[0][1] is True     # won

    def test_drift_detector_integration(self):
        detector = MagicMock()
        j = TradeJournal(drift_detector=detector)
        j.open_trade("t1", "trend", "BTC/USDT", "long")
        j.record_entry_fill(
            "t1", fill_id="f1", order_id="o1",
            side="buy", price=Decimal("100"), qty=Decimal("1"), fee=Decimal("0"),
        )
        j.record_exit_fill(
            "t1", fill_id="e1", order_id="eo1",
            side="sell", price=Decimal("110"), qty=Decimal("1"), fee=Decimal("0"),
        )
        assert detector.update_live_metric.called
