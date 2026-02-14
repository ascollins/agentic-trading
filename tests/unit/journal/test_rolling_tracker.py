"""Tests for RollingTracker — sliding-window analytics."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from agentic_trading.journal.rolling_tracker import RollingTracker
from agentic_trading.journal.record import TradeRecord, TradePhase

from .conftest import make_fill, make_winning_trade, make_losing_trade


class TestRollingTrackerBasics:
    """Test basic recording and snapshot generation."""

    def test_empty_snapshot(self, rolling_tracker):
        assert rolling_tracker.snapshot("nonexistent") == {}

    def test_ignores_non_closed_trades(self, rolling_tracker):
        trade = TradeRecord(strategy_id="trend", symbol="BTC/USDT")
        trade.add_entry_fill(make_fill())
        # Trade is OPEN, not CLOSED
        rolling_tracker.add_trade(trade)
        assert rolling_tracker.snapshot("trend") == {}

    def test_single_winning_trade(self, rolling_tracker):
        trade = make_winning_trade()
        rolling_tracker.add_trade(trade)
        snap = rolling_tracker.snapshot("trend")
        assert snap["trade_count"] == 1
        assert snap["win_rate"] == 1.0
        assert snap["total_pnl"] > 0

    def test_single_losing_trade(self, rolling_tracker):
        trade = make_losing_trade()
        rolling_tracker.add_trade(trade)
        snap = rolling_tracker.snapshot("trend")
        assert snap["trade_count"] == 1
        assert snap["win_rate"] == 0.0
        assert snap["total_pnl"] < 0

    def test_multiple_trades_stats(self, rolling_tracker):
        for i in range(5):
            rolling_tracker.add_trade(make_winning_trade(
                entry_price=100.0, exit_price=110.0,
                base_time=datetime(2024, 1, 1 + i, 12, 0),
            ))
        for i in range(3):
            rolling_tracker.add_trade(make_losing_trade(
                entry_price=100.0, exit_price=95.0,
                base_time=datetime(2024, 1, 10 + i, 12, 0),
            ))
        snap = rolling_tracker.snapshot("trend")
        assert snap["trade_count"] == 8
        assert abs(snap["win_rate"] - 5 / 8) < 0.01
        assert snap["profit_factor"] > 0


class TestRollingTrackerWindow:
    """Test sliding window behavior."""

    def test_window_eviction(self):
        tracker = RollingTracker(window_size=3)
        for i in range(5):
            tracker.add_trade(make_winning_trade(
                base_time=datetime(2024, 1, 1 + i, 12, 0),
            ))
        snap = tracker.snapshot("trend")
        assert snap["trade_count"] == 3  # Only most recent 3

    def test_cache_invalidation(self, rolling_tracker):
        rolling_tracker.add_trade(make_winning_trade())
        snap1 = rolling_tracker.snapshot("trend")
        rolling_tracker.add_trade(make_losing_trade(
            base_time=datetime(2024, 1, 2, 12, 0),
        ))
        snap2 = rolling_tracker.snapshot("trend")
        assert snap1["trade_count"] != snap2["trade_count"]


class TestRollingTrackerMetrics:
    """Test specific metric computations."""

    def test_sharpe_ratio(self, rolling_tracker):
        # Add enough trades for meaningful Sharpe
        for i in range(20):
            if i % 3 == 0:
                rolling_tracker.add_trade(make_losing_trade(
                    base_time=datetime(2024, 1, 1 + i, 12, 0),
                ))
            else:
                rolling_tracker.add_trade(make_winning_trade(
                    base_time=datetime(2024, 1, 1 + i, 12, 0),
                ))
        snap = rolling_tracker.snapshot("trend")
        assert "sharpe" in snap
        assert snap["sharpe"] != 0.0

    def test_sortino_ratio(self, rolling_tracker):
        for i in range(10):
            rolling_tracker.add_trade(make_winning_trade(
                base_time=datetime(2024, 1, 1 + i, 12, 0),
            ))
        for i in range(5):
            rolling_tracker.add_trade(make_losing_trade(
                base_time=datetime(2024, 1, 15 + i, 12, 0),
            ))
        snap = rolling_tracker.snapshot("trend")
        assert "sortino" in snap

    def test_max_drawdown(self, rolling_tracker):
        # Win, win, loss, loss, loss pattern
        for i in range(2):
            rolling_tracker.add_trade(make_winning_trade(
                base_time=datetime(2024, 1, 1 + i, 12, 0),
            ))
        for i in range(3):
            rolling_tracker.add_trade(make_losing_trade(
                base_time=datetime(2024, 1, 5 + i, 12, 0),
            ))
        snap = rolling_tracker.snapshot("trend")
        assert snap["max_drawdown"] > 0

    def test_consecutive_streaks(self, rolling_tracker):
        for i in range(3):
            rolling_tracker.add_trade(make_winning_trade(
                base_time=datetime(2024, 1, 1 + i, 12, 0),
            ))
        snap = rolling_tracker.snapshot("trend")
        assert snap["consecutive_wins"] == 3
        assert snap["consecutive_losses"] == 0

    def test_get_all_strategy_ids(self, rolling_tracker):
        rolling_tracker.add_trade(make_winning_trade(strategy_id="a"))
        rolling_tracker.add_trade(make_winning_trade(strategy_id="b"))
        ids = rolling_tracker.get_all_strategy_ids()
        assert set(ids) == {"a", "b"}

    def test_trade_count(self, rolling_tracker):
        rolling_tracker.add_trade(make_winning_trade())
        assert rolling_tracker.trade_count("trend") == 1
        assert rolling_tracker.trade_count("nonexistent") == 0
