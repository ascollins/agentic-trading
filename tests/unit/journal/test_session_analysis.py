"""Tests for SessionAnalyser — time-of-day and session performance analysis."""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from agentic_trading.journal.session_analysis import SessionAnalyser, SESSIONS, DAY_NAMES
from agentic_trading.journal.record import (
    FillLeg,
    MarkSample,
    TradePhase,
    TradeOutcome,
    TradeRecord,
)

from .conftest import make_fill, make_winning_trade, make_losing_trade


@pytest.fixture
def analyser():
    return SessionAnalyser(max_trades=500)


def _make_trade_at_hour(hour: int, won: bool = True, strategy_id: str = "trend", weekday: int = 0) -> TradeRecord:
    """Create a trade opened at a specific hour (UTC)."""
    # 2024-01-01 was a Monday (weekday=0). Offset to desired weekday.
    bt = datetime(2024, 1, 1 + weekday, hour, 0, 0)
    entry_price = 100.0
    exit_price = 110.0 if won else 95.0
    trade = TradeRecord(
        trace_id=f"trace_{hour}_{weekday}_{won}",
        strategy_id=strategy_id,
        symbol="BTC/USDT",
        direction="long",
        signal_confidence=0.7,
        initial_risk_price=Decimal("95"),
    )
    trade.add_entry_fill(make_fill(price=entry_price, qty=1.0, timestamp=bt))
    trade.compute_initial_risk()
    trade.add_exit_fill(make_fill(
        fill_id="exit_1", order_id="exit_order", side="sell",
        price=exit_price, qty=1.0,
        timestamp=bt + timedelta(hours=1),
    ))
    return trade


class TestSessionAnalyserBasics:
    """Test basic session analysis."""

    def test_empty_report(self, analyser):
        report = analyser.report("nonexistent")
        assert report["total_trades"] == 0
        assert report["by_hour"] == {}
        assert report["by_session"] == {}
        assert report["by_day"] == {}

    def test_single_trade_report(self, analyser):
        trade = _make_trade_at_hour(10)  # London session
        analyser.add_trade(trade)
        report = analyser.report("trend")
        assert report["total_trades"] == 1
        assert 10 in report["by_hour"]
        assert "london" in report["by_session"]

    def test_ignores_open_trades(self, analyser):
        trade = TradeRecord(
            trace_id="t1", strategy_id="trend",
            symbol="BTC/USDT", direction="long",
        )
        trade.add_entry_fill(make_fill(price=100.0))
        analyser.add_trade(trade)
        report = analyser.report("trend")
        assert report["total_trades"] == 0


class TestSessionBucketing:
    """Test correct bucketing into sessions."""

    def test_asia_session(self, analyser):
        """Hour 3 should bucket to Asia session."""
        trade = _make_trade_at_hour(3)
        analyser.add_trade(trade)
        report = analyser.report("trend")
        assert "asia" in report["by_session"]
        assert report["by_session"]["asia"]["trades"] == 1

    def test_london_session(self, analyser):
        """Hour 10 should bucket to London session."""
        trade = _make_trade_at_hour(10)
        analyser.add_trade(trade)
        report = analyser.report("trend")
        assert "london" in report["by_session"]
        assert report["by_session"]["london"]["trades"] == 1

    def test_new_york_session(self, analyser):
        """Hour 15 should bucket to both London and NY."""
        trade = _make_trade_at_hour(15)
        analyser.add_trade(trade)
        report = analyser.report("trend")
        # 15 is in both london (8-16) and new_york (13-21)
        assert "london" in report["by_session"]
        assert "new_york" in report["by_session"]

    def test_off_hours(self, analyser):
        """Hour 22 should bucket to off_hours."""
        trade = _make_trade_at_hour(22)
        analyser.add_trade(trade)
        report = analyser.report("trend")
        assert "off_hours" in report["by_session"]
        assert report["by_session"]["off_hours"]["trades"] == 1


class TestDayOfWeek:
    """Test day-of-week bucketing."""

    def test_weekday_bucketing(self, analyser):
        """Trades on different weekdays should bucket correctly."""
        for day in range(5):  # Mon-Fri
            trade = _make_trade_at_hour(10, weekday=day)
            trade.trade_id = f"t_{day}"
            trade.trace_id = f"tr_{day}"
            analyser.add_trade(trade)

        report = analyser.report("trend")
        assert "monday" in report["by_day"]
        assert "tuesday" in report["by_day"]
        assert "wednesday" in report["by_day"]
        assert "thursday" in report["by_day"]
        assert "friday" in report["by_day"]


class TestBestWorst:
    """Test best/worst hour/session/day identification."""

    def test_best_hour_identified(self, analyser):
        """Best performing hour should be identified."""
        # Add 5 winning trades at hour 10
        for i in range(5):
            trade = _make_trade_at_hour(10, won=True)
            trade.trade_id = f"win_{i}"
            trade.trace_id = f"win_tr_{i}"
            analyser.add_trade(trade)

        # Add 5 losing trades at hour 3
        for i in range(5):
            trade = _make_trade_at_hour(3, won=False)
            trade.trade_id = f"loss_{i}"
            trade.trace_id = f"loss_tr_{i}"
            analyser.add_trade(trade)

        report = analyser.report("trend")
        assert report["best_hour"] == 10
        assert report["worst_hour"] == 3

    def test_best_session_identified(self, analyser):
        """Best performing session should be identified."""
        # 5 wins in London
        for i in range(5):
            trade = _make_trade_at_hour(10, won=True)
            trade.trade_id = f"lon_{i}"
            trade.trace_id = f"lon_tr_{i}"
            analyser.add_trade(trade)

        # 5 losses in Asia
        for i in range(5):
            trade = _make_trade_at_hour(3, won=False)
            trade.trade_id = f"asia_{i}"
            trade.trace_id = f"asia_tr_{i}"
            analyser.add_trade(trade)

        report = analyser.report("trend")
        assert report["best_session"] == "london"


class TestSessionStats:
    """Test per-bucket statistics."""

    def test_stats_structure(self, analyser):
        """Each bucket should have correct stat keys."""
        for i in range(5):
            trade = _make_trade_at_hour(10, won=i < 3)
            trade.trade_id = f"t_{i}"
            trade.trace_id = f"tr_{i}"
            analyser.add_trade(trade)

        report = analyser.report("trend")
        stats = report["by_hour"][10]
        assert "trades" in stats
        assert "wins" in stats
        assert "losses" in stats
        assert "win_rate" in stats
        assert "total_pnl" in stats
        assert "avg_pnl" in stats
        assert "avg_r" in stats
        assert "profit_factor" in stats
        assert "avg_hold_minutes" in stats

    def test_win_rate_calculation(self, analyser):
        """Win rate should be correctly calculated."""
        for i in range(10):
            trade = _make_trade_at_hour(10, won=i < 7)
            trade.trade_id = f"t_{i}"
            trade.trace_id = f"tr_{i}"
            analyser.add_trade(trade)

        report = analyser.report("trend")
        assert report["by_hour"][10]["win_rate"] == pytest.approx(0.7, abs=0.01)


class TestMultiStrategy:
    """Test multi-strategy support."""

    def test_separate_strategy_reports(self, analyser):
        """Different strategies should have separate reports."""
        trade_a = _make_trade_at_hour(10, strategy_id="strat_a")
        trade_b = _make_trade_at_hour(15, strategy_id="strat_b")
        analyser.add_trade(trade_a)
        analyser.add_trade(trade_b)

        report_a = analyser.report("strat_a")
        report_b = analyser.report("strat_b")
        assert report_a["total_trades"] == 1
        assert report_b["total_trades"] == 1

    def test_get_all_strategy_ids(self, analyser):
        trade = _make_trade_at_hour(10, strategy_id="test_strat")
        analyser.add_trade(trade)
        assert "test_strat" in analyser.get_all_strategy_ids()
