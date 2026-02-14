"""Tests for TradeRecord — the core data model."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from agentic_trading.journal.record import (
    FillLeg,
    MarkSample,
    TradePhase,
    TradeOutcome,
    TradeRecord,
)

from .conftest import make_fill


class TestTradeRecordLifecycle:
    """Test trade lifecycle state transitions."""

    def test_initial_state_is_pending(self):
        trade = TradeRecord(strategy_id="test", symbol="BTC/USDT")
        assert trade.phase == TradePhase.PENDING

    def test_transitions_to_open_on_entry_fill(self):
        trade = TradeRecord(strategy_id="test", symbol="BTC/USDT")
        fill = make_fill()
        trade.add_entry_fill(fill)
        assert trade.phase == TradePhase.OPEN
        assert trade.opened_at == fill.timestamp

    def test_stays_open_with_partial_exit(self):
        trade = TradeRecord(strategy_id="test", symbol="BTC/USDT", direction="long")
        trade.add_entry_fill(make_fill(qty=10.0))
        trade.add_exit_fill(make_fill(fill_id="e1", side="sell", qty=5.0))
        assert trade.phase == TradePhase.OPEN
        assert trade.remaining_qty == Decimal("5")

    def test_transitions_to_closed_on_full_exit(self):
        trade = TradeRecord(strategy_id="test", symbol="BTC/USDT", direction="long")
        t0 = datetime(2024, 1, 1, 12, 0)
        t1 = datetime(2024, 1, 1, 13, 0)
        trade.add_entry_fill(make_fill(qty=10.0, timestamp=t0))
        trade.add_exit_fill(make_fill(fill_id="e1", side="sell", qty=10.0, timestamp=t1))
        assert trade.phase == TradePhase.CLOSED
        assert trade.closed_at == t1

    def test_cancel_from_pending(self):
        trade = TradeRecord(strategy_id="test", symbol="BTC/USDT")
        trade.cancel()
        assert trade.phase == TradePhase.CANCELLED

    def test_cancel_ignored_if_open(self):
        trade = TradeRecord(strategy_id="test", symbol="BTC/USDT")
        trade.add_entry_fill(make_fill())
        trade.cancel()  # Should be ignored
        assert trade.phase == TradePhase.OPEN


class TestTradeRecordPnL:
    """Test P&L calculations."""

    def test_long_winning_trade(self):
        trade = TradeRecord(direction="long")
        trade.add_entry_fill(make_fill(price=100.0, qty=2.0, fee=0.2))
        trade.add_exit_fill(make_fill(fill_id="e1", side="sell", price=110.0, qty=2.0, fee=0.2))
        assert trade.gross_pnl == Decimal("20")  # (110-100)*2
        assert trade.net_pnl == Decimal("19.6")   # 20 - 0.4
        assert trade.outcome == TradeOutcome.WIN

    def test_short_winning_trade(self):
        trade = TradeRecord(direction="short")
        trade.add_entry_fill(make_fill(side="sell", price=100.0, qty=2.0, fee=0.2))
        trade.add_exit_fill(make_fill(fill_id="e1", side="buy", price=90.0, qty=2.0, fee=0.2))
        assert trade.gross_pnl == Decimal("20")
        assert trade.outcome == TradeOutcome.WIN

    def test_long_losing_trade(self):
        trade = TradeRecord(direction="long")
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0, fee=0.1))
        trade.add_exit_fill(make_fill(fill_id="e1", side="sell", price=95.0, qty=1.0, fee=0.1))
        assert trade.gross_pnl == Decimal("-5")
        assert trade.net_pnl == Decimal("-5.2")
        assert trade.outcome == TradeOutcome.LOSS

    def test_breakeven_trade(self):
        trade = TradeRecord(direction="long")
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0, fee=0.0))
        trade.add_exit_fill(make_fill(fill_id="e1", side="sell", price=100.0, qty=1.0, fee=0.0))
        assert trade.net_pnl == Decimal("0")
        assert trade.outcome == TradeOutcome.BREAKEVEN

    def test_net_pnl_pct(self):
        trade = TradeRecord(direction="long")
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0, fee=0.0))
        trade.add_exit_fill(make_fill(fill_id="e1", side="sell", price=110.0, qty=1.0, fee=0.0))
        assert abs(trade.net_pnl_pct - 0.10) < 1e-9

    def test_avg_entry_price_multiple_fills(self):
        trade = TradeRecord(direction="long")
        trade.add_entry_fill(make_fill(fill_id="f1", price=100.0, qty=1.0))
        trade.add_entry_fill(make_fill(fill_id="f2", price=110.0, qty=1.0))
        assert trade.avg_entry_price == Decimal("105")  # (100+110)/2
        assert trade.entry_qty == Decimal("2")


class TestRMultiple:
    """Test R-multiple computation."""

    def test_r_multiple_winning_long(self):
        trade = TradeRecord(
            direction="long",
            initial_risk_price=Decimal("95"),
        )
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0, fee=0.0))
        trade.compute_initial_risk()
        trade.add_exit_fill(make_fill(fill_id="e1", side="sell", price=110.0, qty=1.0, fee=0.0))
        # Risk = (100-95)*1 = 5, PnL = 10, R = 10/5 = 2.0
        assert trade.initial_risk_amount == Decimal("5")
        assert trade.r_multiple == 2.0

    def test_r_multiple_losing_long(self):
        trade = TradeRecord(
            direction="long",
            initial_risk_price=Decimal("95"),
        )
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0, fee=0.0))
        trade.compute_initial_risk()
        trade.add_exit_fill(make_fill(fill_id="e1", side="sell", price=93.0, qty=1.0, fee=0.0))
        # Risk = 5, PnL = -7, R = -7/5 = -1.4
        assert abs(trade.r_multiple - (-1.4)) < 0.01

    def test_r_multiple_zero_when_no_risk(self):
        trade = TradeRecord(direction="long")
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0))
        assert trade.r_multiple == 0.0

    def test_compute_initial_risk_short(self):
        trade = TradeRecord(
            direction="short",
            initial_risk_price=Decimal("105"),
        )
        trade.add_entry_fill(make_fill(side="sell", price=100.0, qty=2.0))
        risk = trade.compute_initial_risk()
        # Short: risk = (stop - entry) * qty = (105-100)*2 = 10
        assert risk == Decimal("10")


class TestMAEMFE:
    """Test Maximum Adverse/Favorable Excursion."""

    def test_mae_mfe_with_samples(self):
        trade = TradeRecord(direction="long")
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0))
        bt = datetime(2024, 1, 1, 12, 0)
        trade.add_mark_sample(MarkSample(
            timestamp=bt + timedelta(minutes=10),
            mark_price=Decimal("98"),
            unrealized_pnl=Decimal("-2"),
        ))
        trade.add_mark_sample(MarkSample(
            timestamp=bt + timedelta(minutes=20),
            mark_price=Decimal("112"),
            unrealized_pnl=Decimal("12"),
        ))
        trade.add_mark_sample(MarkSample(
            timestamp=bt + timedelta(minutes=30),
            mark_price=Decimal("97"),
            unrealized_pnl=Decimal("-3"),
        ))
        assert trade.mae == Decimal("-3")
        assert trade.mfe == Decimal("12")
        assert trade.mae_price == Decimal("97")
        assert trade.mfe_price == Decimal("112")

    def test_mae_mfe_empty(self):
        trade = TradeRecord(direction="long")
        assert trade.mae == Decimal("0")
        assert trade.mfe == Decimal("0")

    def test_management_efficiency(self):
        trade = TradeRecord(direction="long")
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0, fee=0.0))
        bt = datetime(2024, 1, 1, 12, 0)
        # MFE = 15 (price hit 115)
        trade.add_mark_sample(MarkSample(
            timestamp=bt + timedelta(minutes=10),
            mark_price=Decimal("115"),
            unrealized_pnl=Decimal("15"),
        ))
        # But we only captured 10 of the 15
        trade.add_exit_fill(make_fill(
            fill_id="e1", side="sell", price=110.0, qty=1.0, fee=0.0,
            timestamp=bt + timedelta(hours=1),
        ))
        # efficiency = net_pnl / mfe = 10 / 15 ≈ 0.6667
        assert abs(trade.management_efficiency - 0.6667) < 0.001

    def test_mae_r_and_mfe_r(self):
        trade = TradeRecord(
            direction="long",
            initial_risk_price=Decimal("95"),
        )
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0))
        trade.compute_initial_risk()
        bt = datetime(2024, 1, 1, 12, 0)
        trade.add_mark_sample(MarkSample(
            timestamp=bt, mark_price=Decimal("97"),
            unrealized_pnl=Decimal("-3"),
        ))
        trade.add_mark_sample(MarkSample(
            timestamp=bt + timedelta(minutes=30), mark_price=Decimal("115"),
            unrealized_pnl=Decimal("15"),
        ))
        # Risk = 5, MAE = -3, MFE = 15
        assert abs(trade.mae_r - (-0.6)) < 0.01
        assert abs(trade.mfe_r - 3.0) < 0.01


class TestPlannedVsActual:
    """Test planned vs actual risk-reward ratios."""

    def test_planned_rr_ratio(self):
        trade = TradeRecord(
            direction="long",
            initial_risk_price=Decimal("95"),
            planned_target_price=Decimal("115"),
        )
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0))
        # risk = 5, reward = 15, ratio = 3.0
        assert abs(trade.planned_rr_ratio - 3.0) < 0.01

    def test_actual_rr_ratio(self):
        trade = TradeRecord(
            direction="long",
            initial_risk_price=Decimal("95"),
        )
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0, fee=0.0))
        trade.compute_initial_risk()
        trade.add_exit_fill(make_fill(
            fill_id="e1", side="sell", price=110.0, qty=1.0, fee=0.0
        ))
        # R = 10/5 = 2.0, actual_rr = abs(2.0) = 2.0
        assert abs(trade.actual_rr_ratio - 2.0) < 0.01


class TestHoldDuration:
    """Test hold duration calculations."""

    def test_hold_duration_closed(self):
        trade = TradeRecord()
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        t1 = datetime(2024, 1, 1, 14, 30, 0)
        trade.add_entry_fill(make_fill(timestamp=t0))
        trade.add_exit_fill(make_fill(fill_id="e1", side="sell", timestamp=t1))
        assert trade.hold_duration_seconds == 9000.0  # 2.5 hours

    def test_hold_duration_zero_when_pending(self):
        trade = TradeRecord()
        assert trade.hold_duration_seconds == 0.0


class TestSerialization:
    """Test to_dict serialisation."""

    def test_to_dict_closed_trade(self):
        trade = TradeRecord(
            strategy_id="test", symbol="BTC/USDT", direction="long",
            signal_confidence=0.75,
        )
        trade.add_entry_fill(make_fill(price=100.0, qty=1.0))
        trade.add_exit_fill(make_fill(
            fill_id="e1", side="sell", price=110.0, qty=1.0
        ))
        d = trade.to_dict()
        assert d["strategy_id"] == "test"
        assert d["symbol"] == "BTC/USDT"
        assert d["phase"] == "closed"
        assert d["outcome"] == "win"
        assert d["signal_confidence"] == 0.75
        assert d["entry_fills"] == 1
        assert d["exit_fills"] == 1

    def test_to_dict_pending_trade(self):
        trade = TradeRecord(strategy_id="test", symbol="ETH/USDT")
        d = trade.to_dict()
        assert d["phase"] == "pending"
        assert d["outcome"] is None
