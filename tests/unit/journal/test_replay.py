"""Tests for TradeReplayer — structured mark-to-market replay."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from agentic_trading.journal.replay import TradeReplayer
from agentic_trading.journal.record import (
    FillLeg,
    MarkSample,
    TradePhase,
    TradeRecord,
)

from .conftest import make_fill, make_winning_trade, make_losing_trade


@pytest.fixture
def replayer():
    return TradeReplayer()


class TestReplayBasics:
    """Test basic replay construction."""

    def test_winning_trade_replay(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        assert replay["trade_id"] == trade.trade_id
        assert "summary" in replay
        assert "timeline" in replay
        assert "zones" in replay
        assert "excursion_chart" in replay

    def test_losing_trade_replay(self, replayer):
        trade = make_losing_trade()
        replay = replayer.build_replay(trade)
        assert replay["trade_id"] == trade.trade_id
        assert replay["summary"]["net_pnl"] < 0


class TestTimeline:
    """Test timeline construction."""

    def test_timeline_has_entry_fill(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        events = replay["timeline"]
        entry_events = [e for e in events if e["event_type"] == "entry_fill"]
        assert len(entry_events) == 1

    def test_timeline_has_exit_fill(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        events = replay["timeline"]
        exit_events = [e for e in events if e["event_type"] == "exit_fill"]
        assert len(exit_events) == 1

    def test_timeline_has_marks(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        events = replay["timeline"]
        mark_events = [e for e in events if e["event_type"] == "mark"]
        assert len(mark_events) == 2  # make_winning_trade adds 2 marks

    def test_timeline_is_sorted(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        timestamps = [
            e["timestamp"] for e in replay["timeline"] if e["timestamp"]
        ]
        assert timestamps == sorted(timestamps)

    def test_cumulative_qty_computed(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        entry = [e for e in replay["timeline"] if e["event_type"] == "entry_fill"][0]
        assert entry["cumulative_qty"] > 0


class TestZones:
    """Test zone construction."""

    def test_zone_keys(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        zones = replay["zones"]
        assert "entry_price" in zones
        assert "exit_price" in zones
        assert "stop_price" in zones
        assert "target_price" in zones
        assert "mae_price" in zones
        assert "mfe_price" in zones

    def test_zone_entry_price(self, replayer):
        trade = make_winning_trade(entry_price=100.0)
        replay = replayer.build_replay(trade)
        assert replay["zones"]["entry_price"] == pytest.approx(100.0)

    def test_zone_exit_price(self, replayer):
        trade = make_winning_trade(exit_price=110.0)
        replay = replayer.build_replay(trade)
        assert replay["zones"]["exit_price"] == pytest.approx(110.0)


class TestExcursionChart:
    """Test excursion chart data."""

    def test_excursion_chart_populated(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        assert len(replay["excursion_chart"]) == 2

    def test_excursion_chart_has_pnl(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        for point in replay["excursion_chart"]:
            assert "unrealized_pnl" in point
            assert "mark_price" in point
            assert "timestamp" in point

    def test_mae_mfe_points(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        assert replay["mae_point"] is not None
        assert replay["mfe_point"] is not None
        assert replay["mfe_point"]["unrealized_pnl"] >= replay["mae_point"]["unrealized_pnl"]


class TestSummary:
    """Test replay summary."""

    def test_summary_keys(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        summary = replay["summary"]
        assert summary["trade_id"] == trade.trade_id
        assert summary["strategy_id"] == trade.strategy_id
        assert summary["symbol"] == trade.symbol
        assert summary["direction"] == trade.direction
        assert "net_pnl" in summary
        assert "r_multiple" in summary
        assert "management_efficiency" in summary
        assert "hold_duration_s" in summary

    def test_summary_outcome(self, replayer):
        trade = make_winning_trade()
        replay = replayer.build_replay(trade)
        assert replay["summary"]["outcome"] == "win"

    def test_summary_losing(self, replayer):
        trade = make_losing_trade()
        replay = replayer.build_replay(trade)
        assert replay["summary"]["outcome"] == "loss"


class TestBatchReplay:
    """Test batch replay construction."""

    def test_batch_replay(self, replayer):
        trades = [make_winning_trade(), make_losing_trade()]
        replays = replayer.build_batch_replay(trades)
        assert len(replays) == 2
        assert replays[0]["summary"]["outcome"] == "win"
        assert replays[1]["summary"]["outcome"] == "loss"

    def test_batch_replay_empty(self, replayer):
        replays = replayer.build_batch_replay([])
        assert replays == []
