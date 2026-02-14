"""Tests for TradeExporter — CSV/JSON export and periodic reports."""

import csv
import io
import json
import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from agentic_trading.journal.export import TradeExporter
from agentic_trading.journal.record import (
    TradePhase,
    TradeOutcome,
    TradeRecord,
)

from .conftest import make_fill, make_winning_trade, make_losing_trade


@pytest.fixture
def exporter():
    return TradeExporter(decimal_places=4)


def _make_trades_over_days(n_days: int = 10) -> list[TradeRecord]:
    """Create a series of trades spread over multiple days."""
    trades = []
    for i in range(n_days):
        bt = datetime(2024, 1, 1 + i, 12, 0, 0)
        won = i % 3 != 0
        trade = TradeRecord(
            trace_id=f"tr_{i}",
            strategy_id="trend",
            symbol="BTC/USDT",
            direction="long",
            signal_confidence=0.7,
            initial_risk_price=Decimal("95"),
        )
        trade.add_entry_fill(make_fill(
            fill_id=f"fill_e_{i}", price=100.0, qty=1.0, timestamp=bt,
        ))
        trade.compute_initial_risk()
        exit_px = 110.0 if won else 93.0
        trade.add_exit_fill(make_fill(
            fill_id=f"fill_x_{i}", order_id=f"exit_{i}", side="sell",
            price=exit_px, qty=1.0,
            timestamp=bt + timedelta(hours=2),
        ))
        trades.append(trade)
    return trades


class TestCSVExport:
    """Test CSV export."""

    def test_csv_header(self, exporter):
        trades = [make_winning_trade()]
        csv_str = exporter.to_csv(trades)
        reader = csv.DictReader(io.StringIO(csv_str))
        assert "trade_id" in reader.fieldnames
        assert "net_pnl" in reader.fieldnames
        assert "r_multiple" in reader.fieldnames

    def test_csv_row_count(self, exporter):
        trades = _make_trades_over_days(5)
        csv_str = exporter.to_csv(trades)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 6  # header + 5 data rows

    def test_csv_custom_columns(self, exporter):
        trades = [make_winning_trade()]
        csv_str = exporter.to_csv(trades, columns=["trade_id", "net_pnl"])
        reader = csv.DictReader(io.StringIO(csv_str))
        assert set(reader.fieldnames) == {"trade_id", "net_pnl"}

    def test_csv_winning_trade_values(self, exporter):
        trade = make_winning_trade(entry_price=100.0, exit_price=110.0)
        csv_str = exporter.to_csv([trade])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert float(row["net_pnl"]) > 0
        assert row["direction"] == "long"
        assert row["symbol"] == "BTC/USDT"

    def test_csv_empty_trades(self, exporter):
        csv_str = exporter.to_csv([])
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 0


class TestJSONExport:
    """Test JSON export."""

    def test_json_valid(self, exporter):
        trades = _make_trades_over_days(3)
        json_str = exporter.to_json(trades)
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_json_keys(self, exporter):
        trade = make_winning_trade()
        json_str = exporter.to_json([trade])
        data = json.loads(json_str)
        row = data[0]
        assert "trade_id" in row
        assert "net_pnl" in row
        assert "r_multiple" in row
        assert "strategy_id" in row

    def test_json_losing_trade(self, exporter):
        trade = make_losing_trade()
        json_str = exporter.to_json([trade])
        data = json.loads(json_str)
        assert data[0]["net_pnl"] < 0

    def test_json_indent(self, exporter):
        trade = make_winning_trade()
        json_str = exporter.to_json([trade], indent=4)
        assert "    " in json_str  # 4-space indent


class TestPeriodicReport:
    """Test periodic report generation."""

    def test_daily_report(self, exporter):
        trades = _make_trades_over_days(10)
        report = exporter.periodic_report(trades, period="daily")
        assert report["period"] == "daily"
        assert len(report["buckets"]) == 10
        assert "totals" in report

    def test_weekly_report(self, exporter):
        trades = _make_trades_over_days(14)
        report = exporter.periodic_report(trades, period="weekly")
        assert report["period"] == "weekly"
        assert len(report["buckets"]) >= 2

    def test_monthly_report(self, exporter):
        trades = _make_trades_over_days(10)
        report = exporter.periodic_report(trades, period="monthly")
        assert report["period"] == "monthly"
        assert len(report["buckets"]) >= 1

    def test_report_bucket_stats(self, exporter):
        trades = _make_trades_over_days(5)
        report = exporter.periodic_report(trades, period="daily")
        bucket = report["buckets"][0]
        assert "period_key" in bucket
        assert "trades" in bucket
        assert "wins" in bucket
        assert "losses" in bucket
        assert "win_rate" in bucket
        assert "total_pnl" in bucket
        assert "profit_factor" in bucket
        assert "best_trade" in bucket
        assert "worst_trade" in bucket

    def test_report_totals(self, exporter):
        trades = _make_trades_over_days(10)
        report = exporter.periodic_report(trades, period="daily")
        totals = report["totals"]
        assert totals["trades"] == 10
        assert totals["wins"] + totals["losses"] == 10

    def test_empty_report(self, exporter):
        report = exporter.periodic_report([], period="daily")
        assert report["totals"]["trades"] == 0
        assert report["buckets"] == []

    def test_open_trades_excluded(self, exporter):
        """Only closed trades should appear in periodic reports."""
        trade = TradeRecord(
            trace_id="t1", strategy_id="trend",
            symbol="BTC/USDT", direction="long",
        )
        trade.add_entry_fill(make_fill(price=100.0))
        report = exporter.periodic_report([trade], period="daily")
        assert report["totals"]["trades"] == 0
