"""Tests for CorrelationMatrix — cross-strategy and cross-asset correlation."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from agentic_trading.journal.correlation import CorrelationMatrix
from agentic_trading.journal.record import (
    FillLeg,
    TradePhase,
    TradeRecord,
)

from .conftest import make_fill, make_winning_trade, make_losing_trade


@pytest.fixture
def matrix():
    return CorrelationMatrix(max_days=365)


def _make_trade_on_day(
    day_offset: int,
    strategy_id: str = "trend",
    symbol: str = "BTC/USDT",
    pnl_positive: bool = True,
) -> TradeRecord:
    """Create a closed trade on a specific day offset from base date."""
    bt = datetime(2024, 1, 1 + day_offset, 12, 0, 0)
    entry = 100.0
    exit_price = 110.0 if pnl_positive else 90.0

    trade = TradeRecord(
        trace_id=f"tr_{strategy_id}_{day_offset}",
        strategy_id=strategy_id,
        symbol=symbol,
        direction="long",
        signal_confidence=0.7,
        initial_risk_price=Decimal("95"),
    )
    trade.add_entry_fill(make_fill(price=entry, qty=1.0, timestamp=bt))
    trade.compute_initial_risk()
    trade.add_exit_fill(make_fill(
        fill_id="exit_1", order_id="exit_order", side="sell",
        price=exit_price, qty=1.0,
        timestamp=bt + timedelta(hours=1),
    ))
    return trade


class TestCorrelationBasics:
    """Test basic correlation functionality."""

    def test_empty_report(self, matrix):
        report = matrix.report()
        assert report["strategy_correlation"] == {}
        assert report["symbol_correlation"] == {}
        assert report["data_days"] == 0

    def test_single_strategy_no_correlation(self, matrix):
        """A single strategy produces no correlation pairs."""
        for day in range(10):
            matrix.add_trade(_make_trade_on_day(day, strategy_id="trend"))
        report = matrix.report()
        assert report["strategy_correlation"] == {}
        assert len(report["strategy_ids"]) == 1

    def test_ignores_open_trades(self, matrix):
        trade = TradeRecord(
            trace_id="t1", strategy_id="trend",
            symbol="BTC/USDT", direction="long",
        )
        trade.add_entry_fill(make_fill(price=100.0))
        matrix.add_trade(trade)
        report = matrix.report()
        assert report["data_days"] == 0


class TestStrategyCorrelation:
    """Test cross-strategy correlation."""

    def test_two_strategies_correlated(self, matrix):
        """Two strategies that win on the same days should have positive correlation."""
        for day in range(20):
            win = day % 3 != 0  # Win 2/3 of the time
            matrix.add_trade(_make_trade_on_day(day, "strat_a", pnl_positive=win))
            matrix.add_trade(_make_trade_on_day(day, "strat_b", pnl_positive=win))

        report = matrix.report()
        assert "strat_a|strat_b" in report["strategy_correlation"]
        r = report["strategy_correlation"]["strat_a|strat_b"]
        # Both win/lose on same days → positive correlation
        assert r > 0.5

    def test_anticorrelated_strategies(self, matrix):
        """Two strategies with opposite outcomes should have negative correlation."""
        for day in range(20):
            win_a = day % 2 == 0
            matrix.add_trade(_make_trade_on_day(day, "strat_a", pnl_positive=win_a))
            matrix.add_trade(_make_trade_on_day(day, "strat_b", pnl_positive=not win_a))

        report = matrix.report()
        r = report["strategy_correlation"]["strat_a|strat_b"]
        assert r < -0.5

    def test_insufficient_data_returns_zero(self, matrix):
        """With very few data points, correlation should return 0."""
        for day in range(3):  # Only 3 days (< 5 required)
            matrix.add_trade(_make_trade_on_day(day, "strat_a"))
            matrix.add_trade(_make_trade_on_day(day, "strat_b"))

        report = matrix.report()
        r = report["strategy_correlation"].get("strat_a|strat_b", 0.0)
        assert r == 0.0


class TestSymbolCorrelation:
    """Test cross-symbol correlation."""

    def test_two_symbols_correlated(self, matrix):
        """Two symbols that move together should have positive correlation."""
        for day in range(20):
            win = day % 3 != 0
            matrix.add_trade(_make_trade_on_day(
                day, "trend", symbol="BTC/USDT", pnl_positive=win
            ))
            matrix.add_trade(_make_trade_on_day(
                day, "trend", symbol="ETH/USDT", pnl_positive=win
            ))

        report = matrix.report()
        assert "BTC/USDT|ETH/USDT" in report["symbol_correlation"]
        r = report["symbol_correlation"]["BTC/USDT|ETH/USDT"]
        assert r > 0.5


class TestReportStructure:
    """Test report structure and contents."""

    def test_report_keys(self, matrix):
        for day in range(10):
            matrix.add_trade(_make_trade_on_day(day, "strat_a"))
            matrix.add_trade(_make_trade_on_day(day, "strat_b"))

        report = matrix.report()
        assert "strategy_correlation" in report
        assert "symbol_correlation" in report
        assert "strategy_ids" in report
        assert "symbols" in report
        assert "data_days" in report

    def test_data_days_count(self, matrix):
        for day in range(15):
            matrix.add_trade(_make_trade_on_day(day, "trend"))

        report = matrix.report()
        assert report["data_days"] == 15

    def test_correlation_bounded(self, matrix):
        """Correlation values should be between -1 and 1."""
        for day in range(30):
            win = day % 2 == 0
            matrix.add_trade(_make_trade_on_day(day, "a", pnl_positive=win))
            matrix.add_trade(_make_trade_on_day(day, "b", pnl_positive=not win))

        report = matrix.report()
        for key, r in report["strategy_correlation"].items():
            assert -1.0 <= r <= 1.0


class TestMaxDays:
    """Test max_days limiting."""

    def test_max_days_limits_data(self):
        matrix = CorrelationMatrix(max_days=10)
        for day in range(30):
            matrix.add_trade(_make_trade_on_day(day, "trend"))

        report = matrix.report()
        assert report["data_days"] <= 10
