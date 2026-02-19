"""Tests for the FactTable component."""

from __future__ import annotations

import threading

from agentic_trading.context.fact_table import (
    FactTable,
    FactTableSnapshot,
    PortfolioSnapshot,
    PriceLevels,
    RiskSnapshot,
)


def _make_fact_table() -> FactTable:
    """Create a clean FactTable for testing."""
    return FactTable()


class TestPriceLevels:
    def test_defaults(self):
        p = PriceLevels(symbol="BTC/USDT")
        assert p.symbol == "BTC/USDT"
        assert p.bid == 0.0
        assert p.ask == 0.0
        assert p.last == 0.0
        assert p.funding_rate == 0.0


class TestFactTablePrices:
    def test_get_price_returns_none_for_unknown(self):
        ft = _make_fact_table()
        assert ft.get_price("BTC/USDT") is None

    def test_update_price_creates_entry(self):
        ft = _make_fact_table()
        ft.update_price("BTC/USDT", last=65000.0, bid=64999.0)
        result = ft.get_price("BTC/USDT")
        assert result is not None
        assert result.last == 65000.0
        assert result.bid == 64999.0
        assert result.symbol == "BTC/USDT"

    def test_update_price_partial_update(self):
        ft = _make_fact_table()
        ft.update_price("BTC/USDT", last=65000.0, bid=64999.0)
        ft.update_price("BTC/USDT", last=65500.0)
        result = ft.get_price("BTC/USDT")
        assert result is not None
        assert result.last == 65500.0
        assert result.bid == 64999.0  # Preserved from first update

    def test_get_all_prices(self):
        ft = _make_fact_table()
        ft.update_price("BTC/USDT", last=65000.0)
        ft.update_price("ETH/USDT", last=3500.0)
        prices = ft.get_all_prices()
        assert len(prices) == 2
        assert prices["BTC/USDT"].last == 65000.0
        assert prices["ETH/USDT"].last == 3500.0

    def test_get_price_returns_copy(self):
        ft = _make_fact_table()
        ft.update_price("BTC/USDT", last=65000.0)
        p1 = ft.get_price("BTC/USDT")
        p2 = ft.get_price("BTC/USDT")
        assert p1 is not p2
        assert p1 == p2


class TestFactTableRisk:
    def test_default_risk(self):
        ft = _make_fact_table()
        risk = ft.get_risk()
        assert risk.max_portfolio_leverage == 3.0
        assert risk.kill_switch_active is False

    def test_update_risk_partial(self):
        ft = _make_fact_table()
        ft.update_risk(kill_switch_active=True, current_drawdown_pct=0.03)
        risk = ft.get_risk()
        assert risk.kill_switch_active is True
        assert risk.current_drawdown_pct == 0.03
        assert risk.max_portfolio_leverage == 3.0  # Preserved

    def test_get_risk_returns_copy(self):
        ft = _make_fact_table()
        r1 = ft.get_risk()
        r2 = ft.get_risk()
        assert r1 is not r2


class TestFactTablePortfolio:
    def test_default_portfolio(self):
        ft = _make_fact_table()
        port = ft.get_portfolio()
        assert port.total_equity == 0.0
        assert port.open_position_count == 0

    def test_update_portfolio(self):
        ft = _make_fact_table()
        snapshot = PortfolioSnapshot(
            total_equity=100_000.0,
            gross_exposure=30_000.0,
            open_position_count=2,
        )
        ft.update_portfolio(snapshot)
        result = ft.get_portfolio()
        assert result.total_equity == 100_000.0
        assert result.open_position_count == 2

    def test_update_portfolio_fields(self):
        ft = _make_fact_table()
        ft.update_portfolio_fields(total_equity=50_000.0, daily_pnl=500.0)
        result = ft.get_portfolio()
        assert result.total_equity == 50_000.0
        assert result.daily_pnl == 500.0


class TestFactTableRegime:
    def test_get_regime_empty(self):
        ft = _make_fact_table()
        assert ft.get_regime("BTC/USDT") == {}

    def test_update_regime(self):
        ft = _make_fact_table()
        ft.update_regime("BTC/USDT", {"type": "trend", "confidence": 0.85})
        result = ft.get_regime("BTC/USDT")
        assert result["type"] == "trend"
        assert result["confidence"] == 0.85


class TestFactTableSnapshot:
    def test_snapshot_captures_all(self):
        ft = _make_fact_table()
        ft.update_price("BTC/USDT", last=65000.0)
        ft.update_risk(kill_switch_active=True)
        ft.update_portfolio_fields(total_equity=100_000.0)
        ft.update_regime("BTC/USDT", {"type": "trend"})

        snap = ft.snapshot()
        assert isinstance(snap, FactTableSnapshot)
        assert "BTC/USDT" in snap.prices
        assert snap.risk.kill_switch_active is True
        assert snap.portfolio.total_equity == 100_000.0
        assert snap.regimes["BTC/USDT"]["type"] == "trend"

    def test_snapshot_is_isolated(self):
        ft = _make_fact_table()
        ft.update_price("BTC/USDT", last=65000.0)
        snap = ft.snapshot()
        ft.update_price("BTC/USDT", last=70000.0)
        assert snap.prices["BTC/USDT"].last == 65000.0


class TestFactTableClear:
    def test_clear_resets_all(self):
        ft = _make_fact_table()
        ft.update_price("BTC/USDT", last=65000.0)
        ft.update_risk(kill_switch_active=True)
        ft.clear()
        assert ft.get_price("BTC/USDT") is None
        assert ft.get_risk().kill_switch_active is False


class TestFactTableThreadSafety:
    def test_concurrent_reads_and_writes(self):
        ft = _make_fact_table()
        errors: list[str] = []

        def writer():
            for i in range(100):
                ft.update_price("BTC/USDT", last=float(60000 + i))

        def reader():
            for _ in range(100):
                result = ft.get_price("BTC/USDT")
                if result is not None and not isinstance(result, PriceLevels):
                    errors.append("Got non-PriceLevels type")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
