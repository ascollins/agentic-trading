"""Tests for Prometheus metrics helper functions."""

from __future__ import annotations

import pytest

from agentic_trading.observability.metrics import (
    record_signal,
    record_fill,
    record_order,
    update_equity,
    update_drawdown,
    update_position,
    update_daily_pnl,
    update_kill_switch,
    record_candle_processed,
    record_decision_latency,
    SIGNALS_TOTAL,
    FILLS_TOTAL,
    EQUITY,
    DRAWDOWN,
    DAILY_PNL,
    KILL_SWITCH_ACTIVE,
)


class TestMetricsHelpers:
    """Test that metric helper functions correctly update Prometheus counters."""

    def test_record_signal_increments_counter(self):
        """record_signal should increment SIGNALS_TOTAL."""
        before = SIGNALS_TOTAL.labels(
            strategy_id="test_strat",
            symbol="BTC/USDT",
            direction="long",
        )._value.get()

        record_signal("test_strat", "BTC/USDT", "long")

        after = SIGNALS_TOTAL.labels(
            strategy_id="test_strat",
            symbol="BTC/USDT",
            direction="long",
        )._value.get()

        assert after == before + 1

    def test_record_fill_increments_counter(self):
        """record_fill should increment FILLS_TOTAL."""
        before = FILLS_TOTAL.labels(
            symbol="ETH/USDT",
            side="buy",
        )._value.get()

        record_fill("ETH/USDT", "buy")

        after = FILLS_TOTAL.labels(
            symbol="ETH/USDT",
            side="buy",
        )._value.get()

        assert after == before + 1

    def test_update_equity_sets_gauge(self):
        """update_equity should set EQUITY gauge."""
        update_equity(150_000.0)
        assert EQUITY._value.get() == 150_000.0

    def test_update_drawdown_sets_gauge(self):
        """update_drawdown should set DRAWDOWN gauge."""
        update_drawdown(0.05)
        assert DRAWDOWN._value.get() == pytest.approx(0.05)

    def test_update_daily_pnl_sets_gauge(self):
        """update_daily_pnl should set DAILY_PNL gauge."""
        update_daily_pnl(500.0)
        assert DAILY_PNL._value.get() == 500.0

    def test_update_kill_switch_active(self):
        """update_kill_switch should set gauge to 1 when active."""
        update_kill_switch(True)
        assert KILL_SWITCH_ACTIVE._value.get() == 1

    def test_update_kill_switch_inactive(self):
        """update_kill_switch should set gauge to 0 when inactive."""
        update_kill_switch(False)
        assert KILL_SWITCH_ACTIVE._value.get() == 0

    def test_record_decision_latency(self):
        """record_decision_latency should observe without error."""
        # Just verify it doesn't raise
        record_decision_latency(0.05)
        record_decision_latency(0.001)

    def test_record_candle_processed(self):
        """record_candle_processed should not raise."""
        record_candle_processed("BTC/USDT", "1m")

    def test_update_position(self):
        """update_position should not raise."""
        update_position("BTC/USDT", "long", 0.5)
