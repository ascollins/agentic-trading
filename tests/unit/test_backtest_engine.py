"""Tests for BacktestEngine position management.

Verifies:
- No position stacking (same-direction signals don't create new positions)
- Close-then-reverse on opposite-direction signals
- Cash constraint limits position size
- Short trade PnL tracking
- FLAT signals close positions
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agentic_trading.backtester.engine import BacktestEngine, _SimPosition
from agentic_trading.core.enums import Exchange, SignalDirection, Timeframe
from agentic_trading.core.events import FeatureVector, Signal
from agentic_trading.core.models import Candle


def _make_candle(
    close: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    open_: float = 100.0,
    volume: float = 1000.0,
    symbol: str = "BTC/USDT",
    ts: datetime | None = None,
) -> Candle:
    return Candle(
        symbol=symbol,
        exchange=Exchange.BINANCE,
        timeframe=Timeframe.M1,
        timestamp=ts or datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_signal(
    direction: str = "long",
    confidence: float = 0.8,
    symbol: str = "BTC/USDT",
    atr: float = 2.0,
) -> Signal:
    return Signal(
        strategy_id="test",
        symbol=symbol,
        direction=SignalDirection(direction),
        confidence=confidence,
        rationale="test signal",
        risk_constraints={"atr": atr},
    )


def _make_feature_vector(
    symbol: str = "BTC/USDT",
    adx: float = 30.0,
    atr: float = 2.0,
    ema_12: float = 100.0,
    ema_26: float = 98.0,
) -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        timeframe=Timeframe.M1,
        features={
            "ema_12": ema_12,
            "ema_26": ema_26,
            "adx": adx,
            "atr": atr,
            "close": 100.0,
        },
    )


class TestPositionManagement:
    """Test position-aware signal processing."""

    def _make_engine(self, initial_capital: float = 100_000.0) -> BacktestEngine:
        """Create a minimal BacktestEngine for testing."""
        from agentic_trading.strategies.trend_following import TrendFollowingStrategy

        return BacktestEngine(
            strategies=[TrendFollowingStrategy()],
            feature_engine=None,
            initial_capital=initial_capital,
            slippage_model="fixed_bps",
            slippage_bps=0.0,  # No slippage for easier testing
            fee_maker=0.0,
            fee_taker=0.0,
            funding_enabled=False,
            partial_fills=False,
            seed=42,
        )

    def test_no_position_stacking(self):
        """Same-direction signal should not create a new position."""
        engine = self._make_engine()
        candle = _make_candle(close=100.0)

        # Open a long position
        long_signal = _make_signal(direction="long")
        engine._process_signal(long_signal, candle)

        assert "BTC/USDT" in engine._sim_positions
        initial_qty = engine._sim_positions["BTC/USDT"].qty

        # Same direction signal should be skipped
        long_signal2 = _make_signal(direction="long")
        engine._process_signal(long_signal2, candle)

        # Position should be unchanged
        assert engine._sim_positions["BTC/USDT"].qty == initial_qty

    def test_close_then_reverse(self):
        """Opposite-direction signal should close existing and open new."""
        engine = self._make_engine()
        candle = _make_candle(close=100.0)

        # Open a long position
        long_signal = _make_signal(direction="long")
        engine._process_signal(long_signal, candle)

        assert engine._sim_positions["BTC/USDT"].side == "long"
        initial_trades = len(engine._trade_returns)

        # Opposite direction: close long, open short
        short_signal = _make_signal(direction="short")
        engine._process_signal(short_signal, candle)

        pos = engine._sim_positions.get("BTC/USDT")
        assert pos is not None
        assert pos.side == "short"
        # The close should have logged a trade return
        assert len(engine._trade_returns) > initial_trades

    def test_flat_signal_closes_position(self):
        """FLAT signal should close existing position."""
        engine = self._make_engine()
        candle = _make_candle(close=100.0)

        # Open a long position
        long_signal = _make_signal(direction="long")
        engine._process_signal(long_signal, candle)

        assert "BTC/USDT" in engine._sim_positions

        # FLAT signal should close
        flat_signal = _make_signal(direction="flat")
        engine._process_signal(flat_signal, candle)

        assert "BTC/USDT" not in engine._sim_positions

    def test_cash_constraint(self):
        """Position size should be limited by available cash."""
        engine = self._make_engine(initial_capital=100.0)  # Very small capital
        candle = _make_candle(close=50000.0)  # Expensive asset

        long_signal = _make_signal(direction="long", confidence=1.0, atr=100.0)
        engine._process_signal(long_signal, candle)

        pos = engine._sim_positions.get("BTC/USDT")
        if pos is not None:
            # Position value should not exceed cash
            assert pos.qty * candle.close <= 100.0 * 1.05  # 5% buffer

    def test_short_trade_pnl_tracked(self):
        """Short trade PnL should be recorded correctly."""
        engine = self._make_engine()

        # Open short at 100
        candle_entry = _make_candle(close=100.0)
        short_signal = _make_signal(direction="short")
        engine._process_signal(short_signal, candle_entry)

        assert "BTC/USDT" in engine._sim_positions
        initial_returns = len(engine._trade_returns)

        # Close short at 90 (profit for short)
        candle_exit = _make_candle(close=90.0)
        long_signal = _make_signal(direction="long")
        engine._process_signal(long_signal, candle_exit)

        # Should have recorded a trade return
        assert len(engine._trade_returns) > initial_returns

    def test_no_position_opens_new(self):
        """Signal with no existing position should open a new one."""
        engine = self._make_engine()
        candle = _make_candle(close=100.0)

        assert len(engine._sim_positions) == 0

        long_signal = _make_signal(direction="long")
        engine._process_signal(long_signal, candle)

        assert "BTC/USDT" in engine._sim_positions
        assert engine._sim_positions["BTC/USDT"].side == "long"

    def test_equity_tracking_long(self):
        """Equity update reflects unrealized PnL for long positions."""
        engine = self._make_engine(initial_capital=100_000.0)

        # Open long at 100
        candle1 = _make_candle(close=100.0)
        long_signal = _make_signal(direction="long")
        engine._process_signal(long_signal, candle1)

        # Record equity after opening
        engine._update_equity(candle1)
        equity_at_open = engine._equity

        # Price goes up to 110 — equity should increase
        candle2 = _make_candle(close=110.0)
        engine._update_equity(candle2)

        assert engine._equity > equity_at_open

    def test_equity_tracking_short(self):
        """Equity update reflects unrealized PnL for short positions."""
        engine = self._make_engine(initial_capital=100_000.0)

        # Open short at 100
        candle1 = _make_candle(close=100.0)
        short_signal = _make_signal(direction="short")
        engine._process_signal(short_signal, candle1)

        # Record equity after opening
        engine._update_equity(candle1)
        equity_at_open = engine._equity

        # Price goes down to 90 (profit for short) — equity should increase
        candle2 = _make_candle(close=90.0)
        engine._update_equity(candle2)

        assert engine._equity > equity_at_open


class TestFullBacktest:
    """Integration test: run a full backtest and verify reasonable results."""

    @pytest.mark.asyncio
    async def test_backtest_produces_reasonable_returns(self):
        """Backtest should not produce extreme returns like -1707%."""
        from agentic_trading.features.engine import FeatureEngine
        from agentic_trading.strategies.trend_following import TrendFollowingStrategy

        # Create 500 candles with a simple uptrend
        candles = []
        base_price = 100.0
        for i in range(500):
            # Gentle uptrend with noise
            price = base_price + i * 0.01 + (i % 7 - 3) * 0.5
            candles.append(Candle(
                symbol="TEST/USDT",
                exchange=Exchange.BINANCE,
                timeframe=Timeframe.M1,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i),
                open=price - 0.1,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1000.0,
            ))

        engine = BacktestEngine(
            strategies=[TrendFollowingStrategy()],
            feature_engine=FeatureEngine(),
            initial_capital=100_000.0,
            fee_maker=0.0002,
            fee_taker=0.0004,
            seed=42,
        )

        result = await engine.run({"TEST/USDT": candles})

        # Return should be within reasonable bounds (not -1707%)
        assert result.total_return > -1.0, f"Total return {result.total_return} is too negative"
        assert result.total_return < 10.0, f"Total return {result.total_return} is unrealistically high"
