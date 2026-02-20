"""Tests for BacktestEngine TradeDetail capture and MAE/MFE tracking.

Verifies that per-trade details are correctly captured during backtest
execution, including entry/exit prices, fees, exit reason, hold time,
and MAE/MFE excursions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agentic_trading.backtester.engine import BacktestEngine, _SimPosition
from agentic_trading.backtester.results import TradeDetail
from agentic_trading.core.models import Candle


# ---------------------------------------------------------------------------
# TradeDetail dataclass tests
# ---------------------------------------------------------------------------


class TestTradeDetail:
    """Test TradeDetail dataclass defaults and construction."""

    def test_default_values(self):
        td = TradeDetail()
        assert td.strategy_id == ""
        assert td.symbol == ""
        assert td.direction == ""
        assert td.entry_price == 0.0
        assert td.exit_price == 0.0
        assert td.return_pct == 0.0
        assert td.mae_pct == 0.0
        assert td.mfe_pct == 0.0
        assert td.exit_reason == ""
        assert td.hold_seconds == 0.0

    def test_custom_values(self):
        td = TradeDetail(
            strategy_id="bb_squeeze",
            symbol="BTC/USDT",
            direction="long",
            entry_price=50000.0,
            exit_price=51000.0,
            return_pct=0.02,
            mae_pct=-0.01,
            mfe_pct=0.03,
            exit_reason="signal",
            hold_seconds=3600.0,
            fee_paid=5.0,
        )
        assert td.strategy_id == "bb_squeeze"
        assert td.symbol == "BTC/USDT"
        assert td.direction == "long"
        assert td.entry_price == 50000.0
        assert td.exit_price == 51000.0
        assert td.return_pct == 0.02
        assert td.exit_reason == "signal"
        assert td.fee_paid == 5.0


# ---------------------------------------------------------------------------
# _SimPosition MAE/MFE fields
# ---------------------------------------------------------------------------


class TestSimPositionMAEMFE:
    """Test that _SimPosition tracks MAE/MFE fields."""

    def test_initial_mae_mfe(self):
        pos = _SimPosition(
            symbol="BTC/USDT",
            side="long",
            qty=1.0,
            entry_price=50000.0,
            mae_price=50000.0,
            mfe_price=50000.0,
            entry_fee=10.0,
        )
        assert pos.mae_price == 50000.0
        assert pos.mfe_price == 50000.0
        assert pos.entry_fee == 10.0

    def test_default_mae_mfe_zero(self):
        pos = _SimPosition(
            symbol="ETH/USDT",
            side="short",
            qty=10.0,
            entry_price=3000.0,
        )
        assert pos.mae_price == 0.0
        assert pos.mfe_price == 0.0
        assert pos.entry_fee == 0.0


# ---------------------------------------------------------------------------
# BacktestEngine trade_details integration
# ---------------------------------------------------------------------------


def _make_candle(
    symbol: str = "BTC/USDT",
    timestamp: datetime | None = None,
    open_: float = 50000.0,
    high: float = 50100.0,
    low: float = 49900.0,
    close: float = 50050.0,
    volume: float = 100.0,
) -> Candle:
    """Create a test candle."""
    ts = timestamp or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Candle(
        symbol=symbol,
        exchange="binance",
        timeframe="1m",
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class TestBacktestTradeDetails:
    """Integration tests for trade detail capture in BacktestEngine."""

    def _make_engine(self) -> BacktestEngine:
        """Create a BacktestEngine with a simple mock strategy."""
        mock_strategy = MagicMock()
        mock_strategy.strategy_id = "test_strat"
        mock_strategy.on_candle.return_value = None

        mock_fe = MagicMock()
        mock_fe.add_candle = MagicMock()
        mock_fe.get_buffer = MagicMock(return_value=[])
        mock_fe.compute_features = MagicMock(
            return_value=MagicMock(features={"atr": 100.0, "adx": 25.0})
        )

        return BacktestEngine(
            strategies=[mock_strategy],
            feature_engine=mock_fe,
            initial_capital=100_000.0,
            slippage_bps=0.0,  # No slippage for test clarity
            fee_maker=0.0001,
            fee_taker=0.0002,
            funding_enabled=False,
            partial_fills=False,
            seed=42,
        )

    @pytest.mark.asyncio
    async def test_trade_details_empty_for_no_trades(self):
        """Verify empty trade_details when no signals are generated."""
        engine = self._make_engine()
        candles = {"BTC/USDT": [_make_candle()]}
        result = await engine.run(candles)
        assert result.trade_details == []

    def test_trade_details_list_on_result(self):
        """Verify BacktestResult has trade_details field."""
        from agentic_trading.backtester.results import BacktestResult

        result = BacktestResult()
        assert hasattr(result, "trade_details")
        assert result.trade_details == []

    def test_update_mae_mfe_long(self):
        """Test MAE/MFE update for a long position."""
        engine = self._make_engine()

        # Manually create a position
        pos = _SimPosition(
            symbol="BTC/USDT",
            side="long",
            qty=1.0,
            entry_price=50000.0,
            mae_price=50000.0,
            mfe_price=50000.0,
        )
        engine._sim_positions["BTC/USDT"] = pos

        # Candle with lower low and higher high
        candle = _make_candle(
            low=49500.0,
            high=50500.0,
        )
        engine._update_mae_mfe(candle)

        assert pos.mae_price == 49500.0  # Tracked lowest
        assert pos.mfe_price == 50500.0  # Tracked highest

    def test_update_mae_mfe_short(self):
        """Test MAE/MFE update for a short position."""
        engine = self._make_engine()

        pos = _SimPosition(
            symbol="BTC/USDT",
            side="short",
            qty=1.0,
            entry_price=50000.0,
            mae_price=50000.0,
            mfe_price=50000.0,
        )
        engine._sim_positions["BTC/USDT"] = pos

        candle = _make_candle(
            low=49500.0,
            high=50500.0,
        )
        engine._update_mae_mfe(candle)

        assert pos.mae_price == 50500.0  # For short, highest is adverse
        assert pos.mfe_price == 49500.0  # For short, lowest is favorable

    def test_update_mae_mfe_only_updates_matching_symbol(self):
        """MAE/MFE should only update for candle's symbol."""
        engine = self._make_engine()

        pos_btc = _SimPosition(
            symbol="BTC/USDT",
            side="long",
            qty=1.0,
            entry_price=50000.0,
            mae_price=50000.0,
            mfe_price=50000.0,
        )
        pos_eth = _SimPosition(
            symbol="ETH/USDT",
            side="long",
            qty=1.0,
            entry_price=3000.0,
            mae_price=3000.0,
            mfe_price=3000.0,
        )
        engine._sim_positions["BTC/USDT"] = pos_btc
        engine._sim_positions["ETH/USDT"] = pos_eth

        # BTC candle should not affect ETH position
        candle = _make_candle(symbol="BTC/USDT", low=49000.0, high=51000.0)
        engine._update_mae_mfe(candle)

        assert pos_btc.mae_price == 49000.0  # Updated
        assert pos_eth.mae_price == 3000.0   # Unchanged
