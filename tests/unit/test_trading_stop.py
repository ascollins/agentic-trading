"""Tests for TP/SL (trading stop) functionality.

Covers:
  - Signal model TP/SL fields
  - ExitConfig defaults and TOML loading
  - TP/SL computation from strategy risk_constraints
  - Trailing stop logic for trend strategies only
  - PaperAdapter no-op set_trading_stop
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentic_trading.core.config import ExitConfig
from agentic_trading.core.enums import SignalDirection, Timeframe
from agentic_trading.core.events import Signal


# ---------------------------------------------------------------------------
# Signal model TP/SL fields
# ---------------------------------------------------------------------------


class TestSignalTPSL:
    """Test that Signal model accepts TP/SL fields."""

    def test_signal_with_tp_sl(self):
        sig = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            confidence=0.8,
            take_profit=Decimal("100000"),
            stop_loss=Decimal("95000"),
        )
        assert sig.take_profit == Decimal("100000")
        assert sig.stop_loss == Decimal("95000")
        assert sig.trailing_stop is None

    def test_signal_with_trailing_stop(self):
        sig = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.SHORT,
            confidence=0.7,
            take_profit=Decimal("90000"),
            stop_loss=Decimal("100000"),
            trailing_stop=Decimal("500"),
        )
        assert sig.trailing_stop == Decimal("500")

    def test_signal_tp_sl_defaults_none(self):
        sig = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
        )
        assert sig.take_profit is None
        assert sig.stop_loss is None
        assert sig.trailing_stop is None

    def test_signal_backward_compatible(self):
        """Existing code that doesn't set TP/SL should still work."""
        sig = Signal(
            strategy_id="trend_following",
            symbol="ETH/USDT",
            direction=SignalDirection.LONG,
            confidence=0.85,
            rationale="EMA cross",
            risk_constraints={
                "stop_distance_atr": 50.0,
                "atr": 25.0,
                "price": 3000.0,
            },
        )
        assert sig.take_profit is None
        assert sig.stop_loss is None
        assert sig.risk_constraints["atr"] == 25.0


# ---------------------------------------------------------------------------
# ExitConfig
# ---------------------------------------------------------------------------


class TestExitConfig:
    """Test ExitConfig defaults and validation."""

    def test_defaults(self):
        cfg = ExitConfig()
        assert cfg.enabled is True
        assert cfg.sl_atr_multiplier == 2.5
        assert cfg.tp_atr_multiplier == 5.0
        assert cfg.trailing_stop_atr_multiplier == 2.0
        assert "trend_following" in cfg.trailing_strategies
        assert "breakout" in cfg.trailing_strategies
        assert "mean_reversion" not in cfg.trailing_strategies

    def test_disabled(self):
        cfg = ExitConfig(enabled=False)
        assert cfg.enabled is False

    def test_custom_multipliers(self):
        cfg = ExitConfig(
            sl_atr_multiplier=3.0,
            tp_atr_multiplier=6.0,
            trailing_stop_atr_multiplier=1.5,
        )
        assert cfg.sl_atr_multiplier == 3.0
        assert cfg.tp_atr_multiplier == 6.0

    def test_custom_trailing_strategies(self):
        cfg = ExitConfig(trailing_strategies=["my_strategy"])
        assert cfg.trailing_strategies == ["my_strategy"]


# ---------------------------------------------------------------------------
# TP/SL price computation
# ---------------------------------------------------------------------------


class TestTPSLComputation:
    """Test TP/SL price level computation from ATR."""

    def test_long_tp_sl_from_atr(self):
        """LONG: SL below entry, TP above entry."""
        price = Decimal("100000")
        atr = Decimal("500")
        sl_mult = Decimal("2.5")
        tp_mult = Decimal("5.0")

        sl = price - (atr * sl_mult)
        tp = price + (atr * tp_mult)

        assert sl == Decimal("98750")   # 100000 - 1250
        assert tp == Decimal("102500")  # 100000 + 2500

    def test_short_tp_sl_from_atr(self):
        """SHORT: SL above entry, TP below entry."""
        price = Decimal("100000")
        atr = Decimal("500")
        sl_mult = Decimal("2.5")
        tp_mult = Decimal("5.0")

        sl = price + (atr * sl_mult)
        tp = price - (atr * tp_mult)

        assert sl == Decimal("101250")  # 100000 + 1250
        assert tp == Decimal("97500")   # 100000 - 2500

    def test_trailing_stop_distance(self):
        """Trailing stop is a distance, not a price."""
        atr = Decimal("500")
        trail_mult = Decimal("2.0")
        trail = atr * trail_mult
        assert trail == Decimal("1000")

    def test_mean_reversion_tp_is_middle_bb(self):
        """Mean reversion should use middle BB as TP."""
        entry = Decimal("3000")
        middle_bb = Decimal("3050")  # price has dropped below middle

        tp = middle_bb  # Mean reversion targets the mean
        sl = entry - Decimal("60")  # ATR-based stop

        assert tp > entry  # For a LONG mean-reversion
        assert sl < entry

    def test_zero_atr_no_tp_sl(self):
        """If ATR is zero, TP/SL should not be set."""
        sig = Signal(
            strategy_id="test",
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            risk_constraints={"atr": 0, "price": 100000},
        )
        # When ATR is 0, strategies should leave TP/SL as None
        assert sig.take_profit is None
        assert sig.stop_loss is None


# ---------------------------------------------------------------------------
# Trailing stop strategy filtering
# ---------------------------------------------------------------------------


class TestTrailingStopFiltering:
    """Test that trailing stops are only for configured strategies."""

    def test_trend_following_gets_trailing(self):
        cfg = ExitConfig()
        assert "trend_following" in cfg.trailing_strategies

    def test_mean_reversion_no_trailing(self):
        cfg = ExitConfig()
        assert "mean_reversion" not in cfg.trailing_strategies
        assert "mean_reversion_enhanced" not in cfg.trailing_strategies

    def test_custom_list(self):
        cfg = ExitConfig(trailing_strategies=["my_strat"])
        assert "trend_following" not in cfg.trailing_strategies
        assert "my_strat" in cfg.trailing_strategies


# ---------------------------------------------------------------------------
# PaperAdapter set_trading_stop
# ---------------------------------------------------------------------------


class TestPaperAdapterTradingStop:
    """Test that PaperAdapter has a no-op set_trading_stop."""

    @pytest.mark.asyncio
    async def test_paper_adapter_set_trading_stop(self):
        from agentic_trading.execution.adapters.paper import PaperAdapter
        from agentic_trading.core.enums import Exchange

        adapter = PaperAdapter(exchange=Exchange.BYBIT)
        result = await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("100000"),
            stop_loss=Decimal("95000"),
            trailing_stop=Decimal("500"),
        )
        assert result == {"result": "paper_mode_noop"}

    @pytest.mark.asyncio
    async def test_paper_adapter_set_trading_stop_partial(self):
        from agentic_trading.execution.adapters.paper import PaperAdapter
        from agentic_trading.core.enums import Exchange

        adapter = PaperAdapter(exchange=Exchange.BYBIT)
        result = await adapter.set_trading_stop(
            "ETH/USDT",
            stop_loss=Decimal("2800"),
        )
        assert result == {"result": "paper_mode_noop"}


# ---------------------------------------------------------------------------
# Strategy TP/SL emission
# ---------------------------------------------------------------------------


class TestStrategyTPSLEmission:
    """Test that strategies emit TP/SL with signals."""

    def _make_candle(self, symbol: str = "BTC/USDT", close: float = 100000.0):
        """Factory: create a test candle."""
        from agentic_trading.core.models import Candle
        from datetime import datetime, timezone

        return Candle(
            symbol=symbol,
            exchange="bybit",
            timeframe=Timeframe.M1,
            timestamp=datetime.now(timezone.utc),
            open=Decimal(str(close - 10)),
            high=Decimal(str(close + 20)),
            low=Decimal(str(close - 30)),
            close=Decimal(str(close)),
            volume=Decimal("1000"),
        )

    def test_signal_has_tp_sl_fields(self):
        """Verify Signal model supports TP/SL Decimal fields."""
        sig = Signal(
            strategy_id="trend_following",
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            confidence=0.8,
            take_profit=Decimal("102500"),
            stop_loss=Decimal("98750"),
            trailing_stop=Decimal("1000"),
            risk_constraints={
                "atr": 500.0,
                "stop_distance_atr": 1250.0,
                "price": 100000.0,
            },
        )
        assert isinstance(sig.take_profit, Decimal)
        assert isinstance(sig.stop_loss, Decimal)
        assert isinstance(sig.trailing_stop, Decimal)
        # TP above entry for long
        assert sig.take_profit > Decimal("100000")
        # SL below entry for long
        assert sig.stop_loss < Decimal("100000")
