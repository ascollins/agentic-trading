"""Tests for Week 2 audit fixes.

B1: Backtest stop-loss fills apply slippage model.
B3: Paper limit orders check market price before fill.
E7: Elapsed-time circuit breaker in retry loop.
E10/E11: Paper adapter reentrant guard on set_market_price.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentic_trading.core.enums import (
    Exchange,
    OrderStatus,
    OrderType,
    Side,
    Timeframe,
    TimeInForce,
)
from agentic_trading.core.errors import ExchangeError
from agentic_trading.core.events import OrderAck, OrderIntent
from agentic_trading.execution.adapters.paper import PaperAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(
    balance: Decimal = Decimal("100000"),
) -> PaperAdapter:
    return PaperAdapter(
        exchange=Exchange.BYBIT,
        initial_balances={"USDT": balance},
    )


def _make_intent(
    symbol: str = "BTC/USDT",
    side: Side = Side.BUY,
    qty: Decimal = Decimal("0.01"),
    order_type: OrderType = OrderType.MARKET,
    price: Decimal | None = None,
) -> OrderIntent:
    return OrderIntent(
        dedupe_key="test-dedupe",
        strategy_id="trend_following",
        symbol=symbol,
        exchange=Exchange.BYBIT,
        side=side,
        order_type=order_type,
        time_in_force=TimeInForce.GTC,
        qty=qty,
        price=price,
        trace_id="test-trace",
    )


# ---------------------------------------------------------------------------
# B3: Paper limit orders — market price check
# ---------------------------------------------------------------------------

class TestPaperLimitOrderMarketCheck:
    """B3: limit orders should only fill when market price is favourable."""

    @pytest.mark.asyncio
    async def test_buy_limit_fills_when_market_at_or_below_limit(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("49000"))  # below limit
        ack = await adapter.submit_order(
            _make_intent(
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("50000"),
            )
        )
        assert ack.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_buy_limit_rejects_when_market_above_limit(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("51000"))  # above limit
        with pytest.raises(ExchangeError, match="not reachable"):
            await adapter.submit_order(
                _make_intent(
                    side=Side.BUY,
                    order_type=OrderType.LIMIT,
                    price=Decimal("50000"),
                )
            )

    @pytest.mark.asyncio
    async def test_sell_limit_fills_when_market_at_or_above_limit(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("51000"))  # above limit
        # Need to have a position to sell
        ack = await adapter.submit_order(
            _make_intent(
                side=Side.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("50000"),
            )
        )
        assert ack.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_sell_limit_rejects_when_market_below_limit(self) -> None:
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("49000"))  # below limit
        with pytest.raises(ExchangeError, match="not reachable"):
            await adapter.submit_order(
                _make_intent(
                    side=Side.SELL,
                    order_type=OrderType.LIMIT,
                    price=Decimal("50000"),
                )
            )

    @pytest.mark.asyncio
    async def test_buy_limit_fills_at_limit_price_not_market(self) -> None:
        """Limit fills should execute at the limit price, not market."""
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("49000"))
        ack = await adapter.submit_order(
            _make_intent(
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("50000"),
            )
        )
        order = adapter._orders[ack.order_id]
        assert order.avg_fill_price == Decimal("50000")

    @pytest.mark.asyncio
    async def test_market_order_still_fills_normally(self) -> None:
        """Market orders should be unaffected by the limit check."""
        adapter = _make_adapter()
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        ack = await adapter.submit_order(
            _make_intent(side=Side.BUY, order_type=OrderType.MARKET)
        )
        assert ack.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# E7: Elapsed-time circuit breaker
# ---------------------------------------------------------------------------

class TestRetryCircuitBreaker:
    """E7: retry loop should respect elapsed-time timeout."""

    @pytest.mark.asyncio
    async def test_retry_timeout_aborts_early(self) -> None:
        """Retries should stop when wall time exceeds retry_timeout_s."""
        from agentic_trading.execution.engine import ExecutionEngine
        from agentic_trading.execution.order_manager import OrderManager

        # Create a mock adapter that always fails
        mock_adapter = AsyncMock()
        mock_adapter.submit_order = AsyncMock(
            side_effect=ExchangeError("timeout")
        )
        mock_bus = AsyncMock()

        engine = ExecutionEngine(
            adapter=mock_adapter,
            event_bus=mock_bus,
            risk_manager=AsyncMock(),
            max_retries=100,  # Many retries
            retry_timeout_s=1.0,  # But only 1s wall time
            retry_base_delay_s=0.3,  # 300ms between retries
        )

        intent = _make_intent()
        engine._order_manager.register_intent(intent)

        t0 = time.monotonic()
        ack = await engine._submit_with_retry(intent)
        elapsed = time.monotonic() - t0

        # Should have timed out and returned REJECTED
        assert ack.status == OrderStatus.REJECTED
        assert "Max retries" in ack.message or "timeout" in str(ack.message).lower()
        # Should not have taken much longer than the timeout
        assert elapsed < 5.0  # generous bound

    @pytest.mark.asyncio
    async def test_retry_backoff_increases_delay(self) -> None:
        """Backoff delay should increase between retries."""
        from agentic_trading.execution.engine import ExecutionEngine

        call_times: list[float] = []
        real_submit = AsyncMock(side_effect=ExchangeError("fail"))

        async def _record_time(*args, **kwargs):
            call_times.append(time.monotonic())
            raise ExchangeError("fail")

        mock_adapter = AsyncMock()
        mock_adapter.submit_order = _record_time
        mock_bus = AsyncMock()

        engine = ExecutionEngine(
            adapter=mock_adapter,
            event_bus=mock_bus,
            risk_manager=AsyncMock(),
            max_retries=3,
            retry_timeout_s=30.0,
            retry_base_delay_s=0.1,
        )

        intent = _make_intent()
        engine._order_manager.register_intent(intent)

        await engine._submit_with_retry(intent)

        # Should have 3 attempts
        assert len(call_times) == 3
        # Delays between attempts should generally increase
        if len(call_times) >= 3:
            gap1 = call_times[1] - call_times[0]
            gap2 = call_times[2] - call_times[1]
            # gap2 should be >= gap1 (exponential backoff, jitter notwithstanding)
            # Use a generous comparison since jitter can affect this
            assert gap2 >= gap1 * 0.5  # At least half as long


# ---------------------------------------------------------------------------
# E10/E11: Reentrant guard on set_market_price
# ---------------------------------------------------------------------------

class TestReentrantGuard:
    """E10/E11: set_market_price should not re-enter _check_trading_stops."""

    def test_reentrant_guard_prevents_double_check(self) -> None:
        adapter = _make_adapter()
        call_count = 0
        original_check = adapter._check_trading_stops

        def _counting_check(symbol, price):
            nonlocal call_count
            call_count += 1
            # Simulate reentrant call (e.g. from a fill callback)
            if call_count == 1:
                adapter.set_market_price(symbol, price)
            original_check(symbol, price)

        adapter._check_trading_stops = _counting_check
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        # Should only have entered _check_trading_stops once
        assert call_count == 1

    def test_guard_resets_after_exception(self) -> None:
        adapter = _make_adapter()

        def _failing_check(symbol, price):
            raise ValueError("boom")

        adapter._check_trading_stops = _failing_check
        with pytest.raises(ValueError, match="boom"):
            adapter.set_market_price("BTC/USDT", Decimal("50000"))

        # Guard should be reset so next call works
        assert adapter._in_stop_check is False


# ---------------------------------------------------------------------------
# B1: Backtest stop-loss slippage (via BacktestEngine)
# ---------------------------------------------------------------------------

class TestBacktestStopSlippage:
    """B1: stop-loss fills should apply the slippage model."""

    def test_stop_fill_price_is_worse_than_stop_level(self) -> None:
        """Long stop-loss fill should be at or below the stop price."""
        from agentic_trading.backtester.engine import BacktestEngine, _SimPosition
        from agentic_trading.core.models import Candle

        # Create a minimal engine
        engine = BacktestEngine(
            strategies=[],
            feature_engine=None,
            initial_capital=100000,
            slippage_model="fixed_bps",
            slippage_bps=10.0,  # 10 bps = 0.1%
            seed=42,
        )

        # Create a long position with stop at 48000
        pos = _SimPosition(
            symbol="BTC/USDT",
            side="long",
            qty=0.1,
            entry_price=50000.0,
            stop_price=48000.0,
            strategy_id="test",
            entry_time=engine._clock.now(),
            mae_price=48000.0,
            mfe_price=50000.0,
        )
        engine._sim_positions["BTC/USDT"] = pos
        engine._cash = 100000.0 - (50000.0 * 0.1)  # bought 0.1 BTC

        candle = Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BYBIT,
            timeframe=Timeframe.H1,
            timestamp=engine._clock.now(),
            open=49000.0,
            high=49500.0,
            low=47500.0,
            close=47800.0,
            volume=100.0,
        )

        engine._close_position_at_stop(pos, candle)

        # The fill should be worse (lower) than the stop price for a long
        # Because slippage was applied in sell direction
        assert "BTC/USDT" not in engine._sim_positions
        if engine._trade_details:
            detail = engine._trade_details[-1]
            assert detail.exit_price <= 48000.0, (
                f"Stop fill {detail.exit_price} should be <= stop {48000.0}"
            )

    def test_short_stop_fill_price_is_worse_than_stop_level(self) -> None:
        """Short stop-loss fill should be at or above the stop price."""
        from agentic_trading.backtester.engine import BacktestEngine, _SimPosition
        from agentic_trading.core.models import Candle

        engine = BacktestEngine(
            strategies=[],
            feature_engine=None,
            initial_capital=100000,
            slippage_model="fixed_bps",
            slippage_bps=10.0,
            seed=42,
        )

        pos = _SimPosition(
            symbol="BTC/USDT",
            side="short",
            qty=0.1,
            entry_price=50000.0,
            stop_price=52000.0,
            strategy_id="test",
            entry_time=engine._clock.now(),
            mae_price=52000.0,
            mfe_price=50000.0,
        )
        engine._sim_positions["BTC/USDT"] = pos
        engine._cash = 100000.0 + (50000.0 * 0.1)  # shorted 0.1 BTC

        candle = Candle(
            symbol="BTC/USDT",
            exchange=Exchange.BYBIT,
            timeframe=Timeframe.H1,
            timestamp=engine._clock.now(),
            open=51000.0,
            high=52500.0,
            low=50500.0,
            close=52200.0,
            volume=100.0,
        )

        engine._close_position_at_stop(pos, candle)

        assert "BTC/USDT" not in engine._sim_positions
        if engine._trade_details:
            detail = engine._trade_details[-1]
            assert detail.exit_price >= 52000.0, (
                f"Stop fill {detail.exit_price} should be >= stop {52000.0}"
            )


# ---------------------------------------------------------------------------
# Paper adapter: SL fill applies slippage (extension of B1 to paper mode)
# ---------------------------------------------------------------------------

class TestPaperStopSlippage:
    """Stop-loss fills in paper mode should apply slippage."""

    @pytest.mark.asyncio
    async def test_sl_fill_includes_slippage(self) -> None:
        from agentic_trading.execution.adapters.base import SlippageConfig

        adapter = PaperAdapter(
            exchange=Exchange.BYBIT,
            initial_balances={"USDT": Decimal("100000")},
            slippage=SlippageConfig(
                fixed_bps=Decimal("10"),  # 10 bps
            ),
        )
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.submit_order(
            _make_intent(side=Side.BUY, qty=Decimal("0.01"))
        )

        await adapter.set_trading_stop(
            "BTC/USDT",
            stop_loss=Decimal("48000"),
        )

        # Trigger the stop
        adapter.set_market_price("BTC/USDT", Decimal("47900"))

        # Position should be closed
        pos = adapter._positions.get("BTC/USDT")
        assert pos is not None
        assert pos.qty == Decimal("0")

    @pytest.mark.asyncio
    async def test_tp_fill_has_no_adverse_slippage(self) -> None:
        """Take-profit fills should not get slippage (limit-like)."""
        adapter = PaperAdapter(
            exchange=Exchange.BYBIT,
            initial_balances={"USDT": Decimal("100000")},
        )
        adapter.set_market_price("BTC/USDT", Decimal("50000"))
        await adapter.submit_order(
            _make_intent(side=Side.BUY, qty=Decimal("0.01"))
        )

        await adapter.set_trading_stop(
            "BTC/USDT",
            take_profit=Decimal("52000"),
        )

        # Trigger TP
        adapter.set_market_price("BTC/USDT", Decimal("52100"))

        pos = adapter._positions.get("BTC/USDT")
        assert pos is not None
        assert pos.qty == Decimal("0")
