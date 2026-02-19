"""Unit tests for TpSlWatchdog.

Tests cover:
- Positions already fully protected are skipped
- Missing TP only → re-applied
- Missing SL only → re-applied
- Missing trailing → re-applied
- Missing everything → all re-applied
- Retry logic (3 attempts) on failures
- CRITICAL log after 3 failed attempts
- read_only mode: logs but never calls adapter
- Stats counters track checks and repairs
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_trading.execution.tpsl_watchdog import TpSlWatchdog


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_exit_cfg(
    sl_mult: float = 2.5,
    tp_mult: float = 5.0,
    trail_mult: float = 2.0,
    trailing_strategies: list[str] | None = None,
):
    cfg = MagicMock()
    cfg.sl_atr_multiplier = sl_mult
    cfg.tp_atr_multiplier = tp_mult
    cfg.trailing_stop_atr_multiplier = trail_mult
    cfg.trailing_strategies = trailing_strategies or ["trend_following", "breakout"]
    return cfg


def _make_position(symbol: str = "BTC/USDT", side: str = "long", entry: float = 50000.0):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = Decimal("0.01")
    pos.entry_price = Decimal(str(entry))
    side_mock = MagicMock()
    side_mock.value = side
    pos.side = side_mock
    return pos


def _make_bybit_position(
    symbol: str = "BTC/USDT:USDT",
    tp: float = 0,
    sl: float = 0,
    trailing: float = 0,
):
    """Simulate the raw CCXT-normalized position dict from fetch_positions()."""
    return {
        "symbol": symbol,
        "takeProfitPrice": tp if tp > 0 else None,
        "stopLossPrice": sl if sl > 0 else None,
        "info": {
            "trailingStop": str(trailing) if trailing > 0 else "0",
        },
    }


def _make_adapter(
    positions: list | None = None,
    raw_bybit_pos: dict | None = None,
):
    adapter = MagicMock()
    adapter.get_positions = AsyncMock(return_value=positions or [])
    adapter._ccxt = MagicMock()
    adapter._ccxt.fetch_positions = AsyncMock(
        return_value=[raw_bybit_pos] if raw_bybit_pos else []
    )
    adapter.set_trading_stop = AsyncMock(return_value={"retCode": 0, "retMsg": "OK"})
    return adapter


_SENTINEL = object()


def _make_watchdog(
    adapter=None,
    exit_cfg=None,
    tool_gateway=None,
    trailing_strategies=_SENTINEL,
    read_only: bool = False,
):
    if trailing_strategies is _SENTINEL:
        trailing_strategies = ["trend_following", "breakout"]
    return TpSlWatchdog(
        adapter=adapter or _make_adapter(),
        exit_cfg=exit_cfg or _make_exit_cfg(),
        interval=300.0,
        tool_gateway=tool_gateway,
        trailing_strategies=trailing_strategies,
        read_only=read_only,
    )


# ---------------------------------------------------------------------------
# Identity / capabilities
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_agent_type_is_risk_gate(self):
        from agentic_trading.core.enums import AgentType
        w = _make_watchdog()
        assert w.agent_type == AgentType.RISK_GATE

    def test_capabilities_returns_description(self):
        w = _make_watchdog()
        caps = w.capabilities()
        assert "300" in caps.description
        assert caps.subscribes_to == []
        assert caps.publishes_to == []


# ---------------------------------------------------------------------------
# No-op when fully protected
# ---------------------------------------------------------------------------

class TestFullyProtectedPositions:
    @pytest.mark.asyncio
    async def test_skips_when_tp_sl_trail_all_present(self):
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=55000, sl=48000, trailing=500)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter)

        await w._check_all_positions()

        adapter.set_trading_stop.assert_not_called()
        assert w.positions_repaired == 0

    @pytest.mark.asyncio
    async def test_skips_when_no_trailing_expected_and_tp_sl_present(self):
        """When trailing_strategies is empty, only TP+SL protection is required."""
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=55000, sl=48000, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        await w._check_all_positions()

        adapter.set_trading_stop.assert_not_called()
        assert w.positions_repaired == 0

    @pytest.mark.asyncio
    async def test_skips_zero_qty_positions(self):
        pos = _make_position("BTC/USDT", "long", 50000)
        pos.qty = Decimal("0")
        adapter = _make_adapter(positions=[pos])
        w = _make_watchdog(adapter=adapter)

        await w._check_all_positions()

        adapter._ccxt.fetch_positions.assert_not_called()
        adapter.set_trading_stop.assert_not_called()


# ---------------------------------------------------------------------------
# Re-apply missing protection
# ---------------------------------------------------------------------------

class TestMissingProtection:
    @pytest.mark.asyncio
    async def test_sets_tp_when_missing(self):
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=0, sl=48000, trailing=500)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        await w._check_all_positions()

        adapter.set_trading_stop.assert_called_once()
        call_kwargs = adapter.set_trading_stop.call_args[1]
        assert call_kwargs["take_profit"] is not None
        assert call_kwargs["stop_loss"] is None   # already set → not passed

    @pytest.mark.asyncio
    async def test_sets_sl_when_missing(self):
        pos = _make_position("ETH/USDT", "short", 2000)
        raw = _make_bybit_position("ETH/USDT:USDT", tp=1900, sl=0, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        await w._check_all_positions()

        adapter.set_trading_stop.assert_called_once()
        call_kwargs = adapter.set_trading_stop.call_args[1]
        assert call_kwargs["stop_loss"] is not None
        assert call_kwargs["take_profit"] is None   # already set

    @pytest.mark.asyncio
    async def test_sets_trailing_when_missing_tp_sl_present(self):
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=55000, sl=48000, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter, trailing_strategies=["trend_following"])

        await w._check_all_positions()

        adapter.set_trading_stop.assert_called_once()
        call_kwargs = adapter.set_trading_stop.call_args[1]
        assert call_kwargs["trailing_stop"] is not None
        assert call_kwargs["active_price"] is not None
        assert call_kwargs["take_profit"] is None   # already present

    @pytest.mark.asyncio
    async def test_sets_everything_when_all_missing(self):
        pos = _make_position("SOL/USDT", "long", 100)
        raw = _make_bybit_position("SOL/USDT:USDT", tp=0, sl=0, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter, trailing_strategies=["trend_following"])

        await w._check_all_positions()

        adapter.set_trading_stop.assert_called_once()
        call_kwargs = adapter.set_trading_stop.call_args[1]
        assert call_kwargs["take_profit"] is not None
        assert call_kwargs["stop_loss"] is not None
        assert call_kwargs["trailing_stop"] is not None
        assert call_kwargs["active_price"] is not None

    @pytest.mark.asyncio
    async def test_active_price_correct_for_long(self):
        """active_price = entry + sl_distance for long (breakeven activation)."""
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=0, sl=0, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        exit_cfg = _make_exit_cfg(sl_mult=2.5, tp_mult=5.0, trail_mult=2.0)
        w = _make_watchdog(adapter=adapter, exit_cfg=exit_cfg, trailing_strategies=["t"])

        await w._check_all_positions()

        call_kwargs = adapter.set_trading_stop.call_args[1]
        entry = Decimal("50000")
        atr = entry * Decimal("0.004")
        sl_dist = atr * Decimal("2.5")
        expected_active = entry + sl_dist
        assert call_kwargs["active_price"] == expected_active

    @pytest.mark.asyncio
    async def test_active_price_correct_for_short(self):
        """active_price = entry - sl_distance for short."""
        pos = _make_position("BTC/USDT", "short", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=0, sl=0, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        exit_cfg = _make_exit_cfg(sl_mult=2.5)
        w = _make_watchdog(adapter=adapter, exit_cfg=exit_cfg, trailing_strategies=["t"])

        await w._check_all_positions()

        call_kwargs = adapter.set_trading_stop.call_args[1]
        entry = Decimal("50000")
        atr = entry * Decimal("0.004")
        sl_dist = atr * Decimal("2.5")
        expected_active = entry - sl_dist
        assert call_kwargs["active_price"] == expected_active


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_up_to_3_times_on_failure(self):
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=0, sl=0, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        adapter.set_trading_stop = AsyncMock(side_effect=RuntimeError("API error"))
        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await w._check_all_positions()

        assert adapter.set_trading_stop.call_count == 3
        assert w.positions_repaired == 0

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self):
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=0, sl=0, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        adapter.set_trading_stop = AsyncMock(
            side_effect=[RuntimeError("fail"), {"retCode": 0}]
        )
        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await w._check_all_positions()

        assert adapter.set_trading_stop.call_count == 2
        assert w.positions_repaired == 1

    @pytest.mark.asyncio
    async def test_critical_log_after_all_retries_exhausted(self, caplog):
        import logging
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=0, sl=0, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        adapter.set_trading_stop = AsyncMock(side_effect=RuntimeError("dead"))
        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with caplog.at_level(logging.CRITICAL):
                await w._check_all_positions()

        assert any("FAILED to protect" in r.message for r in caplog.records)
        assert any(r.levelno == logging.CRITICAL for r in caplog.records)


# ---------------------------------------------------------------------------
# Read-only mode
# ---------------------------------------------------------------------------

class TestReadOnlyMode:
    @pytest.mark.asyncio
    async def test_read_only_never_calls_set_trading_stop(self):
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=0, sl=0, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter, read_only=True)

        await w._check_all_positions()

        adapter.set_trading_stop.assert_not_called()
        assert w.positions_repaired == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    @pytest.mark.asyncio
    async def test_positions_checked_increments(self):
        pos1 = _make_position("BTC/USDT", "long", 50000)
        pos2 = _make_position("ETH/USDT", "long", 2000)
        # Both fully protected
        raw1 = _make_bybit_position("BTC/USDT:USDT", tp=55000, sl=48000, trailing=500)
        raw2 = _make_bybit_position("ETH/USDT:USDT", tp=2200, sl=1900, trailing=20)
        adapter = MagicMock()
        adapter.get_positions = AsyncMock(return_value=[pos1, pos2])
        adapter._ccxt = MagicMock()
        adapter._ccxt.fetch_positions = AsyncMock(side_effect=[[raw1], [raw2]])
        adapter.set_trading_stop = AsyncMock()

        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        await w._check_all_positions()
        assert w.positions_checked == 2
        assert w.positions_repaired == 0

    @pytest.mark.asyncio
    async def test_positions_repaired_increments(self):
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=0, sl=0, trailing=0)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        await w._check_all_positions()
        assert w.positions_repaired == 1

    @pytest.mark.asyncio
    async def test_stats_accumulate_across_calls(self):
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=55000, sl=48000, trailing=500)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        await w._check_all_positions()
        await w._check_all_positions()

        assert w.positions_checked == 2
        assert w.positions_repaired == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_no_positions_returns_early(self):
        adapter = _make_adapter(positions=[])
        w = _make_watchdog(adapter=adapter)

        await w._check_all_positions()

        adapter._ccxt.fetch_positions.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_positions_failure_is_swallowed(self):
        adapter = MagicMock()
        adapter.get_positions = AsyncMock(side_effect=RuntimeError("exchange down"))
        w = _make_watchdog(adapter=adapter)

        # Should not raise
        await w._check_all_positions()

    @pytest.mark.asyncio
    async def test_individual_position_error_does_not_abort_others(self):
        pos1 = _make_position("BTC/USDT", "long", 50000)
        pos2 = _make_position("ETH/USDT", "long", 2000)
        raw_ok = _make_bybit_position("ETH/USDT:USDT", tp=0, sl=0, trailing=0)

        adapter = MagicMock()
        adapter.get_positions = AsyncMock(return_value=[pos1, pos2])
        adapter._ccxt = MagicMock()
        # First call (BTC) raises, second call (ETH) succeeds
        adapter._ccxt.fetch_positions = AsyncMock(
            side_effect=[RuntimeError("BTC api error"), [raw_ok]]
        )
        adapter.set_trading_stop = AsyncMock(return_value={"retCode": 0})

        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        await w._check_all_positions()

        # ETH should still be checked and repaired
        assert adapter.set_trading_stop.call_count == 1

    @pytest.mark.asyncio
    async def test_symbol_matching_with_settle_suffix(self):
        """Position with 'BTC/USDT' matches raw position with 'BTC/USDT:USDT'."""
        pos = _make_position("BTC/USDT", "long", 50000)
        raw = _make_bybit_position("BTC/USDT:USDT", tp=55000, sl=48000, trailing=500)
        adapter = _make_adapter(positions=[pos], raw_bybit_pos=raw)
        w = _make_watchdog(adapter=adapter, trailing_strategies=[])

        await w._check_all_positions()

        adapter.set_trading_stop.assert_not_called()  # already protected
