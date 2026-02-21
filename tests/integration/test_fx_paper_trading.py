"""Integration test: FX paper trading end-to-end flow.

Verifies: FX instrument → PaperAdapter → fill → position → reconciliation.
Uses the real PaperAdapter with FX instruments.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentic_trading.core.enums import (
    AssetClass,
    Exchange,
    InstrumentType,
    OrderStatus,
    OrderType,
    PositionSide,
    QtyUnit,
    Side,
)
from agentic_trading.core.events import OrderIntent
from agentic_trading.core.models import Instrument, TradingSession
from agentic_trading.execution.adapters.paper import PaperAdapter


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_eurusd() -> Instrument:
    return Instrument(
        symbol="EUR/USD",
        exchange=Exchange.OANDA,
        instrument_type=InstrumentType.FX_SPOT,
        base="EUR",
        quote="USD",
        asset_class=AssetClass.FX,
        price_precision=5,
        qty_precision=2,
        tick_size=Decimal("0.00001"),
        step_size=Decimal("0.01"),
        min_qty=Decimal("0.01"),
        max_qty=Decimal("10000"),
        min_notional=Decimal("0"),
        pip_size=Decimal("0.0001"),
        pip_value_per_lot=Decimal("10"),
        lot_size=Decimal("100000"),
        venue_symbol="EUR_USD",
        max_leverage=50,
        weekend_close=True,
        trading_sessions=[
            TradingSession(name="london", open_utc="08:00", close_utc="17:00"),
            TradingSession(name="new_york", open_utc="13:00", close_utc="22:00"),
        ],
    )


def _make_gbpusd() -> Instrument:
    return Instrument(
        symbol="GBP/USD",
        exchange=Exchange.OANDA,
        instrument_type=InstrumentType.FX_SPOT,
        base="GBP",
        quote="USD",
        asset_class=AssetClass.FX,
        price_precision=5,
        qty_precision=2,
        tick_size=Decimal("0.00001"),
        step_size=Decimal("0.01"),
        min_qty=Decimal("0.01"),
        max_qty=Decimal("10000"),
        min_notional=Decimal("0"),
        pip_size=Decimal("0.0001"),
        pip_value_per_lot=Decimal("10"),
        lot_size=Decimal("100000"),
        venue_symbol="GBP_USD",
        max_leverage=50,
    )


def _make_paper_adapter(
    instruments: list[Instrument] | None = None,
    balance_usd: Decimal = Decimal("100000"),
) -> PaperAdapter:
    """Build a PaperAdapter pre-loaded with FX instruments."""
    inst_dict = {i.symbol: i for i in (instruments or [_make_eurusd()])}
    return PaperAdapter(
        exchange=Exchange.OANDA,
        initial_balances={"USD": balance_usd},
        instruments=inst_dict,
    )


def _make_fx_intent(
    symbol: str = "EUR/USD",
    side: Side = Side.BUY,
    qty: Decimal = Decimal("0.1"),
    order_type: OrderType = OrderType.MARKET,
    dedupe_key: str = "test_fx_001",
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        exchange=Exchange.OANDA,
        side=side,
        qty=qty,
        order_type=order_type,
        dedupe_key=dedupe_key,
        strategy_id="test_fx_strategy",
        asset_class=AssetClass.FX,
        qty_unit=QtyUnit.BASE,
        source_module="test",
    )


# ---------------------------------------------------------------------------
# Test 1: Basic FX buy fills successfully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFXPaperBuy:
    async def test_buy_eurusd_fills(self):
        adapter = _make_paper_adapter()
        adapter.set_market_price("EUR/USD", Decimal("1.08500"))

        intent = _make_fx_intent()
        ack = await adapter.submit_order(intent)

        assert ack.status == OrderStatus.FILLED
        assert ack.symbol == "EUR/USD"
        assert ack.exchange == Exchange.OANDA

    async def test_position_created_after_buy(self):
        adapter = _make_paper_adapter()
        adapter.set_market_price("EUR/USD", Decimal("1.08500"))

        await adapter.submit_order(_make_fx_intent())

        positions = await adapter.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "EUR/USD"
        assert positions[0].qty > Decimal("0")

    async def test_balance_decreases_after_buy(self):
        adapter = _make_paper_adapter(balance_usd=Decimal("100000"))
        adapter.set_market_price("EUR/USD", Decimal("1.08500"))

        await adapter.submit_order(_make_fx_intent())

        balances = await adapter.get_balances()
        usd_bal = next(b for b in balances if b.currency == "USD")
        assert usd_bal.total < Decimal("100000")


# ---------------------------------------------------------------------------
# Test 2: FX sell (short) works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFXPaperSell:
    async def test_sell_creates_short_position(self):
        adapter = _make_paper_adapter()
        adapter.set_market_price("EUR/USD", Decimal("1.08500"))

        intent = _make_fx_intent(side=Side.SELL, dedupe_key="test_sell_001")
        ack = await adapter.submit_order(intent)

        assert ack.status == OrderStatus.FILLED
        positions = await adapter.get_positions()
        assert len(positions) == 1
        assert positions[0].qty < Decimal("0")


# ---------------------------------------------------------------------------
# Test 3: Round-trip (buy then sell) closes position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFXRoundTrip:
    async def test_buy_then_sell_closes_position(self):
        adapter = _make_paper_adapter()
        adapter.set_market_price("EUR/USD", Decimal("1.08500"))

        buy = _make_fx_intent(dedupe_key="buy_001")
        await adapter.submit_order(buy)

        # Price moves up, then sell to close
        adapter.set_market_price("EUR/USD", Decimal("1.08700"))
        sell = _make_fx_intent(
            side=Side.SELL, dedupe_key="sell_001"
        )
        await adapter.submit_order(sell)

        # Position should be closed (qty == 0)
        positions = await adapter.get_positions()
        assert len(positions) == 0  # get_positions filters out zero-qty


# ---------------------------------------------------------------------------
# Test 4: Multiple FX instruments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMultipleFXInstruments:
    async def test_two_fx_pairs(self):
        eurusd = _make_eurusd()
        gbpusd = _make_gbpusd()
        adapter = _make_paper_adapter(instruments=[eurusd, gbpusd])

        adapter.set_market_price("EUR/USD", Decimal("1.08500"))
        adapter.set_market_price("GBP/USD", Decimal("1.26500"))

        eur_intent = _make_fx_intent(dedupe_key="eur_001")
        gbp_intent = _make_fx_intent(
            symbol="GBP/USD", dedupe_key="gbp_001"
        )

        ack1 = await adapter.submit_order(eur_intent)
        ack2 = await adapter.submit_order(gbp_intent)

        assert ack1.status == OrderStatus.FILLED
        assert ack2.status == OrderStatus.FILLED

        positions = await adapter.get_positions()
        symbols = {p.symbol for p in positions}
        assert symbols == {"EUR/USD", "GBP/USD"}


# ---------------------------------------------------------------------------
# Test 5: Instrument not loaded → rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFXMissingInstrument:
    async def test_unknown_symbol_gets_price_error(self):
        adapter = _make_paper_adapter()
        # Do NOT set a market price for an unknown pair
        intent = _make_fx_intent(
            symbol="TRY/ZAR", dedupe_key="unknown_001"
        )
        with pytest.raises(Exception):
            await adapter.submit_order(intent)


# ---------------------------------------------------------------------------
# Test 6: Idempotent submit (same dedupe_key)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFXIdempotency:
    async def test_paper_fills_both_same_dedupe(self):
        """PaperAdapter fills all submits (no built-in dedup).

        In the real pipeline, the ExecutionEngine's OrderManager handles
        dedup.  Here we verify PaperAdapter doesn't crash on same key.
        """
        adapter = _make_paper_adapter()
        adapter.set_market_price("EUR/USD", Decimal("1.08500"))

        intent = _make_fx_intent(dedupe_key="abc123")
        ack1 = await adapter.submit_order(intent)
        ack2 = await adapter.submit_order(intent)

        assert ack1.status == OrderStatus.FILLED
        assert ack2.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# Test 7: FX notional calculation via Instrument model
# ---------------------------------------------------------------------------


class TestFXNotionalIntegration:
    def test_eurusd_notional_matches_lot_convention(self):
        inst = _make_eurusd()
        # 0.1 lots * 100,000 * 1.0850 = 10,850 USD
        notional = inst.notional_value(Decimal("0.1"), Decimal("1.0850"))
        assert notional == Decimal("10850.00")

    def test_gbpusd_notional(self):
        inst = _make_gbpusd()
        # 1 lot * 100,000 * 1.2650 = 126,500 USD
        notional = inst.notional_value(Decimal("1"), Decimal("1.2650"))
        assert notional == Decimal("126500.0")


# ---------------------------------------------------------------------------
# Test 8: FX instrument hash pinning
# ---------------------------------------------------------------------------


class TestInstrumentHashPinning:
    def test_same_spec_same_hash(self):
        inst1 = _make_eurusd()
        inst2 = _make_eurusd()
        assert inst1.instrument_hash == inst2.instrument_hash

    def test_different_leverage_different_hash(self):
        inst1 = _make_eurusd()
        # Build inst2 with different max_leverage — hash should differ
        inst2_data = inst1.model_dump()
        inst2_data["max_leverage"] = 20
        inst2_data["instrument_hash"] = ""  # reset so model_post_init recomputes
        inst2 = Instrument(**inst2_data)
        assert inst1.instrument_hash != inst2.instrument_hash

    def test_hash_on_intent_matches_instrument(self):
        inst = _make_eurusd()
        intent = OrderIntent(
            symbol="EUR/USD",
            exchange=Exchange.OANDA,
            side=Side.BUY,
            qty=Decimal("0.1"),
            order_type=OrderType.MARKET,
            dedupe_key="hash_test_001",
            strategy_id="test",
            asset_class=AssetClass.FX,
            qty_unit=QtyUnit.BASE,
            instrument_hash=inst.instrument_hash,
            source_module="test",
        )
        assert intent.instrument_hash == inst.instrument_hash


# ---------------------------------------------------------------------------
# Test 9: FXBrokerAdapter rejects unknown instrument
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFXBrokerAdapterReject:
    async def test_submit_unknown_instrument_rejected(self):
        from agentic_trading.execution.adapters.fx_broker import FXBrokerAdapter

        adapter = FXBrokerAdapter(
            exchange=Exchange.OANDA,
            instruments={},
        )
        intent = _make_fx_intent(dedupe_key="reject_001")
        ack = await adapter.submit_order(intent)
        assert ack.status == OrderStatus.REJECTED
        assert "not loaded" in ack.message.lower()


# ---------------------------------------------------------------------------
# Test 10: Reconciliation with FX tolerance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFXReconciliation:
    async def test_fx_position_tolerance(self):
        from unittest.mock import AsyncMock

        from agentic_trading.core.models import Position
        from agentic_trading.event_bus.memory_bus import MemoryEventBus
        from agentic_trading.execution.order_manager import OrderManager
        from agentic_trading.reconciliation.loop import ReconciliationLoop

        bus = MemoryEventBus()
        order_mgr = OrderManager()
        eurusd = _make_eurusd()

        local_positions = {
            "EUR/USD": Position(
                symbol="EUR/USD",
                exchange=Exchange.OANDA,
                side=PositionSide.LONG,
                qty=Decimal("10000.00"),
                entry_price=Decimal("1.08500"),
                mark_price=Decimal("1.08500"),
                leverage=1,
            ),
        }

        recon = ReconciliationLoop(
            adapter=AsyncMock(),
            event_bus=bus,
            order_manager=order_mgr,
            exchange=Exchange.OANDA,
            local_positions=local_positions,
            instruments={"EUR/USD": eurusd},
        )

        # Exchange reports qty that differs by step_size (0.01) —
        # within FX tolerance (2 * step_size = 0.02)
        exchange_pos = Position(
            symbol="EUR/USD",
            exchange=Exchange.OANDA,
            side=PositionSide.LONG,
            qty=Decimal("10000.01"),
            entry_price=Decimal("1.08500"),
            mark_price=Decimal("1.08500"),
            leverage=1,
        )

        recon._adapter.get_open_orders = AsyncMock(return_value=[])
        recon._adapter.get_positions = AsyncMock(return_value=[exchange_pos])
        recon._adapter.get_balances = AsyncMock(return_value=[])

        result = await recon.reconcile()

        # No position mismatch because difference is within FX tolerance
        pos_mismatches = [
            d for d in result.discrepancies
            if d.get("type") == "position_mismatch"
        ]
        assert len(pos_mismatches) == 0

    async def test_fx_position_large_mismatch_detected(self):
        from unittest.mock import AsyncMock

        from agentic_trading.core.models import Position
        from agentic_trading.event_bus.memory_bus import MemoryEventBus
        from agentic_trading.execution.order_manager import OrderManager
        from agentic_trading.reconciliation.loop import ReconciliationLoop

        bus = MemoryEventBus()
        order_mgr = OrderManager()
        eurusd = _make_eurusd()

        local_positions = {
            "EUR/USD": Position(
                symbol="EUR/USD",
                exchange=Exchange.OANDA,
                side=PositionSide.LONG,
                qty=Decimal("10000.00"),
                entry_price=Decimal("1.08500"),
                mark_price=Decimal("1.08500"),
                leverage=1,
            ),
        }

        recon = ReconciliationLoop(
            adapter=AsyncMock(),
            event_bus=bus,
            order_manager=order_mgr,
            exchange=Exchange.OANDA,
            local_positions=local_positions,
            auto_repair=False,
            instruments={"EUR/USD": eurusd},
        )

        # Exchange reports qty that differs by a large amount
        exchange_pos = Position(
            symbol="EUR/USD",
            exchange=Exchange.OANDA,
            side=PositionSide.LONG,
            qty=Decimal("10100.00"),
            entry_price=Decimal("1.08500"),
            mark_price=Decimal("1.08500"),
            leverage=1,
        )

        recon._adapter.get_open_orders = AsyncMock(return_value=[])
        recon._adapter.get_positions = AsyncMock(return_value=[exchange_pos])
        recon._adapter.get_balances = AsyncMock(return_value=[])

        result = await recon.reconcile()

        pos_mismatches = [
            d for d in result.discrepancies
            if d.get("type") == "position_mismatch"
        ]
        assert len(pos_mismatches) == 1
