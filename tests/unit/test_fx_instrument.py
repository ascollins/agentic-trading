"""Tests for the generalized Instrument model with FX support."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentic_trading.core.enums import (
    AssetClass,
    Exchange,
    InstrumentType,
    QtyUnit,
)
from agentic_trading.core.models import Instrument, TradingSession


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_crypto_instrument() -> Instrument:
    """Factory: BTC/USDT crypto instrument."""
    return Instrument(
        symbol="BTC/USDT",
        exchange=Exchange.BYBIT,
        instrument_type=InstrumentType.PERP,
        base="BTC",
        quote="USDT",
        price_precision=1,
        qty_precision=3,
        tick_size=Decimal("0.1"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100"),
        min_notional=Decimal("10"),
        max_leverage=100,
    )


def _make_fx_instrument(
    symbol: str = "EUR/USD",
    pip_size: Decimal = Decimal("0.0001"),
    lot_size: Decimal = Decimal("100000"),
) -> Instrument:
    """Factory: FX spot instrument."""
    return Instrument(
        symbol=symbol,
        exchange=Exchange.OANDA,
        instrument_type=InstrumentType.FX_SPOT,
        base=symbol[:3],
        quote=symbol[4:],
        asset_class=AssetClass.FX,
        price_precision=5,
        qty_precision=2,
        tick_size=Decimal("0.00001"),
        step_size=Decimal("0.01"),
        min_qty=Decimal("0.01"),
        max_qty=Decimal("10000"),
        min_notional=Decimal("0"),
        pip_size=pip_size,
        pip_value_per_lot=Decimal("10"),
        lot_size=lot_size,
        venue_symbol=symbol.replace("/", "_"),
        max_leverage=50,
        weekend_close=True,
        trading_sessions=[
            TradingSession(name="london", open_utc="08:00", close_utc="17:00"),
            TradingSession(name="new_york", open_utc="13:00", close_utc="22:00"),
        ],
    )


# ---------------------------------------------------------------------------
# Crypto backward compatibility
# ---------------------------------------------------------------------------


class TestCryptoBackwardCompat:
    def test_default_asset_class_is_crypto(self):
        inst = _make_crypto_instrument()
        assert inst.asset_class == AssetClass.CRYPTO

    def test_pip_size_is_none_for_crypto(self):
        inst = _make_crypto_instrument()
        assert inst.pip_size is None

    def test_notional_value_crypto(self):
        inst = _make_crypto_instrument()
        result = inst.notional_value(Decimal("0.5"), Decimal("50000"))
        assert result == Decimal("25000")

    def test_normalize_qty_base_passthrough(self):
        inst = _make_crypto_instrument()
        result = inst.normalize_qty(Decimal("0.5"), QtyUnit.BASE)
        assert result == Decimal("0.5")

    def test_instrument_hash_computed(self):
        inst = _make_crypto_instrument()
        assert inst.instrument_hash != ""
        assert len(inst.instrument_hash) == 16

    def test_instrument_hash_deterministic(self):
        inst1 = _make_crypto_instrument()
        inst2 = _make_crypto_instrument()
        assert inst1.instrument_hash == inst2.instrument_hash

    def test_round_price_crypto(self):
        inst = _make_crypto_instrument()
        assert inst.round_price(Decimal("50000.45")) == Decimal("50000.4")

    def test_round_qty_crypto(self):
        inst = _make_crypto_instrument()
        assert inst.round_qty(Decimal("0.0015")) == Decimal("0.002")


# ---------------------------------------------------------------------------
# FX Instrument model
# ---------------------------------------------------------------------------


class TestFXInstrument:
    def test_asset_class_fx(self):
        inst = _make_fx_instrument()
        assert inst.asset_class == AssetClass.FX

    def test_pip_size(self):
        inst = _make_fx_instrument()
        assert inst.pip_size == Decimal("0.0001")

    def test_lot_size(self):
        inst = _make_fx_instrument()
        assert inst.lot_size == Decimal("100000")

    def test_venue_symbol(self):
        inst = _make_fx_instrument()
        assert inst.venue_symbol == "EUR_USD"

    def test_weekend_close(self):
        inst = _make_fx_instrument()
        assert inst.weekend_close is True

    def test_trading_sessions(self):
        inst = _make_fx_instrument()
        assert len(inst.trading_sessions) == 2
        assert inst.trading_sessions[0].name == "london"


class TestFXPipConversions:
    def test_price_to_pips(self):
        inst = _make_fx_instrument()
        result = inst.price_to_pips(Decimal("0.0023"))
        assert result == Decimal("23")

    def test_price_to_pips_negative(self):
        inst = _make_fx_instrument()
        result = inst.price_to_pips(Decimal("-0.0010"))
        assert result == Decimal("-10")

    def test_pips_to_price(self):
        inst = _make_fx_instrument()
        result = inst.pips_to_price(Decimal("15"))
        assert result == Decimal("0.0015")

    def test_price_to_pips_crypto_returns_zero(self):
        inst = _make_crypto_instrument()
        result = inst.price_to_pips(Decimal("100"))
        assert result == Decimal("0")

    def test_pips_to_price_crypto_returns_zero(self):
        inst = _make_crypto_instrument()
        result = inst.pips_to_price(Decimal("5"))
        assert result == Decimal("0")


class TestFXNotional:
    def test_notional_value_fx_lots(self):
        inst = _make_fx_instrument()
        # 0.1 lots * 100,000 * 1.0850 = 10,850
        result = inst.notional_value(Decimal("0.1"), Decimal("1.0850"))
        assert result == Decimal("10850.00")

    def test_notional_value_fx_one_lot(self):
        inst = _make_fx_instrument()
        result = inst.notional_value(Decimal("1"), Decimal("1.1000"))
        assert result == Decimal("110000.0")


class TestFXNormalizeQty:
    def test_lots_to_base(self):
        inst = _make_fx_instrument()
        result = inst.normalize_qty(Decimal("0.1"), QtyUnit.LOTS)
        assert result == Decimal("10000.0")

    def test_base_passthrough(self):
        inst = _make_fx_instrument()
        result = inst.normalize_qty(Decimal("10000"), QtyUnit.BASE)
        assert result == Decimal("10000")


class TestInstrumentHash:
    def test_hash_changes_with_spec(self):
        inst1 = _make_fx_instrument(pip_size=Decimal("0.0001"))
        inst2 = _make_fx_instrument(pip_size=Decimal("0.01"))
        assert inst1.instrument_hash != inst2.instrument_hash

    def test_compute_hash_matches(self):
        inst = _make_fx_instrument()
        assert inst.compute_hash() == inst.instrument_hash

    def test_hash_is_16_chars(self):
        inst = _make_fx_instrument()
        assert len(inst.instrument_hash) == 16

    def test_usdjpy_pip_size(self):
        inst = _make_fx_instrument(
            symbol="USD/JPY", pip_size=Decimal("0.01")
        )
        assert inst.pip_size == Decimal("0.01")
        assert inst.price_to_pips(Decimal("1.5")) == Decimal("150")


class TestTradingSession:
    def test_session_creation(self):
        session = TradingSession(
            name="london", open_utc="08:00", close_utc="17:00"
        )
        assert session.name == "london"
        assert session.days == [1, 2, 3, 4, 5]

    def test_session_custom_days(self):
        session = TradingSession(
            name="test", open_utc="00:00", close_utc="23:59", days=[1, 2, 3]
        )
        assert session.days == [1, 2, 3]
