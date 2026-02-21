"""Tests for FX intent enrichment.

Validates that OrderIntents are enriched with:
- asset_class (CRYPTO or FX)
- qty_unit (BASE)
- instrument_hash (from instrument metadata)

Tests both the intent_converter and the staleness session awareness.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from agentic_trading.core.enums import (
    AssetClass,
    Exchange,
    InstrumentType,
    QtyUnit,
    Side,
)
from agentic_trading.core.events import TargetPosition
from agentic_trading.core.fx_instruments import build_fx_instruments
from agentic_trading.core.models import Instrument, TradingSession
from agentic_trading.intelligence.data_qa import DataQualityChecker
from agentic_trading.signal.portfolio.intent_converter import build_order_intents


def _make_fx_instrument(symbol: str = "EUR/USD") -> Instrument:
    """Create a minimal FX instrument."""
    return Instrument(
        symbol=symbol,
        exchange=Exchange.OANDA,
        instrument_type=InstrumentType.FX_SPOT,
        base="EUR",
        quote="USD",
        price_precision=5,
        qty_precision=2,
        tick_size=Decimal("0.00001"),
        step_size=Decimal("0.01"),
        min_qty=Decimal("0.01"),
        asset_class=AssetClass.FX,
        pip_size=Decimal("0.0001"),
        lot_size=Decimal("100000"),
    )


def _make_target(
    symbol: str = "EUR/USD",
    qty: Decimal = Decimal("1"),
    side: Side = Side.BUY,
    strategy_id: str = "trend_fx",
) -> TargetPosition:
    """Create a minimal TargetPosition."""
    return TargetPosition(
        strategy_id=strategy_id,
        symbol=symbol,
        target_qty=qty,
        side=side,
    )


class TestIntentConverterFXEnrichment:
    """Test that build_order_intents enriches FX intents."""

    def test_fx_intent_has_fx_asset_class(self):
        """FX intents should have asset_class=FX."""
        inst = _make_fx_instrument()
        targets = [_make_target()]
        now = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)

        intents = build_order_intents(
            targets=targets,
            exchange=Exchange.OANDA,
            timestamp=now,
            instruments={inst.symbol: inst},
        )

        assert len(intents) == 1
        assert intents[0].asset_class == AssetClass.FX

    def test_fx_intent_has_base_qty_unit(self):
        """FX intents should have qty_unit=BASE."""
        inst = _make_fx_instrument()
        targets = [_make_target()]
        now = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)

        intents = build_order_intents(
            targets=targets,
            exchange=Exchange.OANDA,
            timestamp=now,
            instruments={inst.symbol: inst},
        )

        assert intents[0].qty_unit == QtyUnit.BASE

    def test_fx_intent_has_instrument_hash(self):
        """FX intents should carry instrument_hash."""
        inst = _make_fx_instrument()
        targets = [_make_target()]
        now = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)

        intents = build_order_intents(
            targets=targets,
            exchange=Exchange.OANDA,
            timestamp=now,
            instruments={inst.symbol: inst},
        )

        assert intents[0].instrument_hash != ""
        assert intents[0].instrument_hash == inst.instrument_hash

    def test_crypto_intent_defaults_to_crypto(self):
        """Without instrument, intents default to CRYPTO."""
        targets = [_make_target(symbol="BTC/USDT")]
        now = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)

        intents = build_order_intents(
            targets=targets,
            exchange=Exchange.BINANCE,
            timestamp=now,
        )

        assert len(intents) == 1
        assert intents[0].asset_class == AssetClass.CRYPTO
        assert intents[0].instrument_hash == ""

    def test_multiple_targets_enriched(self):
        """Multiple targets with mixed instruments are all enriched."""
        fx_inst = _make_fx_instrument("EUR/USD")
        targets = [
            _make_target("EUR/USD"),
            _make_target("GBP/USD"),  # no instrument loaded
        ]
        now = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)

        intents = build_order_intents(
            targets=targets,
            exchange=Exchange.OANDA,
            timestamp=now,
            instruments={"EUR/USD": fx_inst},
        )

        assert len(intents) == 2
        assert intents[0].asset_class == AssetClass.FX
        assert intents[1].asset_class == AssetClass.CRYPTO  # fallback


class TestFXInstrumentDefinitions:
    """Test the hardcoded FX instrument builder."""

    def test_build_known_pairs(self):
        """Known G10 pairs should be built."""
        instruments = build_fx_instruments(
            ["EUR/USD", "USD/JPY", "GBP/USD"]
        )
        assert len(instruments) == 3
        assert "EUR/USD" in instruments
        assert "USD/JPY" in instruments
        assert "GBP/USD" in instruments

    def test_unknown_pairs_skipped(self):
        """Unknown pairs should be silently skipped."""
        instruments = build_fx_instruments(
            ["EUR/USD", "FAKE/PAIR"]
        )
        assert len(instruments) == 1
        assert "EUR/USD" in instruments

    def test_fx_instrument_properties(self):
        """FX instruments should have correct asset_class and lot_size."""
        instruments = build_fx_instruments(["EUR/USD"])
        inst = instruments["EUR/USD"]

        assert inst.asset_class == AssetClass.FX
        assert inst.lot_size == Decimal("100000")
        assert inst.pip_size == Decimal("0.0001")
        assert inst.weekend_close is True
        assert inst.rollover_enabled is True
        assert len(inst.trading_sessions) > 0

    def test_jpy_pair_has_different_pip_size(self):
        """JPY pairs should have pip_size=0.01."""
        instruments = build_fx_instruments(["USD/JPY"])
        inst = instruments["USD/JPY"]

        assert inst.pip_size == Decimal("0.01")
        assert inst.price_precision == 3

    def test_instrument_hash_computed(self):
        """Instruments should have a non-empty hash."""
        instruments = build_fx_instruments(["EUR/USD"])
        inst = instruments["EUR/USD"]

        assert inst.instrument_hash != ""
        assert len(inst.instrument_hash) == 16  # SHA256[:16]

    def test_venue_symbol_oanda_format(self):
        """OANDA venue symbols use underscore format."""
        instruments = build_fx_instruments(["EUR/USD"], exchange=Exchange.OANDA)
        inst = instruments["EUR/USD"]

        assert inst.venue_symbol == "EUR_USD"


class TestSessionAwareStaleness:
    """Test that staleness checks are session-aware for FX."""

    def test_staleness_suppressed_during_weekend(self):
        """FX staleness should be suppressed on weekends."""
        inst = _make_fx_instrument()
        inst = inst.model_copy(
            update={
                "weekend_close": True,
                "trading_sessions": [
                    TradingSession(
                        name="london",
                        open_utc="08:00",
                        close_utc="17:00",
                    ),
                ],
            }
        )

        checker = DataQualityChecker()
        # Saturday at noon — market is closed
        now = datetime(2024, 1, 13, 12, 0, tzinfo=timezone.utc)  # Saturday
        last_candle = datetime(2024, 1, 12, 22, 0, tzinfo=timezone.utc)  # Friday

        issue = checker.check_staleness(
            last_candle_time=last_candle,
            max_age_seconds=300,  # 5 minutes
            symbol="EUR/USD",
            now=now,
            instrument=inst,
        )
        # Should return None (suppressed)
        assert issue is None

    def test_staleness_fires_during_open_session(self):
        """FX staleness should fire when session is open."""
        inst = _make_fx_instrument()
        inst = inst.model_copy(
            update={
                "weekend_close": True,
                "trading_sessions": [
                    TradingSession(
                        name="london",
                        open_utc="08:00",
                        close_utc="17:00",
                    ),
                ],
            }
        )

        checker = DataQualityChecker()
        # Tuesday at noon (London session open)
        now = datetime(2024, 1, 16, 12, 0, tzinfo=timezone.utc)  # Tuesday
        last_candle = datetime(2024, 1, 16, 11, 50, tzinfo=timezone.utc)

        issue = checker.check_staleness(
            last_candle_time=last_candle,
            max_age_seconds=300,  # 5 minutes
            symbol="EUR/USD",
            now=now,
            instrument=inst,
        )
        # 600 seconds staleness > 300 threshold → should fire
        assert issue is not None
        assert issue.check == "staleness"

    def test_staleness_without_instrument_unchanged(self):
        """Without instrument, staleness behaves as before."""
        checker = DataQualityChecker()
        now = datetime(2024, 1, 13, 12, 0, tzinfo=timezone.utc)  # Saturday
        last_candle = datetime(2024, 1, 12, 22, 0, tzinfo=timezone.utc)

        issue = checker.check_staleness(
            last_candle_time=last_candle,
            max_age_seconds=300,
            symbol="BTC/USDT",
            now=now,
        )
        # No instrument → regular staleness check → should fire
        assert issue is not None
