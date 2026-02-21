"""Hardcoded G10 FX instrument definitions for paper trading.

Since FX paper trading (e.g. OANDA demo) does not require querying
an exchange API for instrument specs, we define them statically here.
All pip sizes, lot sizes, and tick sizes follow standard FX conventions.
"""

from __future__ import annotations

from decimal import Decimal

from .enums import AssetClass, Exchange, InstrumentType
from .models import Instrument, TradingSession

# Standard FX trading sessions (UTC times).
_LONDON = TradingSession(name="london", open_utc="08:00", close_utc="17:00")
_NEW_YORK = TradingSession(name="new_york", open_utc="13:00", close_utc="22:00")
_TOKYO = TradingSession(name="tokyo", open_utc="00:00", close_utc="09:00")
_SYDNEY = TradingSession(name="sydney", open_utc="22:00", close_utc="07:00")

_ALL_SESSIONS = [_SYDNEY, _TOKYO, _LONDON, _NEW_YORK]

# Instrument spec templates for G10 major pairs.
_FX_SPECS: dict[str, dict] = {
    "EUR/USD": {
        "base": "EUR",
        "quote": "USD",
        "pip_size": Decimal("0.0001"),
        "pip_value_per_lot": Decimal("10"),
        "tick_size": Decimal("0.00001"),
        "price_precision": 5,
        "min_qty": Decimal("0.01"),
        "maker_fee": Decimal("0"),
        "taker_fee": Decimal("0"),
    },
    "GBP/USD": {
        "base": "GBP",
        "quote": "USD",
        "pip_size": Decimal("0.0001"),
        "pip_value_per_lot": Decimal("10"),
        "tick_size": Decimal("0.00001"),
        "price_precision": 5,
        "min_qty": Decimal("0.01"),
        "maker_fee": Decimal("0"),
        "taker_fee": Decimal("0"),
    },
    "AUD/USD": {
        "base": "AUD",
        "quote": "USD",
        "pip_size": Decimal("0.0001"),
        "pip_value_per_lot": Decimal("10"),
        "tick_size": Decimal("0.00001"),
        "price_precision": 5,
        "min_qty": Decimal("0.01"),
        "maker_fee": Decimal("0"),
        "taker_fee": Decimal("0"),
    },
    "NZD/USD": {
        "base": "NZD",
        "quote": "USD",
        "pip_size": Decimal("0.0001"),
        "pip_value_per_lot": Decimal("10"),
        "tick_size": Decimal("0.00001"),
        "price_precision": 5,
        "min_qty": Decimal("0.01"),
        "maker_fee": Decimal("0"),
        "taker_fee": Decimal("0"),
    },
    "USD/JPY": {
        "base": "USD",
        "quote": "JPY",
        "pip_size": Decimal("0.01"),
        "pip_value_per_lot": Decimal("1000"),
        "tick_size": Decimal("0.001"),
        "price_precision": 3,
        "min_qty": Decimal("0.01"),
        "maker_fee": Decimal("0"),
        "taker_fee": Decimal("0"),
    },
    "USD/CHF": {
        "base": "USD",
        "quote": "CHF",
        "pip_size": Decimal("0.0001"),
        "pip_value_per_lot": Decimal("10"),
        "tick_size": Decimal("0.00001"),
        "price_precision": 5,
        "min_qty": Decimal("0.01"),
        "maker_fee": Decimal("0"),
        "taker_fee": Decimal("0"),
    },
    "USD/CAD": {
        "base": "USD",
        "quote": "CAD",
        "pip_size": Decimal("0.0001"),
        "pip_value_per_lot": Decimal("10"),
        "tick_size": Decimal("0.00001"),
        "price_precision": 5,
        "min_qty": Decimal("0.01"),
        "maker_fee": Decimal("0"),
        "taker_fee": Decimal("0"),
    },
}

# OANDA venue symbol format: "EUR_USD" instead of "EUR/USD"
_OANDA_FORMAT = {sym: sym.replace("/", "_") for sym in _FX_SPECS}


def build_fx_instruments(
    symbols: list[str],
    exchange: Exchange = Exchange.OANDA,
) -> dict[str, Instrument]:
    """Build Instrument objects for known FX pairs.

    Parameters
    ----------
    symbols:
        List of unified symbols, e.g. ``["EUR/USD", "USD/JPY"]``.
    exchange:
        Target exchange enum (default ``OANDA``).

    Returns
    -------
    dict mapping symbol to :class:`Instrument`.
    Unknown symbols are silently skipped.
    """
    result: dict[str, Instrument] = {}
    for sym in symbols:
        spec = _FX_SPECS.get(sym)
        if spec is None:
            continue
        inst = Instrument(
            symbol=sym,
            exchange=exchange,
            instrument_type=InstrumentType.FX_SPOT,
            base=spec["base"],
            quote=spec["quote"],
            price_precision=spec["price_precision"],
            qty_precision=2,
            tick_size=spec["tick_size"],
            step_size=Decimal("0.01"),
            min_qty=spec["min_qty"],
            max_qty=Decimal("500"),
            min_notional=Decimal("1000"),
            max_leverage=50,
            maker_fee=spec["maker_fee"],
            taker_fee=spec["taker_fee"],
            asset_class=AssetClass.FX,
            pip_size=spec["pip_size"],
            pip_value_per_lot=spec["pip_value_per_lot"],
            lot_size=Decimal("100000"),
            venue_symbol=_OANDA_FORMAT.get(sym, sym),
            trading_sessions=list(_ALL_SESSIONS),
            weekend_close=True,
            rollover_enabled=True,
        )
        result[sym] = inst
    return result
