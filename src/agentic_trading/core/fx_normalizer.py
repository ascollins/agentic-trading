"""FX-specific normalization utilities.

Used by adapters, the feature engine, and the policy evaluator to convert
between pip-based and price-based representations and to enforce
session/rollover awareness.
"""

from __future__ import annotations

from decimal import Decimal

from .enums import AssetClass, QtyUnit
from .models import Instrument


def spread_in_pips(
    bid: Decimal, ask: Decimal, instrument: Instrument
) -> Decimal:
    """Compute bid-ask spread in pips."""
    if instrument.pip_size is None or instrument.pip_size == Decimal("0"):
        return Decimal("0")
    return (Decimal(str(ask)) - Decimal(str(bid))) / instrument.pip_size


def notional_usd(
    qty: Decimal,
    price: Decimal,
    instrument: Instrument,
    *,
    cross_rate_to_usd: Decimal = Decimal("1"),
) -> Decimal:
    """Compute approximate USD notional.

    For USD-denominated pairs this is exact.  For cross pairs, the caller
    must supply an external ``cross_rate_to_usd``.
    """
    raw = instrument.notional_value(qty, price)
    if instrument.quote == "USD":
        return raw
    if instrument.base == "USD":
        # e.g. USD/JPY: notional in base IS USD
        q = Decimal(str(qty))
        if instrument.asset_class == AssetClass.FX:
            return q * instrument.lot_size
        return q
    # Cross pair: caller enriches
    return raw * cross_rate_to_usd


def pip_value_in_account_ccy(
    instrument: Instrument,
    lots: Decimal,
    account_ccy: str = "USD",
    cross_rate: Decimal = Decimal("1"),
) -> Decimal:
    """Pip value for position sizing.

    Standard formula:
        pip_value = pip_value_per_lot * lots
    When account_ccy differs from quote, multiply by cross_rate.
    """
    if instrument.pip_value_per_lot is not None:
        base_pv = instrument.pip_value_per_lot * Decimal(str(lots))
    elif instrument.pip_size is not None:
        base_pv = (
            instrument.pip_size
            * instrument.lot_size
            * Decimal(str(lots))
        )
    else:
        return Decimal("0")

    if instrument.quote == account_ccy:
        return base_pv
    return base_pv * cross_rate


def is_session_open(
    instrument: Instrument,
    utc_hour: int,
    utc_minute: int,
    weekday: int,
) -> bool:
    """Check if any trading session is currently open.

    Parameters
    ----------
    weekday:
        ISO weekday (1=Monday .. 7=Sunday).

    Returns ``True`` if no sessions are defined (e.g. crypto is 24/7).
    """
    if not instrument.trading_sessions:
        return True
    if instrument.weekend_close and weekday in (6, 7):
        return False
    current_mins = utc_hour * 60 + utc_minute
    for session in instrument.trading_sessions:
        if weekday not in session.days:
            continue
        parts_open = session.open_utc.split(":")
        parts_close = session.close_utc.split(":")
        open_mins = int(parts_open[0]) * 60 + int(parts_open[1])
        close_mins = int(parts_close[0]) * 60 + int(parts_close[1])
        if close_mins > open_mins:
            # Normal same-day window
            if open_mins <= current_mins < close_mins:
                return True
        else:
            # Overnight window (e.g. 22:00–07:00)
            if current_mins >= open_mins or current_mins < close_mins:
                return True
    return False


def rollover_cost(
    instrument: Instrument, qty: Decimal, is_long: bool
) -> Decimal:
    """Daily rollover/swap cost for holding a position overnight.

    Returns a signed value (negative = cost, positive = credit).
    """
    if not instrument.rollover_enabled:
        return Decimal("0")
    rate = instrument.long_swap_rate if is_long else instrument.short_swap_rate
    q = Decimal(str(qty))
    if instrument.asset_class == AssetClass.FX:
        return q * instrument.lot_size * rate
    return q * rate


def round_to_tick(price: Decimal, tick_size: Decimal) -> Decimal:
    """Quantize *price* to the nearest tick boundary."""
    d = Decimal(str(price))
    return (d / tick_size).quantize(Decimal("1")) * tick_size


def normalize_order_qty(
    qty: Decimal, unit: QtyUnit, instrument: Instrument
) -> Decimal:
    """Convert lots to base if needed, then round to step_size."""
    base_qty = instrument.normalize_qty(Decimal(str(qty)), unit)
    return instrument.round_qty(base_qty)
