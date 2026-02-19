"""Position sizing methods.

Supports: volatility-adjusted, Kelly criterion, fixed-fractional.
All methods respect instrument constraints (min/max qty, notional).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

from agentic_trading.core.models import Instrument


def volatility_adjusted_size(
    capital: float,
    risk_per_trade_pct: float,
    atr: float,
    price: float,
    atr_multiplier: float = 2.0,
    instrument: Instrument | None = None,
) -> Decimal:
    """Volatility-adjusted position sizing using ATR.

    Size = (capital * risk_pct) / (ATR * multiplier)
    """
    if atr <= 0 or price <= 0:
        return Decimal("0")

    risk_amount = capital * risk_per_trade_pct
    stop_distance = atr * atr_multiplier
    qty = risk_amount / stop_distance

    result = Decimal(str(qty))
    if instrument:
        result = instrument.round_qty(result)
        result = max(instrument.min_qty, min(result, instrument.max_qty))

    return result


def fixed_fractional_size(
    capital: float,
    fraction: float,
    price: float,
    instrument: Instrument | None = None,
) -> Decimal:
    """Fixed-fractional sizing: allocate fraction of capital.

    Size = (capital * fraction) / price
    """
    if price <= 0:
        return Decimal("0")

    notional = capital * fraction
    qty = notional / price

    result = Decimal(str(qty))
    if instrument:
        result = instrument.round_qty(result)
        result = max(instrument.min_qty, min(result, instrument.max_qty))

    return result


def kelly_size(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    price: float,
    kelly_fraction: float = 0.25,  # Use fractional Kelly (safer)
    instrument: Instrument | None = None,
) -> Decimal:
    """Kelly criterion position sizing.

    Kelly% = W - (1-W)/R
    where W = win rate, R = avg_win/avg_loss ratio

    Uses fractional Kelly (default 25%) for safety.
    """
    if price <= 0 or avg_loss <= 0 or win_rate <= 0:
        return Decimal("0")

    r = avg_win / avg_loss  # Win/loss ratio
    kelly_pct = win_rate - (1 - win_rate) / r
    kelly_pct = max(0.0, kelly_pct) * kelly_fraction

    # Cap at 20% of capital
    kelly_pct = min(kelly_pct, 0.20)

    notional = capital * kelly_pct
    qty = notional / price

    result = Decimal(str(qty))
    if instrument:
        result = instrument.round_qty(result)
        result = max(instrument.min_qty, min(result, instrument.max_qty))

    return result


def liquidity_adjusted_size(
    base_qty: Decimal,
    liquidity_score: float,
    max_market_impact_pct: float = 0.01,
    instrument: Instrument | None = None,
) -> Decimal:
    """Reduce size based on liquidity score.

    If liquidity is low, scale down to avoid market impact.
    """
    if liquidity_score <= 0:
        return Decimal("0")

    scale = min(1.0, liquidity_score)
    adjusted = Decimal(str(float(base_qty) * scale))

    if instrument:
        adjusted = instrument.round_qty(adjusted)
        adjusted = max(instrument.min_qty, min(adjusted, instrument.max_qty))

    return adjusted


def stop_loss_based_size(
    capital: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss_price: float,
    instrument: Instrument | None = None,
) -> Decimal:
    """Position size based on explicit entry and stop-loss prices.

    Size = (capital * risk_pct) / |entry_price - stop_loss_price|

    Unlike :func:`volatility_adjusted_size` which derives the stop distance
    from ATR, this function uses explicit price levels provided by the trader.

    Args:
        capital: Account capital.
        risk_per_trade_pct: Risk as a decimal fraction (e.g. 0.01 for 1%).
        entry_price: Planned entry price.
        stop_loss_price: Stop-loss price level.
        instrument: Optional instrument for qty rounding / clamping.

    Returns:
        Position quantity in base units (Decimal).
    """
    if entry_price <= 0 or stop_loss_price <= 0 or capital <= 0:
        return Decimal("0")

    stop_distance = abs(entry_price - stop_loss_price)
    if stop_distance == 0:
        return Decimal("0")

    risk_amount = capital * risk_per_trade_pct
    qty = risk_amount / stop_distance

    result = Decimal(str(qty))
    if instrument:
        result = instrument.round_qty(result)
        result = max(instrument.min_qty, min(result, instrument.max_qty))

    return result


def scaled_entry_size(
    capital: float,
    risk_per_trade_pct: float,
    entries: list[tuple[float, float]],
    stop_loss_price: float,
    instrument: Instrument | None = None,
) -> list[tuple[Decimal, float]]:
    """Position sizing for scaled / laddered entries.

    Distributes the total risk budget across multiple entry price levels.
    The stop distance is measured from the *weighted-average* entry to
    ``stop_loss_price``, so total risk equals ``capital * risk_per_trade_pct``
    regardless of how many entry levels are used.

    Args:
        capital: Account capital.
        risk_per_trade_pct: Risk as a decimal fraction (e.g. 0.01 for 1%).
        entries: List of ``(price, allocation_weight)`` tuples.
            Weights are normalised internally; they need not sum to 1.0.
        stop_loss_price: Common stop-loss price.
        instrument: Optional instrument for qty rounding / clamping.

    Returns:
        List of ``(qty, price)`` tuples, one per entry level.
        Returns an empty list if inputs are invalid.
    """
    if not entries or stop_loss_price <= 0 or capital <= 0:
        return []

    total_weight = sum(w for _, w in entries)
    if total_weight <= 0:
        return []

    # Normalise allocations
    normalised = [(p, w / total_weight) for p, w in entries]

    # Weighted-average entry price
    avg_entry = sum(p * a for p, a in normalised)
    if avg_entry <= 0:
        return []

    stop_distance = abs(avg_entry - stop_loss_price)
    if stop_distance == 0:
        return []

    total_risk = capital * risk_per_trade_pct
    total_qty = total_risk / stop_distance

    result: list[tuple[Decimal, float]] = []
    for price, alloc in normalised:
        if price <= 0:
            continue
        qty = total_qty * alloc
        qty_dec = Decimal(str(qty))
        if instrument:
            qty_dec = instrument.round_qty(qty_dec)
            qty_dec = max(instrument.min_qty, min(qty_dec, instrument.max_qty))
        result.append((qty_dec, price))

    return result
