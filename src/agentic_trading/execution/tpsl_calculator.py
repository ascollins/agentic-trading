"""TP/SL level calculator with multiple distance methods.

Supports five methods for computing take-profit and stop-loss levels:

1. **Price** — absolute price level.
2. **Percentage** — percentage of entry price.
3. **ATR** — ATR-multiple from entry (existing pattern).
4. **Ticks** — number of tick_size units from entry.
5. **Currency** — fixed quote-currency distance from entry.

Usage::

    from agentic_trading.execution.tpsl_calculator import compute_tpsl

    levels = compute_tpsl(
        entry_price=Decimal("50000"),
        side="long",
        method="ticks",
        tp_distance=200,
        sl_distance=100,
        tick_size=Decimal("0.01"),
    )
    print(levels)
    # TpSlLevels(take_profit=Decimal('50002.00'),
    #            stop_loss=Decimal('49999.00'))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TpSlLevels:
    """Computed TP/SL price levels."""

    take_profit: Decimal | None = None
    stop_loss: Decimal | None = None
    trailing_stop_distance: Decimal | None = None


def compute_tpsl(
    entry_price: Decimal,
    side: str,
    method: str = "percentage",
    tp_distance: float | Decimal | None = None,
    sl_distance: float | Decimal | None = None,
    trailing_distance: float | Decimal | None = None,
    tick_size: Decimal | None = None,
    atr_value: Decimal | None = None,
) -> TpSlLevels:
    """Compute TP/SL levels using the specified distance method.

    Parameters
    ----------
    entry_price:
        Position entry price.
    side:
        ``"long"`` or ``"short"``.
    method:
        Distance calculation method. One of:
        ``"price"`` — tp/sl_distance are absolute price levels.
        ``"percentage"`` — tp/sl_distance are percentage values (e.g. 2.0 = 2%).
        ``"atr"`` — tp/sl_distance are ATR multiples.
        ``"ticks"`` — tp/sl_distance are integer tick counts.
        ``"currency"`` — tp/sl_distance are fixed quote-currency amounts.
    tp_distance:
        Take-profit distance (interpretation depends on *method*).
    sl_distance:
        Stop-loss distance (interpretation depends on *method*).
    trailing_distance:
        Trailing stop distance (same interpretation as *method*).
    tick_size:
        Instrument tick size. Required when method is ``"ticks"``.
    atr_value:
        Current ATR value. Required when method is ``"atr"``.

    Returns
    -------
    TpSlLevels
        Computed TP and SL price levels.

    Raises
    ------
    ValueError
        If a required parameter for the chosen method is missing.
    """
    is_long = side.lower() == "long"
    tp_price: Decimal | None = None
    sl_price: Decimal | None = None
    trail_dist: Decimal | None = None

    if method == "price":
        # Distances are already absolute price levels
        if tp_distance is not None:
            tp_price = Decimal(str(tp_distance))
        if sl_distance is not None:
            sl_price = Decimal(str(sl_distance))
        if trailing_distance is not None:
            trail_dist = Decimal(str(trailing_distance))

    elif method == "percentage":
        if tp_distance is not None:
            pct = Decimal(str(tp_distance)) / Decimal("100")
            offset = entry_price * pct
            tp_price = entry_price + offset if is_long else entry_price - offset
        if sl_distance is not None:
            pct = Decimal(str(sl_distance)) / Decimal("100")
            offset = entry_price * pct
            sl_price = entry_price - offset if is_long else entry_price + offset
        if trailing_distance is not None:
            pct = Decimal(str(trailing_distance)) / Decimal("100")
            trail_dist = entry_price * pct

    elif method == "atr":
        if atr_value is None:
            raise ValueError("atr_value is required when method='atr'")
        atr = Decimal(str(atr_value))
        if tp_distance is not None:
            offset = atr * Decimal(str(tp_distance))
            tp_price = entry_price + offset if is_long else entry_price - offset
        if sl_distance is not None:
            offset = atr * Decimal(str(sl_distance))
            sl_price = entry_price - offset if is_long else entry_price + offset
        if trailing_distance is not None:
            trail_dist = atr * Decimal(str(trailing_distance))

    elif method == "ticks":
        if tick_size is None:
            raise ValueError("tick_size is required when method='ticks'")
        ts = Decimal(str(tick_size))
        if tp_distance is not None:
            offset = ts * Decimal(str(int(tp_distance)))
            tp_price = entry_price + offset if is_long else entry_price - offset
        if sl_distance is not None:
            offset = ts * Decimal(str(int(sl_distance)))
            sl_price = entry_price - offset if is_long else entry_price + offset
        if trailing_distance is not None:
            trail_dist = ts * Decimal(str(int(trailing_distance)))

    elif method == "currency":
        if tp_distance is not None:
            offset = Decimal(str(tp_distance))
            tp_price = entry_price + offset if is_long else entry_price - offset
        if sl_distance is not None:
            offset = Decimal(str(sl_distance))
            sl_price = entry_price - offset if is_long else entry_price + offset
        if trailing_distance is not None:
            trail_dist = Decimal(str(trailing_distance))

    else:
        raise ValueError(f"Unknown TP/SL method: {method!r}")

    return TpSlLevels(
        take_profit=tp_price,
        stop_loss=sl_price,
        trailing_stop_distance=trail_dist,
    )
