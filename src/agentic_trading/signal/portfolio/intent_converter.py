"""Convert TargetPosition events into OrderIntent events.

Bridges the gap between PortfolioManager (which produces targets) and
ExecutionEngine (which consumes order intents).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from agentic_trading.core.enums import (
    AssetClass,
    Exchange,
    OrderType,
    QtyUnit,
    Side,
    TimeInForce,
)
from agentic_trading.core.events import OrderIntent, TargetPosition
from agentic_trading.core.ids import content_hash
from agentic_trading.core.models import Instrument


def build_order_intents(
    targets: list[TargetPosition],
    exchange: Exchange,
    timestamp: datetime,
    order_type: OrderType = OrderType.MARKET,
    bucket_seconds: int = 60,
    instruments: dict[str, Instrument] | None = None,
) -> list[OrderIntent]:
    """Convert a list of TargetPosition events into OrderIntent events.

    Parameters
    ----------
    targets:
        Target positions from PortfolioManager.generate_targets().
    exchange:
        Exchange to route orders to.
    timestamp:
        Current clock time, used for deduplication bucketing.
    order_type:
        Order type (default MARKET).
    bucket_seconds:
        Time bucket for deduplication (default 60s).
    instruments:
        Optional instrument metadata for asset-class enrichment.
    """
    intents: list[OrderIntent] = []
    _instruments = instruments or {}

    for target in targets:
        # Build deterministic dedupe key from strategy+symbol+time bucket
        ts_bucket = int(timestamp.timestamp()) // bucket_seconds
        raw = f"{target.strategy_id}:{target.symbol}:{ts_bucket}"
        dedupe_key = content_hash(raw)

        # Enrich with instrument metadata
        inst = _instruments.get(target.symbol)
        asset_class = inst.asset_class if inst is not None else AssetClass.CRYPTO
        instrument_hash = inst.instrument_hash if inst is not None else ""

        intent = OrderIntent(
            dedupe_key=dedupe_key,
            strategy_id=target.strategy_id,
            symbol=target.symbol,
            exchange=exchange,
            side=target.side,
            order_type=order_type,
            time_in_force=TimeInForce.GTC,
            qty=target.target_qty,
            price=None,  # Market orders — no price needed
            trace_id=target.trace_id,
            asset_class=asset_class,
            qty_unit=QtyUnit.BASE,
            instrument_hash=instrument_hash,
        )
        intents.append(intent)

    return intents


def build_partial_exit_intents(
    symbol: str,
    strategy_id: str,
    exchange: Exchange,
    current_qty: Decimal,
    portions: list[float],
    timestamp: datetime,
    side: Side = Side.SELL,
    order_type: OrderType = OrderType.MARKET,
    bucket_seconds: int = 60,
    trace_id: str = "",
    instruments: dict[str, Instrument] | None = None,
) -> list[OrderIntent]:
    """Build a sequence of partial exit OrderIntents.

    Parameters
    ----------
    symbol:
        Trading pair to exit.
    strategy_id:
        Strategy that owns the position.
    exchange:
        Target exchange.
    current_qty:
        Current open position size (absolute value).
    portions:
        List of exit fractions summing to <= 1.0.
        Example: [0.5, 0.25, 0.25] exits 50%, then 25%, then 25%.
    timestamp:
        Current clock time for deduplication.
    side:
        Exit side (SELL to close longs, BUY to close shorts).
    order_type:
        Order type for exit orders.
    bucket_seconds:
        Time bucket for deduplication.
    trace_id:
        Correlation trace ID.
    instruments:
        Optional instrument metadata for asset-class enrichment.

    Returns
    -------
    list[OrderIntent]
        One intent per portion, each with ``reduce_only=True`` and
        ``exit_portion`` set.
    """
    intents: list[OrderIntent] = []
    remaining = current_qty
    _instruments = instruments or {}

    inst = _instruments.get(symbol)
    asset_class = inst.asset_class if inst is not None else AssetClass.CRYPTO
    instrument_hash = inst.instrument_hash if inst is not None else ""

    for idx, portion in enumerate(portions):
        if portion <= 0.0 or portion > 1.0:
            continue
        exit_qty = (current_qty * Decimal(str(portion))).quantize(
            Decimal("0.000001")
        )
        if exit_qty <= 0 or exit_qty > remaining:
            continue

        ts_bucket = int(timestamp.timestamp()) // bucket_seconds
        raw = f"{strategy_id}:{symbol}:{ts_bucket}:exit_{idx}"
        dedupe_key = content_hash(raw)

        intent = OrderIntent(
            dedupe_key=dedupe_key,
            strategy_id=strategy_id,
            symbol=symbol,
            exchange=exchange,
            side=side,
            order_type=order_type,
            time_in_force=TimeInForce.GTC,
            qty=exit_qty,
            price=None,
            reduce_only=True,
            exit_portion=portion,
            trace_id=trace_id,
            asset_class=asset_class,
            instrument_hash=instrument_hash,
        )
        intents.append(intent)
        remaining -= exit_qty

    return intents
