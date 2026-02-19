"""Convert TargetPosition events into OrderIntent events.

Bridges the gap between PortfolioManager (which produces targets) and
ExecutionEngine (which consumes order intents).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from decimal import Decimal

from agentic_trading.core.enums import Exchange, OrderType, Side, TimeInForce
from agentic_trading.core.events import OrderIntent, TargetPosition


def build_order_intents(
    targets: list[TargetPosition],
    exchange: Exchange,
    timestamp: datetime,
    order_type: OrderType = OrderType.MARKET,
    bucket_seconds: int = 60,
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
    """
    intents: list[OrderIntent] = []

    for target in targets:
        # Build deterministic dedupe key from strategy+symbol+time bucket
        ts_bucket = int(timestamp.timestamp()) // bucket_seconds
        raw = f"{target.strategy_id}:{target.symbol}:{ts_bucket}"
        dedupe_key = hashlib.sha256(raw.encode()).hexdigest()[:16]

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
        )
        intents.append(intent)

    return intents
