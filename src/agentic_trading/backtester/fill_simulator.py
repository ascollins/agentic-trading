"""Fill simulation for backtesting.

Simulates partial fills, latency injection, and realistic execution.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from decimal import Decimal

from agentic_trading.core.enums import Exchange, OrderStatus, OrderType, Side
from agentic_trading.core.events import FillEvent, OrderIntent
from agentic_trading.core.models import Candle

from .fee_model import FeeModel
from .slippage import SlippageModel


class FillSimulator:
    """Simulates order fills against candle data."""

    def __init__(
        self,
        slippage_model: SlippageModel,
        fee_model: FeeModel,
        partial_fills: bool = True,
        latency_ms: int = 50,
        seed: int = 42,
    ) -> None:
        self._slippage = slippage_model
        self._fees = fee_model
        self._partial_fills = partial_fills
        self._latency_ms = latency_ms
        self._rng = random.Random(seed)

    def simulate_fill(
        self,
        intent: OrderIntent,
        candle: Candle,
        current_time: datetime,
    ) -> list[FillEvent]:
        """Simulate filling an order against a candle.

        For market orders: fill at candle OHLC with slippage.
        For limit orders: fill only if price is within candle range.
        Partial fills: split into 1-3 fills based on volume.

        Returns list of FillEvent (may be empty if not filled).
        """
        fills: list[FillEvent] = []
        is_buy = intent.side == Side.BUY
        qty = float(intent.qty)

        # Check if order can be filled
        if intent.order_type == OrderType.LIMIT:
            if intent.price is None:
                return []
            limit_price = float(intent.price)
            # Buy limit: fills if candle low <= limit price
            if is_buy and candle.low > limit_price:
                return []
            # Sell limit: fills if candle high >= limit price
            if not is_buy and candle.high < limit_price:
                return []
            base_price = limit_price
        else:
            # Market order: use open price (next candle's open in realistic sim)
            base_price = candle.open

        # Apply slippage
        exec_price = self._slippage.compute_slippage(
            price=base_price,
            qty=qty,
            is_buy=is_buy,
            atr=candle.high - candle.low,  # Use candle range as ATR proxy
            volume=candle.volume,
        )

        # Ensure price is within candle range
        exec_price = max(candle.low, min(exec_price, candle.high))

        # Partial fills
        if self._partial_fills and qty > 0:
            fill_splits = self._split_fills(qty, candle.volume)
        else:
            fill_splits = [qty]

        # Apply latency
        fill_time = current_time + timedelta(milliseconds=self._latency_ms)

        for fill_qty in fill_splits:
            if fill_qty <= 0:
                continue

            # Slight price variation for each partial fill
            price_jitter = self._rng.uniform(-0.0001, 0.0001)
            fill_price = exec_price * (1 + price_jitter)

            fee = self._fees.compute_fee(
                price=fill_price,
                qty=fill_qty,
                is_maker=intent.order_type == OrderType.LIMIT,
            )

            fills.append(
                FillEvent(
                    fill_id=str(uuid.uuid4()),
                    order_id=intent.dedupe_key,
                    client_order_id=intent.dedupe_key,
                    symbol=intent.symbol,
                    exchange=intent.exchange,
                    side=intent.side,
                    price=Decimal(str(round(fill_price, 8))),
                    qty=Decimal(str(round(fill_qty, 8))),
                    fee=fee,
                    fee_currency=intent.symbol.split("/")[-1] if "/" in intent.symbol else "USDT",
                    is_maker=intent.order_type == OrderType.LIMIT,
                    timestamp=fill_time,
                )
            )

            # Stagger fill times
            fill_time += timedelta(milliseconds=self._rng.randint(10, 100))

        return fills

    def _split_fills(self, total_qty: float, volume: float) -> list[float]:
        """Split an order into partial fills.

        If order is > 5% of candle volume, split into 2-3 fills.
        """
        if volume <= 0:
            return [total_qty]

        participation = total_qty / volume
        if participation < 0.05:
            return [total_qty]

        # Split into 2-3 fills
        n_fills = self._rng.randint(2, 3)
        splits = []
        remaining = total_qty

        for i in range(n_fills - 1):
            pct = self._rng.uniform(0.2, 0.5)
            fill = remaining * pct
            splits.append(fill)
            remaining -= fill

        splits.append(remaining)
        return splits
