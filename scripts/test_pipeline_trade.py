"""Test: submit a trade through the full execution pipeline via Redis event bus.

Publishes an OrderIntent using the same wire format as RedisStreamsBus.publish(),
so the running ExecutionEngine picks it up, submits to Bybit demo via CCXTAdapter,
and the FillEvent flows to the Journal.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import redis.asyncio as redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("test_pipeline")

REDIS_URL = os.environ.get("TRADING_REDIS_URL", "redis://redis:6379/0")


def build_intent_payload(
    dedupe_key: str,
    trace_id: str,
    side: str = "buy",
    reduce_only: bool = False,
) -> dict[str, str]:
    """Build a Redis Stream message matching RedisStreamsBus._deserialize format."""
    from agentic_trading.core.events import OrderIntent
    from agentic_trading.core.enums import Exchange, Side, OrderType, TimeInForce

    intent = OrderIntent(
        trace_id=trace_id,
        dedupe_key=dedupe_key,
        strategy_id="test_manual",
        symbol="BTC/USDT",
        exchange=Exchange.BYBIT,
        side=Side.BUY if side == "buy" else Side.SELL,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        qty=Decimal("0.001"),
        price=None,
        reduce_only=reduce_only,
    )

    return {
        "_type": "OrderIntent",
        "_data": intent.model_dump_json(),
    }


async def wait_for_event(
    r: redis.Redis,
    stream: str,
    after_id: str,
    target_trace: str,
    target_type: str,
    timeout: float = 30,
) -> dict | None:
    """Wait for a specific event type with a matching trace_id."""
    last_id = after_id
    deadline = asyncio.get_event_loop().time() + timeout

    while asyncio.get_event_loop().time() < deadline:
        result = await r.xread({stream: last_id}, count=20, block=2000)
        if not result:
            continue
        for _stream_name, messages in result:
            for mid, fields in messages:
                last_id = mid
                _type = fields.get("_type", "")
                _data = fields.get("_data", "{}")

                if _type not in ("OrderAck", "FillEvent", "PositionUpdate"):
                    continue

                try:
                    evt = json.loads(_data)
                except json.JSONDecodeError:
                    continue

                evt_trace = evt.get("trace_id", "")
                if evt_trace != target_trace:
                    continue

                log.info("  >> %s: %s", _type, {
                    k: v for k, v in evt.items()
                    if k in ("order_id", "status", "fill_id", "side", "qty", "price", "symbol", "message")
                })

                if _type == target_type:
                    return evt

    return None


async def main() -> None:
    r = redis.from_url(REDIS_URL, decode_responses=True)

    trace_id = str(uuid.uuid4())
    dedupe_key = f"test-pipeline-{trace_id[:8]}"

    # ---- ENTRY (BUY) ----
    payload = build_intent_payload(dedupe_key, trace_id, side="buy")

    log.info("=== ENTRY ORDER ===")
    log.info("  dedupe_key=%s  symbol=BTC/USDT  side=buy  qty=0.001  type=market", dedupe_key)
    log.info("  trace_id=%s", trace_id)

    msg_id = await r.xadd("execution", payload)
    log.info("Published to Redis stream 'execution': msg_id=%s", msg_id)

    log.info("Waiting for OrderAck...")
    ack = await wait_for_event(r, "execution", msg_id, trace_id, "OrderAck", timeout=30)

    if ack:
        log.info("Waiting for FillEvent...")
        fill = await wait_for_event(r, "execution", msg_id, trace_id, "FillEvent", timeout=15)
        if fill:
            log.info("=== ENTRY FILL CONFIRMED ===")
        else:
            log.warning("No FillEvent received — check if status was FILLED in ack")
    else:
        log.error("No OrderAck received — ExecutionEngine may not be processing")

    # ---- Hold for 5 seconds ----
    log.info("")
    log.info("--- Holding for 5 seconds ---")
    await asyncio.sleep(5)

    # ---- EXIT (SELL, reduceOnly) ----
    close_trace = str(uuid.uuid4())
    close_dedupe = f"test-close-{close_trace[:8]}"

    close_payload = build_intent_payload(close_dedupe, close_trace, side="sell", reduce_only=True)

    log.info("=== EXIT ORDER ===")
    log.info("  dedupe_key=%s  side=sell  reduceOnly=True", close_dedupe)
    log.info("  trace_id=%s", close_trace)

    close_msg_id = await r.xadd("execution", close_payload)
    log.info("Published to Redis stream 'execution': msg_id=%s", close_msg_id)

    log.info("Waiting for close OrderAck...")
    close_ack = await wait_for_event(r, "execution", close_msg_id, close_trace, "OrderAck", timeout=30)

    if close_ack:
        log.info("Waiting for close FillEvent...")
        close_fill = await wait_for_event(r, "execution", close_msg_id, close_trace, "FillEvent", timeout=15)
        if close_fill:
            log.info("=== EXIT FILL CONFIRMED — Trade fully round-tripped ===")
        else:
            log.warning("No close FillEvent received")
    else:
        log.error("No close OrderAck received")

    log.info("")
    log.info("=== PIPELINE TEST COMPLETE ===")
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
