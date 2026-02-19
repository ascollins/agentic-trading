"""One-shot script to set TP/SL on existing Bybit positions that lack them.

Uses the same logic as main.py startup reconciliation:
    SL = entry_price - (ATR_est × sl_atr_multiplier)  for longs
    TP = entry_price + (ATR_est × tp_atr_multiplier)  for longs
    ATR_est = entry_price × 0.004  (0.4% fallback when no candle data)

Config defaults from live.toml:
    sl_atr_multiplier = 2.5  → SL distance = 1.0% of entry
    tp_atr_multiplier = 5.0  → TP distance = 2.0% of entry

Usage:
    python3 scripts/set_tp_sl.py [--dry-run]
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("set_tp_sl")

# Load .env file if present
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key not in os.environ:
                    os.environ[key] = value

# Verify safety environment
if not os.environ.get("I_UNDERSTAND_LIVE_TRADING", "").lower() == "true":
    logger.error("Set I_UNDERSTAND_LIVE_TRADING=true to run this script.")
    sys.exit(1)


async def main() -> None:
    dry_run = "--dry-run" in sys.argv

    # ---- Load config ----
    from agentic_trading.core.config import load_settings

    settings = load_settings("configs/live.toml")
    exit_cfg = settings.exits

    if not exit_cfg.enabled:
        logger.info("Exit config disabled, nothing to do.")
        return

    # ---- Create exchange adapter ----
    from agentic_trading.execution.adapters.ccxt_adapter import CCXTAdapter

    exc_cfg = settings.exchanges[0]

    adapter = CCXTAdapter(
        exchange_name=exc_cfg.name.value,
        api_key=exc_cfg.api_key,
        api_secret=exc_cfg.secret,
        sandbox=exc_cfg.testnet,
        demo=exc_cfg.demo,
        default_type="swap",
    )

    try:
        # ---- Fetch positions ----
        positions = await adapter.get_positions()
        open_positions = [p for p in positions if float(p.qty) != 0]

        if not open_positions:
            logger.info("No open positions found.")
            return

        logger.info("Found %d open positions", len(open_positions))

        # ---- Check raw Bybit data for existing TP/SL ----
        for pos in open_positions:
            try:
                raw_positions = await adapter._ccxt.fetch_positions([pos.symbol])
                # CCXT normalises Bybit perpetual symbols to 'BTC/USDT:USDT'
                # (with settle suffix) so we match on both plain and suffixed form.
                bybit_pos = next(
                    (p for p in raw_positions
                     if (p.get("symbol") == pos.symbol
                         or p.get("symbol", "").split(":")[0] == pos.symbol)),
                    None,
                )

                # takeProfitPrice / stopLossPrice / trailingStop are CCXT-normalised
                # top-level numeric fields (None or 0.0 when unset).
                has_tp = float((bybit_pos or {}).get("takeProfitPrice") or 0) > 0
                has_sl = float((bybit_pos or {}).get("stopLossPrice") or 0) > 0
                # trailingStop lives in the raw Bybit info dict; CCXT doesn't
                # normalise it to a top-level field.
                _info = (bybit_pos or {}).get("info") or {}
                has_trail = float(_info.get("trailingStop") or 0) > 0

                current_tp = (bybit_pos or {}).get("takeProfitPrice") or 0
                current_sl = (bybit_pos or {}).get("stopLossPrice") or 0
                current_trail = _info.get("trailingStop") or 0

                logger.info(
                    "%s: side=%s entry=%s qty=%s | current TP=%s SL=%s Trail=%s",
                    pos.symbol,
                    pos.side.value,
                    pos.entry_price,
                    pos.qty,
                    current_tp,
                    current_sl,
                    current_trail,
                )

                if has_tp and has_sl and has_trail:
                    logger.info("  → Already has TP/SL/Trail, skipping")
                    continue

                # ---- Compute TP/SL ----
                entry = pos.entry_price
                atr_est = entry * Decimal("0.004")
                sl_distance = atr_est * Decimal(str(exit_cfg.sl_atr_multiplier))
                tp_distance = atr_est * Decimal(str(exit_cfg.tp_atr_multiplier))

                direction = pos.side.value  # "long" or "short"
                if direction == "long":
                    sl_price = entry - sl_distance
                    tp_price = entry + tp_distance
                else:
                    sl_price = entry + sl_distance
                    tp_price = entry - tp_distance

                # Compute trailing stop distance + breakeven active_price.
                # All 5 default strategies (BTC/ETH/SOL/XRP/DOGE) are in the
                # trailing_strategies list, so apply trailing to all positions.
                trail_multiplier = Decimal(str(exit_cfg.trailing_stop_atr_multiplier))
                trail_distance = atr_est * trail_multiplier
                # active_price = entry ± sl_distance (1× SL in profit = breakeven)
                if direction == "long":
                    active_price = entry + sl_distance
                else:
                    active_price = entry - sl_distance

                # Round to sensible precision based on price magnitude
                if entry > 10000:
                    quant = Decimal("0.01")
                elif entry > 100:
                    quant = Decimal("0.01")
                elif entry > 1:
                    quant = Decimal("0.0001")
                else:
                    quant = Decimal("0.00001")

                sl_price = sl_price.quantize(quant, rounding=ROUND_HALF_UP)
                tp_price = tp_price.quantize(quant, rounding=ROUND_HALF_UP)
                trail_distance = trail_distance.quantize(quant, rounding=ROUND_HALF_UP)
                active_price = active_price.quantize(quant, rounding=ROUND_HALF_UP)

                logger.info(
                    "  → Computed: TP=%s (+%.2f%%) SL=%s (-%.2f%%) "
                    "Trail=%s ActivePrice=%s [ATR_est=%s]",
                    tp_price,
                    float((tp_price - entry) / entry * 100),
                    sl_price,
                    float((entry - sl_price) / entry * 100),
                    trail_distance,
                    active_price,
                    atr_est,
                )

                if dry_run:
                    logger.info(
                        "  → DRY RUN: would set TP=%s SL=%s Trail=%s ActivePrice=%s",
                        tp_price, sl_price, trail_distance, active_price,
                    )
                    continue

                # ---- Apply to Bybit ----
                tp_to_set = tp_price if not has_tp else None
                sl_to_set = sl_price if not has_sl else None

                result = await adapter.set_trading_stop(
                    pos.symbol,
                    take_profit=tp_to_set,
                    stop_loss=sl_to_set,
                    trailing_stop=trail_distance,
                    active_price=active_price,
                )
                logger.info(
                    "  ✓ TP/SL+Trail SET: %s tp=%s sl=%s trail=%s active_price=%s (result=%s)",
                    pos.symbol, tp_to_set, sl_to_set,
                    trail_distance, active_price,
                    result.get("retMsg", "ok"),
                )

            except Exception:
                logger.exception("  ✗ Failed to set TP/SL for %s", pos.symbol)

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
