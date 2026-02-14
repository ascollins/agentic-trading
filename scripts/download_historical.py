#!/usr/bin/env python3
"""Download historical OHLCV data to Parquet files.

Usage:
  python scripts/download_historical.py --symbols BTC/USDT,ETH/USDT --since 2024-01-01
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import click

sys.path.insert(0, "src")


@click.command()
@click.option("--symbols", required=True, help="Comma-separated symbols")
@click.option("--exchange", default="binance", help="Exchange name")
@click.option("--timeframe", default="1m", help="Candle timeframe")
@click.option("--since", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--until", default=None, help="End date (YYYY-MM-DD)")
@click.option("--output-dir", default="data/historical", help="Output directory")
def main(
    symbols: str,
    exchange: str,
    timeframe: str,
    since: str,
    until: str | None,
    output_dir: str,
) -> None:
    """Download historical OHLCV data."""
    asyncio.run(
        _download(
            symbols.split(","),
            exchange,
            timeframe,
            since,
            until,
            output_dir,
        )
    )


async def _download(
    symbols: list[str],
    exchange_name: str,
    timeframe: str,
    since: str,
    until: str | None,
    output_dir: str,
) -> None:
    import ccxt.async_support as ccxt
    import pandas as pd

    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls({"enableRateLimit": True})

    since_ts = int(datetime.strptime(since, "%Y-%m-%d").timestamp() * 1000)
    until_ts = (
        int(datetime.strptime(until, "%Y-%m-%d").timestamp() * 1000)
        if until
        else None
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        for symbol in symbols:
            click.echo(f"Downloading {symbol} {timeframe} from {since}...")
            all_candles = []
            current_since = since_ts

            while True:
                candles = await exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=1000
                )
                if not candles:
                    break

                all_candles.extend(candles)
                current_since = candles[-1][0] + 1

                if until_ts and current_since >= until_ts:
                    break

                click.echo(f"  {len(all_candles)} candles...")

            if all_candles:
                df = pd.DataFrame(
                    all_candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

                # Save in HistoricalDataLoader layout:
                #   {output_dir}/{exchange}/{BTC-USDT}/{1m}.parquet
                slug = symbol.replace("/", "-").replace(":", "-")
                symbol_dir = out / exchange_name / slug
                symbol_dir.mkdir(parents=True, exist_ok=True)
                path = symbol_dir / f"{timeframe}.parquet"
                df.to_parquet(path, index=False)
                click.echo(f"  Saved {len(df)} candles to {path}")
    finally:
        await exchange.close()


if __name__ == "__main__":
    main()
