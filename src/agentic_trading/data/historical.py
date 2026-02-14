"""Historical data loader for backtesting.

``HistoricalDataLoader`` reads OHLCV candles from Parquet files on disk and
yields them in chronological order, optionally filtered by date range.  It is
the primary data source for the backtest engine, which replays candles through
the same event-driven pipeline used in live trading.

Parquet file layout
-------------------
The loader expects the following directory convention::

    {data_dir}/{exchange}/{symbol_slug}/{timeframe}.parquet

Where:
    * ``symbol_slug`` is the unified symbol with ``/`` replaced by ``-``
      (e.g. ``BTC-USDT``).
    * Each Parquet file contains columns:
      ``timestamp`` (datetime64[ms, UTC]), ``open``, ``high``, ``low``,
      ``close``, ``volume``, and optionally ``quote_volume``, ``trades``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterator

import pyarrow.parquet as pq

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.models import Candle

logger = logging.getLogger(__name__)


def _symbol_to_slug(symbol: str) -> str:
    """Convert a unified symbol to a filesystem-safe slug.

    ``"BTC/USDT"`` -> ``"BTC-USDT"``
    """
    return symbol.replace("/", "-")


def _resolve_parquet_path(
    data_dir: str | Path,
    exchange: Exchange,
    symbol: str,
    timeframe: Timeframe,
) -> Path:
    """Build the expected Parquet file path."""
    slug = _symbol_to_slug(symbol)
    return Path(data_dir) / exchange.value / slug / f"{timeframe.value}.parquet"


class HistoricalDataLoader:
    """Loads candle data from Parquet files for backtesting.

    Parameters
    ----------
    data_dir:
        Root directory for historical data.  Defaults to ``"data/historical"``.
    """

    def __init__(self, data_dir: str | Path = "data/historical") -> None:
        self._data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_candles(
        self,
        exchange: Exchange,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        """Load candles synchronously and return as a sorted list.

        Parameters
        ----------
        exchange:
            Exchange enum value.
        symbol:
            Unified symbol (e.g. ``"BTC/USDT"``).
        timeframe:
            Candle timeframe.
        start:
            Inclusive lower bound on candle open time (UTC).
        end:
            Exclusive upper bound on candle open time (UTC).

        Returns
        -------
        list[Candle]
            Candles in ascending chronological order.
        """
        return list(self.iter_candles(exchange, symbol, timeframe, start, end))

    def iter_candles(
        self,
        exchange: Exchange,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[Candle]:
        """Yield candles lazily in chronological order.

        Memory-efficient for large datasets: reads the Parquet file in a
        streaming fashion via PyArrow's ``read_table`` with row-group
        iteration, then yields one ``Candle`` at a time.
        """
        path = _resolve_parquet_path(
            self._data_dir, exchange, symbol, timeframe
        )
        if not path.exists():
            logger.warning("Parquet file not found: %s", path)
            return

        logger.info(
            "Loading historical candles: %s %s %s [%s .. %s)",
            exchange.value,
            symbol,
            timeframe.value,
            start.isoformat() if start else "beginning",
            end.isoformat() if end else "end",
        )

        table = pq.read_table(str(path))
        df = table.to_pandas()

        # Ensure timestamp column is timezone-aware UTC.
        if df.empty:
            logger.warning("Parquet file is empty: %s", path)
            return

        ts_col = "timestamp"
        if ts_col not in df.columns:
            raise ValueError(
                f"Parquet file {path} missing required column '{ts_col}'. "
                f"Available columns: {list(df.columns)}"
            )

        # Convert to UTC-aware datetime if not already.
        if df[ts_col].dt.tz is None:
            df[ts_col] = df[ts_col].dt.tz_localize("UTC")
        else:
            df[ts_col] = df[ts_col].dt.tz_convert("UTC")

        # Sort by timestamp ascending.
        df = df.sort_values(ts_col).reset_index(drop=True)

        # Apply date filters.
        if start is not None:
            _start = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
            df = df[df[ts_col] >= _start]
        if end is not None:
            _end = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
            df = df[df[ts_col] < _end]

        # Optional columns with defaults.
        has_quote_volume = "quote_volume" in df.columns
        has_trades = "trades" in df.columns

        count = 0
        for row in df.itertuples(index=False):
            ts = row.timestamp.to_pydatetime()
            # Ensure timezone-aware.
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            candle = Candle(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                timestamp=ts,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                quote_volume=float(row.quote_volume) if has_quote_volume else 0.0,
                trades=int(row.trades) if has_trades else 0,
                is_closed=True,
            )
            yield candle
            count += 1

        logger.info("Loaded %d candles from %s", count, path)

    async def aiter_candles(
        self,
        exchange: Exchange,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> AsyncIterator[Candle]:
        """Async wrapper around ``iter_candles`` for use in async replay loops.

        The actual file I/O is synchronous (PyArrow), but wrapping in an
        async iterator lets the backtest engine interleave candle processing
        with other coroutines.
        """
        for candle in self.iter_candles(exchange, symbol, timeframe, start, end):
            yield candle

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def available_symbols(self, exchange: Exchange) -> list[str]:
        """List symbols that have Parquet data for the given exchange.

        Returns unified symbols (e.g. ``["BTC/USDT", "ETH/USDT"]``).
        """
        exchange_dir = self._data_dir / exchange.value
        if not exchange_dir.is_dir():
            return []

        symbols: list[str] = []
        for child in sorted(exchange_dir.iterdir()):
            if child.is_dir():
                # Convert slug back to unified symbol.
                symbol = child.name.replace("-", "/")
                symbols.append(symbol)
        return symbols

    def available_timeframes(
        self, exchange: Exchange, symbol: str
    ) -> list[Timeframe]:
        """List timeframes available for a given exchange/symbol pair."""
        slug = _symbol_to_slug(symbol)
        symbol_dir = self._data_dir / exchange.value / slug
        if not symbol_dir.is_dir():
            return []

        timeframes: list[Timeframe] = []
        tf_values = {tf.value for tf in Timeframe}
        for path in sorted(symbol_dir.glob("*.parquet")):
            stem = path.stem  # e.g. "1m", "5m"
            if stem in tf_values:
                timeframes.append(Timeframe(stem))
        return timeframes

    def get_date_range(
        self,
        exchange: Exchange,
        symbol: str,
        timeframe: Timeframe,
    ) -> tuple[datetime, datetime] | None:
        """Return (earliest, latest) candle timestamps in the Parquet file.

        Returns ``None`` if the file does not exist or is empty.
        """
        path = _resolve_parquet_path(
            self._data_dir, exchange, symbol, timeframe
        )
        if not path.exists():
            return None

        table = pq.read_table(str(path), columns=["timestamp"])
        if table.num_rows == 0:
            return None

        ts_array = table.column("timestamp").to_pandas()
        earliest = ts_array.min().to_pydatetime()
        latest = ts_array.max().to_pydatetime()

        if earliest.tzinfo is None:
            earliest = earliest.replace(tzinfo=timezone.utc)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)

        return earliest, latest
