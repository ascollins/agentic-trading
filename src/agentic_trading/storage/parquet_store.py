"""Parquet-based storage for OHLCV candle data.

Provides efficient columnar read/write for historical market data using
PyArrow. Data is partitioned by ``symbol`` and ``timeframe`` for fast
range scans without reading unrelated files.

Directory layout::

    {base_dir}/
        candles/
            symbol=BTC_USDT/
                timeframe=1m/
                    part-0000.parquet
                    part-0001.parquet
                timeframe=1h/
                    ...
            symbol=ETH_USDT/
                ...

Symbol names are sanitised (``/`` replaced with ``_``) to be
filesystem-safe.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from agentic_trading.core.enums import Timeframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arrow schema for OHLCV candles
# ---------------------------------------------------------------------------

CANDLE_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("quote_volume", pa.float64()),
        pa.field("trades", pa.int64()),
    ]
)


def _sanitize_symbol(symbol: str) -> str:
    """Convert ``BTC/USDT`` to ``BTC_USDT`` for filesystem safety."""
    return symbol.replace("/", "_")


def _unsanitize_symbol(sanitized: str) -> str:
    """Convert ``BTC_USDT`` back to ``BTC/USDT``."""
    return sanitized.replace("_", "/")


class ParquetStore:
    """Read/write OHLCV candle data in partitioned Parquet format.

    Args:
        base_dir: Root directory for the parquet data store.
            Defaults to ``"data/historical"``.
    """

    def __init__(self, base_dir: str | Path = "data/historical") -> None:
        self._base = Path(base_dir)
        self._candles_dir = self._base / "candles"
        # Ensure the root directory exists
        self._candles_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ParquetStore initialised at %s", self._candles_dir)

    # -- helpers -------------------------------------------------------------

    def _partition_dir(self, symbol: str, timeframe: str) -> Path:
        """Return the partition directory for a given symbol + timeframe."""
        safe_symbol = _sanitize_symbol(symbol)
        return self._candles_dir / f"symbol={safe_symbol}" / f"timeframe={timeframe}"

    # -- write ---------------------------------------------------------------

    def write_candles(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        df: pd.DataFrame,
        *,
        coalesce: bool = True,
    ) -> Path:
        """Write a DataFrame of OHLCV candles to partitioned Parquet.

        The DataFrame must contain columns matching :data:`CANDLE_SCHEMA`:
        ``timestamp, open, high, low, close, volume`` (plus optional
        ``quote_volume, trades``).

        Args:
            symbol: Unified symbol string (e.g. ``"BTC/USDT"``).
            timeframe: Timeframe enum value or string (e.g. ``"1h"``).
            df: DataFrame with candle data.
            coalesce: If ``True`` (default), merge with any existing data
                in the partition, de-duplicating on ``timestamp``.

        Returns:
            Path to the written Parquet file.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
        """
        if isinstance(timeframe, Timeframe):
            timeframe = timeframe.value

        if df.empty:
            raise ValueError("Cannot write empty DataFrame.")

        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Ensure optional columns exist with defaults
        if "quote_volume" not in df.columns:
            df = df.assign(quote_volume=0.0)
        if "trades" not in df.columns:
            df = df.assign(trades=0)

        # Ensure timestamp column is timezone-aware UTC
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
        elif df["timestamp"].dt.tz is None:
            df = df.assign(timestamp=df["timestamp"].dt.tz_localize("UTC"))

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        partition_dir = self._partition_dir(symbol, timeframe)
        partition_dir.mkdir(parents=True, exist_ok=True)
        out_path = partition_dir / "data.parquet"

        if coalesce and out_path.exists():
            # Read existing data and merge
            existing_df = pd.read_parquet(out_path)
            if not existing_df.empty:
                combined = pd.concat([existing_df, df], ignore_index=True)
                combined = (
                    combined
                    .drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                df = combined

        # Select only the schema columns in the correct order
        cols = [f.name for f in CANDLE_SCHEMA]
        df = df[cols]

        # Write via PyArrow for schema enforcement
        table = pa.Table.from_pandas(df, schema=CANDLE_SCHEMA, preserve_index=False)
        pq.write_table(
            table,
            out_path,
            compression="snappy",
            write_statistics=True,
        )

        logger.info(
            "Wrote %d candles for %s/%s -> %s",
            len(df), symbol, timeframe, out_path,
        )
        return out_path

    # -- read ----------------------------------------------------------------

    def read_candles(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        *,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read OHLCV candle data for a symbol/timeframe with optional date filtering.

        Args:
            symbol: Unified symbol string.
            timeframe: Timeframe string or enum.
            start: Inclusive lower bound for the timestamp filter.
            end: Inclusive upper bound for the timestamp filter.
            columns: Optional subset of columns to read. Defaults to all.

        Returns:
            DataFrame with candle data, sorted by timestamp ascending.
            Returns an empty DataFrame (with correct schema) if no data
            is found.
        """
        if isinstance(timeframe, Timeframe):
            timeframe = timeframe.value

        partition_dir = self._partition_dir(symbol, timeframe)
        data_path = partition_dir / "data.parquet"

        if not data_path.exists():
            logger.debug("No data found for %s/%s", symbol, timeframe)
            return pd.DataFrame(columns=[f.name for f in CANDLE_SCHEMA])

        # Build predicate filters for PyArrow
        filters: list[tuple] = []
        if start is not None:
            if isinstance(start, str):
                start = pd.Timestamp(start, tz="UTC")
            elif isinstance(start, datetime) and start.tzinfo is None:
                start = pd.Timestamp(start, tz="UTC")
            filters.append(("timestamp", ">=", start))
        if end is not None:
            if isinstance(end, str):
                end = pd.Timestamp(end, tz="UTC")
            elif isinstance(end, datetime) and end.tzinfo is None:
                end = pd.Timestamp(end, tz="UTC")
            filters.append(("timestamp", "<=", end))

        try:
            table = pq.read_table(
                data_path,
                columns=columns,
                filters=filters if filters else None,
            )
            df = table.to_pandas()
        except Exception:
            logger.exception("Error reading parquet file %s", data_path)
            return pd.DataFrame(columns=[f.name for f in CANDLE_SCHEMA])

        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.debug(
            "Read %d candles for %s/%s [%s -> %s]",
            len(df), symbol, timeframe, start, end,
        )
        return df

    # -- metadata queries ----------------------------------------------------

    def list_symbols(self) -> list[str]:
        """List all symbols that have stored data.

        Returns:
            Sorted list of unified symbol strings (e.g. ``["BTC/USDT", "ETH/USDT"]``).
        """
        symbols: list[str] = []
        if not self._candles_dir.exists():
            return symbols

        for entry in sorted(self._candles_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("symbol="):
                sanitized = entry.name.removeprefix("symbol=")
                symbols.append(_unsanitize_symbol(sanitized))

        return symbols

    def list_timeframes(self, symbol: str) -> list[str]:
        """List available timeframes for a given symbol.

        Args:
            symbol: Unified symbol string.

        Returns:
            Sorted list of timeframe strings (e.g. ``["1m", "1h", "1d"]``).
        """
        safe_symbol = _sanitize_symbol(symbol)
        symbol_dir = self._candles_dir / f"symbol={safe_symbol}"
        if not symbol_dir.exists():
            return []

        timeframes: list[str] = []
        for entry in sorted(symbol_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("timeframe="):
                tf = entry.name.removeprefix("timeframe=")
                timeframes.append(tf)

        return timeframes

    def get_date_range(
        self,
        symbol: str,
        timeframe: str | Timeframe,
    ) -> tuple[datetime | None, datetime | None]:
        """Return the earliest and latest timestamps for a symbol/timeframe.

        Args:
            symbol: Unified symbol string.
            timeframe: Timeframe string or enum.

        Returns:
            Tuple of ``(earliest, latest)`` timestamps, or ``(None, None)``
            if no data exists.
        """
        if isinstance(timeframe, Timeframe):
            timeframe = timeframe.value

        partition_dir = self._partition_dir(symbol, timeframe)
        data_path = partition_dir / "data.parquet"

        if not data_path.exists():
            return None, None

        try:
            metadata = pq.read_metadata(data_path)
            # Read only the timestamp column for efficiency
            table = pq.read_table(data_path, columns=["timestamp"])
            ts_col = table.column("timestamp")

            if len(ts_col) == 0:
                return None, None

            earliest = ts_col[0].as_py()
            latest = ts_col[-1].as_py()

            # Data is written sorted, but verify
            min_val = min(ts_col.to_pylist())
            max_val = max(ts_col.to_pylist())

            return min_val, max_val
        except Exception:
            logger.exception(
                "Error reading date range for %s/%s", symbol, timeframe,
            )
            return None, None

    def delete_candles(
        self,
        symbol: str,
        timeframe: str | Timeframe,
    ) -> bool:
        """Delete all candle data for a symbol/timeframe partition.

        Args:
            symbol: Unified symbol string.
            timeframe: Timeframe string or enum.

        Returns:
            ``True`` if the file was deleted, ``False`` if it did not exist.
        """
        if isinstance(timeframe, Timeframe):
            timeframe = timeframe.value

        partition_dir = self._partition_dir(symbol, timeframe)
        data_path = partition_dir / "data.parquet"

        if not data_path.exists():
            return False

        data_path.unlink()
        logger.info("Deleted candles for %s/%s", symbol, timeframe)

        # Clean up empty directories
        try:
            partition_dir.rmdir()
        except OSError:
            pass  # Not empty, that's fine
        try:
            partition_dir.parent.rmdir()
        except OSError:
            pass

        return True

    def get_row_count(
        self,
        symbol: str,
        timeframe: str | Timeframe,
    ) -> int:
        """Return the number of candles stored for a symbol/timeframe.

        Args:
            symbol: Unified symbol string.
            timeframe: Timeframe string or enum.

        Returns:
            Number of rows, or ``0`` if no data exists.
        """
        if isinstance(timeframe, Timeframe):
            timeframe = timeframe.value

        partition_dir = self._partition_dir(symbol, timeframe)
        data_path = partition_dir / "data.parquet"

        if not data_path.exists():
            return 0

        try:
            metadata = pq.read_metadata(data_path)
            return metadata.num_rows
        except Exception:
            logger.exception(
                "Error reading row count for %s/%s", symbol, timeframe,
            )
            return 0
