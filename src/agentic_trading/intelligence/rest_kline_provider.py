"""REST kline provider for Binance spot and USD-M futures.

Fetches historical candles via the Binance REST API with automatic
pagination, rate limiting, retry with exponential backoff, and
deduplication.  Used to bootstrap FeatureEngine ring buffers on
startup so that indicators (e.g. 200-period SMA) are immediately
available instead of waiting 8+ hours for live candles to fill.

Usage::

    async with BinanceKlineProvider(MarketType.FUTURES) as provider:
        candles = await provider.fetch_historical(
            symbol="BTC/USDT",
            timeframe=Timeframe.M1,
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 8, tzinfo=timezone.utc),
        )
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

import httpx

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.core.models import Candle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class MarketType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"


_BASE_URLS: dict[MarketType, str] = {
    MarketType.SPOT: "https://api.binance.com/api/v3/klines",
    MarketType.FUTURES: "https://fapi.binance.com/fapi/v1/klines",
}

_TIMEFRAME_TO_BINANCE: dict[Timeframe, str] = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
}

# Binance kline array indices
_K_OPEN_TIME = 0
_K_OPEN = 1
_K_HIGH = 2
_K_LOW = 3
_K_CLOSE = 4
_K_VOLUME = 5
_K_CLOSE_TIME = 6
_K_QUOTE_VOLUME = 7
_K_TRADE_COUNT = 8


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """Simple async token-bucket rate limiter.

    Parameters
    ----------
    rate:
        Maximum requests per minute.
    """

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = rate
        self._max_tokens = rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + elapsed * (self._rate / 60.0),
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

            # Wait a short interval before retrying
            await asyncio.sleep(60.0 / self._rate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symbol_to_binance(symbol: str) -> str:
    """Convert unified symbol (``BTC/USDT``) to Binance format (``BTCUSDT``)."""
    return symbol.replace("/", "")


def _binance_interval(timeframe: Timeframe) -> str:
    """Map a ``Timeframe`` enum to the Binance interval string."""
    interval = _TIMEFRAME_TO_BINANCE.get(timeframe)
    if interval is None:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r} for Binance REST API. "
            f"Supported: {list(_TIMEFRAME_TO_BINANCE.keys())}"
        )
    return interval


def _normalize_kline(
    raw: list,
    symbol: str,
    timeframe: Timeframe,
) -> Candle:
    """Convert a single Binance kline array to a canonical ``Candle``."""
    ts_ms = int(raw[_K_OPEN_TIME])
    timestamp = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return Candle(
        symbol=symbol,
        exchange=Exchange.BINANCE,
        timeframe=timeframe,
        timestamp=timestamp,
        open=float(raw[_K_OPEN]),
        high=float(raw[_K_HIGH]),
        low=float(raw[_K_LOW]),
        close=float(raw[_K_CLOSE]),
        volume=float(raw[_K_VOLUME]),
        quote_volume=float(raw[_K_QUOTE_VOLUME]),
        trades=int(raw[_K_TRADE_COUNT]),
        is_closed=True,
    )


def _validate_ordering(candles: Sequence[Candle]) -> None:
    """Assert strictly increasing timestamps."""
    for i in range(1, len(candles)):
        if candles[i].timestamp <= candles[i - 1].timestamp:
            raise ValueError(
                f"Non-monotonic timestamps at index {i}: "
                f"{candles[i - 1].timestamp} >= {candles[i].timestamp}"
            )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class BinanceKlineProvider:
    """Fetch historical klines from Binance REST API.

    Supports both spot and USD-M futures endpoints with automatic
    pagination, deduplication, rate limiting, and retry.

    Parameters
    ----------
    market_type:
        Default market type (``SPOT`` or ``FUTURES``).
    max_limit:
        Maximum candles per API request (Binance max is 1500).
    rate_limit_rpm:
        Requests per minute budget.
    max_retries:
        Maximum retry attempts for transient errors.
    base_backoff:
        Base backoff duration in seconds for retries.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        market_type: MarketType = MarketType.FUTURES,
        *,
        max_limit: int = 1500,
        rate_limit_rpm: float = 1200,
        max_retries: int = 5,
        base_backoff: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        self._market_type = market_type
        self._max_limit = max_limit
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._timeout = timeout
        self._rate_limiter = TokenBucketRateLimiter(rate_limit_rpm)
        self._client: httpx.AsyncClient | None = None

    # -- Lifecycle -----------------------------------------------------------

    async def open(self) -> None:
        """Create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                http2=False,
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> BinanceKlineProvider:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- Single page fetch ---------------------------------------------------

    async def get_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        market_type: MarketType | None = None,
    ) -> list[Candle]:
        """Fetch a single page of klines.

        Parameters
        ----------
        symbol:
            Unified symbol, e.g. ``"BTC/USDT"``.
        timeframe:
            Candle timeframe.
        start_time:
            Start time (inclusive).
        end_time:
            End time (inclusive).
        limit:
            Max candles to return (default ``max_limit``).
        market_type:
            Override default market type for this request.

        Returns
        -------
        list[Candle]
            Normalized candles sorted by timestamp.
        """
        if self._client is None:
            raise RuntimeError("Provider not opened. Use 'async with' or call open().")

        mt = market_type or self._market_type
        url = _BASE_URLS[mt]
        interval = _binance_interval(timeframe)
        api_symbol = _symbol_to_binance(symbol)

        params: dict[str, str | int] = {
            "symbol": api_symbol,
            "interval": interval,
            "limit": limit or self._max_limit,
        }
        if start_time is not None:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.timestamp() * 1000)

        data = await self._request_with_retry(url, params)
        return [_normalize_kline(k, symbol, timeframe) for k in data]

    # -- Paginated fetch -----------------------------------------------------

    async def fetch_historical(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime,
        *,
        market_type: MarketType | None = None,
    ) -> list[Candle]:
        """Fetch historical klines with automatic pagination and dedup.

        Parameters
        ----------
        symbol:
            Unified symbol, e.g. ``"BTC/USDT"``.
        timeframe:
            Candle timeframe.
        start_time:
            Start of the range (inclusive).
        end_time:
            End of the range (inclusive).
        market_type:
            Override default market type.

        Returns
        -------
        list[Candle]
            Deduplicated candles sorted by timestamp.
        """
        all_candles: list[Candle] = []
        seen_timestamps: set[int] = set()
        current_start = start_time
        page = 0

        while True:
            page += 1
            batch = await self.get_klines(
                symbol,
                timeframe,
                start_time=current_start,
                end_time=end_time,
                limit=self._max_limit,
                market_type=market_type,
            )

            if not batch:
                break

            # Deduplicate
            new_candles = []
            for candle in batch:
                ts_ms = int(candle.timestamp.timestamp() * 1000)
                if ts_ms not in seen_timestamps:
                    seen_timestamps.add(ts_ms)
                    new_candles.append(candle)

            all_candles.extend(new_candles)

            logger.debug(
                "Kline page %d: %d candles (%d new) for %s:%s",
                page, len(batch), len(new_candles), symbol, timeframe.value,
            )

            # Stop on partial page (last page)
            if len(batch) < self._max_limit:
                break

            # Advance start to last open_time + 1ms
            last_ts_ms = int(batch[-1].timestamp.timestamp() * 1000)
            current_start = datetime.fromtimestamp(
                (last_ts_ms + 1) / 1000.0, tz=timezone.utc,
            )

        # Validate ordering
        all_candles.sort(key=lambda c: c.timestamp)
        if all_candles:
            _validate_ordering(all_candles)

        logger.info(
            "Fetched %d candles for %s:%s (%d pages)",
            len(all_candles), symbol, timeframe.value, page,
        )
        return all_candles

    # -- Retry logic ---------------------------------------------------------

    async def _request_with_retry(
        self,
        url: str,
        params: dict,
    ) -> list:
        """Execute a GET request with rate limiting, retry, and backoff.

        Retries on:
        - 5xx server errors
        - 429/418 rate limit responses
        - Network / timeout errors

        Raises immediately on non-retryable 4xx errors.
        """
        assert self._client is not None

        for attempt in range(1, self._max_retries + 1):
            await self._rate_limiter.acquire()

            try:
                resp = await self._client.get(url, params=params)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self._max_retries:
                    raise RuntimeError(
                        f"Max retries ({self._max_retries}) exhausted for "
                        f"{url}: {exc}"
                    ) from exc
                wait = self._backoff_delay(attempt)
                logger.warning(
                    "Network error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, self._max_retries, wait, exc,
                )
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 200:
                return resp.json()

            # Rate limit: 429 or 418 (IP ban warning)
            if resp.status_code in (429, 418):
                retry_after = float(
                    resp.headers.get("Retry-After", "60")
                )
                logger.warning(
                    "Rate limited (%d), sleeping %.0fs (attempt %d/%d)",
                    resp.status_code, retry_after, attempt, self._max_retries,
                )
                if attempt == self._max_retries:
                    raise RuntimeError(
                        f"Max retries ({self._max_retries}) exhausted: "
                        f"rate limited {resp.status_code}"
                    )
                await asyncio.sleep(retry_after)
                continue

            # Server error: 5xx
            if resp.status_code >= 500:
                if attempt == self._max_retries:
                    raise RuntimeError(
                        f"Max retries ({self._max_retries}) exhausted: "
                        f"server error {resp.status_code}"
                    )
                wait = self._backoff_delay(attempt)
                logger.warning(
                    "Server error %d (attempt %d/%d), retrying in %.1fs",
                    resp.status_code, attempt, self._max_retries, wait,
                )
                await asyncio.sleep(wait)
                continue

            # Non-retryable client error
            raise httpx.HTTPStatusError(
                f"Binance API error {resp.status_code}: {resp.text}",
                request=resp.request,
                response=resp,
            )

        # Should not reach here, but safety net
        raise RuntimeError(f"Max retries ({self._max_retries}) exhausted")

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        base = self._base_backoff * (2 ** (attempt - 1))
        return base + random.uniform(0, base * 0.5)
