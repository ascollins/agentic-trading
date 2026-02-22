"""Unit tests for the Binance REST kline provider.

Tests cover normalisation, pagination, deduplication, rate limiting,
retry logic, URL routing, and schema compatibility.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agentic_trading.core.enums import Exchange, Timeframe
from agentic_trading.intelligence.rest_kline_provider import (
    BinanceKlineProvider,
    MarketType,
    TokenBucketRateLimiter,
    _binance_interval,
    _normalize_kline,
    _symbol_to_binance,
    _validate_ordering,
    _BASE_URLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_kline(
    open_time_ms: int,
    o: str = "100.0",
    h: str = "105.0",
    l: str = "95.0",
    c: str = "102.0",
    vol: str = "1000.0",
    q_vol: str = "100000.0",
    trades: int = 500,
) -> list:
    """Build a raw Binance kline array."""
    close_time_ms = open_time_ms + 59_999
    return [
        open_time_ms,
        o, h, l, c, vol,
        close_time_ms,
        q_vol,
        trades,
        "500.0",       # taker buy base volume
        "50000.0",     # taker buy quote volume
        "0",           # ignore
    ]


@pytest.fixture
def sample_binance_klines() -> list[list]:
    """Three consecutive 1m klines starting at 2024-01-01 00:00 UTC."""
    base_ms = 1_704_067_200_000  # 2024-01-01T00:00:00Z
    return [
        _make_raw_kline(base_ms, "100.0", "105.0", "95.0", "102.0", "1000.0", "100000.0", 500),
        _make_raw_kline(base_ms + 60_000, "102.0", "108.0", "101.0", "106.0", "1200.0", "126000.0", 600),
        _make_raw_kline(base_ms + 120_000, "106.0", "110.0", "104.0", "109.0", "800.0", "86000.0", 400),
    ]


def _make_transport(pages: list[list[list]], status_codes: list[int] | None = None):
    """Create an httpx.MockTransport that returns sequential pages.

    After all pages are exhausted, returns empty ``[]`` (which
    terminates pagination).

    Parameters
    ----------
    pages:
        List of pages, each page being a list of raw kline arrays.
    status_codes:
        Optional list of HTTP status codes for each call.
        Defaults to 200 for every call.
    """
    call_count = 0
    codes = status_codes or [200] * (len(pages) + 5)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        idx = call_count
        code_idx = min(call_count, len(codes) - 1)
        call_count += 1
        data = pages[idx] if idx < len(pages) else []
        return httpx.Response(
            status_code=codes[code_idx],
            json=data,
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# 1. Single kline normalisation
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_single_kline_fields(self):
        """Single kline → correct Candle fields."""
        raw = _make_raw_kline(
            1_704_067_200_000,  # 2024-01-01T00:00:00Z
            "42000.5", "42500.0", "41800.0", "42200.0",
            "150.5", "6331050.0", 12345,
        )
        candle = _normalize_kline(raw, "BTC/USDT", Timeframe.M1)

        assert candle.symbol == "BTC/USDT"
        assert candle.exchange == Exchange.BINANCE
        assert candle.timeframe == Timeframe.M1
        assert candle.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert candle.open == 42000.5
        assert candle.high == 42500.0
        assert candle.low == 41800.0
        assert candle.close == 42200.0
        assert candle.volume == 150.5
        assert candle.quote_volume == 6331050.0
        assert candle.trades == 12345
        assert candle.is_closed is True

    def test_batch_normalization(self, sample_binance_klines):
        """Batch of klines maintains order and all fields populated."""
        candles = [
            _normalize_kline(k, "ETH/USDT", Timeframe.M5)
            for k in sample_binance_klines
        ]
        assert len(candles) == 3
        for c in candles:
            assert c.symbol == "ETH/USDT"
            assert c.timeframe == Timeframe.M5
            assert c.volume > 0
            assert c.quote_volume > 0
            assert c.trades > 0

        # Verify chronological order
        for i in range(1, len(candles)):
            assert candles[i].timestamp > candles[i - 1].timestamp


# ---------------------------------------------------------------------------
# 2. Ordering validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_monotonic_passes(self, sample_binance_klines):
        """Strictly increasing timestamps → no error."""
        candles = [
            _normalize_kline(k, "BTC/USDT", Timeframe.M1)
            for k in sample_binance_klines
        ]
        _validate_ordering(candles)  # Should not raise

    def test_non_monotonic_raises(self, sample_binance_klines):
        """Non-monotonic timestamps → ValueError."""
        candles = [
            _normalize_kline(k, "BTC/USDT", Timeframe.M1)
            for k in sample_binance_klines
        ]
        # Swap to break ordering
        candles[0], candles[1] = candles[1], candles[0]
        with pytest.raises(ValueError, match="Non-monotonic"):
            _validate_ordering(candles)


# ---------------------------------------------------------------------------
# 3. Unsupported timeframe
# ---------------------------------------------------------------------------

class TestIntervalMapping:
    def test_supported_timeframes(self):
        """All supported timeframes map correctly."""
        assert _binance_interval(Timeframe.M1) == "1m"
        assert _binance_interval(Timeframe.M5) == "5m"
        assert _binance_interval(Timeframe.M15) == "15m"
        assert _binance_interval(Timeframe.H1) == "1h"
        assert _binance_interval(Timeframe.H4) == "4h"
        assert _binance_interval(Timeframe.D1) == "1d"

    def test_unsupported_timeframe_raises(self):
        """Unsupported Timeframe values raise ValueError."""
        # All Timeframe enum members are supported, so we test via
        # passing a string that doesn't match
        with pytest.raises((ValueError, KeyError)):
            _binance_interval("3m")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. Symbol conversion
# ---------------------------------------------------------------------------

class TestSymbolConversion:
    def test_slash_removed(self):
        assert _symbol_to_binance("BTC/USDT") == "BTCUSDT"

    def test_no_slash(self):
        assert _symbol_to_binance("BTCUSDT") == "BTCUSDT"


# ---------------------------------------------------------------------------
# 5. URL routing
# ---------------------------------------------------------------------------

class TestURLRouting:
    def test_spot_url(self):
        assert _BASE_URLS[MarketType.SPOT] == "https://api.binance.com/api/v3/klines"

    def test_futures_url(self):
        assert _BASE_URLS[MarketType.FUTURES] == "https://fapi.binance.com/fapi/v1/klines"


# ---------------------------------------------------------------------------
# 6. Pagination
# ---------------------------------------------------------------------------

class TestPagination:
    @pytest.mark.asyncio
    async def test_multi_page_concatenation(self, sample_binance_klines):
        """Three pages of 1 kline each → 3 candles concatenated."""
        pages = [[k] for k in sample_binance_klines]
        transport = _make_transport(pages)

        provider = BinanceKlineProvider(max_limit=1, rate_limit_rpm=60000)
        provider._client = httpx.AsyncClient(transport=transport)

        try:
            candles = await provider.fetch_historical(
                "BTC/USDT",
                Timeframe.M1,
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
            )
            assert len(candles) == 3
            # Verify order
            for i in range(1, len(candles)):
                assert candles[i].timestamp > candles[i - 1].timestamp
        finally:
            await provider._client.aclose()
            provider._client = None

    @pytest.mark.asyncio
    async def test_partial_page_stops(self, sample_binance_klines):
        """Page with < limit candles → pagination terminates."""
        # Single page with 3 candles but limit=5 → partial, stops immediately
        pages = [sample_binance_klines]
        transport = _make_transport(pages)

        provider = BinanceKlineProvider(max_limit=5, rate_limit_rpm=60000)
        provider._client = httpx.AsyncClient(transport=transport)

        try:
            candles = await provider.fetch_historical(
                "BTC/USDT",
                Timeframe.M1,
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
            )
            assert len(candles) == 3
        finally:
            await provider._client.aclose()
            provider._client = None


# ---------------------------------------------------------------------------
# 7. Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    @pytest.mark.asyncio
    async def test_overlapping_pages_dedup(self, sample_binance_klines):
        """Overlapping pages → no duplicate timestamps."""
        # Page 1: klines 0, 1; Page 2: klines 1, 2 (overlap on kline 1)
        pages = [
            sample_binance_klines[:2],
            sample_binance_klines[1:],
            [],  # Empty page to terminate
        ]
        transport = _make_transport(pages)

        provider = BinanceKlineProvider(max_limit=2, rate_limit_rpm=60000)
        provider._client = httpx.AsyncClient(transport=transport)

        try:
            candles = await provider.fetch_historical(
                "BTC/USDT",
                Timeframe.M1,
                start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
            )
            # Should be 3 unique candles despite overlap
            assert len(candles) == 3
            timestamps = [c.timestamp for c in candles]
            assert len(set(timestamps)) == 3
        finally:
            await provider._client.aclose()
            provider._client = None


# ---------------------------------------------------------------------------
# 8. Rate limiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_burst_capacity(self):
        """Rate limiter allows burst up to capacity."""
        limiter = TokenBucketRateLimiter(rate=10.0)
        # Should be able to acquire 10 tokens immediately
        for _ in range(10):
            await asyncio.wait_for(limiter.acquire(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_blocks_when_exhausted(self):
        """Rate limiter blocks when tokens exhausted."""
        limiter = TokenBucketRateLimiter(rate=2.0)
        # Drain tokens
        await limiter.acquire()
        await limiter.acquire()
        # Next acquire should block (tokens exhausted)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limiter.acquire(), timeout=0.05)


# ---------------------------------------------------------------------------
# 9. Retry on 5xx
# ---------------------------------------------------------------------------

class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_on_500(self, sample_binance_klines):
        """Mock 2x 500 then 200 → succeeds on 3rd attempt."""
        pages = [[], [], sample_binance_klines]
        codes = [500, 500, 200]

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            idx = min(call_count, len(pages) - 1)
            code = codes[min(call_count, len(codes) - 1)]
            call_count += 1
            return httpx.Response(
                status_code=code,
                json=pages[idx],
                headers={"Content-Type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        provider = BinanceKlineProvider(
            max_limit=5,
            rate_limit_rpm=60000,
            max_retries=5,
            base_backoff=0.01,  # Fast backoff for tests
        )
        provider._client = httpx.AsyncClient(transport=transport)

        try:
            candles = await provider.get_klines(
                "BTC/USDT",
                Timeframe.M1,
            )
            assert len(candles) == 3
            assert call_count == 3
        finally:
            await provider._client.aclose()
            provider._client = None

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """All 5xx → raises RuntimeError."""
        pages = [[]] * 6
        codes = [500] * 6

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                status_code=500,
                json=[],
                headers={"Content-Type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        provider = BinanceKlineProvider(
            max_limit=5,
            rate_limit_rpm=60000,
            max_retries=3,
            base_backoff=0.01,
        )
        provider._client = httpx.AsyncClient(transport=transport)

        try:
            with pytest.raises(RuntimeError, match="Max retries"):
                await provider.get_klines("BTC/USDT", Timeframe.M1)
            assert call_count == 3
        finally:
            await provider._client.aclose()
            provider._client = None


# ---------------------------------------------------------------------------
# 10. 429 respects Retry-After
# ---------------------------------------------------------------------------

class TestRateLimit429:
    @pytest.mark.asyncio
    async def test_429_retry_after(self, sample_binance_klines):
        """Mock 429 with Retry-After → waits then succeeds."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    status_code=429,
                    json={"msg": "rate limited"},
                    headers={
                        "Content-Type": "application/json",
                        "Retry-After": "0.01",  # Very short for testing
                    },
                )
            return httpx.Response(
                status_code=200,
                json=sample_binance_klines,
                headers={"Content-Type": "application/json"},
            )

        transport = httpx.MockTransport(handler)
        provider = BinanceKlineProvider(
            max_limit=5,
            rate_limit_rpm=60000,
            max_retries=3,
            base_backoff=0.01,
        )
        provider._client = httpx.AsyncClient(transport=transport)

        try:
            candles = await provider.get_klines("BTC/USDT", Timeframe.M1)
            assert len(candles) == 3
            assert call_count == 2
        finally:
            await provider._client.aclose()
            provider._client = None


# ---------------------------------------------------------------------------
# 11. Schema compatibility with normalize_ccxt_ohlcv
# ---------------------------------------------------------------------------

class TestSchemaCompatibility:
    def test_rest_candle_matches_ccxt_schema(self):
        """REST Candle has identical schema to normalize_ccxt_ohlcv output."""
        from agentic_trading.intelligence.normalizer import normalize_ccxt_ohlcv

        # REST kline
        raw = _make_raw_kline(
            1_704_067_200_000,
            "42000.0", "42500.0", "41800.0", "42200.0",
            "150.0", "6300000.0", 1000,
        )
        rest_candle = _normalize_kline(raw, "BTC/USDT", Timeframe.M1)

        # CCXT kline (same data, CCXT format: [ts_ms, o, h, l, c, v])
        ccxt_raw = [1_704_067_200_000, 42000.0, 42500.0, 41800.0, 42200.0, 150.0]
        ccxt_candle = normalize_ccxt_ohlcv(
            Exchange.BINANCE, "BTC/USDT", Timeframe.M1, ccxt_raw,
        )

        # Both should produce Candle with matching core OHLCV fields
        assert rest_candle.symbol == ccxt_candle.symbol
        assert rest_candle.exchange == ccxt_candle.exchange
        assert rest_candle.timeframe == ccxt_candle.timeframe
        assert rest_candle.timestamp == ccxt_candle.timestamp
        assert rest_candle.open == ccxt_candle.open
        assert rest_candle.high == ccxt_candle.high
        assert rest_candle.low == ccxt_candle.low
        assert rest_candle.close == ccxt_candle.close
        assert rest_candle.volume == ccxt_candle.volume
        assert rest_candle.is_closed == ccxt_candle.is_closed

        # REST additionally provides quote_volume and trades
        assert rest_candle.quote_volume == 6300000.0
        assert rest_candle.trades == 1000


# ---------------------------------------------------------------------------
# 12. Context manager
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Context manager opens and closes client."""
        provider = BinanceKlineProvider()
        assert provider._client is None

        async with provider:
            assert provider._client is not None

        assert provider._client is None

    @pytest.mark.asyncio
    async def test_not_opened_raises(self):
        """Calling get_klines without open raises RuntimeError."""
        provider = BinanceKlineProvider()
        with pytest.raises(RuntimeError, match="not opened"):
            await provider.get_klines("BTC/USDT", Timeframe.M1)
