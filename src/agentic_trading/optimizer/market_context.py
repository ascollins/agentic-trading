"""Live market context fetcher using Bybit V5 API.

Fetches funding rates, open interest, ticker data, and historical klines
to build a comprehensive market context for strategy optimization.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BASE_URL = "https://api.bybit.com"


@dataclass
class FundingSummary:
    """Summary of funding rate data."""

    current_rate: float = 0.0
    avg_24h: float = 0.0
    avg_72h: float = 0.0
    bias: str = "neutral"  # "long-biased", "short-biased", "neutral"
    is_extreme: bool = False
    data_points: int = 0


@dataclass
class OISummary:
    """Summary of open interest data."""

    current_oi: float = 0.0
    change_pct: float = 0.0
    trend: str = "flat"  # "rising", "falling", "flat"
    data_points: int = 0


@dataclass
class TickerData:
    """Current ticker snapshot."""

    last_price: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    change_24h_pct: float = 0.0
    volume_24h: float = 0.0
    turnover_24h: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0


@dataclass
class MarketContext:
    """Complete market context for a symbol."""

    symbol: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ticker: TickerData = field(default_factory=TickerData)
    funding: FundingSummary = field(default_factory=FundingSummary)
    oi: OISummary = field(default_factory=OISummary)
    smc_features: dict[str, float] = field(default_factory=dict)
    htf_bias: str = "unclear"
    error: str | None = None


class MarketContextFetcher:
    """Fetches live market context from Bybit V5 public API.

    No authentication required — uses only public market data endpoints.

    Usage::

        fetcher = MarketContextFetcher()
        ctx = fetcher.fetch_context("BTCUSDT")
        print(ctx.funding.bias)
    """

    def __init__(self, rate_limit_seconds: float = 0.1) -> None:
        self._rate_limit = rate_limit_seconds

    def fetch_context(
        self,
        symbol: str,
        category: str = "linear",
    ) -> MarketContext:
        """Fetch comprehensive market context for a symbol.

        Args:
            symbol: Bybit symbol (e.g. "BTCUSDT").
            category: "linear" for USDT perps, "spot" for spot.

        Returns:
            MarketContext with ticker, funding, OI data.
        """
        try:
            import requests
        except ImportError:
            return MarketContext(
                symbol=symbol,
                error="requests library not installed",
            )

        ctx = MarketContext(symbol=symbol)

        # Fetch ticker
        try:
            ticker_data = self._fetch_ticker(requests, symbol, category)
            if ticker_data:
                ctx.ticker = ticker_data
        except Exception as e:
            logger.warning("Failed to fetch ticker for %s: %s", symbol, e)

        time.sleep(self._rate_limit)

        # Fetch funding
        try:
            funding_data = self._fetch_funding(requests, symbol, category)
            if funding_data:
                ctx.funding = funding_data
        except Exception as e:
            logger.warning("Failed to fetch funding for %s: %s", symbol, e)

        time.sleep(self._rate_limit)

        # Fetch OI
        try:
            oi_data = self._fetch_oi(requests, symbol, category)
            if oi_data:
                ctx.oi = oi_data
        except Exception as e:
            logger.warning("Failed to fetch OI for %s: %s", symbol, e)

        return ctx

    def _fetch_ticker(
        self, requests: Any, symbol: str, category: str
    ) -> TickerData | None:
        """Fetch current ticker data."""
        resp = requests.get(
            f"{BASE_URL}/v5/market/tickers",
            params={"category": category, "symbol": symbol},
            timeout=10,
        )
        data = resp.json()
        if data["retCode"] != 0:
            logger.error("Ticker API error: %s", data["retMsg"])
            return None

        items = data["result"]["list"]
        if not items:
            return None

        t = items[0]
        last_price = float(t["lastPrice"])
        bid = float(t.get("bid1Price", 0))
        ask = float(t.get("ask1Price", 0))
        spread = ask - bid if ask > 0 and bid > 0 else 0
        spread_pct = (spread / last_price * 100) if last_price > 0 else 0

        return TickerData(
            last_price=last_price,
            high_24h=float(t.get("highPrice24h", 0)),
            low_24h=float(t.get("lowPrice24h", 0)),
            change_24h_pct=float(t.get("price24hPcnt", 0)) * 100,
            volume_24h=float(t.get("volume24h", 0)),
            turnover_24h=float(t.get("turnover24h", 0)),
            bid=bid,
            ask=ask,
            spread_pct=round(spread_pct, 4),
        )

    def _fetch_funding(
        self, requests: Any, symbol: str, category: str, hours: int = 72
    ) -> FundingSummary | None:
        """Fetch funding rate history."""
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (hours * 3600 * 1000)

        resp = requests.get(
            f"{BASE_URL}/v5/market/funding/history",
            params={
                "category": category,
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 200,
            },
            timeout=10,
        )
        data = resp.json()
        if data["retCode"] != 0:
            return None

        items = data["result"]["list"]
        if not items:
            return None

        rates = [float(f["fundingRate"]) for f in items]

        current = rates[0]
        avg_24h = np.mean(rates[:3]) if len(rates) >= 3 else current
        avg_72h = np.mean(rates) if rates else current

        if current > 0.0001:
            bias = "long-biased"
        elif current < -0.0001:
            bias = "short-biased"
        else:
            bias = "neutral"

        return FundingSummary(
            current_rate=current,
            avg_24h=float(avg_24h),
            avg_72h=float(avg_72h),
            bias=bias,
            is_extreme=abs(current) > 0.0005,
            data_points=len(rates),
        )

    def _fetch_oi(
        self, requests: Any, symbol: str, category: str, hours: int = 48
    ) -> OISummary | None:
        """Fetch open interest data."""
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (hours * 3600 * 1000)

        resp = requests.get(
            f"{BASE_URL}/v5/market/open-interest",
            params={
                "category": category,
                "symbol": symbol,
                "intervalTime": "1h",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 200,
            },
            timeout=10,
        )
        data = resp.json()
        if data["retCode"] != 0:
            return None

        items = data["result"]["list"]
        if not items:
            return None

        values = [float(o["openInterest"]) for o in items]
        current = values[0]
        oldest = values[-1]
        change_pct = ((current - oldest) / oldest * 100) if oldest > 0 else 0

        if change_pct > 1:
            trend = "rising"
        elif change_pct < -1:
            trend = "falling"
        else:
            trend = "flat"

        return OISummary(
            current_oi=current,
            change_pct=round(change_pct, 2),
            trend=trend,
            data_points=len(values),
        )

    def print_context(self, ctx: MarketContext) -> None:
        """Print market context in a formatted display."""
        print(f"\n{'=' * 60}")
        print(f"MARKET CONTEXT: {ctx.symbol}")
        print(f"{'=' * 60}")
        print(f"Timestamp: {ctx.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        t = ctx.ticker
        if t.last_price > 0:
            print(f"\n--- TICKER ---")
            print(f"  Last Price:    ${t.last_price:,.2f}")
            print(f"  24h High:      ${t.high_24h:,.2f}")
            print(f"  24h Low:       ${t.low_24h:,.2f}")
            print(f"  24h Change:    {t.change_24h_pct:+.2f}%")
            print(f"  24h Volume:    {t.volume_24h:,.2f}")
            print(f"  Spread:        {t.spread_pct:.4f}%")

        f = ctx.funding
        if f.data_points > 0:
            print(f"\n--- FUNDING RATES ---")
            print(f"  Current:       {f.current_rate:.6f} ({f.bias})")
            print(f"  Avg 24h:       {f.avg_24h:.6f}")
            print(f"  Avg 72h:       {f.avg_72h:.6f}")
            print(f"  Extreme:       {'YES' if f.is_extreme else 'No'}")

        o = ctx.oi
        if o.data_points > 0:
            print(f"\n--- OPEN INTEREST ---")
            print(f"  Current:       {o.current_oi:,.0f}")
            print(f"  Change:        {o.change_pct:+.2f}%")
            print(f"  Trend:         {o.trend}")

        print(f"\n{'=' * 60}\n")
