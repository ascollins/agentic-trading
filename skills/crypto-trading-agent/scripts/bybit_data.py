#!/usr/bin/env python3
"""
Bybit V5 API Data Helper for Crypto Trading Agent Skill

Provides convenience functions for fetching and processing market data
from Bybit's public API endpoints. No authentication required for
market data.

Usage:
    python bybit_data.py --symbol BTCUSDT --interval 240 --days 30
    python bybit_data.py --symbol ETHUSDT --funding --oi
    python bybit_data.py --symbol BTCUSDT --full-context
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

BASE_URL = "https://api.bybit.com"


def fetch_klines(symbol, interval, start_ms, end_ms, category="linear"):
    """Fetch all klines between start and end timestamps, handling pagination."""
    all_klines = []
    current_end = end_ms

    while current_end > start_ms:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": str(interval),
            "end": current_end,
            "limit": 1000
        }

        response = requests.get(f"{BASE_URL}/v5/market/kline", params=params)
        data = response.json()

        if data["retCode"] != 0:
            raise Exception(f"API error: {data['retMsg']}")

        klines = data["result"]["list"]
        if not klines:
            break

        all_klines.extend(klines)
        oldest_timestamp = int(klines[-1][0])
        if oldest_timestamp <= start_ms:
            break
        current_end = oldest_timestamp - 1
        time.sleep(0.1)

    all_klines = [k for k in all_klines if int(k[0]) >= start_ms]
    all_klines.sort(key=lambda x: int(x[0]))

    # Remove duplicates
    seen = set()
    unique = []
    for k in all_klines:
        if k[0] not in seen:
            seen.add(k[0])
            unique.append(k)

    return unique


def fetch_ticker(symbol, category="linear"):
    """Fetch current ticker data."""
    response = requests.get(
        f"{BASE_URL}/v5/market/tickers",
        params={"category": category, "symbol": symbol}
    )
    data = response.json()
    if data["retCode"] != 0:
        raise Exception(f"API error: {data['retMsg']}")
    return data["result"]["list"][0] if data["result"]["list"] else None


def fetch_funding_history(symbol, hours=72, category="linear"):
    """Fetch funding rate history."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (hours * 3600 * 1000)

    response = requests.get(
        f"{BASE_URL}/v5/market/funding/history",
        params={
            "category": category,
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 200
        }
    )
    data = response.json()
    if data["retCode"] != 0:
        raise Exception(f"API error: {data['retMsg']}")
    return data["result"]["list"]


def fetch_open_interest(symbol, interval="1h", hours=48, category="linear"):
    """Fetch open interest data."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (hours * 3600 * 1000)

    response = requests.get(
        f"{BASE_URL}/v5/market/open-interest",
        params={
            "category": category,
            "symbol": symbol,
            "intervalTime": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 200
        }
    )
    data = response.json()
    if data["retCode"] != 0:
        raise Exception(f"API error: {data['retMsg']}")
    return data["result"]["list"]


def fetch_orderbook(symbol, depth=50, category="linear"):
    """Fetch current orderbook."""
    response = requests.get(
        f"{BASE_URL}/v5/market/orderbook",
        params={"category": category, "symbol": symbol, "limit": depth}
    )
    data = response.json()
    if data["retCode"] != 0:
        raise Exception(f"API error: {data['retMsg']}")
    return data["result"]


def build_analysis_dataframe(klines):
    """Convert raw klines to a pandas DataFrame with technical indicators."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required for DataFrame analysis")

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

    # EMAs
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()

    # ATR
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = df['tr'].rolling(14).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Volume metrics
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Candle anatomy
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['is_bullish'] = df['close'] > df['open']

    return df


def summarise_funding(funding_data):
    """Summarise funding rate data."""
    rates = [float(f['fundingRate']) for f in funding_data]
    if not rates:
        return {"error": "No funding data available"}

    return {
        "current_rate": f"{rates[0]:.6f}",
        "current_rate_pct": f"{rates[0] * 100:.4f}%",
        "avg_24h": f"{sum(rates[:3]) / min(3, len(rates)):.6f}",
        "avg_72h": f"{sum(rates) / len(rates):.6f}",
        "max": f"{max(rates):.6f}",
        "min": f"{min(rates):.6f}",
        "bias": "long-biased" if rates[0] > 0.0001 else "short-biased" if rates[0] < -0.0001 else "neutral",
        "extreme": abs(rates[0]) > 0.0005,
        "data_points": len(rates)
    }


def summarise_oi(oi_data):
    """Summarise open interest data."""
    values = [float(o['openInterest']) for o in oi_data]
    if not values:
        return {"error": "No OI data available"}

    current = values[0]
    oldest = values[-1]
    change_pct = ((current - oldest) / oldest) * 100

    return {
        "current_oi": f"{current:,.0f}",
        "change_pct": f"{change_pct:+.2f}%",
        "trend": "rising" if change_pct > 1 else "falling" if change_pct < -1 else "flat",
        "high": f"{max(values):,.0f}",
        "low": f"{min(values):,.0f}",
        "data_points": len(values)
    }


def full_context(symbol, category="linear"):
    """Fetch comprehensive market context for a symbol."""
    print(f"\n{'='*60}")
    print(f"MARKET CONTEXT: {symbol}")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Ticker
    print("--- TICKER ---")
    ticker = fetch_ticker(symbol, category)
    if ticker:
        print(f"  Last Price:    ${float(ticker['lastPrice']):,.2f}")
        print(f"  24h High:      ${float(ticker['highPrice24h']):,.2f}")
        print(f"  24h Low:       ${float(ticker['lowPrice24h']):,.2f}")
        print(f"  24h Change:    {float(ticker['price24hPcnt'])*100:+.2f}%")
        print(f"  24h Volume:    {float(ticker['volume24h']):,.2f}")
        print(f"  24h Turnover:  ${float(ticker['turnover24h']):,.0f}")
        print(f"  Bid:           ${float(ticker['bid1Price']):,.2f}")
        print(f"  Ask:           ${float(ticker['ask1Price']):,.2f}")
        spread = float(ticker['ask1Price']) - float(ticker['bid1Price'])
        spread_pct = spread / float(ticker['lastPrice']) * 100
        print(f"  Spread:        ${spread:,.2f} ({spread_pct:.4f}%)")

    # Funding
    print("\n--- FUNDING RATES ---")
    try:
        funding = fetch_funding_history(symbol, hours=72, category=category)
        summary = summarise_funding(funding)
        for k, v in summary.items():
            print(f"  {k:16s}: {v}")
    except Exception as e:
        print(f"  Error fetching funding: {e}")

    # Open Interest
    print("\n--- OPEN INTEREST ---")
    try:
        oi = fetch_open_interest(symbol, interval="1h", hours=48, category=category)
        summary = summarise_oi(oi)
        for k, v in summary.items():
            print(f"  {k:16s}: {v}")
    except Exception as e:
        print(f"  Error fetching OI: {e}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Bybit V5 Data Helper")
    parser.add_argument("--symbol", required=True, help="Trading pair (e.g. BTCUSDT)")
    parser.add_argument("--category", default="linear", help="Category: linear, inverse, spot")
    parser.add_argument("--interval", type=str, default="240", help="Kline interval")
    parser.add_argument("--days", type=int, default=30, help="Days of kline history")
    parser.add_argument("--funding", action="store_true", help="Fetch funding rates")
    parser.add_argument("--oi", action="store_true", help="Fetch open interest")
    parser.add_argument("--orderbook", action="store_true", help="Fetch orderbook")
    parser.add_argument("--full-context", action="store_true", help="Fetch full market context")
    parser.add_argument("--output", help="Output file path (JSON)")

    args = parser.parse_args()

    if args.full_context:
        full_context(args.symbol, args.category)
        return

    results = {"symbol": args.symbol, "category": args.category}

    if args.funding:
        funding = fetch_funding_history(args.symbol, category=args.category)
        results["funding"] = summarise_funding(funding)
        print(json.dumps(results["funding"], indent=2))

    if args.oi:
        oi = fetch_open_interest(args.symbol, category=args.category)
        results["open_interest"] = summarise_oi(oi)
        print(json.dumps(results["open_interest"], indent=2))

    if args.orderbook:
        book = fetch_orderbook(args.symbol, category=args.category)
        results["orderbook"] = {
            "bids_top5": book["b"][:5],
            "asks_top5": book["a"][:5],
            "timestamp": book.get("ts")
        }
        print(json.dumps(results["orderbook"], indent=2))

    if not (args.funding or args.oi or args.orderbook):
        # Default: fetch klines
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (args.days * 24 * 3600 * 1000)
        klines = fetch_klines(args.symbol, args.interval, start_ms, end_ms, args.category)
        results["klines"] = {
            "count": len(klines),
            "interval": args.interval,
            "start": datetime.fromtimestamp(int(klines[0][0])/1000, tz=timezone.utc).isoformat() if klines else None,
            "end": datetime.fromtimestamp(int(klines[-1][0])/1000, tz=timezone.utc).isoformat() if klines else None,
        }
        print(f"Fetched {len(klines)} klines for {args.symbol}")
        print(f"  From: {results['klines']['start']}")
        print(f"  To:   {results['klines']['end']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
