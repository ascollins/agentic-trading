# Bybit V5 API Data Integration Reference

## Table of Contents
1. API Overview
2. Market Data Endpoints
3. Derivatives-Specific Data
4. Data Fetching Patterns
5. Rate Limits & Best Practices

---

## 1. API Overview

Base URL: `https://api.bybit.com`
API Version: V5 (unified across spot, derivatives, options)

All market data endpoints are public (no authentication required).
Authentication is only needed for account/order management.

---

## 2. Market Data Endpoints

### Kline (OHLCV) Data
```
GET /v5/market/kline
```
Parameters:
- `category`: "linear" (USDT perps), "inverse" (coin-margined), "spot"
- `symbol`: e.g. "BTCUSDT"
- `interval`: "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"
- `start`: Start timestamp in milliseconds
- `end`: End timestamp in milliseconds
- `limit`: Max 1000 per request

Response fields: `[timestamp, open, high, low, close, volume, turnover]`

Note: Results are returned in reverse chronological order (newest first).

### Ticker Data
```
GET /v5/market/tickers
```
Parameters:
- `category`: "linear", "inverse", "spot"
- `symbol`: Optional, returns all tickers if omitted

Key response fields:
- `lastPrice`, `highPrice24h`, `lowPrice24h`
- `volume24h`, `turnover24h`
- `price24hPcnt` (24h percentage change)
- `bid1Price`, `ask1Price` (best bid/ask)

### Orderbook
```
GET /v5/market/orderbook
```
Parameters:
- `category`: "linear", "inverse", "spot"
- `symbol`: e.g. "BTCUSDT"
- `limit`: 1-500 (default 25)

Useful for: Identifying large resting orders (potential institutional levels),
assessing current liquidity depth, and detecting orderbook imbalances.

---

## 3. Derivatives-Specific Data

### Funding Rate History
```
GET /v5/market/funding/history
```
Parameters:
- `category`: "linear" or "inverse"
- `symbol`: e.g. "BTCUSDT"
- `startTime`, `endTime`: Millisecond timestamps
- `limit`: Max 200

Response fields: `fundingRate`, `fundingRateTimestamp`

**Interpretation framework**:
| Funding Rate | Market State | Trading Implication |
|-------------|-------------|-------------------|
| > 0.03% | Heavily long-biased | Longs are crowded; short squeeze less likely, long squeeze more likely |
| 0.01-0.03% | Moderately bullish | Normal trending conditions |
| -0.01 to 0.01% | Neutral | No directional funding pressure |
| -0.03 to -0.01% | Moderately bearish | Normal downtrend conditions |
| < -0.03% | Heavily short-biased | Shorts are crowded; short squeeze more likely |

Extreme funding (>0.05% or <-0.05%) often precedes violent reversals.

### Open Interest
```
GET /v5/market/open-interest
```
Parameters:
- `category`: "linear" or "inverse"
- `symbol`: e.g. "BTCUSDT"
- `intervalTime`: "5min", "15min", "30min", "1h", "4h", "1d"
- `startTime`, `endTime`: Millisecond timestamps
- `limit`: Max 200

**OI + Price interpretation matrix**:
| Price | OI | Interpretation |
|-------|-----|---------------|
| Rising | Rising | New longs opening — strong bullish (if trend aligned) |
| Rising | Falling | Short covering rally — weaker bullish, watch for exhaustion |
| Falling | Rising | New shorts opening — strong bearish (if trend aligned) |
| Falling | Falling | Long liquidation — potential capitulation (watch for reversal) |
| Flat | Rising | Tension building — preparing for a large move (direction TBD) |

### Long/Short Ratio
```
GET /v5/market/account-ratio
```
Parameters:
- `category`: "linear" or "inverse"  
- `symbol`: e.g. "BTCUSDT"
- `period`: "5min", "15min", "30min", "1h", "4h", "1d"
- `limit`: Max 500

This shows the ratio of long accounts to short accounts. Useful as a contrarian
indicator at extremes: when everyone is long, the squeeze will be to the downside.

---

## 4. Data Fetching Patterns

### Fetching Historical Klines (Pagination)

The API returns max 1000 candles per request. For longer histories, paginate:

```python
import requests
import time

def fetch_klines(symbol, interval, start_ms, end_ms, category="linear"):
    """Fetch all klines between start and end, handling pagination."""
    all_klines = []
    current_end = end_ms
    
    while current_end > start_ms:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "end": current_end,
            "limit": 1000
        }
        
        response = requests.get(
            "https://api.bybit.com/v5/market/kline",
            params=params
        )
        data = response.json()
        
        if data["retCode"] != 0:
            raise Exception(f"API error: {data['retMsg']}")
        
        klines = data["result"]["list"]
        if not klines:
            break
        
        all_klines.extend(klines)
        
        # Results are newest-first; last element is oldest
        oldest_timestamp = int(klines[-1][0])
        if oldest_timestamp <= start_ms:
            break
        current_end = oldest_timestamp - 1
        
        time.sleep(0.1)  # Rate limit courtesy
    
    # Filter to exact range and sort chronologically
    all_klines = [k for k in all_klines if int(k[0]) >= start_ms]
    all_klines.sort(key=lambda x: int(x[0]))
    
    return all_klines
```

### Building a Multi-Data Analysis DataFrame

```python
import pandas as pd

def build_analysis_df(symbol, interval, start_ms, end_ms):
    """Build a comprehensive DataFrame with OHLCV + derived metrics."""
    
    # Fetch raw klines
    raw = fetch_klines(symbol, interval, start_ms, end_ms)
    
    df = pd.DataFrame(raw, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    
    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    
    # Add technical indicators
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
    
    # Volume analysis
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # Candle classification
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['is_bullish'] = df['close'] > df['open']
    
    return df
```

### Fetching Funding Rate and OI Context

```python
def fetch_funding_context(symbol, lookback_hours=72):
    """Get recent funding rates for context."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (lookback_hours * 3600 * 1000)
    
    response = requests.get(
        "https://api.bybit.com/v5/market/funding/history",
        params={
            "category": "linear",
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 200
        }
    )
    data = response.json()
    rates = data["result"]["list"]
    
    df = pd.DataFrame(rates)
    df['fundingRate'] = df['fundingRate'].astype(float)
    
    return {
        'current_rate': df['fundingRate'].iloc[0],
        'avg_rate_24h': df['fundingRate'].head(3).mean(),  # 3 × 8h = 24h
        'avg_rate_72h': df['fundingRate'].mean(),
        'max_rate': df['fundingRate'].max(),
        'min_rate': df['fundingRate'].min(),
        'trend': 'increasing' if df['fundingRate'].iloc[0] > df['fundingRate'].iloc[-1] else 'decreasing'
    }


def fetch_oi_context(symbol, interval="1h", lookback_hours=48):
    """Get recent open interest data."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (lookback_hours * 3600 * 1000)
    
    response = requests.get(
        "https://api.bybit.com/v5/market/open-interest",
        params={
            "category": "linear",
            "symbol": symbol,
            "intervalTime": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 200
        }
    )
    data = response.json()
    oi_data = data["result"]["list"]
    
    df = pd.DataFrame(oi_data)
    df['openInterest'] = df['openInterest'].astype(float)
    
    current_oi = df['openInterest'].iloc[0]
    oi_24h_ago = df['openInterest'].iloc[min(23, len(df)-1)]
    
    return {
        'current_oi': current_oi,
        'oi_change_24h': ((current_oi - oi_24h_ago) / oi_24h_ago) * 100,
        'oi_trend': 'rising' if current_oi > oi_24h_ago else 'falling',
        'oi_high_48h': df['openInterest'].max(),
        'oi_low_48h': df['openInterest'].min()
    }
```

---

## 5. Rate Limits & Best Practices

### Bybit API Rate Limits (Public Endpoints)
- Market data: 120 requests per second (per IP)
- Historical data: Be courteous — add 100ms delays between paginated requests

### Best Practices
1. Cache kline data locally after fetching — no need to re-fetch historical candles
2. For live monitoring, use WebSocket instead of polling REST endpoints
3. Always check `retCode` in responses (0 = success)
4. Use ISO 8601 timestamps in your analysis, convert from milliseconds at fetch time
5. Store raw data separately from derived metrics — recompute indicators as needed
6. When fetching multiple pairs, stagger requests to avoid burst rate limits

### WebSocket (For Live Data)

Bybit WebSocket URL: `wss://stream.bybit.com/v5/public/linear`

Subscribe to channels:
```json
{
    "op": "subscribe",
    "args": [
        "kline.15.BTCUSDT",
        "tickers.BTCUSDT",
        "orderbook.50.BTCUSDT"
    ]
}
```

Use WebSocket for:
- Real-time candle updates during active trading
- Live orderbook monitoring for execution
- Ticker updates for funding rate monitoring
