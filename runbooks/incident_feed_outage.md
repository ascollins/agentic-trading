# Incident Runbook: Market Data Feed Outage

## Symptoms

- No new candle events being published to the event bus.
- `staleness` circuit breaker tripped (logs show `Circuit breaker TRIPPED: type=staleness`).
- Feed loop error count climbing: `Error in feed loop binance BTC/USDT (attempt N/20)`.
- Feed loop exited after 20 consecutive errors: `Feed loop ... exceeded max errors (20); stopping`.
- Strategies not generating signals because `on_candle()` is never called.
- `FeedManager.active_task_count` drops to 0 or below expected count.

Example log lines:

```
ERROR   agentic_trading.data.feed_manager  Error in feed loop binance BTC/USDT (attempt 12/20, backoff 32.0s)
CRITICAL agentic_trading.data.feed_manager  Feed loop binance BTC/USDT exceeded max errors (20); stopping.
WARNING agentic_trading.risk.circuit_breakers  Circuit breaker TRIPPED: type=staleness, symbol=BTC/USDT, value=120.0000, threshold=60.0000
```

## Diagnosis

### Step 1: Check feed manager state

```bash
# Look for feed loop status messages
grep "feed loop\|Feed loop\|FeedManager" data/logs/*.log | tail -30
```

Key things to identify:

- **Which exchanges and symbols** are affected (all of them, or just one?).
- **Error pattern**: Is it connection errors, authentication errors, or data parsing errors?
- **Backoff state**: The feed manager uses exponential backoff from 1s to 60s. High backoff values indicate persistent failures.

### Step 2: Check WebSocket connections

The feed manager uses CCXT Pro's `watch_ohlcv()` which manages WebSocket connections internally. Common WebSocket issues:

```bash
# Look for WebSocket-specific errors
grep -i "websocket\|ws\|connection\|disconnect\|reconnect" data/logs/*.log | tail -20
```

Possible causes:

| Error pattern                     | Likely cause                                  |
|----------------------------------|-----------------------------------------------|
| `ConnectionClosed`               | Exchange closed the WebSocket (maintenance, rate limit) |
| `ConnectionRefused`              | Exchange WebSocket endpoint is down           |
| `AuthenticationError`            | API key expired or invalid                    |
| `RateLimitExceeded`              | Too many WebSocket connections or subscriptions |
| `InvalidNonce`                   | Clock drift between client and exchange       |

### Step 3: Check exchange WebSocket limits

Exchanges impose limits on WebSocket connections:

- **Binance**: 5 connections per IP, 1024 streams per connection (combined stream).
- **Bybit**: 20 connections per IP, 10 subscriptions per connection.

If you are subscribing to many symbols, you may be hitting these limits. Check:

```bash
# Count active feed tasks
grep "FeedManager launched" data/logs/*.log | tail -1
```

The feed manager creates one task per (exchange, symbol). If you have 50 symbols on 2 exchanges, that is 100 feed tasks. Binance can handle this with combined streams; Bybit may require reducing the symbol count.

### Step 4: Verify network and DNS

```bash
# Test WebSocket connectivity to Binance
python -c "
import asyncio, ccxt.pro as ccxtpro
async def test():
    ex = ccxtpro.binance()
    try:
        ohlcv = await ex.watch_ohlcv('BTC/USDT', '1m')
        print(f'Received {len(ohlcv)} candles -- WebSocket OK')
    finally:
        await ex.close()
asyncio.run(test())
"
```

If this fails, the issue is network-level (firewall, DNS, exchange endpoint down).

### Step 5: Check circuit breaker state

```bash
grep "circuit breaker\|TRIPPED\|RESET" data/logs/*.log | tail -20
```

If `staleness` breakers are tripped, the system has detected the feed outage. This is expected behavior -- the breakers prevent trading on stale data.

## Resolution

### Scenario 1: Single exchange feed down

If only one exchange's feed is affected and the exchange is confirmed down:

1. The circuit breakers for that exchange's symbols will trip automatically, preventing trading on stale data.
2. Strategies running on the other exchange will continue normally.
3. Wait for the exchange to recover. The feed manager's backoff loop will automatically reconnect.

### Scenario 2: All feeds down

1. **Activate kill switch** if you have open positions and no data:

```bash
python scripts/killswitch.py --activate --reason "All market data feeds down"
```

2. Diagnose the root cause (network, exchange, or application issue).

### Scenario 3: Feed loop stopped (max errors exceeded)

If a feed loop has exited after 20 consecutive errors, it will not auto-restart. You need to restart the trading process:

```bash
# If running directly
# Stop (Ctrl+C) and restart
python scripts/run_paper.py --config configs/paper.toml

# If using Docker
docker compose restart trading
```

On restart, the `FeedManager.start()` method reinitializes all CCXT Pro exchange connections and spawns fresh feed tasks.

### Scenario 4: WebSocket rate limit

If you are hitting exchange WebSocket limits:

1. Reduce the number of subscribed symbols in `configs/instruments.toml`.
2. Remove less important symbols from the whitelist.
3. Consider running separate instances per exchange to spread the connection load.

### Recovery verification

After resolving the issue, confirm feeds are healthy:

```bash
# Look for successful candle emissions
grep "Emitted closed candle" data/logs/*.log | tail -10

# Verify feed tasks are running
grep "Feed loop started" data/logs/*.log | tail -10

# Verify circuit breakers have reset
grep "Circuit breaker RESET" data/logs/*.log | tail -10
```

Expected recovery sequence:

1. Feed loops reconnect and start receiving data.
2. Staleness circuit breakers reset after cooldown (default: 300 seconds) once fresh data arrives.
3. Strategies resume receiving `on_candle()` calls and generating signals.
4. If kill switch was activated, deactivate it:

```bash
python scripts/killswitch.py --deactivate
```

5. Monitor for 10-15 minutes to confirm stability before stepping away.
