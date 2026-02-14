# Incident Runbook: Exchange Outage

## Symptoms

- No new candle data arriving (stale `last_candle_ts` in feed manager).
- WebSocket connection errors or timeouts in logs.
- HTTP 5xx errors from CCXT adapter calls (`ExchangeError` exceptions).
- Circuit breakers tripping for `error_rate` or `staleness` breaker types.
- Prometheus metric `exchange_request_errors_total` spiking.
- Reconciliation failures: `fetch_error` discrepancies for orders, positions, or balances.

Example log lines:

```
ERROR  agentic_trading.data.feed_manager  Error in feed loop binance BTC/USDT (attempt 5/20, backoff 16.0s)
ERROR  agentic_trading.execution.reconciliation  Failed to fetch open orders for recon: ExchangeNotAvailable
WARNING agentic_trading.risk.circuit_breakers  Circuit breaker TRIPPED: type=error_rate, symbol=*, value=15.0000, threshold=10.0000
```

## Diagnosis

### Step 1: Check circuit breaker state

Look at current circuit breaker status in the logs or metrics:

```bash
# Search for tripped breakers in recent logs
grep -i "circuit breaker TRIPPED" data/logs/*.log | tail -20
```

If `error_rate` or `staleness` breakers are tripped, the system has already detected the outage and is self-protecting.

### Step 2: Check exchange status page

Verify whether the exchange itself is experiencing issues:

- Binance: https://www.binance.com/en/support/announcement (system maintenance notices)
- Bybit: https://announcements.bybit.com
- General: https://downdetector.com

### Step 3: Check feed manager task state

Look for consecutive error counts and backoff behavior in the feed loop logs:

```bash
grep "feed loop" data/logs/*.log | tail -30
```

The feed manager uses exponential backoff (1s to 60s) and stops after 20 consecutive errors per symbol.

### Step 4: Check network connectivity

```bash
# Test REST API connectivity
python -c "import ccxt; print(ccxt.binance().fetch_status())"

# Test from within Docker
docker compose exec trading python -c "import ccxt; print(ccxt.binance().fetch_status())"
```

## Mitigation

### Activate kill switch (if positions are at risk)

If you have open positions and the exchange is unreachable, activate the kill switch to prevent the system from attempting further order submissions:

```bash
python scripts/killswitch.py --activate --reason "Exchange outage: [binance|bybit] unreachable"
```

This sets `kill_switch:active = 1` in Redis. The execution engine checks this before every order submission.

### Switch to paper mode (if partial outage)

If only one exchange is affected and you want to continue trading on others, you can restart with a modified config that excludes the affected exchange. Edit `configs/paper.toml` or `configs/live.toml` to comment out the affected `[[exchanges]]` block, then restart.

### Reduce exposure

If the exchange is intermittently available, enable safe mode to reduce position sizes while the situation is unstable:

```toml
# In your active config file
[safe_mode]
enabled = true
max_symbols = 3
max_leverage = 2
position_size_multiplier = 0.25
```

## Recovery

### Step 1: Confirm exchange is back

Verify REST API and WebSocket connectivity:

```bash
python -c "import ccxt; e = ccxt.binance(); print(e.fetch_status()); print(e.fetch_ticker('BTC/USDT'))"
```

### Step 2: Deactivate kill switch

```bash
python scripts/killswitch.py --deactivate
```

### Step 3: Force reconciliation

After an outage, local state may be stale. The reconciliation loop will automatically sync, but if you want to force it immediately, restart the trading process. The reconciliation loop runs on startup and then every `reconciliation_interval_seconds` (default: 30s for paper, 15s for live).

### Step 4: Review position state

Check for any discrepancies detected by reconciliation:

```bash
grep "Reconciliation found" data/logs/*.log | tail -10
grep "position_mismatch\|stale_local_order\|missing_local_order" data/logs/*.log | tail -20
```

If `auto_repair` is enabled (the default), the reconciliation loop will have updated local state to match the exchange. Review the repairs to confirm they are correct.

### Step 5: Monitor recovery

Watch for:

- Circuit breakers resetting (logged as `Circuit breaker RESET`).
- Feed loops resuming (`Feed loop started` log messages).
- Clean reconciliation cycles (`Reconciliation clean (no discrepancies)`).

### Step 6: Disable safe mode if it was enabled

Once you are confident the exchange is stable, set `safe_mode.enabled = false` in your config and restart.
