# Incident Runbook: Position/State Desync

## Symptoms

- `position_mismatch` discrepancies in reconciliation logs.
- `balance_mismatch` discrepancies in reconciliation logs.
- `ReconciliationResult` events with non-empty `discrepancies` list.
- Portfolio P&L calculations diverging from expected values.
- Alerts triggered by the risk engine for unexpected exposure levels.

Example log lines:

```
WARNING agentic_trading.execution.reconciliation  Reconciliation found 3 discrepancies (orders=2 positions=1 balances=0 repairs=2)
```

Discrepancy details appear as structured log fields:

```json
{"type": "position_mismatch", "symbol": "BTC/USDT", "local_qty": "0.05", "exchange_qty": "0.08", "detail": "Quantity mismatch"}
{"type": "stale_local_order", "client_order_id": "abc123", "local_status": "submitted"}
{"type": "missing_local_order", "client_order_id": "def456", "exchange_status": "open"}
```

## Diagnosis

### Step 1: Check reconciliation logs

```bash
# Recent reconciliation results
grep "Reconciliation found\|position_mismatch\|balance_mismatch\|stale_local_order\|missing_local_order" data/logs/*.log | tail -30
```

Identify:

- **Which symbols** are affected.
- **Which direction** the mismatch goes (local > exchange or exchange > local).
- **Whether auto-repair applied** (look for `repairs_applied` count in the log).

### Step 2: Compare local vs exchange state

Query the exchange directly to get the current truth:

```bash
# Check positions on exchange
python -c "
import asyncio, ccxt.pro as ccxtpro
async def check():
    ex = ccxtpro.binance({'apiKey': 'YOUR_KEY', 'secret': 'YOUR_SECRET'})
    positions = await ex.fetch_positions()
    for p in positions:
        if float(p['contracts']) != 0:
            print(f\"{p['symbol']}: {p['contracts']} contracts, side={p['side']}\")
    await ex.close()
asyncio.run(check())
"
```

Compare against what the platform believes it holds. The reconciliation loop stores its view in `ReconciliationLoop.local_positions`.

### Step 3: Check order manager state

Look for orders stuck in non-terminal states that may be causing the desync:

```bash
grep "stale_local_order" data/logs/*.log | tail -10
```

A stale local order means the platform thinks an order is active, but the exchange no longer reports it. This typically happens when:

- An order was filled or cancelled but the WebSocket update was missed.
- The process restarted and lost in-memory order state.
- The exchange had an API hiccup during order submission.

### Step 4: Check for recent process restarts

```bash
grep "Starting FeedManager\|ReconciliationLoop started\|Application started" data/logs/*.log | tail -10
```

Process restarts lose in-memory order state. The reconciliation loop is designed to recover from this, but there may be a brief window of inconsistency.

## Resolution

### Auto-repair (default behavior)

The `ReconciliationLoop` runs with `auto_repair = True` by default. When it detects discrepancies:

- **Missing local orders**: Creates a stub entry in the `OrderManager` matching the exchange state.
- **Stale local orders**: Transitions the local order to `CANCELLED` status.
- **Position mismatches**: Updates local position to match the exchange quantity.
- **Balance mismatches**: Updates local balance to match the exchange (with >0.01 tolerance).

Auto-repair is logged. Verify repairs were applied:

```bash
grep "Auto-repair\|repairs_applied" data/logs/*.log | tail -10
```

### Force reconciliation

If auto-repair is not resolving the issue (e.g., it is disabled or errors are preventing the loop from running), force a reconciliation by restarting the trading process:

```bash
# If using Docker
docker compose restart trading

# If running directly
# Stop the process (Ctrl+C), then restart
python scripts/run_paper.py --config configs/paper.toml
```

The reconciliation loop starts immediately on boot, before any new trading decisions are made.

### Manual position correction

If auto-repair is insufficient or you need to correct a position that the reconciliation loop cannot handle:

1. **Activate kill switch** to prevent new orders while you fix state:

```bash
python scripts/killswitch.py --activate --reason "Manual position correction in progress"
```

2. **Close the mismatched position on the exchange manually** (via exchange UI or API).

3. **Restart the trading process** to let reconciliation sync the clean state.

4. **Deactivate kill switch**:

```bash
python scripts/killswitch.py --deactivate
```

### Persistent desync

If discrepancies keep recurring after reconciliation:

- Check if the reconciliation interval is too long. For live trading, 15 seconds is recommended (`reconciliation_interval_seconds = 15` in `configs/live.toml`).
- Check for WebSocket connectivity issues (see `runbooks/incident_feed_outage.md`).
- Check for order execution errors that might be causing the platform to lose track of fills (see `runbooks/incident_stuck_orders.md`).
- Verify that the exchange adapter is returning complete position data (some exchanges exclude small dust positions).
