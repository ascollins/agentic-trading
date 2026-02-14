# Incident Runbook: Stuck Orders

## Symptoms

- Orders remain in `SUBMITTED` or `PARTIALLY_FILLED` state for an abnormally long time.
- `OrderManager.active_count` grows without orders reaching terminal states (`FILLED`, `CANCELLED`, `REJECTED`, `EXPIRED`).
- Reconciliation detects `stale_local_order` discrepancies (local order is active, exchange no longer reports it).
- Retry attempts exhausted: log messages showing `Attempt 3/3 for dedupe_key=...`.
- Risk engine flags unexpected exposure because the platform counts pending orders toward position limits.

Example log lines:

```
WARNING agentic_trading.execution.order_manager  Attempt 3/3 for dedupe_key=trend_BTC_USDT_1706000000
ERROR   agentic_trading.execution.order_manager  Invalid order transition: submitted -> submitted for dedupe_key=...
WARNING agentic_trading.execution.reconciliation  stale_local_order client_order_id=abc123 local_status=submitted
```

## Diagnosis

### Step 1: Check order manager state

Search for active (non-terminal) orders that may be stuck:

```bash
# Find orders that have been in SUBMITTED state
grep "state transition.*submitted" data/logs/*.log | tail -20

# Check for retry exhaustion
grep "Attempt.*dedupe_key" data/logs/*.log | tail -20
```

### Step 2: Check the exchange order status

Query the exchange directly to see if the order exists and what its true status is:

```bash
python -c "
import asyncio, ccxt.pro as ccxtpro
async def check():
    ex = ccxtpro.binance({'apiKey': 'YOUR_KEY', 'secret': 'YOUR_SECRET'})
    # Check open orders
    orders = await ex.fetch_open_orders('BTC/USDT')
    for o in orders:
        print(f\"id={o['id']} client_id={o['clientOrderId']} status={o['status']} filled={o['filled']}/{o['amount']}\")
    # Also check closed/cancelled orders (last 24h)
    closed = await ex.fetch_closed_orders('BTC/USDT', limit=20)
    for o in closed:
        print(f\"[closed] id={o['id']} client_id={o['clientOrderId']} status={o['status']}\")
    await ex.close()
asyncio.run(check())
"
```

Possible findings:

| Exchange status | Local status   | Cause                                                        |
|----------------|----------------|--------------------------------------------------------------|
| Not found      | `SUBMITTED`    | Order was rejected by exchange but update was missed         |
| `closed`       | `SUBMITTED`    | Fill notification was missed (WebSocket gap)                 |
| `cancelled`    | `SUBMITTED`    | Exchange auto-cancelled (e.g., expired, margin insufficient) |
| `open`         | `SUBMITTED`    | Order is legitimately open; may need patience or manual cancel |

### Step 3: Check for execution engine errors

```bash
grep "ExchangeError\|OrderRejected\|order submission" data/logs/*.log | tail -20
```

Common causes:

- **Rate limiting**: The exchange rejected the order due to API rate limits. CCXT should handle this, but check for `RateLimitExceeded` errors.
- **Insufficient margin**: The order was rejected but the rejection callback was lost.
- **Network timeout**: The order submission timed out, leaving the local state ambiguous.

### Step 4: Check valid state transitions

The `OrderManager` enforces this state machine:

```
PENDING -> SUBMITTED | REJECTED | CANCELLED
SUBMITTED -> PARTIALLY_FILLED | FILLED | CANCELLED | REJECTED | EXPIRED
PARTIALLY_FILLED -> PARTIALLY_FILLED | FILLED | CANCELLED
FILLED -> (terminal)
CANCELLED -> (terminal)
REJECTED -> (terminal)
EXPIRED -> (terminal)
```

If you see `Invalid order transition` errors in the logs, the system received an unexpected status update. This usually indicates a missed intermediate state.

## Resolution

### Let reconciliation handle it (preferred)

The reconciliation loop (running every 15-30 seconds) automatically detects and repairs stuck orders:

- If the exchange no longer reports the order: reconciliation transitions it to `CANCELLED` locally.
- If the exchange reports it as filled: reconciliation creates a stub and updates the state.

Wait for at least 2 reconciliation cycles and check if the issue self-resolves:

```bash
grep "Reconciliation\|stale_local_order\|Auto-repair" data/logs/*.log | tail -20
```

### Manual cancel on exchange

If the order is genuinely stuck on the exchange (still `open`), cancel it manually:

```bash
python -c "
import asyncio, ccxt.pro as ccxtpro
async def cancel():
    ex = ccxtpro.binance({'apiKey': 'YOUR_KEY', 'secret': 'YOUR_SECRET'})
    result = await ex.cancel_order('EXCHANGE_ORDER_ID', 'BTC/USDT')
    print(result)
    await ex.close()
asyncio.run(cancel())
"
```

After manual cancellation, the next reconciliation cycle will detect the exchange status change and update local state.

### Force state transition (last resort)

If auto-repair is not resolving the stuck order and you need to clear local state:

1. Activate the kill switch to prevent new orders:

```bash
python scripts/killswitch.py --activate --reason "Clearing stuck orders"
```

2. Restart the trading process. On restart, the `OrderManager` starts with a clean in-memory state. The reconciliation loop will immediately sync against the exchange.

3. Verify clean state:

```bash
grep "Reconciliation clean" data/logs/*.log | tail -5
```

4. Deactivate the kill switch:

```bash
python scripts/killswitch.py --deactivate
```

### Prevent recurrence

- Ensure `reconciliation_interval_seconds` is set to 15s or less for live trading (default in `configs/live.toml`).
- Check that WebSocket connections are stable (see `runbooks/incident_feed_outage.md`). Order status updates arrive via WebSocket; if the connection drops, updates are missed.
- Review exchange rate limits. If `rate_limit` in the exchange config is set too aggressively, CCXT may queue or drop requests. The default of 1200 (requests per minute) is safe for most use cases.
- Monitor the `order_manager_active_count` metric in Prometheus/Grafana. A steadily growing count indicates orders are not reaching terminal states.
