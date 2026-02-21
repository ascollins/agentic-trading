# Event Log & Audit Spine

## Overview

The telemetry spine is a **parallel append-only event log** that sits alongside the existing event bus. It provides a unified view of everything that happened for a given trace across all components — something the scattered audit trails (PostgreSQL, JSONL, in-memory) cannot offer today.

The spine does **not** replace the event bus. It is a secondary, opt-in layer that components write to explicitly via `EventWriter.write()`.

```
BaseEvent (existing, unchanged)
    ↓ mapper.map_base_event_to_spine()
SpineEvent (new, parallel telemetry)
    ↓ EventWriter.write()
ISpineStorage (PostgreSQL / Memory)
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Event Bus (existing)                │
│   RedisStreamsBus / MemoryEventBus — 28 topics      │
└────────────────────────┬────────────────────────────┘
                         │ BaseEvent
                         ▼
┌─────────────────────────────────────────────────────┐
│              Telemetry Mapper                        │
│   map_base_event_to_spine() — taxonomy classifier   │
└────────────────────────┬────────────────────────────┘
                         │ SpineEvent
                         ▼
┌─────────────────────────────────────────────────────┐
│              EventWriter (async, buffered)           │
│   batch_size=100, flush_interval=5.0s               │
└────────────────────────┬────────────────────────────┘
                         │ write_batch()
                         ▼
┌─────────────────────────────────────────────────────┐
│              ISpineStorage                           │
│   MemorySpineStorage (backtest/test)                │
│   PostgresSpineStorage (paper/live)                 │
└─────────────────────────────────────────────────────┘
```

## Field Reference

Every spine event carries these 15 fields:

| # | Field | Type | Source | Description |
|---|-------|------|--------|-------------|
| 1 | `event_id` | UUID v4 | `core.ids.new_id()` | Unique per spine event |
| 2 | `trace_id` | UUID v4 | Copied from BaseEvent | Full decision chain correlation |
| 3 | `span_id` | UUID v4 | `core.ids.new_id()` | Local span within trace |
| 4 | `tenant_id` | string | Default `"default"` | Multi-tenant isolation |
| 5 | `actor` | string | agent_id or `"system"` | Who emitted this |
| 6 | `component` | string | = BaseEvent.source_module | Module/subsystem |
| 7 | `timestamp` | datetime (UTC) | `core.ids.utc_now()` | Event creation time |
| 8 | `input_hash` | string | `core.ids.payload_hash()` | SHA256[:16] of input |
| 9 | `output_hash` | string | `core.ids.payload_hash()` | SHA256[:16] of output |
| 10 | `schema_version` | int | Default `1` | For schema evolution |
| 11 | `event_type` | EventTaxonomy | Set per subclass | Taxonomy classification |
| 12 | `causation_id` | string | From BaseEvent | Parent event_id (DAG) |
| 13 | `payload` | dict | Event-specific data | JSONB in storage |
| 14 | `latency_ms` | float or null | Optional | Duration in ms |
| 15 | `error` | string or null | Optional | Error message |

## Event Taxonomy (9 types)

| Type | Description | Example Sources |
|------|-------------|-----------------|
| `user_action` | Human-initiated action | CLI backtest command, API call |
| `agent_step_started` | Agent processing step begins | Default for unmapped events |
| `agent_step_completed` | Agent step finishes with output | Signal, FillEvent, FeatureVector |
| `tool_call` | Exchange/tool gateway call | ToolCallRecorded |
| `retrieval` | Data fetch operation | CandleEvent, TickEvent |
| `decision` | Deterministic decision | RiskCheckResult, GovernanceDecision |
| `validation` | Schema/data validation | ReconciliationResult |
| `exception` | Error or circuit breaker | CircuitBreakerEvent, KillSwitchEvent |
| `cost_metric` | Cost/usage tracking | LLM token counts, API costs |

### BaseEvent -> SpineEvent Mapping

| BaseEvent Type | Spine Taxonomy |
|---|---|
| `ToolCallRecorded` | `tool_call` |
| `RiskCheckResult` | `decision` |
| `GovernanceDecision` | `decision` |
| `Signal` | `agent_step_completed` |
| `OrderIntent` | `agent_step_completed` |
| `FillEvent` | `agent_step_completed` |
| `FeatureVector` | `agent_step_completed` |
| `CMTAssessment` | `agent_step_completed` |
| `CandleEvent` | `retrieval` |
| `CircuitBreakerEvent` | `exception` |
| `KillSwitchEvent` | `exception` |
| `ReconciliationResult` | `validation` |
| Everything else | `agent_step_started` (safe default) |

## Storage

### PostgreSQL Table: `spine_events`

```sql
CREATE TABLE spine_events (
    event_id      VARCHAR(64)   PRIMARY KEY,
    trace_id      VARCHAR(64)   NOT NULL,
    span_id       VARCHAR(64)   NOT NULL,
    causation_id  VARCHAR(64)   DEFAULT '',
    tenant_id     VARCHAR(64)   NOT NULL DEFAULT 'default',
    event_type    VARCHAR(32)   NOT NULL,
    component     VARCHAR(128)  NOT NULL,
    actor         VARCHAR(128)  NOT NULL DEFAULT 'system',
    timestamp     TIMESTAMPTZ   NOT NULL,
    schema_version INTEGER      DEFAULT 1,
    input_hash    VARCHAR(32)   DEFAULT '',
    output_hash   VARCHAR(32)   DEFAULT '',
    latency_ms    NUMERIC(12,3),
    error         TEXT          DEFAULT '',
    payload       JSONB         NOT NULL DEFAULT '{}'
);
```

### Indexes

```sql
CREATE INDEX ix_spine_tenant_trace      ON spine_events (tenant_id, trace_id);
CREATE INDEX ix_spine_tenant_timestamp   ON spine_events (tenant_id, timestamp);
CREATE INDEX ix_spine_tenant_event_type  ON spine_events (tenant_id, event_type);
CREATE INDEX ix_spine_tenant_component   ON spine_events (tenant_id, component);
CREATE INDEX ix_spine_tenant_actor       ON spine_events (tenant_id, actor);
CREATE INDEX ix_spine_span_id            ON spine_events (span_id);
```

### Partitioning (Future)

For MVP, use a single unpartitioned table. Add partitioning when data exceeds ~10M rows:

```sql
-- Future: partition by RANGE on timestamp (daily)
CREATE TABLE spine_events (
    ...
) PARTITION BY RANGE (timestamp);

CREATE TABLE spine_events_2025_01_15 PARTITION OF spine_events
    FOR VALUES FROM ('2025-01-15') TO ('2025-01-16');
```

### Retention

90-day default. Old partitions dropped via scheduled job. Configurable per tenant.

```sql
-- Example: drop partitions older than 90 days
DROP TABLE IF EXISTS spine_events_2024_10_15;
```

## Example Payloads

### 1. ToolCall — submit_order

```json
{
  "event_id": "a1b2c3d4-...",
  "trace_id": "t1r2a3c4-...",
  "span_id": "s1p2a3n4-...",
  "tenant_id": "default",
  "actor": "control_plane.tool_gateway",
  "component": "control_plane.tool_gateway",
  "timestamp": "2025-01-15T10:30:00.123Z",
  "input_hash": "3f8a1b2c4d5e6f70",
  "output_hash": "7a8b9c0d1e2f3a4b",
  "schema_version": 1,
  "event_type": "tool_call",
  "causation_id": "prev-event-id-...",
  "payload": {
    "source_event_type": "ToolCallRecorded",
    "topic": "system",
    "action_id": "act-123",
    "tool_name": "submit_order",
    "success": true,
    "latency_ms": 45.2,
    "was_idempotent_replay": false
  },
  "latency_ms": 45.2,
  "error": null
}
```

### 2. Decision — risk_check (passed)

```json
{
  "event_id": "b2c3d4e5-...",
  "trace_id": "t1r2a3c4-...",
  "span_id": "s2p3a4n5-...",
  "tenant_id": "default",
  "actor": "risk",
  "component": "risk",
  "timestamp": "2025-01-15T10:29:59.800Z",
  "input_hash": "1a2b3c4d5e6f7890",
  "output_hash": "1a2b3c4d5e6f7890",
  "schema_version": 1,
  "event_type": "decision",
  "causation_id": "intent-event-id-...",
  "payload": {
    "source_event_type": "RiskCheckResult",
    "topic": "risk",
    "check_name": "max_position_size",
    "passed": true,
    "reason": "",
    "details": {"current_exposure": 0.15, "max_allowed": 0.25},
    "order_intent_id": "intent-123"
  },
  "latency_ms": null,
  "error": null
}
```

### 3. Decision — policy_eval (blocked)

```json
{
  "event_id": "c3d4e5f6-...",
  "trace_id": "t1r2a3c4-...",
  "span_id": "s3p4a5n6-...",
  "tenant_id": "default",
  "actor": "governance",
  "component": "governance",
  "timestamp": "2025-01-15T10:29:59.500Z",
  "schema_version": 1,
  "event_type": "decision",
  "causation_id": "signal-event-id-...",
  "payload": {
    "source_event_type": "GovernanceDecision",
    "topic": "governance",
    "strategy_id": "smc_btcusdt",
    "symbol": "BTCUSDT",
    "action": "BLOCK",
    "reason": "Maturity L0_SHADOW: paper-only",
    "sizing_multiplier": 0.0,
    "maturity_level": "L0_SHADOW",
    "impact_tier": "MEDIUM"
  },
  "latency_ms": null,
  "error": null
}
```

### 4. AgentStepStarted — signal_generation

```json
{
  "event_id": "d4e5f6a7-...",
  "trace_id": "t1r2a3c4-...",
  "span_id": "s4p5a6n7-...",
  "tenant_id": "default",
  "actor": "strategy",
  "component": "strategy",
  "timestamp": "2025-01-15T10:29:58.000Z",
  "schema_version": 1,
  "event_type": "agent_step_started",
  "causation_id": "candle-event-id-...",
  "payload": {
    "source_event_type": "RegimeState",
    "topic": "strategy.signal",
    "symbol": "BTCUSDT",
    "regime": "TRENDING_UP",
    "volatility": "NORMAL",
    "confidence": 0.85
  },
  "latency_ms": null,
  "error": null
}
```

### 5. AgentStepCompleted — signal_created

```json
{
  "event_id": "e5f6a7b8-...",
  "trace_id": "t1r2a3c4-...",
  "span_id": "s5p6a7n8-...",
  "tenant_id": "default",
  "actor": "strategy",
  "component": "strategy",
  "timestamp": "2025-01-15T10:29:59.000Z",
  "schema_version": 1,
  "event_type": "agent_step_completed",
  "causation_id": "regime-event-id-...",
  "payload": {
    "source_event_type": "Signal",
    "topic": "strategy.signal",
    "strategy_id": "smc_btcusdt",
    "symbol": "BTCUSDT",
    "direction": "LONG",
    "confidence": 0.78,
    "rationale": "Bullish order block at 42500"
  },
  "latency_ms": null,
  "error": null
}
```

### 6. Exception — circuit_breaker_tripped

```json
{
  "event_id": "f6a7b8c9-...",
  "trace_id": "t2r3a4c5-...",
  "span_id": "s6p7a8n9-...",
  "tenant_id": "default",
  "actor": "risk",
  "component": "risk",
  "timestamp": "2025-01-15T11:00:00.000Z",
  "schema_version": 1,
  "event_type": "exception",
  "causation_id": "",
  "payload": {
    "source_event_type": "CircuitBreakerEvent",
    "topic": "risk",
    "breaker_type": "DAILY_LOSS",
    "tripped": true,
    "symbol": "BTCUSDT",
    "reason": "Daily loss limit exceeded: -2.5% > -2.0%",
    "threshold": 2.0,
    "current_value": 2.5
  },
  "latency_ms": null,
  "error": "Daily loss limit exceeded: -2.5% > -2.0%"
}
```

### 7. UserAction — cli_backtest

```json
{
  "event_id": "a7b8c9d0-...",
  "trace_id": "t3r4a5c6-...",
  "span_id": "s7p8a9n0-...",
  "tenant_id": "default",
  "actor": "cli",
  "component": "cli",
  "timestamp": "2025-01-15T09:00:00.000Z",
  "schema_version": 1,
  "event_type": "user_action",
  "causation_id": "",
  "payload": {
    "action": "start_backtest",
    "params": {
      "config": "configs/backtest.toml",
      "symbol": "BTCUSDT",
      "start_date": "2024-01-01",
      "end_date": "2024-12-31"
    },
    "session_id": "cli-session-abc123"
  },
  "latency_ms": null,
  "error": null
}
```

### 8. CostMetric — llm_tokens

```json
{
  "event_id": "b8c9d0e1-...",
  "trace_id": "t4r5a6c7-...",
  "span_id": "s8p9a0n1-...",
  "tenant_id": "default",
  "actor": "intelligence.cmt",
  "component": "intelligence.cmt",
  "timestamp": "2025-01-15T10:30:15.000Z",
  "schema_version": 1,
  "event_type": "cost_metric",
  "causation_id": "cmt-step-id-...",
  "payload": {
    "metric_type": "llm_tokens",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "input_tokens": 3200,
    "output_tokens": 1500,
    "cost_usd": 0.0285
  },
  "latency_ms": 4500.0,
  "error": null
}
```

### 9. Retrieval — candle_data

```json
{
  "event_id": "c9d0e1f2-...",
  "trace_id": "t1r2a3c4-...",
  "span_id": "s9p0a1n2-...",
  "tenant_id": "default",
  "actor": "data",
  "component": "data",
  "timestamp": "2025-01-15T10:29:57.000Z",
  "schema_version": 1,
  "event_type": "retrieval",
  "causation_id": "",
  "payload": {
    "source_event_type": "CandleEvent",
    "topic": "market.candle",
    "symbol": "BTCUSDT",
    "exchange": "bybit",
    "timeframe": "1m",
    "open": 42500.0,
    "high": 42550.0,
    "low": 42480.0,
    "close": 42530.0,
    "volume": 125.5,
    "is_closed": true
  },
  "latency_ms": null,
  "error": null
}
```

### 10. Validation — reconciliation

```json
{
  "event_id": "d0e1f2a3-...",
  "trace_id": "t5r6a7c8-...",
  "span_id": "s0p1a2n3-...",
  "tenant_id": "default",
  "actor": "execution",
  "component": "execution",
  "timestamp": "2025-01-15T10:35:00.000Z",
  "schema_version": 1,
  "event_type": "validation",
  "causation_id": "",
  "payload": {
    "source_event_type": "ReconciliationResult",
    "topic": "system",
    "exchange": "bybit",
    "discrepancies": [],
    "orders_synced": 12,
    "positions_synced": 3,
    "balances_synced": 1,
    "repairs_applied": 0
  },
  "latency_ms": null,
  "error": null
}
```

## Minimum Viable Replay Procedure

Given a `trace_id`, reconstruct the full decision chain:

### Step 1: Find the trace

```sql
SELECT event_type, component, timestamp, causation_id, event_id,
       payload->>'source_event_type' AS source_type
FROM   spine_events
WHERE  tenant_id = 'default'
  AND  trace_id  = 'YOUR_TRACE_ID'
ORDER BY timestamp;
```

### Step 2: Retrieve the causation chain

```sql
WITH RECURSIVE chain AS (
    -- Start from the root (no causation_id)
    SELECT event_id, causation_id, event_type, component, timestamp,
           payload, 0 AS depth
    FROM   spine_events
    WHERE  tenant_id = 'default'
      AND  trace_id  = 'YOUR_TRACE_ID'
      AND  causation_id = ''

    UNION ALL

    -- Follow causation links
    SELECT s.event_id, s.causation_id, s.event_type, s.component, s.timestamp,
           s.payload, c.depth + 1
    FROM   spine_events s
    JOIN   chain c ON s.causation_id = c.event_id
    WHERE  s.tenant_id = 'default'
)
SELECT * FROM chain ORDER BY depth, timestamp;
```

### Step 3: Reconstruct DAG

```sql
-- All parent-child edges in this trace
SELECT causation_id AS parent_event_id,
       event_id     AS child_event_id,
       event_type,
       component,
       timestamp
FROM   spine_events
WHERE  tenant_id = 'default'
  AND  trace_id  = 'YOUR_TRACE_ID'
  AND  causation_id != ''
ORDER BY timestamp;
```

### Step 4: Inspect decisions

```sql
SELECT event_id, component, timestamp,
       payload->>'source_event_type' AS decision_type,
       payload->>'passed'            AS passed,
       payload->>'reason'            AS reason,
       payload->>'action'            AS action
FROM   spine_events
WHERE  tenant_id  = 'default'
  AND  trace_id   = 'YOUR_TRACE_ID'
  AND  event_type = 'decision'
ORDER BY timestamp;
```

### Step 5: Verify integrity

```sql
-- Check that input/output hashes are consistent
SELECT event_id, event_type, input_hash, output_hash,
       payload->>'source_event_type' AS source_type
FROM   spine_events
WHERE  tenant_id = 'default'
  AND  trace_id  = 'YOUR_TRACE_ID'
  AND  input_hash != ''
ORDER BY timestamp;
```

## Grafana Query Examples

### Events by type (time series)

```sql
SELECT date_trunc('minute', timestamp) AS time,
       event_type,
       COUNT(*) AS event_count
FROM   spine_events
WHERE  tenant_id = 'default'
  AND  timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY 1, 2
ORDER BY 1;
```

### Error rate

```sql
SELECT date_trunc('minute', timestamp) AS time,
       COUNT(*) FILTER (WHERE error != '') AS errors,
       COUNT(*) AS total,
       ROUND(100.0 * COUNT(*) FILTER (WHERE error != '') / NULLIF(COUNT(*), 0), 2) AS error_pct
FROM   spine_events
WHERE  tenant_id = 'default'
  AND  timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY 1
ORDER BY 1;
```

### p95 latency by component

```sql
SELECT component,
       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
       PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
       COUNT(*) AS total
FROM   spine_events
WHERE  tenant_id  = 'default'
  AND  timestamp >= NOW() - INTERVAL '1 hour'
  AND  latency_ms IS NOT NULL
GROUP BY component
ORDER BY p95_ms DESC;
```

## Integration Guide

### Writing spine events from existing components

```python
from agentic_trading.telemetry import EventWriter, MemorySpineStorage, map_base_event_to_spine

# Setup (once, during bootstrap)
storage = MemorySpineStorage()  # or PostgresSpineStorage(session_factory)
writer = EventWriter(storage, batch_size=100, flush_interval=5.0)
await writer.start()

# In any event handler:
async def on_signal(signal: Signal) -> None:
    # ... existing logic ...

    # Emit to spine
    spine_event = map_base_event_to_spine(signal, topic="strategy.signal")
    await writer.write(spine_event)

# Shutdown
await writer.stop()
```

### Using SpanContext for nested operations

```python
from agentic_trading.telemetry import SpanContext

span_ctx = SpanContext(trace_id=event.trace_id)

# Enter a span
span_id = span_ctx.push_span()
try:
    # ... do work ...
    pass
finally:
    span_ctx.pop_span()
```

### Querying the spine

```python
# By trace
events = await storage.query_by_trace("trace-id-here")

# By time range
from datetime import datetime, timezone, timedelta
end = datetime.now(timezone.utc)
start = end - timedelta(hours=1)
events = await storage.query_by_time_range(start, end, event_type="decision")
```
