# Canonical Domain Model — Agentic Trading Platform

## Purpose

This document defines the **minimal set of core entities** that every component must agree on. It exists to answer three questions:

1. **"Why did it do that?"** — Full trace from market data through reasoning to execution.
2. **"What is the system of record?"** — One authoritative store per entity, no ambiguity.
3. **"Can I add a new agent without breaking anything?"** — Stable entity contracts with versioned schemas.

All entities are JSON Schema-defined in `schemas/core/`. Pydantic models and SQLAlchemy tables are downstream implementations of these schemas.

---

## 1. Core Entities (18)

The platform has 84+ models, but only **18 entities** are canonical — everything else is derived, ephemeral, or implementation-specific.

### 1.1 Market Data

| Entity | Definition | System of Record |
|--------|-----------|-----------------|
| **Instrument** | Tradeable contract metadata (symbol, precision, fees, leverage limits) | Exchange API (read-only) |
| **Candle** | OHLCV bar for one symbol/timeframe/period | Parquet (`data/historical/{exchange}/{symbol}/{tf}.parquet`) |

### 1.2 Intelligence

| Entity | Definition | System of Record |
|--------|-----------|-----------------|
| **FeatureVector** | Computed indicator values for one symbol/timeframe at one point in time | Event log (`feature.vector` topic) |
| **Signal** | Trading direction + confidence + rationale from one strategy | Event log (`strategy.signal` topic) |

### 1.3 Execution

| Entity | Definition | System of Record |
|--------|-----------|-----------------|
| **Order** | Full lifecycle of one exchange order (intent → submitted → filled/cancelled) | PostgreSQL (`orders` table) |
| **Fill** | Single fill execution against an order | PostgreSQL (`fills` table) |
| **Trade** | Logical trade: entry fill(s) → mark-to-market → exit fill(s), with P&L | PostgreSQL (on close), in-memory journal (while open) |

### 1.4 Portfolio State

| Entity | Definition | System of Record |
|--------|-----------|-----------------|
| **Position** | Current holdings for one symbol (qty, entry price, unrealized P&L) | PostgreSQL (`position_snapshots`), Redis hot cache |
| **Balance** | Account balance for one currency (total, free, used) | PostgreSQL (`balance_snapshots`), Redis hot cache |

### 1.5 Governance & Audit

| Entity | Definition | System of Record |
|--------|-----------|-----------------|
| **GovernanceDecision** | Result of the 10-step governance pipeline for one order intent | PostgreSQL (`governance_logs`) |
| **ApprovalRequest** | Human-in-the-loop approval lifecycle (pending → approved/rejected/expired) | PostgreSQL (`approval_requests` — **Gap G2: not yet implemented**) |
| **DecisionAudit** | Full decision chain snapshot: features → signal → risk → intent → fill | PostgreSQL (`decision_audits`) |
| **AuditEntry** | Control plane tool call record (policy + approval + dispatch + result) | PostgreSQL (`audit_entries` — **Gap G1: not yet implemented**) |

### 1.6 Reasoning

| Entity | Definition | System of Record |
|--------|-----------|-----------------|
| **AgentConversation** | Multi-agent discussion with messages, traces, and outcome | PostgreSQL (`agent_conversations`) + JSONL |
| **MemoryEntry** | Persisted analysis (HTF assessment, CMT report, trade plan) with TTL + relevance | JSONL (`data/memory_store.jsonl`) |

### 1.7 Narration

| Entity | Definition | System of Record |
|--------|-----------|-----------------|
| **NarrationItem** | Plain-English desk script at a specific verbosity level | JSONL (`data/narration_history.jsonl`) |

### 1.8 Infrastructure

| Entity | Definition | System of Record |
|--------|-----------|-----------------|
| **PolicySet** | Versioned collection of declarative governance rules | JSON file (via PolicyStore) |

---

## 2. ID Strategy

### 2.1 ID Categories

All ID generation is centralized in `src/agentic_trading/core/ids.py`. No module defines its own factory.

| Category | Format | Generator | When to Use | Examples |
|----------|--------|-----------|-------------|---------|
| **Entity ID** | UUID v4 string | `new_id()` | Every new entity instance | `event_id`, `trade_id`, `action_id`, `conversation_id` |
| **External ID** | Opaque string | Exchange/adapter | IDs assigned by external systems | `order_id` (exchange), `fill_id` (exchange) |
| **Content-Derived ID** | SHA256[:16] hex | `content_hash(*parts)` | Idempotency & deduplication | `dedupe_key`, `script_id` |
| **Integrity Hash** | SHA256[:16] hex | `payload_hash(dict)` | Tamper detection on payloads | `request_hash`, `response_hash`, `snapshot_hash` |

### 2.2 Why UUID v4, Not ULID

ULIDs offer time-sortability, but the platform already has explicit `timestamp` fields on every entity. UUID v4 was chosen because:

- Existing codebase (84+ models) uses UUID v4 — migration cost is high, benefit is marginal.
- PostgreSQL `gen_random_uuid()` is native; ULID requires an extension or application-side generation.
- Time-ordering queries use `created_at` / `timestamp` columns, not ID ordering.
- Content-derived IDs (dedupe_key, script_id) are SHA256-based and cannot be ULIDs regardless.

**Decision**: Keep UUID v4 for entity IDs. Use explicit timestamps for ordering.

### 2.3 Trace ID Propagation Rules

`trace_id` is the **correlation ID** that links the full decision chain. It is:

1. **Born** when a `Signal` is created (auto-generated by `BaseEvent.trace_id = Field(default_factory=new_id)`).
2. **Forwarded** at every step — never regenerated:

```
Signal.trace_id (origin)
  → TargetPosition.trace_id
    → OrderIntent.trace_id
      → OrderAck.trace_id
        → FillEvent.trace_id
          → TradeRecord.trace_id
            → DecisionAudit.trace_id
```

3. **Indexed** in PostgreSQL on `orders.trace_id`, `fills.trace_id`, `decision_audits.trace_id`, `governance_logs.trace_id`.
4. **Queryable**: `SELECT * FROM orders WHERE trace_id = ?` reconstructs the full chain.

**Rule**: Any component that creates a downstream entity from an upstream entity MUST copy `trace_id` from the source.

### 2.4 correlation_id vs causation_id

| Field | Scope | Propagation | Purpose |
|-------|-------|-------------|---------|
| `trace_id` | Full decision chain (Signal → Fill) | Copied from source entity | "Show me everything about this trade" |
| `correlation_id` | Control plane action chain | New per `ProposedAction`, forwarded through CP pipeline | "Show me everything about this tool call" |
| `causation_id` | Direct parent → child | Set to source entity's `event_id` | "What directly caused this event?" |

**causation_id forms a DAG**:
```
CandleEvent.event_id ──causation_id──→ FeatureVector
FeatureVector.event_id ──causation_id──→ Signal
Signal.event_id ──causation_id──→ OrderIntent
OrderIntent.event_id ──causation_id──→ ProposedAction
ProposedAction.action_id ──causation_id──→ AuditEntry
```

**Current status**: `causation_id` is populated in the control plane (3 locations) but not yet in the core pipeline. See Gap G9 in `docs/architecture/component-data-inventory.md`.

---

## 3. Entity Definitions

### 3.1 Instrument

```
Primary Key:    (symbol, exchange)         — natural key
Lifecycle:      Static (read-only from exchange)
Schema:         schemas/core/instrument.json
Implementation: core/models.py::Instrument (Pydantic)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| symbol | string | yes | e.g. "BTC/USDT:USDT" |
| exchange | string | yes | e.g. "bybit" |
| instrument_type | enum | yes | linear, inverse, spot |
| base | string | yes | Base currency |
| quote | string | yes | Quote currency |
| settle | string | no | Settlement currency (perps) |
| price_precision | integer | yes | Decimal places for price |
| qty_precision | integer | yes | Decimal places for quantity |
| tick_size | decimal | yes | Minimum price increment |
| step_size | decimal | yes | Minimum quantity increment |
| min_qty | decimal | yes | Minimum order size |
| max_qty | decimal | yes | Maximum order size |
| min_notional | decimal | yes | Minimum USD notional |
| max_leverage | integer | yes | Maximum allowed leverage |
| maker_fee | decimal | yes | Maker fee rate |
| taker_fee | decimal | yes | Taker fee rate |
| is_active | boolean | yes | Currently tradeable |

---

### 3.2 Candle

```
Primary Key:    content-derived hash of (symbol, exchange, timeframe, timestamp)
Natural Key:    (symbol, exchange, timeframe, timestamp)
Lifecycle:      Immutable once is_closed=true
Schema:         schemas/core/candle.json
Implementation: core/models.py::Candle (Pydantic), CandleEvent (BaseEvent)
System of Record: Parquet files
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| symbol | string | yes | |
| exchange | string | yes | |
| timeframe | enum | yes | 1m, 5m, 15m, 1h, 4h, 1d |
| timestamp | datetime(utc) | yes | Candle open time |
| open | decimal | yes | |
| high | decimal | yes | |
| low | decimal | yes | |
| close | decimal | yes | |
| volume | decimal | yes | Base currency volume |
| quote_volume | decimal | no | Quote currency volume |
| trades | integer | no | Number of trades in period |
| is_closed | boolean | yes | False while candle is forming |

---

### 3.3 FeatureVector

```
Primary Key:    event_id (UUID v4)
Natural Key:    (symbol, timeframe, timestamp)
Lifecycle:      Immutable
Schema:         schemas/core/feature-vector.json
Implementation: core/events.py::FeatureVector (BaseEvent)
System of Record: Event log (feature.vector topic)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| event_id | string(uuid) | yes | Auto-generated |
| timestamp | datetime(utc) | yes | Computation time |
| trace_id | string(uuid) | yes | Correlation chain |
| causation_id | string | no | CandleEvent.event_id that triggered this |
| schema_version | integer | yes | Default 1 |
| symbol | string | yes | |
| timeframe | enum | yes | |
| features | map[string, number?] | yes | 40+ indicator values, null = not computable |

---

### 3.4 Signal

```
Primary Key:    event_id (UUID v4)
Lifecycle:      Immutable
Schema:         schemas/core/signal.json
Implementation: core/events.py::Signal (BaseEvent)
System of Record: Event log (strategy.signal topic)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| event_id | string(uuid) | yes | |
| timestamp | datetime(utc) | yes | |
| trace_id | string(uuid) | yes | **Origin of trace chain** |
| causation_id | string | no | FeatureVector.event_id |
| schema_version | integer | yes | |
| strategy_id | string | yes | Which strategy emitted this |
| symbol | string | yes | |
| direction | enum | yes | LONG, SHORT, FLAT |
| confidence | decimal(0..1) | yes | |
| rationale | string | yes | Human-readable reasoning |
| features_used | list[string] | no | Which indicators contributed |
| timeframe | enum | no | Primary timeframe |
| take_profit | decimal | no | Target exit price |
| stop_loss | decimal | no | Risk exit price |
| trailing_stop | decimal | no | Trailing stop distance |
| risk_constraints | map | no | Strategy-specific constraints |

---

### 3.5 Order

```
Primary Key:    id (UUID v4, database surrogate)
Natural Key:    order_id (exchange-assigned, unique)
Alternate Key:  client_order_id (= dedupe_key, unique)
Lifecycle:      PENDING → SUBMITTED → PARTIALLY_FILLED → FILLED
                PENDING → SUBMITTED → CANCELLED
                PENDING → REJECTED
                PENDING → EXPIRED
Schema:         schemas/core/order.json
Implementation: core/models.py::Order (Pydantic), storage/postgres/models.py::OrderRecord (SQLAlchemy)
System of Record: PostgreSQL (orders table)
Hot Cache:      Redis (trading:open_orders:{symbol})
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| id | string(uuid) | yes | Database PK (surrogate) |
| order_id | string | yes | Exchange-assigned ID (unique) |
| client_order_id | string | yes | Our dedupe_key (unique) |
| symbol | string | yes | |
| exchange | string | yes | |
| side | enum | yes | BUY, SELL |
| order_type | enum | yes | MARKET, LIMIT, STOP_MARKET, STOP_LIMIT, TP_MARKET, TP_LIMIT |
| time_in_force | enum | yes | GTC, IOC, FOK, GTD, POST_ONLY |
| price | decimal | no | Limit price (null for market) |
| stop_price | decimal | no | Trigger price for stop orders |
| qty | decimal | yes | Order quantity |
| filled_qty | decimal | yes | Default 0 |
| remaining_qty | decimal | yes | Default = qty |
| avg_fill_price | decimal | no | Null until first fill |
| status | enum | yes | See lifecycle above |
| reduce_only | boolean | yes | |
| post_only | boolean | yes | |
| leverage | integer | no | |
| strategy_id | string | no | Which strategy placed this |
| trace_id | string | no | Decision chain correlation |
| metadata | map | no | Free-form extension point |
| created_at | datetime(utc) | yes | |
| updated_at | datetime(utc) | yes | |

**State Machine** (valid transitions):
```
PENDING → SUBMITTED → PARTIALLY_FILLED ⇄ PARTIALLY_FILLED → FILLED
PENDING → SUBMITTED → CANCELLED
PENDING → REJECTED
PENDING → SUBMITTED → EXPIRED
```

---

### 3.6 Fill

```
Primary Key:    id (UUID v4, database surrogate)
Natural Key:    fill_id (exchange-assigned or synthetic, unique)
Lifecycle:      Immutable
Schema:         schemas/core/fill.json
Implementation: core/models.py::Fill (Pydantic), core/events.py::FillEvent (BaseEvent), storage/postgres/models.py::FillRecord (SQLAlchemy)
System of Record: PostgreSQL (fills table)
Relationships:  N fills → 1 order (via order_id)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| id | string(uuid) | yes | Database PK |
| fill_id | string | yes | Exchange fill ID (unique) |
| order_id | string | yes | Links to Order.order_id |
| client_order_id | string | no | Our dedupe_key |
| symbol | string | yes | |
| exchange | string | yes | |
| side | enum | yes | BUY, SELL |
| price | decimal | yes | Fill price |
| qty | decimal | yes | Fill quantity |
| fee | decimal | yes | Fee charged |
| fee_currency | string | yes | Currency of fee |
| is_maker | boolean | yes | Maker or taker |
| trace_id | string | no | Decision chain correlation |
| timestamp | datetime(utc) | yes | Fill execution time |

---

### 3.7 Trade

```
Primary Key:    trade_id (UUID v4)
Natural Key:    trace_id (links to originating Signal)
Lifecycle:      PENDING → OPEN → CLOSED
                PENDING → CANCELLED
Schema:         schemas/core/trade.json
Implementation: reconciliation/journal/record.py::TradeRecord (dataclass)
System of Record: PostgreSQL (on close), in-memory journal (while open)
Relationships:  1 trade → N entry fills, N exit fills (via fill_id)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| trade_id | string(uuid) | yes | |
| trace_id | string | yes | Links full decision chain |
| strategy_id | string | yes | |
| symbol | string | yes | |
| exchange | string | yes | |
| direction | enum | yes | long, short |
| phase | enum | yes | PENDING, OPEN, CLOSED, CANCELLED |
| signal_confidence | decimal | no | From originating Signal |
| signal_rationale | string | no | |
| entry_fills | list[FillLeg] | yes | |
| exit_fills | list[FillLeg] | yes | |
| mark_samples | list[MarkSample] | no | MAE/MFE tracking |
| opened_at | datetime(utc) | no | First entry fill time |
| closed_at | datetime(utc) | no | Last exit fill time |
| maturity_level | string | no | Governance maturity at entry |
| health_score_at_entry | decimal | no | |
| governance_sizing_multiplier | decimal | no | |

**Computed** (not stored, derived from fills):
- `entry_qty`, `exit_qty`, `remaining_qty`
- `avg_entry_price`, `avg_exit_price`
- `total_fees`, `gross_pnl`, `net_pnl`, `net_pnl_pct`
- `outcome` (WIN/LOSS/BREAKEVEN), `r_multiple`
- `mae` (Maximum Adverse Excursion), `mfe` (Maximum Favorable Excursion)
- `hold_duration_seconds`

---

### 3.8 Position

```
Primary Key:    id (UUID v4, database surrogate)
Natural Key:    (symbol, exchange, snapshot_at)
Lifecycle:      Continuous snapshots; is_open when qty != 0
Schema:         schemas/core/position.json
Implementation: core/models.py::Position (Pydantic), storage/postgres/models.py::PositionSnapshot (SQLAlchemy)
System of Record: PostgreSQL (position_snapshots), Redis hot cache
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| symbol | string | yes | |
| exchange | string | yes | |
| side | enum | yes | long, short, both |
| qty | decimal | yes | Absolute quantity |
| entry_price | decimal | yes | Average entry price |
| mark_price | decimal | yes | Current mark price |
| liquidation_price | decimal | no | Exchange liquidation price |
| unrealized_pnl | decimal | yes | |
| realized_pnl | decimal | yes | |
| leverage | integer | yes | Default 1 |
| margin_mode | enum | yes | cross, isolated |
| notional | decimal | yes | USD notional value |
| updated_at | datetime(utc) | yes | |

---

### 3.9 Balance

```
Primary Key:    id (UUID v4, database surrogate)
Natural Key:    (currency, exchange, snapshot_at)
Lifecycle:      Continuous snapshots
Schema:         schemas/core/balance.json
Implementation: core/models.py::Balance (Pydantic), storage/postgres/models.py::BalanceSnapshot (SQLAlchemy)
System of Record: PostgreSQL (balance_snapshots), Redis hot cache
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| currency | string | yes | e.g. "USDT" |
| exchange | string | yes | |
| total | decimal | yes | |
| free | decimal | yes | Available for trading |
| used | decimal | yes | Locked in margin/orders |
| updated_at | datetime(utc) | yes | |

---

### 3.10 GovernanceDecision

```
Primary Key:    id (UUID v4, database surrogate)
Natural Key:    trace_id (links to Signal/Order chain)
Lifecycle:      Immutable
Schema:         schemas/core/governance-decision.json
Implementation: core/events.py::GovernanceDecision (BaseEvent), storage/postgres/models.py::GovernanceLog (SQLAlchemy)
System of Record: PostgreSQL (governance_logs)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| trace_id | string | yes | Decision chain correlation |
| strategy_id | string | yes | |
| symbol | string | yes | |
| action | enum | yes | ALLOW, REDUCE_SIZE, BLOCK, DEMOTE, PAUSE, KILL |
| reason | string | no | |
| sizing_multiplier | decimal | no | 0.0–1.0 |
| maturity_level | enum | no | L0_SHADOW through L4_AUTONOMOUS |
| impact_tier | enum | no | LOW, MEDIUM, HIGH, CRITICAL |
| health_score | decimal | no | |
| details | map | no | Full evaluation context |
| decision_at | datetime(utc) | yes | |

---

### 3.11 ApprovalRequest

```
Primary Key:    request_id (UUID v4)
Lifecycle:      PENDING → APPROVED | REJECTED | EXPIRED | ESCALATED | CANCELLED
Schema:         schemas/core/approval-request.json
Implementation: policy/approval_models.py::ApprovalRequest (Pydantic)
System of Record: PostgreSQL (approval_requests — Gap G2: NOT YET IMPLEMENTED)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| request_id | string(uuid) | yes | |
| created_at | datetime(utc) | yes | |
| expires_at | datetime(utc) | yes | |
| strategy_id | string | yes | |
| symbol | string | yes | |
| action_type | string | yes | |
| trigger | enum | yes | 8 trigger types |
| escalation_level | enum | yes | L1_AUTO through L4_ADMIN |
| notional_usd | decimal | no | |
| impact_tier | enum | no | |
| reason | string | no | |
| status | enum | yes | See lifecycle above |
| decided_at | datetime(utc) | no | |
| decided_by | string | no | |
| decision_reason | string | no | |
| order_intent_data | map | no | Serialized OrderIntent |

---

### 3.12 DecisionAudit

```
Primary Key:    id (UUID v4, database surrogate)
Natural Key:    trace_id
Lifecycle:      Immutable (append-only)
Schema:         schemas/core/decision-audit.json
Implementation: storage/postgres/models.py::DecisionAudit (SQLAlchemy)
System of Record: PostgreSQL (decision_audits)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| trace_id | string | yes | Links to full decision chain |
| strategy_id | string | yes | |
| symbol | string | yes | |
| exchange | string | yes | |
| feature_vector | map | no | Serialized FeatureVector |
| signal | map | no | Serialized Signal |
| risk_check | map | no | Serialized RiskCheckResult |
| order_intent | map | no | Serialized OrderIntent |
| order_result | map | no | Serialized OrderAck |
| fill_result | map | no | Serialized FillEvent |
| signal_direction | string | no | Denormalized for queries |
| signal_confidence | decimal | no | Denormalized for queries |
| risk_passed | boolean | no | |
| final_status | string | no | |
| decision_at | datetime(utc) | yes | |

---

### 3.13 AuditEntry

```
Primary Key:    entry_id (UUID v4)
Lifecycle:      Immutable (append-only, fail-closed)
Schema:         schemas/core/audit-entry.json
Implementation: control_plane/action_types.py::AuditEntry (Pydantic)
System of Record: PostgreSQL (audit_entries — Gap G1: NOT YET IMPLEMENTED, currently in-memory)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| entry_id | string(uuid) | yes | |
| timestamp | datetime(utc) | yes | |
| correlation_id | string(uuid) | yes | Control plane chain |
| causation_id | string | no | ProposedAction.action_id |
| event_type | string | yes | E.g. "order_submitted", "policy_blocked" |
| tool_name | enum | yes | Which tool was called |
| action_id | string | yes | Links to ProposedAction |
| payload | map | yes | Full event context |
| payload_hash | string | yes | Integrity check (SHA256[:16]) |

---

### 3.14 AgentConversation

```
Primary Key:    conversation_id (UUID v4)
Lifecycle:      Active → Completed (finalized with outcome)
Schema:         schemas/core/agent-conversation.json
Implementation: reasoning/agent_conversation.py::AgentConversation (Pydantic), storage/postgres/models.py::AgentConversationRecord (SQLAlchemy)
System of Record: PostgreSQL (agent_conversations) + JSONL backup
Relationships:  1 conversation → N messages, N Soteria traces
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| conversation_id | string(uuid) | yes | |
| symbol | string | yes | |
| timeframe | string | no | |
| trigger_event | string | no | What started the conversation |
| strategy_id | string | no | |
| started_at | datetime(utc) | yes | |
| completed_at | datetime(utc) | no | |
| messages | list[AgentMessage] | yes | Ordered chat log |
| traces | list[SoteriaTrace] | no | Agent reasoning traces |
| outcome | enum | yes | TRADE_ENTERED, NO_TRADE, VETOED, ERROR, etc. |
| outcome_details | map | no | |
| context_snapshot | map | no | FactTable at conversation start |

---

### 3.15 MemoryEntry

```
Primary Key:    entry_id (UUID v4)
Lifecycle:      Active → Expired (TTL-based pruning in memory)
Schema:         schemas/core/memory-entry.json
Implementation: context/memory_store.py::MemoryEntry (Pydantic)
System of Record: JSONL (data/memory_store.jsonl)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| entry_id | string(uuid) | yes | |
| entry_type | enum | yes | HTF_ASSESSMENT, SMC_REPORT, CMT_ASSESSMENT, TRADE_PLAN, SIGNAL, RISK_EVENT, REASONING_TRACE |
| timestamp | datetime(utc) | yes | |
| symbol | string | no | |
| timeframe | string | no | |
| strategy_id | string | no | |
| tags | list[string] | yes | Keyword index for retrieval |
| content | map | yes | Free-form analysis content |
| summary | string | no | One-line summary for retrieval ranking |
| relevance_score | decimal | no | 0.0–1.0 |
| ttl_hours | decimal | no | Time-to-live in hours |

---

### 3.16 NarrationItem

```
Primary Key:    script_id (content-derived SHA256[:16])
Lifecycle:      Immutable
Schema:         schemas/core/narration-item.json
Implementation: narration/schema.py::NarrationItem (Pydantic)
System of Record: JSONL (data/narration_history.jsonl)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| script_id | string | yes | `payload_hash()` of content fields |
| timestamp | datetime(utc) | yes | |
| script_text | string | yes | Plain-English narration |
| verbosity | enum | yes | QUIET, NORMAL, DETAILED, PRESENTER |
| decision_ref | string | no | Links to DecisionExplanation |
| sources | list[string] | no | Data sources used |
| metadata | map | no | action, symbol, regime, etc. |
| playback_url | string | no | Audio/video URL |
| published_text | string | no | After text-to-speech |
| published_avatar | string | no | After avatar generation |

---

### 3.17 PolicySet

```
Primary Key:    version (integer, auto-increment)
Lifecycle:      Active → Superseded (only one active at a time)
Schema:         schemas/core/policy-set.json
Implementation: governance/policy_models.py::PolicySet (Pydantic)
System of Record: JSON file (via PolicyStore)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| version | integer | yes | Auto-incremented |
| name | string | yes | |
| description | string | no | |
| rules | list[PolicyRule] | yes | Ordered rule list |
| created_at | datetime(utc) | yes | |
| is_active | boolean | yes | Only one active set |

**PolicyRule** (embedded):
| Attribute | Type | Required |
|-----------|------|----------|
| rule_id | string | yes |
| name | string | yes |
| field | string | yes | Dot-path (e.g. "risk.leverage") |
| operator | enum | yes | LT, LE, GT, GE, EQ, NE, IN, NOT_IN, BETWEEN |
| value | any | yes | Threshold value |
| action | enum | yes | BLOCK, REDUCE_SIZE, WARN |
| sizing_multiplier | decimal | no | |
| scope | map | no | strategy_ids, symbols filters |
| shadow | boolean | no | Log-only, don't enforce |
| enabled | boolean | yes | |

---

### 3.18 Experiment

```
Primary Key:    id (UUID v4, database surrogate)
Natural Key:    experiment_id (string, unique)
Lifecycle:      running → completed | failed
Schema:         schemas/core/experiment.json
Implementation: storage/postgres/models.py::ExperimentLog (SQLAlchemy)
System of Record: PostgreSQL (experiment_logs)
```

| Attribute | Type | Required | Notes |
|-----------|------|----------|-------|
| experiment_id | string | yes | Unique experiment identifier |
| name | string | yes | |
| strategy_id | string | yes | |
| config | map | yes | Full strategy config snapshot |
| symbols | list[string] | yes | |
| timeframes | list[string] | yes | |
| start_date | string | yes | |
| end_date | string | yes | |
| initial_capital | decimal | yes | |
| final_equity | decimal | no | |
| total_return_pct | decimal | no | |
| sharpe_ratio | decimal | no | |
| max_drawdown_pct | decimal | no | |
| win_rate | decimal | no | |
| total_trades | integer | no | |
| status | enum | yes | running, completed, failed |
| git_sha | string | no | Code version |
| random_seed | integer | no | Reproducibility |
| started_at | datetime(utc) | yes | |
| completed_at | datetime(utc) | no | |

---

## 4. Relationships & Cardinalities

```
Instrument (1) ←───── (N) Candle
Instrument (1) ←───── (N) Signal
Instrument (1) ←───── (N) Order
Instrument (1) ←───── (N) Position

Signal (1) ──trace_id── (1) Order ──trace_id── (1) DecisionAudit
Signal (1) ──trace_id── (1) GovernanceDecision
Signal (1) ──trace_id── (1) Trade

Order (1) ←─order_id── (N) Fill

Trade (1) ←─fill_id─── (N) Fill  [entry_fills + exit_fills]

AgentConversation (1) ←── (N) AgentMessage
AgentConversation (1) ←── (N) SoteriaTrace

PolicySet (1) ←── (N) PolicyRule [embedded]

Experiment (1) ──strategy_id── (1) Strategy config
```

**Key join paths**:
- **"Why did it trade?"**: `DecisionAudit JOIN orders ON trace_id JOIN fills ON order_id`
- **"What did governance decide?"**: `governance_logs WHERE trace_id = ?`
- **"What did the agents discuss?"**: `agent_conversations WHERE strategy_id = ? AND symbol = ?`
- **"Show me the trade P&L"**: `TradeRecord WHERE trace_id = ?` → entry_fills + exit_fills

---

## 5. Multi-Tenant Separation

### Current State

The platform is **single-tenant**. No `tenant_id`, `account_id`, or `user_id` fields exist anywhere.

### Future Extension

To support multi-tenancy without breaking compatibility, add an **optional** `tenant_id` field with a default:

| Layer | Change |
|-------|--------|
| BaseEvent | Add `tenant_id: str = "default"` |
| DomainEvent | Add `tenant_id: str = "default"` |
| All PostgreSQL tables | Add `tenant_id VARCHAR(64) DEFAULT 'default'` column + composite indexes |
| Redis keys | Change `trading:{entity}:{key}` → `trading:{tenant_id}:{entity}:{key}` |
| JSON Schemas | Add optional `tenant_id` field with default |
| Event bus | Add `tenant_id` to consumer group naming for isolation |

**Default value `"default"`** ensures all existing data and code works unchanged. New tenants get their own namespace. Queries add `WHERE tenant_id = ?` filter.

**This is a future concern** — no implementation needed now, but the schema must not preclude it. All JSON Schemas in `schemas/core/` include the optional `tenant_id` field.

---

## 6. Replay & Debug Support

### "Why did it do that?"

Given a `trace_id`, reconstruct the full decision chain:

```sql
-- 1. Find the signal
SELECT * FROM decision_audits WHERE trace_id = :tid;

-- 2. Find governance decision
SELECT * FROM governance_logs WHERE trace_id = :tid;

-- 3. Find order lifecycle
SELECT * FROM orders WHERE trace_id = :tid;

-- 4. Find fills
SELECT f.* FROM fills f
JOIN orders o ON f.order_id = o.order_id
WHERE o.trace_id = :tid;

-- 5. Find agent reasoning
SELECT * FROM agent_conversations
WHERE strategy_id = :strategy AND symbol = :symbol
AND started_at BETWEEN :signal_ts - interval '5 minutes' AND :signal_ts + interval '5 minutes';
```

### Event Replay

All events inherit `BaseEvent` with `event_id`, `timestamp`, `trace_id`, `causation_id`, `schema_version`. The event bus (Redis Streams or MemoryBus) preserves ordering. To replay:

1. Read events from Redis Streams or dead-letter queue.
2. Deserialize using `bus/schemas.py::EVENT_TYPE_MAP[event_type_name]`.
3. Check `schema_version` against `SCHEMA_VERSIONS[event_type_name]` for compatibility.
4. Follow `causation_id` chain to reconstruct the event DAG.

---

## 7. Schema Versioning & Compatibility

### Rules

1. **Additive changes** (new optional field with default): bump `schema_version`, no migration needed.
2. **Breaking changes** (field removed, type changed, required field added): bump `schema_version`, add migration logic to consumers.
3. **JSON Schemas** in `schemas/core/` include `$schema`, `$id`, and `version` fields.
4. **PostgreSQL migrations**: Use Alembic (not yet implemented — Gap G11).

### Forward Compatibility for New Agents

New agents can:
- Subscribe to any existing topic without changes.
- Publish new event types by adding them to `bus/schemas.py::TOPIC_SCHEMAS`.
- Create new memory entry types by adding to `MemoryEntryType` enum.
- Participate in conversations by implementing the `AgentRole` interface.

They MUST:
- Use `core/ids.py` for all ID/timestamp generation.
- Propagate `trace_id` from source events.
- Set `source_module` on all events.
- Respect `schema_version` when deserializing.
