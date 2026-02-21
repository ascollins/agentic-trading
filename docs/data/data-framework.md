# Data Framework

## 1. Purpose

This document is the **master index** for the platform's data architecture. It defines principles, provides quick-reference tables, and links to detailed specifications. It does not duplicate content — each section summarizes and points to the authoritative source.

### Principles

1. **Single source of truth** — One authoritative store per entity. Everything else (Redis caches, in-memory snapshots, event log copies) is a derived view that can be rebuilt.
2. **Trace everything** — Every decision chain carries a `trace_id` from Signal through OrderIntent, OrderAck, FillEvent, to DecisionAudit. Any trade can answer "why did it do that?" via a single query.
3. **Schema-first** — JSON Schema (`schemas/`) defines the contract. Pydantic models and SQLAlchemy tables are downstream implementations of these schemas, not the other way around.
4. **Validate, don't trust** — Every LLM output should pass through the 4-layer validation pipeline (schema, evidence, business rules, critique). Raw output is always stored alongside the parsed result. **Current gap (R11)**: the pipeline is implemented and tested but not yet wired into production agent code paths.
5. **Append-only audit** — Spine events and decision audits are immutable. No updates, no deletes. State changes produce new records, not mutations.

---

## 2. Entity Catalog

18 canonical entities across 7 domains. For full attribute definitions, see [`domain-model.md`](domain-model.md).

| Domain | Entity | Description | Schema |
|--------|--------|-------------|--------|
| **Market** | Instrument | Tradeable contract metadata (read-only from exchange) | `schemas/core/instrument.json` |
| | Candle | OHLCV bar for one symbol/timeframe/period | `schemas/core/candle.json` |
| **Intelligence** | FeatureVector | Computed indicator values (40+ indicators) for one symbol/timeframe | `schemas/core/feature-vector.json` |
| | Signal | Trading direction + confidence + rationale from one strategy | `schemas/core/signal.json` |
| **Execution** | Order | Full lifecycle of one exchange order (intent to filled/cancelled) | `schemas/core/order.json` |
| | Fill | Single fill execution against an order | `schemas/core/fill.json` |
| | Trade | Logical trade: entry fills, mark-to-market, exit fills with P&L | `schemas/core/trade.json` |
| **Portfolio** | Position | Current holdings for one symbol (qty, entry price, unrealized P&L) | `schemas/core/position.json` |
| | Balance | Account balance for one currency (total, free, used) | `schemas/core/balance.json` |
| **Governance** | GovernanceDecision | Result of the 10-step governance pipeline | `schemas/core/governance-decision.json` |
| | ApprovalRequest | Human-in-the-loop approval lifecycle | `schemas/core/approval-request.json` |
| | DecisionAudit | Full decision chain snapshot: features to fill | `schemas/core/decision-audit.json` |
| | AuditEntry | Control plane tool call record | `schemas/core/audit-entry.json` |
| **Reasoning** | AgentConversation | Multi-agent discussion with messages, traces, and outcome | `schemas/core/agent-conversation.json` |
| | MemoryEntry | Persisted analysis with TTL + relevance scoring | `schemas/core/memory-entry.json` |
| **Narration** | NarrationItem | Plain-English desk script at a specific verbosity level | `schemas/core/narration-item.json` |
| **Infrastructure** | PolicySet | Versioned collection of declarative governance rules | `schemas/core/policy-set.json` |
| | Experiment | Backtest or optimization experiment with full results | `schemas/core/experiment.json` |

---

## 3. ID & Timestamp Standards

All ID and timestamp generation is centralized in `src/agentic_trading/core/ids.py`. No module defines its own factory. For trace propagation rules and correlation ID semantics, see [`domain-model.md` Section 2](domain-model.md#2-id-strategy).

| Category | Format | Generator | Example Fields |
|----------|--------|-----------|----------------|
| Entity ID | UUID v4 string | `new_id()` | `event_id`, `trade_id`, `trace_id`, `request_id` |
| External ID | Opaque string | Exchange adapter | `order_id`, `fill_id` |
| Content-derived | SHA256[:16] hex | `content_hash(*parts)` | `dedupe_key`, `script_id` |
| Integrity hash | SHA256[:16] hex | `payload_hash(dict)` | `request_hash`, `envelope_hash` |

**Timestamps**: Always `datetime` with `tzinfo=timezone.utc` (never naive). Factory: `utc_now()`. Serialization: ISO 8601 (Pydantic handles automatically). PostgreSQL: `DateTime(timezone=True)` with `server_default=func.now()`.

> **Hash length note**: `content_hash()` produces 64-bit hashes (16 hex chars). Birthday paradox gives 1% collision probability at ~190 million items. For high-volume entities (`dedupe_key`), consider increasing to `length=32` (128 bits) which pushes 1% collision to ~2.6 x 10^18 items.

---

## 4. Systems of Record

One authoritative store per entity. For relationships and join paths, see [`domain-model.md` Section 4](domain-model.md#4-relationships--cardinalities).

| Entity | Authoritative Store | Hot Cache | Notes |
|--------|-------------------|-----------|-------|
| Order | PostgreSQL `orders` | Redis `trading:open_orders:{symbol}` | |
| Fill | PostgreSQL `fills` | — | Immutable after write |
| Trade | PostgreSQL (on close) | In-memory journal (while open) | |
| Position | PostgreSQL `position_snapshots` | Redis `trading:position:{symbol}` | |
| Balance | PostgreSQL `balance_snapshots` | Redis `trading:balance:{currency}` | |
| DecisionAudit | PostgreSQL `decision_audits` | — | Append-only |
| GovernanceDecision | PostgreSQL `governance_logs` | — | Append-only |
| ApprovalRequest | PostgreSQL (Gap G2: not yet) | In-memory ApprovalManager | |
| AuditEntry | PostgreSQL (Gap G1: not yet) | In-memory | |
| Candle | Parquet `data/historical/` | In-memory ring buffer | |
| Instrument | Exchange API | In-memory cache | Read-only |
| Signal, FeatureVector | Event log (bus topics) | — | Ephemeral |
| AgentConversation | PostgreSQL + JSONL | — | |
| MemoryEntry | JSONL `data/memory_store.jsonl` | In-memory MemoryStore | |
| NarrationItem | JSONL `data/narration_history.jsonl` | In-memory ring buffer | |
| PolicySet | JSON file via PolicyStore | In-memory PolicyEngine | |
| Experiment | PostgreSQL `experiment_logs` | — | |
| LLM Interactions | JSONL `data/llm_interactions.jsonl` | MemoryInteractionStore | |
| Kill Switch | Redis `trading:kill_switch` | — | Manual reset only |
| Dedup Tokens | Redis `trading:dedupe:*` (TTL) | — | Auto-expiry |

---

## 5. Event Spine

The platform has a **dual event architecture**. For the full specification, see [`event-log.md`](event-log.md).

**Layer 1 — Event Bus** (`src/agentic_trading/event_bus/`): 28 topics carrying `BaseEvent` subclasses via Redis Streams (paper/live) or MemoryEventBus (backtest). Real-time, pub/sub, drives the trading pipeline.

**Layer 2 — Telemetry Spine** (`src/agentic_trading/telemetry/`): Parallel append-only log. `EventMapper` classifies BaseEvents into 9 taxonomy types (`SpineEvent`), `EventWriter` batches and flushes to `ISpineStorage` (PostgreSQL or memory).

| Spine Field | Purpose |
|-------------|---------|
| `event_id`, `trace_id`, `span_id` | Identity and correlation |
| `tenant_id`, `actor`, `component` | Context |
| `timestamp`, `latency_ms` | Timing |
| `event_type` | Taxonomy classification |
| `input_hash`, `output_hash` | Integrity |
| `schema_version`, `causation_id` | Versioning and DAG |
| `payload`, `error` | Content |

9 taxonomy types: `USER_ACTION`, `AGENT_STEP_STARTED`, `AGENT_STEP_COMPLETED`, `TOOL_CALL`, `RETRIEVAL`, `DECISION`, `VALIDATION`, `EXCEPTION`, `COST_METRIC`.

> **Trace ID hazard**: `BaseEvent.trace_id` defaults to a new UUID via `Field(default_factory=new_id)`. Any event constructed without explicitly passing `trace_id=upstream.trace_id` silently gets a broken chain. Every event in a decision chain MUST copy `trace_id` from its upstream cause. See R4 in Foundation Risks.

> **source_module gap**: `BaseEvent.source_module` defaults to empty string. Events without a `source_module` cannot be attributed to a component. Future: validate non-empty at construction or add a linting rule.

---

## 6. Metadata Standard

Every platform entity can embed a composable `MetadataStandard` model for provenance, trust, and lifecycle tracking. For the full specification, see [`metadata-standard.md`](metadata-standard.md).

Key fields: `source_id`, `source_type` (8 types), `author`, `trust_level` (5 levels with numeric weights), `quality_score` (0.0-1.0), `status` (active/deprecated/do_not_use/archived/expired), `lineage` (parent_id, root_id, derivation chain).

Trust-weighted relevance formula: `effective_relevance = relevance_score * trust_weight * quality_score`

Schema: `schemas/meta/metadata.json`. Implementation: `src/agentic_trading/llm/envelope.py::MetadataStandard`.

---

## 7. Schema Versioning

All JSON Schemas use Draft 2020-12 with `additionalProperties: false` for strict validation. For forward compatibility rules for new agents, see [`domain-model.md` Section 7](domain-model.md#7-schema-versioning--compatibility).

**Rules**:
1. **Additive** (new optional field with default): bump `schema_version` in `bus/schemas.py::SCHEMA_VERSIONS`. No migration needed.
2. **Breaking** (field removed, type changed, required field added): bump `schema_version`, add migration logic to consumers.
3. JSON Schemas include `$schema`, `$id`, and `version` fields.
4. PostgreSQL migrations: Alembic (Gap G11: not yet implemented).

**Schema registry**: `bus/schemas.py` maps 28 topic names to event classes (`TOPIC_SCHEMAS`) and tracks per-type versions (`SCHEMA_VERSIONS`). Functions: `get_event_class(name)`, `get_topic_for_event(event)`.

> **Current gap**: `SCHEMA_VERSIONS` is defined (all types at version 1) but **never checked during deserialization**. `RedisStreamsBus._deserialize()` calls `model_validate_json()` with no version check. An event with `schema_version: 2` will either silently parse (if additive) or crash the consumer (if breaking) — with no logging, no dead-letter routing, and no alert. See R12 in Foundation Risks.

---

## 8. Data Retention Policy

| Store | Entity | Retention | Rationale |
|-------|--------|-----------|-----------|
| PostgreSQL | `orders`, `fills` | Indefinite | Regulatory audit trail |
| PostgreSQL | `decision_audits` | Indefinite | "Why did it do that?" debugging |
| PostgreSQL | `governance_logs` | Indefinite | Compliance record |
| PostgreSQL | `position_snapshots`, `balance_snapshots` | 90 days live, then archive | Equity curve analysis; archive to Parquet |
| PostgreSQL | `spine_events` | 30 days live, then archive | Debug and replay; archive to Parquet monthly |
| PostgreSQL | `experiment_logs` | Indefinite | Backtest reproducibility |
| Parquet | Candle history | Indefinite | Backtest corpus; append-only |
| Parquet | Archived snapshots/spine | Indefinite | Cold storage after PostgreSQL rotation |
| JSONL | `memory_store.jsonl` | 30 days effective | TTL-managed via decay formula in-code |
| JSONL | `narration_history.jsonl` | 90 days | Review and replay |
| JSONL | `llm_interactions.jsonl` | 30 days | Cost analysis; archive monthly |
| Redis | Positions, balances | Session lifetime | Ephemeral hot cache; **not currently rebuilt on restart (Gap R6)** |
| Redis | Dedup tokens | 5 min TTL | Automatic expiry via Redis TTL |
| Redis | Kill switch | No TTL | Manual reset only |
| Redis | Event streams | 24 hours | Consumer groups auto-trim |

**Archival strategy**: Monthly cron job exports aged PostgreSQL rows to Parquet files in `data/archive/{table}/{year}/{month}.parquet`, then deletes the exported rows. JSONL files are rotated with the same cadence.

> **Implementation status**: All retention periods above are **policy targets, not yet enforced**. No automated archival, rotation, or cleanup exists. See R5, R7 in Foundation Risks.

---

## 9. Access Control

Current enforcement is **trust-based**: each agent only calls its own interfaces. The `actor` field on `SpineEvent` enables post-hoc audit verification. Future: enforce at the event bus layer.

| Actor | Can Read | Can Write | Cannot Touch |
|-------|----------|-----------|-------------|
| Strategy agents | FactTable, MemoryStore, candles, feature vectors | Signal, FeatureVector, MemoryEntry | Orders, fills (only via ExecutionEngine) |
| Execution agent | Orders, fills, positions, balances | OrderAck, FillEvent | Policy rules, governance config |
| Risk agent | All positions, balances, orders, signals | RiskCheckResult, RiskAlert, CircuitBreakerEvent | Order submission (only flags, never submits) |
| Governance gate | All entities (read-only assessment) | GovernanceDecision, ApprovalRequest, TokenEvent | Direct exchange calls |
| Control plane | All entities | AuditEntry, tool call dispatch | Direct event bus publish (routes through ToolGateway) |
| Human operator | All entities (dashboard read) | Approval decisions, kill switch, policy edits | Direct database writes, direct adapter calls |
| Telemetry spine | All BaseEvents (subscribe) | SpineEvent (append-only) | Any trading entity mutation |

**Classification levels** (from MetadataStandard): `public` (market data, candles), `internal` (signals, feature vectors, narration), `confidential` (orders, fills, positions, balances, LLM interactions), `restricted` (API keys, exchange credentials — never stored in data layer).

> **Tenant isolation**: Currently single-tenant. `tenant_id` field exists in JSON schemas and `MetadataStandard` as a future extension point (default: `"default"`) but is **not enforced** in Redis key namespacing, PostgreSQL queries, or event bus consumer groups. Multi-tenant deployment requires adding `tenant_id` to all storage key builders, query WHERE clauses, and consumer group names.

---

## 10. Observability

Key metrics organized by concern. Grafana query templates are in [`event-log.md`](event-log.md). Prometheus scrape targets are defined in `docker-compose.yml`.

**Data freshness**:
- Candle age per symbol/timeframe — alert if > 2x expected interval
- Position snapshot lag vs exchange — alert if > 30s
- Balance sync lag vs exchange — alert if > 60s
- FeatureVector computation latency — p95 target < 100ms

**Validation health**:
- `quality_score` distribution per agent/output type (histogram)
- Uncited claim ratio per output type (should be < 30%)
- Schema validation failure rate (should be ~0% in production)
- Critique trigger rate and cost per day
- Remediation action distribution (retry vs escalate vs accept)

**Event throughput**:
- Events/sec per bus topic (baseline + anomaly detection)
- Dead-letter queue depth (should be 0; alert on any)
- Handler retry rate per topic (should be < 1%)
- SpineEvent write latency p99

**Storage health**:
- PostgreSQL table row counts and sizes (alert on unexpected growth)
- Redis memory usage and key count per namespace
- Parquet partition count and total corpus size
- JSONL file sizes (memory_store, narration, llm_interactions)

**LLM costs**:
- `cost_usd` per agent per day (from CostMetricEvent)
- Token usage: input vs output per model
- Critique budget consumption vs cap ($0.10/call)
- Envelope count per workflow type (analysis, planning, execution)

---

## 11. Foundation Risks

17 data framework risks, ordered by severity. P0 = blocks production. P1 = fix soon. P2 = track.

| # | Risk | Severity | Mitigation | Status |
|---|------|----------|------------|--------|
| R1 | No Alembic migrations — schema drift between environments | P0 | Add Alembic migration framework in `alembic/` | Gap G11 |
| R2 | AuditEntry not persisted to PostgreSQL (in-memory only) | P0 | Implement `audit_entries` table + repository | Gap G1 |
| R4 | `causation_id` not set on OrderAck or FillEvent — event DAG is broken, trace replay impossible | P0 | Set `causation_id=intent.event_id` on OrderAck, `causation_id=ack.event_id` on FillEvent in `execution/engine.py` | Gap G9 |
| R6 | Redis state not rebuilt from PostgreSQL on startup — 30s stale window if Redis restarts | P0 | Add `rebuild_state_from_postgres()` in `main.py` initialization; enable Redis AOF | New |
| R9 | EventWriter buffer cleared before write succeeds — events permanently lost on PostgreSQL failure | P0 | **Fixed**: buffer now retained on failure with `max_buffer_size` overflow cap | **Fixed** |
| R11 | Validation pipeline (`validation/pipeline.py`) and envelope builder have zero production call sites — LLM outputs bypass all validation | P0 | Wire `ValidationPipeline.run()` into agent response path before production | New |
| R14 | `LLMBudget.max_cost_usd` is never enforced — no cost ceiling on LLM calls; `LLMBudgetExhaustedError` exists but is never raised | P1 | Implement pre-call budget check; add global daily limit; raise on exceed | New |
| R3 | ApprovalRequest not persisted to PostgreSQL | P1 | Implement `approval_requests` table + repository | Gap G2 |
| R5 | No retention/archival automation — all retention periods in Section 8 are policy targets only | P1 | Implement monthly archival job (PostgreSQL to Parquet, JSONL rotation) | New |
| R12 | `SCHEMA_VERSIONS` defined but never checked on deserialization — version mismatches silently corrupt or crash | P1 | Add version check in `bus/redis_streams.py::_deserialize()` and log/dead-letter on mismatch | New |
| R13 | Evidence validator does not verify cited source IDs exist in `envelope.retrieved_evidence` — LLM can fabricate citations | P1 | Add source existence check in `evidence_validator.py::_classify_claim()` | New |
| R16 | Dead-letter queue is in-memory only — lost on restart with no persistence or alerting | P1 | Persist dead letters to PostgreSQL or dedicated Redis list; add DLQ depth metric | New |
| R7 | JSONL files grow unbounded (memory_store, narration, llm_interactions, pipeline_log, conversations) | P2 | Add file rotation: rename to `.jsonl.{date}`, start fresh, compress old | New |
| R8 | No continuous data integrity checks between PostgreSQL and Redis | P2 | ReconciliationLoop exists but runs only on startup; add periodic check | Partial |
| R10 | No backup strategy for Parquet candle corpus | P2 | Document rsync/S3 backup procedure; add to ops runbook | New |
| R15 | `LLMResult.output_hash` is computed but never verified — output mutation undetectable | P2 | Add `verify_integrity()` method; call at pipeline boundaries | New |
| R17 | `expected_output_schema` defaults to empty dict — agents can bypass schema validation by omission | P2 | Make schema mandatory in `EnvelopeBuilder.build()` or default to catch-all schema | New |

---

## 12. Cross-Reference Index

| Topic | Detailed Doc | Source Code | Schemas |
|-------|-------------|------------|---------|
| Entity definitions | [`domain-model.md`](domain-model.md) | `core/events.py`, `core/models.py` | `schemas/core/` (18 files) |
| Event spine | [`event-log.md`](event-log.md) | `telemetry/event_writer.py`, `telemetry/models.py`, `telemetry/mapper.py` | `schemas/events/` (9 files) |
| Metadata standard | [`metadata-standard.md`](metadata-standard.md) | `llm/envelope.py::MetadataStandard` | `schemas/meta/metadata.json` |
| LLM contract | [`metadata-standard.md`](metadata-standard.md) | `llm/envelope.py`, `llm/envelope_builder.py`, `llm/store.py` | `schemas/llm/envelope.json` |
| Validation pipeline | [`docs/quality/validation.md`](../quality/validation.md) | `validation/pipeline.py`, `validation/schema_validator.py`, `validation/evidence_validator.py`, `validation/business_rules.py`, `validation/critique_validator.py`, `validation/remediation.py` | `schemas/validation/` (3 files) |
| ID & timestamp factories | [`domain-model.md` Section 2](domain-model.md#2-id-strategy) | `core/ids.py` | — |
| Schema registry | [`domain-model.md` Section 7](domain-model.md#7-schema-versioning--compatibility) | `bus/schemas.py` | — |
| Knowledge taxonomy | [`docs/knowledge/taxonomy.md`](../knowledge/taxonomy.md) | `context/memory_store.py` | — |
| Retrieval policy | — | — | `configs/retrieval_policy.yaml` |
| Error hierarchy | — | `core/errors.py` (30+ exception types) | — |
