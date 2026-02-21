# Component & Data Inventory — Agentic Trading Platform

## Context

This inventory maps every platform component to its purpose, trust boundary, data interfaces, and SLO needs — then catalogs every data object with sensitivity classification, retention policy, and authoritative store. The goal: eliminate hand-waving, name every data flow precisely, and surface gaps that block production readiness.

**Note**: The user-referenced "Data Flow Diagrams – MagicCarpet AI Platform" document was not found in the repository. This inventory is derived entirely from the codebase.

---

## 1. Component Table

| # | Component | Purpose | Owner | Trust Boundary | Ingress (topic → event) | Egress (topic → event) | SLO |
|---|-----------|---------|-------|---------------|------------------------|------------------------|-----|
| 1 | **FeedManager** | Streams real-time 1m OHLCV from exchange WebSockets | `intelligence/feed_manager.py` | EXTERNAL | Exchange WS → raw candles | `market.candle` → `CandleEvent` | <1s latency, HA critical |
| 2 | **CandleBuilder** | Aggregates 1m candles into 5m/15m/1h/4h/1d with UTC-aligned boundaries | `intelligence/candle_builder.py` | INTERNAL | 1m Candle objects (from FeedManager) | `market.candle` → `CandleEvent` (higher TFs) | <100ms aggregation |
| 3 | **FeatureEngine** | Computes 40+ technical indicators (EMA, RSI, ATR, MACD, BB, etc.) | `intelligence/features/engine.py` | INTERNAL | `market.candle` → `CandleEvent` | `feature.vector` → `FeatureVector` | <50ms compute |
| 4 | **Strategies** (×13) | Generate LONG/SHORT/FLAT signals from features + candles | `signal/strategies/*.py` | INTERNAL | `CandleEvent` + `FeatureVector` (via `on_candle()`) | Returns `Signal` objects | <100ms per signal |
| 5 | **PortfolioManager** | Converts signals into sized order intents with position sizing | `signal/portfolio/manager.py` | INTERNAL | `strategy.signal` → `Signal` | `execution.intent` → `OrderIntent`, `strategy.target` → `TargetPosition` | <100ms, HA critical |
| 6 | **ExecutionEngine** | Full order lifecycle: dedup, risk checks, submission, fill handling | `execution/engine.py` | MIXED | `execution.intent` → `OrderIntent`, `system.kill_switch` → `KillSwitchEvent` | `execution.ack` → `OrderAck`, `execution.fill` → `FillEvent`, `state.position` → `PositionUpdate` | <200ms submit, **CRITICAL** |
| 7 | **PaperAdapter** | Simulates order fills with slippage + fees | `execution/adapters/paper.py` | INTERNAL | `OrderIntent` (direct call) | `OrderAck` (status=FILLED) | Instant |
| 8 | **CCXTAdapter** | Live exchange integration (Bybit) via REST + WebSocket | `execution/adapters/ccxt_adapter.py` | **EXTERNAL** | `OrderIntent` (direct call), WS user trades stream | `OrderAck`, `FillEvent`, position/balance snapshots | <500ms REST, **CRITICAL** |
| 9 | **RiskManager** | Pre/post-trade checks, circuit breakers, drawdown, exposure, kill switch | `execution/risk/manager.py` | INTERNAL | `OrderIntent`, `FillEvent`, `PortfolioState` | `risk.check` → `RiskCheckResult`, `risk.alert` → `RiskAlert`, `risk.circuit_breaker` → `CircuitBreakerEvent` | <50ms pre-trade, **CRITICAL** |
| 10 | **GovernanceGate** | 10-step pipeline: maturity, impact, approval, health, policy, token, sizing | `policy/governance_gate.py` | INTERNAL | OrderIntent context, StrategyMetrics | Governance decision (action, sizing_multiplier, maturity_level) | <50ms, **CRITICAL** |
| 11 | **PolicyEngine** | Declarative rule evaluation (9 operators, scoping, shadow mode) | `policy/engine.py` | INTERNAL | Action context dict, PolicySet | `PolicyEvalResult` (allowed, failed_rules, sizing_multiplier) | <20ms |
| 12 | **ApprovalManager** | Human-in-the-loop for high-impact trades (4 escalation levels, 8 triggers) | `policy/approval_manager.py` | INTERNAL | Approval trigger conditions | `governance.approval` → `ApprovalRequested`, `ApprovalResolved` | Minutes (human), **CRITICAL** |
| 13 | **ToolGateway** | **ONLY** path for exchange side effects in CP mode; policy + approval + audit + kill-switch + rate-limit | `control_plane/tool_gateway.py` | **EXTERNAL** | `ProposedAction` | `control_plane.tool_call` → `ToolCallRecorded`, `ToolCallResult` | <300ms total, **CRITICAL** |
| 14 | **TradeJournal** | Records fills, computes per-trade P&L with MAE/MFE | `reconciliation/journal/journal.py` | INTERNAL | `execution.fill` → `FillEvent` | `TradeRecord` → PostgreSQL | Async |
| 15 | **NarrationService** | Converts DecisionExplanation into plain-English scripts (4 verbosity modes) | `narration/service.py` | INTERNAL | `DecisionExplanation` | `NarrationItem` → NarrationStore (JSONL) | Async, non-critical |
| 16 | **ContextManager** | Unified facade: FactTable (real-time) + MemoryStore (historical analysis) | `context/manager.py` | INTERNAL | Syncs from TradingContext | `AgentContext` (snapshot + memories) | <10ms reads |
| 17 | **AgentOrchestrator** | Creates, registers, lifecycle-manages all platform agents | `agents/orchestrator.py` | INTERNAL | Settings config | Registers agents in AgentRegistry | Startup only |
| 18 | **EventBus** | Pub/sub with 28 topics; Redis Streams (paper/live) or MemoryBus (backtest) | `bus/redis_streams.py`, `bus/memory_bus.py` | INTERNAL | `publish(topic, event)` from all components | Delivers to subscribed handlers | <5ms Redis, **CRITICAL** |
| 19 | **BacktestEngine** | Deterministic event replay with SimClock, fee model, slippage | `backtester/engine.py` | INTERNAL | Parquet candles, strategy list | `BacktestResult` (equity curve, per-trade details) | Offline batch |
| 20 | **OptimizerScheduler** | Periodic parameter optimization (grid search + walk-forward) | `optimizer/scheduler.py` | INTERNAL | Strategy configs, historical candles | `optimizer.result` → `OptimizationCompleted`, `ParameterChangeApplied` | 24h cycle |
| 21 | **ReconciliationLoop** | Periodic sync of local state with exchange ground truth | `reconciliation/loop.py` | MIXED | Polls adapter every 300s | `system.reconciliation` → `ReconciliationResult` | Background |
| 22 | **GovernanceCanary** | Infrastructure watchdog: health checks on bus, adapters, DB, clock | `governance/canary.py` | INTERNAL | Polls components every 60s | `CanaryAlert`, can trigger kill switch | Background, moderate HA |
| 23 | **CMT Engine** | 9-layer classical technical analysis with LLM confluence scoring | `intelligence/analysis/cmt_engine.py` | INTERNAL | Candle buffers, FeatureVector | `intelligence.cmt` → `CMTAssessment` | 1-3s (LLM), advisory |
| 24 | **ReasoningLayer** | Multi-agent consensus with Soteria tracing | `reasoning/` | INTERNAL | ContextManager, signals, CMT assessments | `AgentConversation`, `SoteriaTrace`, `PipelineResult` (JSONL) | 3-10s (LLM), advisory |

---

## 2. Data Objects Table

### 2.1 PostgreSQL (Authoritative — Persistent)

| Data Object | Table | Sensitivity | Retention | Producers | Consumers | Schema |
|-------------|-------|-------------|-----------|-----------|-----------|--------|
| `OrderRecord` | `orders` | CONFIDENTIAL | Permanent (audit) | ExecutionEngine | Journal, DecisionAudit, UI | `storage/postgres/models.py` |
| `FillRecord` | `fills` | CONFIDENTIAL | Permanent (audit) | ExecutionEngine | Journal, P&L calc | `storage/postgres/models.py` |
| `PositionSnapshot` | `position_snapshots` | CONFIDENTIAL | Permanent (regulatory) | ReconciliationLoop | Risk analytics, compliance | `storage/postgres/models.py` |
| `BalanceSnapshot` | `balance_snapshots` | CONFIDENTIAL | Permanent (regulatory) | ReconciliationLoop | Equity tracking, risk | `storage/postgres/models.py` |
| `DecisionAudit` | `decision_audits` | CONFIDENTIAL | 7 years (compliance) | GovernanceGate | Compliance, diagnostics | `storage/postgres/models.py` |
| `GovernanceLog` | `governance_logs` | CONFIDENTIAL | Permanent (audit) | GovernanceGate | Compliance, incident review | `storage/postgres/models.py` |
| `ExperimentLog` | `experiment_logs` | INTERNAL | Permanent (research) | BacktestEngine, Optimizer | Strategy research | `storage/postgres/models.py` |
| `AgentConversationRecord` | `agent_conversations` | CONFIDENTIAL | Permanent (Soteria audit) | ReasoningLayer | Decision audit | `storage/postgres/models.py` |
| `AgentMessageRecord` | `agent_messages` | CONFIDENTIAL | Permanent | ReasoningLayer | Reasoning audit | `storage/postgres/models.py` |

### 2.2 Redis (Hot Path — Ephemeral)

| Data Object | Key Pattern | Sensitivity | Retention | Producers | Consumers |
|-------------|-------------|-------------|-----------|-----------|-----------|
| Position cache | `trading:position:{symbol}` | CONFIDENTIAL | Session (rebuilt on startup) | PositionManager | RiskManager, PortfolioManager |
| Balance cache | `trading:balance:{currency}` | CONFIDENTIAL | Session | BalanceTracker | RiskManager, PortfolioManager |
| Open orders | `trading:open_orders:{symbol}` | CONFIDENTIAL | Session | ExecutionEngine | RiskManager, ReconciliationLoop |
| Kill switch flag | `trading:kill_switch` | INTERNAL | Session | KillSwitch, Canary, RiskManager | ExecutionEngine, all agents |
| Dedup tokens | `trading:dedupe:{client_order_id}` | INTERNAL | TTL (1h) | ExecutionEngine | ExecutionEngine |
| Event streams | `trading:stream:{topic}` (x28) | CONFIDENTIAL | Stream retention | All publishers | All subscribers |

### 2.3 Parquet (Historical — Persistent)

| Data Object | Path Pattern | Sensitivity | Retention | Producers | Consumers |
|-------------|-------------|-------------|-----------|-----------|-----------|
| Candle history | `data/historical/{exchange}/{symbol}/{tf}.parquet` | INTERNAL | Permanent (archive) | FeedManager | BacktestEngine, Optimizer, FeatureEngine |

### 2.4 JSONL (Append-Only — Persistent)

| Data Object | Path | Sensitivity | Retention | Producers | Consumers |
|-------------|------|-------------|-----------|-----------|-----------|
| `MemoryEntry` | `data/memory_store.jsonl` | CONFIDENTIAL | Persistent (TTL-pruned in memory) | ContextManager, CMT agents | All reasoning agents |
| `NarrationItem` | `data/narration_history.jsonl` | PUBLIC | Persistent | NarrationService | UI, avatar service |
| `AgentConversation` | `data/historical/conversations.jsonl` | CONFIDENTIAL | Persistent | ReasoningLayer | Governance audit |
| `PipelineLog` | `data/pipeline_logs/pipelines.jsonl` | CONFIDENTIAL | Persistent | Pipeline orchestrator | Debugging |

### 2.5 Event Types by Topic (50+ types across 28 topics)

| Topic | Event Types | Sensitivity | Key Payload Fields |
|-------|-------------|-------------|--------------------|
| `market.tick` | `TickEvent` | INTERNAL | bid, ask, last_price |
| `market.trade` | `TradeEvent` | INTERNAL | price, qty, side |
| `market.orderbook` | `OrderBookSnapshot` | INTERNAL | bids[], asks[] |
| `market.candle` | `CandleEvent` | INTERNAL | open, high, low, close, volume |
| `feature.vector` | `FeatureVector` | INTERNAL | features: dict[str, float] (40+ indicators) |
| `feature.news` | `NewsEvent` | INTERNAL | headline, sentiment_score |
| `feature.whale` | `WhaleEvent` | INTERNAL | amount_usd, wallet |
| `strategy.signal` | `Signal` | CONFIDENTIAL | direction, confidence, rationale, tp/sl |
| `strategy.regime` | `RegimeState` | INTERNAL | regime, volatility, liquidity |
| `strategy.target` | `TargetPosition` | CONFIDENTIAL | target_qty, side, urgency |
| `execution.intent` | `OrderIntent` | CONFIDENTIAL | dedupe_key, strategy_id, qty, price, side |
| `execution.ack` | `OrderAck` | CONFIDENTIAL | order_id, status, fill_price |
| `execution.update` | `OrderUpdate` | CONFIDENTIAL | filled_qty, avg_fill_price |
| `execution.fill` | `FillEvent` | CONFIDENTIAL | price, qty, fee, fill_id |
| `state.position` | `PositionUpdate` | CONFIDENTIAL | qty, entry_price, unrealized_pnl |
| `state.balance` | `BalanceUpdate` | CONFIDENTIAL | total, free, used |
| `state.funding` | `FundingPaymentEvent` | CONFIDENTIAL | funding_rate, payment |
| `state.open_interest` | `OpenInterestEvent` | INTERNAL | open_interest |
| `risk.check` | `RiskCheckResult` | CONFIDENTIAL | passed, reason, details |
| `risk.alert` | `RiskAlert` | CONFIDENTIAL | severity, alert_type, message |
| `risk.circuit_breaker` | `CircuitBreakerEvent` | CONFIDENTIAL | tripped, threshold, current_value |
| `system.health` | `SystemHealth` | INTERNAL | component, healthy, latency_ms |
| `system.kill_switch` | `KillSwitchEvent` | INTERNAL | activated, reason |
| `system.reconciliation` | `ReconciliationResult` | INTERNAL | discrepancies, repairs_applied |
| `system.degraded_mode` | `DegradedModeEnabled` | INTERNAL | reason, disabled_components |
| `system.incident` | `IncidentCreated` | CONFIDENTIAL | severity, description, affected |
| `governance.approval` | `ApprovalRequested`, `ApprovalResolved` | CONFIDENTIAL | notional_usd, impact_tier, escalation_level |
| `control_plane.tool_call` | `ToolCallRecorded` | CONFIDENTIAL | tool_name, success, request_hash, response_hash |
| `intelligence.cmt` | `CMTAssessment` | CONFIDENTIAL | layers, confluence_score, trade_plan |
| `optimizer.result` | `OptimizationCompleted`, `StrategyOptimizationResult`, `ParameterChangeApplied`, `EfficacyAnalysisCompleted` | CONFIDENTIAL | parameters, metrics, improvement_pct |

### 2.6 Configuration Secrets (RESTRICTED)

| Secret | Env Variable | Used By |
|--------|-------------|---------|
| Exchange API key | `TRADING_BYBIT_API_KEY` | CCXTAdapter |
| Exchange secret | `TRADING_BYBIT_SECRET` | CCXTAdapter |
| Claude API key | `ANTHROPIC_API_KEY` | CMT Engine, ReasoningLayer |
| Tavus API key | `TAVUS_API_KEY` | NarrationService (avatar) |
| PostgreSQL URL | `DATABASE_URL` | Storage layer |
| Redis URL | `REDIS_URL` | EventBus, KillSwitch, caches |

### 2.7 Control Plane Audit Objects (CONFIDENTIAL)

| Data Object | Persisted? | Store | Schema |
|-------------|-----------|-------|--------|
| `ProposedAction` | YES (via AuditEntry) | PostgreSQL | `control_plane/action_types.py` |
| `CPPolicyDecision` | YES (via AuditEntry) | PostgreSQL | `control_plane/action_types.py` |
| `ApprovalDecision` | YES (via AuditEntry) | PostgreSQL | `control_plane/action_types.py` |
| `ToolCallResult` | YES (via AuditEntry) | PostgreSQL | `control_plane/action_types.py` |
| `AuditEntry` | **NO — see Gap G1** | In-memory list only | `control_plane/action_types.py` |
| `ApprovalRequest` | **NO — see Gap G2** | In-memory dict only | `policy/approval_models.py` |
| `ExecutionToken` | NO (ephemeral by design) | In-memory with TTL | `policy/tokens.py` |

---

## 3. Missing Details That Block Implementation (Ranked by Impact)

### CRITICAL — Blocks production safety

| # | Gap | Impact | Component | Detail |
|---|-----|--------|-----------|--------|
| G1 | **AuditEntry has no durable persistence** | Audit entries in ToolGateway accumulate in an in-memory list (`_audit_log`). On process restart, all audit history is lost. For a fail-closed audit system, this is a critical gap. | `control_plane/tool_gateway.py` | Needs: PostgreSQL `audit_entries` table + append on every tool call. AuditEntry already has `payload_hash` and `correlation_id` — just needs a writer. |
| G2 | **ApprovalRequest has no durable persistence** | All pending/resolved approvals live in `ApprovalManager._requests` (in-memory dict). Restart = all pending approvals lost, no audit trail of past approvals. | `policy/approval_manager.py` | Needs: PostgreSQL `approval_requests` table with full lifecycle columns + load-on-start. |
| G3 | **No TradeRecord recovery on restart** | TradeJournal holds open trades in memory. Restart during an open position = orphaned trade with no entry record when the exit fill arrives. | `reconciliation/journal/journal.py` | Needs: Persist open `TradeRecord` to PostgreSQL, reload on startup, reconcile against position snapshots. |
| G4 | **Redis Streams have no retention policy** | Event streams (`trading:stream:*`) grow unbounded. No `MAXLEN` or `MINID` trimming configured. Production Redis will run out of memory. | `bus/redis_streams.py` | Needs: `MAXLEN ~10000` on `XADD` calls or periodic `XTRIM`. |

### HIGH — Blocks operational readiness

| # | Gap | Impact | Component | Detail |
|---|-----|--------|-----------|--------|
| G5 | **ToolGateway read operations bypass gateway in legacy mode** | `GET_POSITIONS`, `GET_BALANCES`, etc. are registered as tools but legacy mode calls adapter directly. Inconsistent audit coverage. | `control_plane/tool_gateway.py`, `execution/engine.py` | Needs: Route all adapter reads through ToolGateway (or explicitly document which reads are unaudited). |
| G6 | **No data encryption at rest** | PostgreSQL tables with CONFIDENTIAL data (orders, fills, positions, balances, decision audits) have no column-level encryption. JSONL files are plaintext. | All storage layers | Needs: At minimum, filesystem-level encryption (dm-crypt/LUKS) or PostgreSQL TDE. Column-level encryption for RESTRICTED fields if multi-tenant. |
| G7 | **JSONL files grow unbounded** | `memory_store.jsonl`, `conversations.jsonl`, and `narration_history.jsonl` append forever. In-memory ring buffers cap what's loaded, but disk grows without bound. | `context/memory_store.py`, `narration/store.py` | Needs: Log rotation (daily/size-based) or periodic compaction that rewrites from current ring buffer state. |
| G8 | **No backup/restore strategy documented** | PostgreSQL, Redis, and Parquet all hold critical state, but there's no backup procedure, no point-in-time recovery setup, no disaster recovery runbook. | Infrastructure | Needs: `pg_dump` schedule, Redis RDB/AOF config, Parquet replication strategy. |

### MEDIUM — Degrades observability/compliance

| # | Gap | Impact | Component | Detail |
|---|-----|--------|-----------|--------|
| G9 | **`causation_id` is never populated** | `BaseEvent.causation_id` field exists but no component sets it. The event DAG feature is defined but non-functional. | All event publishers | Needs: Each component that creates a downstream event should set `causation_id = source_event.event_id`. |
| G10 | **"Data Flow Diagrams – MagicCarpet AI Platform" not in repository** | Referenced as source-of-truth by the user but not found. Without it, this inventory cannot be cross-validated. | Documentation | Needs: Add the document to `docs/architecture/` or link it in `CLAUDE.md`. |
| G11 | **No schema migration strategy** | `schema_version` field exists on BaseEvent but no consumer checks it. No Alembic migrations for PostgreSQL tables. | `bus/schemas.py`, `storage/postgres/` | Needs: Version-checking deserializer + Alembic migration scripts. |
| G12 | **Dead-letter queue is in-memory only** | Event bus dead-letter lists are in-memory. Failed messages are lost on restart. | `bus/redis_streams.py`, `bus/memory_bus.py` | Needs: Persist dead-letters to Redis list or PostgreSQL for post-mortem analysis. |
| G13 | **No access control on JSONL files** | `data/*.jsonl` files contain CONFIDENTIAL data but have no file-permission hardening. | All JSONL stores | Needs: `chmod 600` on creation, or move to PostgreSQL for access-controlled storage. |

### LOW — Polish items

| # | Gap | Impact | Component | Detail |
|---|-----|--------|-----------|--------|
| G14 | **Parquet schema not versioned** | Candle Parquet files have no schema version metadata. Adding columns in the future could break readers. | `intelligence/historical/` | Needs: Parquet metadata field `schema_version` or Arrow schema evolution strategy. |
| G15 | **No data lineage for BacktestResult** | `BacktestResult` contains a `deterministic_hash` but doesn't record which Parquet files, config version, or code version produced it. | `backtester/engine.py` | Needs: Add `source_data_hash`, `config_hash`, `git_sha` to BacktestResult. |

---

## Sensitivity Summary

| Classification | Count | Examples |
|----------------|-------|---------|
| **RESTRICTED** | 6 secrets | API keys, DB credentials (env vars only, never logged) |
| **CONFIDENTIAL** | ~45 objects | Orders, fills, positions, balances, signals, P&L, reasoning traces, governance decisions |
| **INTERNAL** | ~15 objects | Market data, indicators, system health, experiment configs |
| **PUBLIC** | 1 object | NarrationItem (narration scripts for avatar/UI) |

**No PII exists in the platform** — no user names, emails, or addresses. Regulatory requirements are financial (trade record-keeping, best execution), not GDPR.

---

## Latency Budget (Signal to Fill, Live CP Mode)

```
FeedManager (exchange WS)     - <1000ms -> CandleEvent
FeatureEngine                  -   <50ms -> FeatureVector
Strategy                       -  <100ms -> Signal
PortfolioManager               -  <100ms -> OrderIntent
GovernanceGate                 -   <50ms -> governance decision
ToolGateway (policy+audit)     -  <100ms -> audit + dispatch
CCXTAdapter (REST)             -  <500ms -> OrderAck
                               ===========
                         Total:  <1900ms (autonomous)
                                 +minutes (if human approval needed)
```

## Fail-Closed Components (unavailable = all trading stops)

1. EventBus (Redis Streams)
2. ExecutionEngine
3. RiskManager
4. ToolGateway (CP mode)
5. Audit log (PostgreSQL)
6. GovernanceGate (when not in shadow mode)
