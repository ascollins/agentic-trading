# Agentic Trading Platform — Institutional Refurb Plan

**Version:** 1.0
**Date:** 2026-02-16
**Status:** DRAFT — Awaiting Review

---

## 1) EXECUTIVE REFURB SUMMARY

### Guiding Principle: Institutional Core, Apple-Simple Shell

The platform has strong bones: event-driven architecture, Protocol-based composition, a mature governance gate (maturity ladder + policy engine + approval workflows), and a rich feature engine with 50+ indicators. The refurb does not tear it down. It hardens the core, fills measurement gaps, and wraps it in a supervision UI that hides complexity behind progressive disclosure.

### What to Keep

| Component | Why |
|-----------|-----|
| Event-driven pipeline (FeedManager → FeatureEngine → Strategy → PortfolioManager → ExecutionEngine) | Clean separation, mode-agnostic |
| GovernanceGate (maturity + impact + health + drift + policy + approval) | Already 80% institutional; needs provenance, not replacement |
| BaseAgent + AgentRegistry lifecycle pattern | Solid ABC; extend, don't replace |
| TradeJournal + QualityScorecard | R-multiple, MAE/MFE, A-F grading — rare in crypto platforms |
| Policy-as-code engine (9 operators, shadow mode, versioned store) | Declarative rules ready for production |
| Redis Streams bus with retry + dead-letter | Correct ack-after-handler semantics already fixed |

### What to Change

| Change | Impact |
|--------|--------|
| **Add event causation chain** (`parent_event_id`, `correlation_id` on BaseEvent) | Enables full decision replay and audit |
| **Add execution quality measurement** (slippage, adverse selection, fill rate) | Currently a blind spot — trades happen but quality is unmeasured |
| **Wire the AgentOrchestrator into main.py** (currently defined but unused) | Single lifecycle manager instead of ad-hoc wiring |
| **Add ToolGateway boundary** between agents and side-effects | Every exchange call, config change, or state mutation goes through a logged gateway |
| **Add strategy lifecycle state machine** (Candidate → Backtest → Paper → Limited → Scale) | MaturityLevel exists but no enforced promotion workflow |
| **Build a supervision web UI** (4 tabs, Apple-calm) replacing Grafana-only monitoring | Grafana is ops-grade; traders need a calm, sparse interface |

### What to Cut / Defer

| Item | Reason |
|------|--------|
| Multi-tenancy (org/fund/subaccount) | Premature — single-operator platform today |
| Full OpenTelemetry integration | P2 — causation chain on events gives 80% of the value |
| ML-based strategy selection | No data to train on yet; focus on measurement first |
| Complex rebalancing / portfolio optimization | Current volatility-adjusted sizing is adequate for P0 |

### Smallest Set of Changes for Step-Change in Credibility

1. **Event envelope upgrade** — 3 fields on BaseEvent unlock replay, audit, attribution
2. **Execution quality tracker** — slippage + fill rate + adverse selection per trade
3. **Strategy lifecycle state machine** — enforced promotion with evidence gates
4. **Daily Effectiveness Scorecard** — 4 scores visible on Home screen
5. **ToolGateway** — all side-effects go through a logged, policy-checked boundary
6. **Supervision UI** — 4-tab web app replacing Grafana for daily use

---

## 2) PLATFORM GAP ANALYSIS

| # | Domain | Current Coverage | Gaps | Impact | Recommended Fix | Priority |
|---|--------|-----------------|------|--------|----------------|----------|
| G1 | **Policy-as-code + approvals** | PolicyEngine (9 ops, shadow mode), ApprovalManager (4 levels, 8 triggers), GovernanceGate step 3.5 + 6.5 | No policy versioning UI; no approval dashboard; approval state is in-memory (lost on restart); no policy diff/review workflow | Approvals lost on restart; no audit of policy changes | Persist approvals to Postgres; add policy diff events; add approval API for UI | **P0** |
| G2 | **Tool boundary (ToolGateway)** | Exchange calls go directly through adapter; no interception layer | No signed calls; no idempotency tokens for exchange submissions; no audit log of adapter calls; agents can call adapter methods without governance check | Unauditable side-effects; potential duplicate submissions on retry | Add `ToolGateway` wrapper around `IExchangeAdapter` that logs, checks policy, enforces idempotency | **P0** |
| G3 | **Strategy lifecycle** | MaturityLevel L0-L4 exists; GovernanceGate blocks by level; no enforced promotion workflow | No `StrategyPromotionRequest` model; no evidence collection (trade count, win rate, Sharpe); no demotion triggers connected to journal stats; maturity set manually | Strategies can be manually promoted without evidence; no auditability | Add StrategyLifecycleManager with state machine (Candidate→Backtest→Paper→Limited→Scale→Demote) | **P0** |
| G4 | **Portfolio/intent layer + constraints** | PortfolioManager computes TargetPosition; multi-method sizing (volatility, stop-loss, Kelly); governance cap | No portfolio-level constraint enforcement (max correlation, sector limits, max gross exposure as hard constraint); no intent validation against existing portfolio | Could exceed aggregate exposure limits even if individual trades pass | Add PortfolioConstraintChecker between signal and intent generation | **P1** |
| G5 | **Execution quality measurement** | FillEvent records price/qty/fee; TradeRecord computes PnL | No slippage tracking (signal_price vs fill_price); no fill rate metric; no adverse selection detection; no venue comparison; EXECUTION_SLIPPAGE metric defined but never emitted | Cannot assess execution quality; no data for venue optimization | Add ExecutionQualityTracker: compute slippage, adverse selection, fill rate per trade; emit metrics | **P0** |
| G6 | **Surveillance / case management** | GovernanceCanary periodic health; DriftDetector compares live vs baseline | No case creation for anomalies; no investigation workflow; no alert → case → resolution pipeline; no suspicious activity detection | Anomalies logged but not tracked to resolution | Add SurveillanceCase model + basic case lifecycle (Open→Investigating→Resolved) | **P2** |
| G7 | **Incident response + degraded modes** | KillSwitch exists; CircuitBreakers (5 types); GovernanceCanary triggers kill on threshold | No degraded mode definition (reduce-only, no-new-trades, read-only); no incident declaration event; no runbook execution; no auto-recovery criteria | Binary: running or killed. No graceful degradation. | Add DegradedMode enum + IncidentManager with state machine; connect to agent health cascade | **P1** |
| G8 | **Audit / replay (decision provenance)** | BaseEvent has `trace_id` and `event_id`; GovernanceDecision has `details` dict | No `parent_event_id` causation chain; no `correlation_id` for business grouping; GovernanceDecision doesn't persist full decision path (steps); no event replay capability | Cannot trace Signal→Intent→Ack→Fill chain; impossible to audit governance decisions retroactively | Add causation fields to BaseEvent; add DecisionPath model; persist to event store | **P0** |
| G9 | **Data quality + lineage** | FeatureEngine computes 50+ indicators; NaN padding for warmup | No input validation (OHLCV sanity, zero volume, gap detection); no feature lineage (which candles produced which features); no data quality score | Bad data silently propagates to strategies; no way to attribute bad trades to bad data | Add DataQualityChecker at FeedManager output; add quality score to FeatureVector | **P1** |
| G10 | **Schema evolution** | Static TOPIC_SCHEMAS map; Pydantic model_validate_json | No schema_version in event envelope; adding required field breaks old messages; no migration layer | Deployment of new event fields breaks in-flight messages | Add `_schema_version` to envelope; add migration registry; default all new fields | **P1** |

---

## 3) ARCHITECTURE DELTA

### Current → Target Component Map

```
CURRENT                                    TARGET (additions in [brackets])
─────────────────────────────────────      ──────────────────────────────────────
FeedManager                                FeedManager
  ↓                                          ↓
CandleBuilder                              CandleBuilder
  ↓                                          ↓ [DataQualityChecker]
FeatureEngine                              FeatureEngine
  ↓                                          ↓
Strategy.on_candle()                       Strategy.on_candle()
  ↓                                          ↓
Signal                                     Signal (+ candle_event_id, fv_id)
  ↓                                          ↓
PortfolioManager                           PortfolioManager
  ↓                                          ↓ [PortfolioConstraintChecker]
OrderIntent                                TradeIntentProposed (new envelope)
  ↓                                          ↓
                                           [ToolGateway]
                                             ↓ PolicyEval → Approval (if req'd)
ExecutionEngine                            ExecutionEngine
  ↓ pre_trade_check                          ↓ pre_trade_check
  ↓ governance_gate.evaluate()               ↓ governance_gate.evaluate()
  ↓ adapter.submit_order()                   ↓ [ToolGateway].submit_order()
  ↓                                          ↓ ToolCallRecorded event
OrderAck → FillEvent                       OrderAck → FillEvent
  ↓                                          ↓ [ExecutionQualityTracker]
Journal                                    Journal (+ slippage, adverse_sel)
  ↓                                          ↓
QualityScorecard                           QualityScorecard
                                             ↓ [DailyEffectivenessScorecard]

AGENTS (current 6)                         AGENTS (target 8, +2 thin wrappers)
├─ MarketIntelligenceAgent                 ├─ MarketIntelligenceAgent
├─ ExecutionAgent                          ├─ ExecutionAgent
├─ RiskGateAgent                           ├─ RiskGateAgent
├─ GovernanceCanary                        ├─ GovernanceCanary
├─ ReconciliationLoop                      ├─ ReconciliationLoop
├─ OptimizerScheduler                      ├─ OptimizerScheduler
                                           ├─ [StrategyLifecycleAgent] (NEW)
                                           └─ [IncidentManager] (NEW)

ORCHESTRATION                              ORCHESTRATION
AgentOrchestrator (UNUSED)                 AgentOrchestrator (WIRED into main.py)
                                           + dependency ordering
                                           + health cascade
                                           + degraded mode transitions
```

### Minimum New Services (Thinnest Viable Set)

| # | Service | Responsibility | Size Estimate | Plugs Into |
|---|---------|---------------|---------------|------------|
| 1 | **ToolGateway** | Wraps `IExchangeAdapter`; logs every call; checks policy; enforces idempotency tokens; emits `ToolCallRecorded` | ~200 LOC | ExecutionEngine replaces `self._adapter` with `self._tool_gateway` |
| 2 | **ExecutionQualityTracker** | Computes slippage, fill rate, adverse selection per fill; emits metrics | ~150 LOC | Called from ExecutionEngine `handle_fill()` after FillEvent |
| 3 | **StrategyLifecycleManager** | State machine for strategy promotion/demotion; collects evidence from Journal; emits `MaturityTransition` | ~300 LOC | New agent; subscribes to journal close events; provides API for UI |
| 4 | **IncidentManager** | State machine (Detect→Triage→Degrade→Recover); manages `DegradedMode` transitions | ~250 LOC | Subscribes to `risk` and `system` topics; publishes `governance` |
| 5 | **DailyEffectivenessScorecard** | Aggregates 4 scores (Edge, Execution, Risk, Ops) from journal + metrics | ~200 LOC | Called from journal on trade close + periodic timer |
| 6 | **SupervisionAPI** | FastAPI/Starlette endpoints for the UI (read-only + approval actions) | ~400 LOC | Reads from journal, registry, scorecard; writes to ApprovalManager |

**Total new code: ~1,500 LOC** (plus ~300 LOC of event model additions)

### The Action Boundary

Every agent side-effect follows this path:

```
AgentProposal (e.g., OrderIntent)
    ↓
PolicyEval (GovernanceGate.evaluate() — maturity, impact, health, drift, policy rules)
    ↓
Approval (if required by ApprovalManager rules — L2+ escalation)
    ↓
ToolCall (ToolGateway.submit_order() — idempotency check, policy re-check, log)
    ↓
AuditEvent (ToolCallRecorded published on governance topic)
```

No agent can bypass ToolGateway. The gateway is the single point of:
- Idempotency enforcement (dedupe token)
- Policy re-validation (belt-and-suspenders with GovernanceGate)
- Call logging (every adapter method call recorded)
- Rate limiting (if configured)

---

## 4) EVENT & DATA MODEL DELTA

### 4.1 BaseEvent Envelope Upgrade

Add 3 fields to `BaseEvent` (backward-compatible — all have defaults):

```python
class BaseEvent(BaseModel):
    # Existing fields
    event_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    trace_id: str = Field(default_factory=_uuid)
    source_module: str = ""

    # NEW: Causation chain
    parent_event_id: str | None = None       # Which event triggered this?
    correlation_id: str = ""                  # Business-level grouping (e.g., strategy signal cycle)

    # NEW: Schema version for evolution
    schema_version: int = 1
```

### 4.2 New Canonical Events

#### PolicyEvaluated

```json
{
  "event_type": "PolicyEvaluated",
  "event_id": "evt-abc123",
  "timestamp": "2026-02-16T10:30:00Z",
  "trace_id": "trc-xyz",
  "parent_event_id": "evt-order-intent-456",
  "correlation_id": "corr-signal-cycle-789",
  "schema_version": 1,
  "source_module": "governance",

  "strategy_id": "trend_following",
  "symbol": "BTC/USDT",
  "intent_event_id": "evt-order-intent-456",
  "decision": "ALLOW",
  "sizing_multiplier": 0.85,
  "evaluation_ms": 4.2,
  "steps": [
    {"step": "maturity_check", "result": "PASS", "level": "L3_PROVEN", "ms": 0.1},
    {"step": "impact_classification", "result": "MEDIUM", "tier": "MEDIUM", "ms": 0.3},
    {"step": "health_score", "result": "PASS", "score": 0.92, "ms": 0.2},
    {"step": "drift_check", "result": "PASS", "max_drift_pct": 8.5, "ms": 0.5},
    {"step": "policy_rules", "result": "PASS", "rules_checked": 9, "violations": 0, "ms": 2.1},
    {"step": "approval_check", "result": "NOT_REQUIRED", "ms": 0.1}
  ],
  "policy_set_version": "v3",
  "shadow_violations": []
}
```

#### ApprovalRequested / ApprovalGranted / ApprovalDenied

These already exist (`ApprovalRequested`, `ApprovalResolved`). Enhancement:

```json
{
  "event_type": "ApprovalResolved",
  "event_id": "evt-appr-resolved-001",
  "parent_event_id": "evt-appr-requested-001",
  "correlation_id": "corr-signal-cycle-789",
  "schema_version": 1,

  "request_id": "appr-req-001",
  "resolution": "APPROVED",
  "resolved_by": "operator@firm.com",
  "resolved_at": "2026-02-16T10:30:45Z",
  "escalation_level": "L2_OPERATOR",
  "time_to_resolve_ms": 45000,
  "conditions": [],
  "audit_note": "Confirmed after reviewing market conditions"
}
```

#### ToolCallRecorded

```json
{
  "event_type": "ToolCallRecorded",
  "event_id": "evt-tool-001",
  "parent_event_id": "evt-order-intent-456",
  "correlation_id": "corr-signal-cycle-789",
  "schema_version": 1,
  "source_module": "tool_gateway",

  "tool_name": "submit_order",
  "agent_id": "execution-agent-01",
  "adapter_type": "CCXTAdapter",
  "idempotency_key": "dedupe-trend_following-BTC-1708075800",
  "request_hash": "sha256:abc123",
  "request_summary": {
    "symbol": "BTC/USDT",
    "side": "BUY",
    "qty": "0.015",
    "order_type": "LIMIT",
    "price": "42150.00"
  },
  "response_summary": {
    "order_id": "bybit-ord-789",
    "status": "SUBMITTED",
    "exchange_ts": "2026-02-16T10:30:01.234Z"
  },
  "latency_ms": 234,
  "success": true,
  "policy_check_passed": true,
  "retries": 0
}
```

#### IncidentDeclared + DegradedModeEnabled

```json
{
  "event_type": "IncidentDeclared",
  "event_id": "evt-incident-001",
  "parent_event_id": "evt-circuit-breaker-trip-123",
  "correlation_id": "corr-incident-001",
  "schema_version": 1,
  "source_module": "incident_manager",

  "incident_id": "inc-001",
  "severity": "HIGH",
  "trigger": "CIRCUIT_BREAKER_DAILY_LOSS",
  "trigger_event_id": "evt-circuit-breaker-trip-123",
  "description": "Daily loss limit breached: -2.3% vs -2.0% threshold",
  "affected_strategies": ["trend_following", "breakout"],
  "affected_symbols": ["BTC/USDT", "ETH/USDT"],
  "auto_actions_taken": ["PAUSE_NEW_TRADES"],
  "requires_human": true
}

{
  "event_type": "DegradedModeEnabled",
  "event_id": "evt-degrade-001",
  "parent_event_id": "evt-incident-001",
  "correlation_id": "corr-incident-001",
  "schema_version": 1,
  "source_module": "incident_manager",

  "mode": "REDUCE_ONLY",
  "previous_mode": "NORMAL",
  "reason": "Incident inc-001: daily loss limit breached",
  "restrictions": [
    "No new position entries",
    "Existing positions may be reduced or closed",
    "Strategy signals still generated (shadow mode)"
  ],
  "recovery_criteria": {
    "type": "MANUAL_APPROVAL",
    "min_cooldown_minutes": 60
  }
}
```

#### TradeIntentProposed + ExecutionPlanProposed

```json
{
  "event_type": "TradeIntentProposed",
  "event_id": "evt-intent-001",
  "parent_event_id": "evt-signal-abc",
  "correlation_id": "corr-signal-cycle-789",
  "schema_version": 1,
  "source_module": "portfolio",

  "strategy_id": "trend_following",
  "symbol": "BTC/USDT",
  "direction": "LONG",
  "signal_confidence": 0.82,
  "signal_event_id": "evt-signal-abc",
  "proposed_qty": "0.015",
  "sizing_method": "volatility_adjusted",
  "stop_price": "41200.00",
  "target_price": "44500.00",
  "risk_amount_usd": "142.50",
  "portfolio_impact": {
    "new_gross_exposure_pct": 12.5,
    "new_concentration_pct": 8.2,
    "correlation_with_existing": 0.15
  }
}

{
  "event_type": "ExecutionPlanProposed",
  "event_id": "evt-plan-001",
  "parent_event_id": "evt-intent-001",
  "correlation_id": "corr-signal-cycle-789",
  "schema_version": 1,
  "source_module": "execution",

  "intent_event_id": "evt-intent-001",
  "execution_style": "LIMIT_WITH_TIMEOUT",
  "orders": [
    {
      "order_type": "LIMIT",
      "side": "BUY",
      "qty": "0.015",
      "limit_price": "42150.00",
      "time_in_force": "GTC",
      "timeout_seconds": 300
    }
  ],
  "estimated_slippage_bps": 3.2,
  "estimated_fee_usd": 1.26
}
```

### 4.3 Event Envelope Fields for Replay Determinism

Every event persisted to the event store must include:

| Field | Purpose | Example |
|-------|---------|---------|
| `event_id` | Unique event identity | `evt-abc123` |
| `timestamp` | Wall clock when created | `2026-02-16T10:30:00.123Z` |
| `trace_id` | Distributed trace group | `trc-xyz` |
| `parent_event_id` | Causation parent | `evt-signal-abc` |
| `correlation_id` | Business cycle group | `corr-signal-cycle-789` |
| `schema_version` | For migration | `1` |
| `source_module` | Producing module | `execution` |
| `_type` | Event class name (existing) | `OrderAck` |
| `_sequence` | **NEW**: Monotonic sequence per topic | `10452` |

The `_sequence` field is assigned by the event bus at publish time. For Redis Streams, this is the message ID. For MemoryEventBus, a per-topic counter. This enables deterministic replay: events replayed in `_sequence` order produce identical state.

### 4.4 Idempotency Strategy

| Boundary | Guarantee | Mechanism |
|----------|-----------|-----------|
| **Event bus publish** | At-least-once | Redis Streams ack-after-handler; DLQ on exhaustion. Consumer groups ensure each message processed by one consumer per group. |
| **Event bus subscribe** | At-least-once | Handlers must be idempotent. Use `event_id` dedup in handlers that mutate state. |
| **Exchange order submission** | Exactly-once (effort) | `dedupe_key` in OrderIntent → ToolGateway checks `idempotency_key` against in-memory set (TTL 1h) + adapter's `clientOrderId`. |
| **Approval resolution** | Exactly-once | ApprovalRequest has `request_id`; `resolve()` checks current status before transition. |
| **Journal trade recording** | Exactly-once | Journal uses `trace_id` as key; `record_entry_fill()` is idempotent by `fill_id`. |
| **Policy evaluation** | Idempotent (stateless) | Same inputs → same outputs. No side effects during evaluation. |

**Where exactly-once matters:** Exchange order submission (financial impact). All others tolerate at-least-once with idempotent handlers.

---

## 5) WORKFLOW STATE MACHINES

### 5.1 Execution Lifecycle

```
                                    ┌─────────────┐
                                    │  PROPOSED    │ (TradeIntentProposed)
                                    └──────┬──────┘
                                           │ validate portfolio constraints
                                    ┌──────▼──────┐
                             ┌──────│  EVALUATING  │ (PolicyEvaluated)
                             │      └──────┬──────┘
                             │             │ policy ALLOW
                     policy  │      ┌──────▼──────┐
                     BLOCK   │      │ APPROVED /   │ (approval not required)
                             │      │ PENDING_APPR │ (approval required → wait)
                             │      └──────┬──────┘
                             │             │ approved (or auto-approved L1)
                      ┌──────▼──────┐      │
                      │  REJECTED   │      │
                      └─────────────┘      │
                                    ┌──────▼──────┐
                                    │ SUBMITTING   │ (ToolGateway.submit)
                                    └──────┬──────┘
                                           │ OrderAck
                          ┌────────────────┼────────────────┐
                          │                │                │
                   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼─────┐
                   │  SUBMITTED  │  │  FILLED     │  │ FAILED    │
                   └──────┬──────┘  └──────┬──────┘  └───────────┘
                          │                │
                          │ FillEvent      │ (post-trade checks)
                          │                │
                   ┌──────▼──────┐  ┌──────▼──────┐
                   │ PARTIALLY   │  │  MONITORED  │ (ExecutionQualityTracker)
                   │ FILLED      │  └──────┬──────┘
                   └──────┬──────┘         │ trade closed
                          │                │
                          │         ┌──────▼──────┐
                          └────────►│  COMPLETED  │ (post-trade audit)
                                    └─────────────┘
```

**Timeouts:**
- PENDING_APPROVAL: 15 min default → auto-escalate to next level
- SUBMITTING: 30s → retry (max 3) → FAILED
- SUBMITTED: exchange-dependent (GTC orders monitored by ReconciliationLoop)
- PARTIALLY_FILLED: monitor for 5 min → decide: cancel remainder or wait

**Failure Modes:**
- Policy BLOCK → REJECTED (emit event, log, no retry)
- Approval DENIED → REJECTED (emit event, notify strategy)
- Approval EXPIRED → auto-escalate or REJECTED (configurable)
- Exchange error → retry with backoff (max 3) → FAILED
- Post-trade risk check fails → emit RiskAlert (may trigger incident)

### 5.2 Strategy Lifecycle

```
     ┌───────────┐
     │ CANDIDATE  │  (code merged, no live data)
     └─────┬─────┘
           │ backtest submitted
     ┌─────▼─────┐
     │ BACKTEST   │  (OptimizerScheduler runs walk-forward)
     └─────┬─────┘
           │ eval pack generated
     ┌─────▼─────┐
     │ EVAL_PACK  │  (QualityScorecard ≥ 60, min 50 trades, PF > 1.1)
     └─────┬─────┘
           │ operator approves → L2_GATED
     ┌─────▼─────┐
     │ PAPER      │  (MaturityLevel.L2_GATED, sizing cap 10%)
     └─────┬─────┘
           │ 100+ paper trades, live Sharpe within 1σ of backtest
     ┌─────▼─────┐
     │ LIMITED    │  (MaturityLevel.L3_PROVEN, sizing cap 25%)
     └─────┬─────┘
           │ 30+ days, drawdown < threshold, QualityScore ≥ 70
     ┌─────▼─────┐
     │ SCALE      │  (MaturityLevel.L4_AUTONOMOUS, sizing cap 100%)
     └─────┬─────┘
           │
           ▼ (continuous monitoring)
           │
           │ demotion triggers:
           │  - drawdown > 10% from peak
           │  - loss streak > 10 consecutive
           │  - QualityScore < 50
           │  - drift > 30% from baseline
           │
     ┌─────▼─────┐
     │ DEMOTED    │  (sizing reduced or paused; review required)
     └─────┬─────┘
           │ operator review + fix
           │
           ▼ (re-enter at PAPER or EVAL_PACK)
```

**Evidence Gates (automated collection from Journal):**

| Transition | Required Evidence |
|------------|-------------------|
| CANDIDATE → BACKTEST | Backtest config + data range submitted |
| BACKTEST → EVAL_PACK | Walk-forward Sharpe > 0.5, PF > 1.1, 50+ trades, max DD < 20% |
| EVAL_PACK → PAPER | Operator approval (L2_OPERATOR) |
| PAPER → LIMITED | 100+ paper trades, live Sharpe within 1σ of backtest, QualityScore ≥ 60 |
| LIMITED → SCALE | 30+ days at LIMITED, QualityScore ≥ 70, no incidents, operator approval (L3_RISK) |
| Any → DEMOTED | Automated trigger (drawdown, streak, drift, score) |
| DEMOTED → PAPER | Operator approval + evidence of fix |

**Rollback:** Demotion always drops to PAPER (not BACKTEST). Strategy keeps running in shadow mode at DEMOTED level so data continues to be collected.

### 5.3 Incident Lifecycle

```
     ┌───────────┐
     │ NOMINAL    │  (all systems healthy)
     └─────┬─────┘
           │ trigger detected (circuit breaker, canary, drift, manual)
     ┌─────▼─────┐
     │ DETECTED   │  (IncidentDeclared event)
     └─────┬─────┘
           │ auto-triage (severity classification)
     ┌─────▼─────┐
     │ TRIAGED    │  (severity: LOW | MEDIUM | HIGH | CRITICAL)
     └─────┬─────┘
           │ degraded mode applied
     ┌─────▼─────┐
     │ DEGRADED   │  (DegradedModeEnabled event)
     │            │  Modes: REDUCE_ONLY | NO_NEW_TRADES | READ_ONLY | KILLED
     └─────┬─────┘
           │ runbook actions (auto or manual)
     ┌─────▼─────┐
     │ MITIGATING │  (actions in progress)
     └─────┬─────┘
           │ recovery criteria met
     ┌─────▼─────┐
     │ RECOVERING │  (cooldown period, health checks passing)
     └─────┬─────┘
           │ operator approval to resume (MEDIUM+)
     ┌─────▼─────┐
     │ RESOLVED   │  (DegradedModeDisabled, back to NOMINAL)
     └───────────┘
```

**Auto-Triage Rules:**

| Trigger | Severity | Auto-Action |
|---------|----------|-------------|
| Single circuit breaker (non-daily-loss) | LOW | Log + alert, continue trading |
| Daily loss limit breached | HIGH | NO_NEW_TRADES, notify operator |
| Kill switch activated | CRITICAL | KILLED, require L3_RISK approval to resume |
| 2+ canary components unhealthy | MEDIUM | REDUCE_ONLY (50% sizing cap) |
| Exchange connection lost > 60s | HIGH | NO_NEW_TRADES, attempt reconnect |
| Reconciliation drift > 5% | MEDIUM | REDUCE_ONLY, flag for investigation |

**Recovery Criteria:**

| Severity | Cooldown | Resume Approval |
|----------|----------|-----------------|
| LOW | None | Auto-resume when trigger clears |
| MEDIUM | 15 min | Auto-resume after cooldown if health checks pass |
| HIGH | 60 min | Operator approval required |
| CRITICAL | 4 hours | L3_RISK approval required |

---

## 6) APPLE-INSPIRED SUPERVISION UX REDESIGN

### Design Language

| Property | Value |
|----------|-------|
| **Typography** | SF Pro Display (headings), SF Mono (numbers/prices). 16px base, 1.5 line height. |
| **Color palette** | Near-white background (#FAFAFA). Near-black text (#1D1D1F). Green (#34C759) for positive. Red (#FF3B30) for negative. Blue (#007AFF) for interactive. Amber (#FF9500) for warnings. |
| **Spacing** | 8px grid. 16px component padding. 24px section gaps. |
| **Border radius** | 12px for cards. 8px for buttons. |
| **Shadow** | `0 1px 3px rgba(0,0,0,0.08)` for cards. No heavy shadows. |
| **Animation** | 200ms ease-out for transitions. No bouncing, no gratuitous animation. |
| **Alert philosophy** | Calm by default. Only CRITICAL alerts use red + sound. Everything else is quiet. |

### Alert Hierarchy

| Level | Visual | Sound | When |
|-------|--------|-------|------|
| **Info** | Gray dot in Activity tab badge | None | Strategy signal, fill, routine event |
| **Notice** | Blue dot + Activity badge count | None | Approval pending, strategy promoted |
| **Warning** | Amber card in Home stack | None | Degraded mode, drift detected, quality score drop |
| **Critical** | Red full-width banner at top of any tab + haptic/sound | Short tone | Kill switch, incident, exchange down |

**Rule:** If the user hasn't seen a critical alert in a month, the system is working correctly.

### Tab Structure

**4 tabs: Home | Strategies | Activity | Settings**

---

### Tab 1: HOME (Status Card Stack)

**Purpose:** "Is everything okay?" — answerable in 2 seconds.

**Layout:**

```
┌────────────────────────────────────────────────┐
│  [CRITICAL BANNER — only if active incident]   │
├────────────────────────────────────────────────┤
│                                                │
│  Daily Effectiveness Score          8.2 / 10   │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░   │
│  Edge 8.5  Exec 7.8  Risk 9.0  Ops 7.5        │
│                                                │
├────────────────────────────────────────────────┤
│                                                │
│  Portfolio               ▲ +1.23% today        │
│  $102,450                                      │
│  2 open positions                              │
│                                                │
├────────────────────────────────────────────────┤
│                                                │
│  System Status           ● All Healthy         │
│  6/6 agents running                            │
│  Last candle: 12s ago                          │
│                                                │
├────────────────────────────────────────────────┤
│                                                │
│  [Pending Approvals card — only if any]        │
│  1 approval waiting        [Review →]          │
│                                                │
└────────────────────────────────────────────────┘
```

**Progressive Disclosure:**
- Tap "Daily Effectiveness Score" → drawer with per-metric breakdown
- Tap "Portfolio" → drawer with position detail, exposure chart
- Tap "System Status" → drawer with per-agent health, event bus stats
- Tap "Pending Approvals" → navigates to Activity tab, Approvals filter

**Empty State:** When no trades today: "No activity yet today. Strategies are monitoring the market." with a subtle pulse on the system status dot.

---

### Tab 2: STRATEGIES (Stage Ribbon + Score)

**Purpose:** What's each strategy doing, and how good is it?

**Layout:**

```
┌────────────────────────────────────────────────┐
│  Strategies                          [+ Add]   │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │ trend_following                    B+ 78  │  │
│  │ SCALE ● ───────────────────────●         │  │
│  │        C  BT  EV  PA  LIM  SCALE         │  │
│  │ +$1,240 today  |  42 trades  |  WR 43%   │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │ mean_reversion                    A- 87  │  │
│  │ LIMITED ● ──────────────────●             │  │
│  │          C  BT  EV  PA  LIM               │  │
│  │ +$380 today   |  28 trades  |  WR 68%    │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │ breakout                          C  62  │  │
│  │ PAPER ● ─────────────●                   │  │
│  │        C  BT  EV  PA                      │  │
│  │ -$45 today    |  12 trades  |  WR 35%    │  │
│  │ ⚠ Below promotion threshold               │  │
│  └──────────────────────────────────────────┘  │
│                                                │
└────────────────────────────────────────────────┘
```

**Stage Ribbon:** Horizontal dots (C → BT → EV → PA → LIM → SCALE) with filled dot at current stage and line showing progress. Compact, always visible.

**Progressive Disclosure:**
- Tap strategy card → **Strategy Detail drawer**:
  - Full QualityScorecard (all metrics with grades)
  - Equity curve (strategy-level)
  - Recent trades (last 10)
  - Parameter snapshot
  - Promotion history
  - "Promote" / "Demote" buttons (if eligible)
  - "View Backtest" link

---

### Tab 3: ACTIVITY (Timeline + Approvals)

**Purpose:** What happened, and what needs my attention?

**Layout:**

```
┌────────────────────────────────────────────────┐
│  Activity                                      │
│                                                │
│  [All] [Trades] [Approvals] [Incidents]        │
│                                                │
│  TODAY                                         │
│                                                │
│  10:32  ● FILL  BTC/USDT +0.015 @ $42,150    │
│         trend_following | slippage: 1.2 bps    │
│                                                │
│  10:31  ○ SIGNAL  BTC/USDT LONG conf 0.82     │
│         trend_following | EMA cross + ADX 31   │
│                                                │
│  10:30  ◆ GOVERNANCE ALLOW                     │
│         Policy passed | sizing × 0.85          │
│                                                │
│  09:15  ▲ APPROVAL PENDING                     │
│         ETH/USDT HIGH impact trade             │
│         [Approve] [Deny] [Details]             │
│                                                │
│  08:00  ■ INCIDENT RESOLVED                    │
│         Daily loss limit — auto-recovered      │
│                                                │
│  YESTERDAY                                     │
│  ...                                           │
│                                                │
└────────────────────────────────────────────────┘
```

**Filters:** Segment control at top. "Approvals" shows only actionable items. "Incidents" shows incident history.

**Progressive Disclosure:**
- Tap any timeline entry → **Detail drawer** with:
  - Full event JSON (collapsed by default)
  - Causation chain ("This fill was caused by...")
  - Related events (all events sharing `correlation_id`)

---

### Tab 4: SETTINGS

**Layout:**

```
┌────────────────────────────────────────────────┐
│  Settings                                      │
│                                                │
│  ACCOUNT                                       │
│  Exchange: Bybit (demo) ✓ Connected            │
│  Mode: Paper Trading                           │
│                                                │
│  RISK LIMITS                                   │
│  Max position size    5% of equity             │
│  Max daily loss       -2.0%                    │
│  Max drawdown         -10.0%                   │
│  Kill switch          [OFF]                    │
│                                                │
│  GOVERNANCE                                    │
│  Policy engine        Enabled (9 rules)        │
│  Approval workflow    Enabled (auto L1)         │
│  Shadow mode          OFF                      │
│  [View Policy Rules →]                         │
│                                                │
│  AGENTS                                        │
│  6/6 running                                   │
│  [View Agent Details →]                        │
│                                                │
│  SYSTEM                                        │
│  Event bus: Redis Streams ✓                    │
│  Metrics: Prometheus :9090 ✓                   │
│  Database: Postgres ✓                          │
│                                                │
└────────────────────────────────────────────────┘
```

---

### Layout Variants

#### Variant A: Ops-First

- Home card stack leads with **System Status** (agents, bus, latency)
- Portfolio summary is second card
- Better for: operator monitoring multiple instances
- Trade-off: daily P&L less prominent; feels like ops dashboard

#### Variant B: Trader-First (Recommended)

- Home card stack leads with **Daily Effectiveness Score** and **Portfolio**
- System status is third card (smaller, green dot = no issues)
- Better for: single operator focused on strategy performance
- Trade-off: system issues less immediately visible (but critical banner handles emergencies)

**Recommendation:** Start with Variant B (Trader-First). The alert hierarchy ensures critical issues surface regardless of tab layout. Most of the time, the operator wants to know "am I making money and are my strategies working?" — not "is Redis up?"

---

## 7) MEASUREMENT SYSTEM

### 7.1 Daily Effectiveness Scorecard (4 Scores, 0-10)

Displayed on Home screen as a single number (weighted average) with 4 sub-scores.

#### Score 1: Edge Quality (Weight: 30%)

*"Are we trading with a genuine statistical edge?"*

| Metric | Formula | Weight | 10 | 7 | 4 | 0 |
|--------|---------|--------|------|------|------|------|
| Win Rate vs Strategy Type Target | actual_wr / target_wr | 25% | ≥1.2x | ≥1.0x | ≥0.8x | <0.6x |
| Profit Factor (rolling 50 trades) | gross_wins / gross_losses | 25% | ≥2.0 | ≥1.5 | ≥1.1 | <1.0 |
| Average R-Multiple | mean(r_multiples) | 25% | ≥0.5R | ≥0.3R | ≥0.1R | <0R |
| Confidence Calibration (Brier) | mean((conf - outcome)^2) | 25% | <0.15 | <0.20 | <0.25 | ≥0.30 |

**Data Dependencies:** TradeJournal `_StrategyStats` (win_rate, profit_factor, avg_r); signal confidence vs actual outcome.

**Home Display:** Single number + green/amber/red badge. Tap for breakdown.

#### Score 2: Execution Quality (Weight: 25%)

*"Are we getting good fills?"*

| Metric | Formula | Weight | 10 | 7 | 4 | 0 |
|--------|---------|--------|------|------|------|------|
| Slippage (bps) | abs(fill_price - signal_price) / signal_price × 10000 | 35% | <2 | <5 | <10 | ≥15 |
| Fill Rate | filled_orders / submitted_orders | 25% | ≥98% | ≥95% | ≥90% | <85% |
| Adverse Selection (1min) | price_move_against_us_1min_post_fill | 20% | <1bps | <3bps | <5bps | ≥8bps |
| Management Efficiency | actual_pnl / mfe | 20% | ≥70% | ≥55% | ≥40% | <25% |

**Data Dependencies:** ExecutionQualityTracker (slippage, fill_rate, adverse_selection); TradeRecord.management_efficiency.

**Home Display:** Single number. Tap for per-trade slippage chart.

#### Score 3: Risk Discipline (Weight: 25%)

*"Are we staying within our risk envelope?"*

| Metric | Formula | Weight | 10 | 7 | 4 | 0 |
|--------|---------|--------|------|------|------|------|
| Drawdown vs Limit | current_dd / max_dd_limit | 30% | <25% | <50% | <75% | ≥90% |
| Position Sizing Adherence | actual_size / planned_size deviation | 25% | <5% | <10% | <20% | ≥30% |
| Circuit Breaker Trips Today | count | 20% | 0 | 1 | 2 | ≥3 |
| Governance Override Rate | overridden_decisions / total_decisions | 25% | 0% | <5% | <10% | ≥15% |

**Data Dependencies:** DrawdownMonitor (current_dd); CircuitBreaker (trip count); GovernanceGate (decisions).

**Home Display:** Single number. Green unless <7 (amber) or <4 (red).

#### Score 4: Operational Integrity (Weight: 20%)

*"Is the platform healthy and auditable?"*

| Metric | Formula | Weight | 10 | 7 | 4 | 0 |
|--------|---------|--------|------|------|------|------|
| Agent Health | healthy_agents / total_agents | 30% | 100% | ≥83% | ≥67% | <50% |
| Data Freshness | max(staleness_seconds) across symbols | 25% | <30s | <60s | <120s | ≥300s |
| Event Bus DLQ | dead_letter_count | 20% | 0 | ≤5 | ≤20 | >20 |
| Reconciliation Drift | max(drift_pct) across positions | 25% | 0% | <1% | <3% | ≥5% |

**Data Dependencies:** AgentRegistry.health_check_all(); FeedManager staleness; EventBus.dead_letters; ReconciliationLoop.

**Home Display:** Single number. This is the "operational heartbeat."

### 7.2 Per-Trade Measurement

Every closed trade in the Journal receives these computed fields:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Slippage (bps)** | `abs(fill_price - signal_price) / signal_price * 10000` | Execution cost |
| **Slippage vs Benchmark** | `(fill_price - arrival_price) / arrival_price * 10000` where arrival = price at signal time | Implementation shortfall |
| **Fill Quality** | `1.0 - (slippage_bps / expected_slippage_bps)` | Fill vs model expectation |
| **Adverse Selection (1min)** | `(price_1min_post_fill - fill_price) * direction_sign / fill_price * 10000` | Were we picked off? |
| **Adverse Selection (5min)** | Same formula, 5min window | Longer-term adverse selection |
| **Management Efficiency** | `net_pnl / mfe` (already computed) | Exit quality |
| **R-Multiple** | `net_pnl / initial_risk_amount` (already computed) | Risk-adjusted return |
| **MAE/MFE** | Min/max unrealized PnL (already computed) | Trade lifecycle quality |
| **Hold Duration** | `closed_at - opened_at` | Are we holding too long? |
| **Venue Latency** | `fill_timestamp - submit_timestamp` | Exchange speed |

---

## 8) P0/P1/P2 DELIVERY PLAN

### P0 — Foundation (Weeks 1-4)

*The minimum changes that create institutional credibility.*

| Epic | Stories | Acceptance Criteria | Week |
|------|---------|-------------------|------|
| **E1: Event Envelope Upgrade** | Add `parent_event_id`, `correlation_id`, `schema_version` to BaseEvent | All existing tests pass. New fields default to empty/1. Signal→Intent→Ack→Fill chain linked via parent_event_id in integration test. | 1 |
| **E2: ToolGateway** | Create ToolGateway wrapping IExchangeAdapter; emit ToolCallRecorded | Every `submit_order` goes through gateway. Idempotency key checked. ToolCallRecorded events appear in event bus log. | 1-2 |
| **E3: Execution Quality Tracker** | Compute slippage, fill rate, adverse selection per fill | Slippage metric emitted on every fill. Per-trade slippage visible in Journal. | 2 |
| **E4: PolicyEvaluated Event** | GovernanceGate emits structured PolicyEvaluated with decision steps | Every governance evaluation produces PolicyEvaluated event with step-by-step audit trail. | 2 |
| **E5: Strategy Lifecycle SM** | StrategyLifecycleManager with state machine + evidence gates | Strategy can be promoted PAPER→LIMITED only with 100+ trades + score ≥ 60. Demotion auto-triggers on drawdown > 10%. MaturityTransition events emitted. | 3 |
| **E6: Daily Scorecard** | 4-score effectiveness scorecard computed from journal + metrics | Scorecard computed on trade close + hourly. Scores queryable via API. | 3-4 |
| **E7: Wire AgentOrchestrator** | Replace ad-hoc agent creation in main.py with orchestrator | Single lifecycle manager. Agents start in dependency order, stop in reverse. Health cascade works. | 4 |
| **E8: Approval Persistence** | Persist ApprovalRequest to Postgres (not in-memory) | Approvals survive restart. Pending approvals queryable after recovery. | 2 |

### P1 — Hardening (Weeks 5-8)

| Epic | Stories | Week |
|------|---------|------|
| **E9: Incident Manager** | IncidentManager agent with state machine, DegradedMode transitions | 5 |
| **E10: Portfolio Constraints** | PortfolioConstraintChecker (max correlation, max gross exposure hard limit) | 5-6 |
| **E11: Data Quality Checker** | OHLCV validation at FeedManager output; quality score on FeatureVector | 6 |
| **E12: Schema Evolution** | Migration registry; `_schema_version` in Redis envelope; backward-compat tests | 6-7 |
| **E13: Supervision API** | FastAPI read-only endpoints + approval actions for UI | 7-8 |
| **E14: Supervision UI v1** | 4-tab web UI: Home, Strategies, Activity, Settings (read-only + approvals) | 7-8 |

### P2 — Polish (Weeks 9-12)

| Epic | Stories | Week |
|------|---------|------|
| **E15: Surveillance Cases** | SurveillanceCase model + basic lifecycle (Open→Investigating→Resolved) | 9 |
| **E16: Event Replay** | Export/import event streams; replay mode for debugging | 9-10 |
| **E17: Walk-Forward Validator** | Automated walk-forward analysis; drift detection vs backtest baseline | 10-11 |
| **E18: Benchmark Tracking** | Alpha/beta vs buy-and-hold; information ratio on Home screen | 11 |
| **E19: UI Polish** | Keyboard shortcuts, dark mode, notification preferences, mobile responsive | 12 |

### What to Cut / Defer

- **Multi-tenancy**: No demand yet; single operator
- **Full OpenTelemetry**: Causation chain on events gives 80% of tracing value
- **ML strategy selection**: Need 6+ months of quality data first
- **Complex portfolio optimization**: Current sizing is adequate
- **Voice narration UI**: Bloomberg Presenter exists but UI integration is P3

### Sprint 1 Plan (Weeks 1-2)

| Day | Task | Owner | DoD |
|-----|------|-------|-----|
| D1 | Add `parent_event_id`, `correlation_id`, `schema_version` to BaseEvent | Dev | Tests pass, no regressions |
| D1 | Update OrderIntent to set `parent_event_id` = Signal.event_id in PortfolioManager | Dev | Integration test: signal→intent linked |
| D2 | Update ExecutionEngine to propagate `parent_event_id` through OrderAck, FillEvent | Dev | Full chain traceable in test |
| D2 | Update GovernanceGate to propagate `correlation_id` | Dev | Governance decision carries correlation |
| D3-D4 | Create ToolGateway class wrapping IExchangeAdapter | Dev | Unit tests: idempotency, logging, policy check |
| D5 | Wire ToolGateway into ExecutionEngine (replace direct adapter calls) | Dev | E2E: order goes through gateway, ToolCallRecorded emitted |
| D6-D7 | Create ExecutionQualityTracker | Dev | Unit tests: slippage calc, adverse selection, fill rate |
| D8 | Wire tracker into ExecutionEngine.handle_fill() | Dev | Integration: fill produces quality metrics |
| D9 | Create PolicyEvaluated event; update GovernanceGate.evaluate() to emit it | Dev | Every governance eval produces structured PolicyEvaluated |
| D10 | Persist ApprovalRequests to Postgres | Dev | Approvals survive restart; integration test |

### Testing Plan

| Level | Scope | Tools | Cadence |
|-------|-------|-------|---------|
| **Unit** | Every new class (ToolGateway, ExecutionQualityTracker, StrategyLifecycleManager, IncidentManager, DailyScorecard) | pytest, mock adapters | Per-commit |
| **Integration** | Full pipeline: Signal → PolicyEval → ToolGateway → Fill → Quality metrics | pytest + MemoryEventBus + PaperAdapter | Per-PR |
| **Simulation** | Paper mode E2E: 24h run, verify scorecard, verify approvals persist, verify incident lifecycle | Docker compose + Redis + Postgres | Weekly |
| **Replay** | Record live events → replay through new code → compare outputs | (after E16) Event replay tool | Post-E16 |
| **Property** | ToolGateway idempotency (N duplicate calls → 1 submission); lifecycle SM (invalid transitions rejected) | Hypothesis | Per-commit for SMs |

---

## 9) CODEBASE REVIEW REQUEST

### Files I've Already Reviewed (Comprehensive)

All core modules have been reviewed through the exploration agents. Below are files/data I'd want to validate specific findings.

### Files Needed for Deeper Review

| File/Module | Why |
|-------------|-----|
| `src/agentic_trading/execution/order_manager.py` | Confirm idempotency mechanism (dedupe_key set + TTL). Need to verify if the seen-keys set is bounded or grows unbounded. |
| `src/agentic_trading/governance/drift_detector.py` | Confirm drift thresholds and how baseline metrics are established. Needed for strategy demotion triggers. |
| `src/agentic_trading/observability/decision_audit.py` | Discovered this file exists — need to understand what audit infrastructure is already in place before building new. |
| `src/agentic_trading/observability/health.py` | Need to understand existing health probe infrastructure. |
| `src/agentic_trading/optimizer/scheduler.py` | Confirm walk-forward support and how results feed back into strategy evaluation. |
| `src/agentic_trading/storage/postgres/models.py` | Need DB schema to plan approval persistence and event store tables. |
| `src/agentic_trading/strategies/research/experiment_log.py` | Discovered during exploration — may have strategy lifecycle tracking already. |

### Logs / Traces Needed

| Data | Format | Purpose |
|------|--------|---------|
| Sample paper trading session log (30 min) | Structured JSON (from `ObservabilityConfig.log_format=json`) | Verify what's logged today; identify gaps in event correlation |
| Sample Grafana dashboard screenshot | PNG / PDF | Verify current monitoring state; identify what's redundant vs missing |
| Sample Redis Streams dump (10 messages per topic) | `XRANGE topic - + COUNT 10` output | Verify actual event envelope format in production |

### Config Samples Needed

| Config | Purpose |
|--------|---------|
| Production `live.toml` (redacted) | Verify risk limits, governance thresholds, strategy params |
| Policy rules file (if persisted to disk) | Verify current policy rule set |
| Docker compose health check config | Verify what's already monitored at infra level |

---

## ASSUMPTIONS

1. **Assumption: Single operator.** The platform is run by one person or small team. Multi-tenancy is not needed now.
2. **Assumption: Bybit primary exchange.** CCXTAdapter targets Bybit; no multi-venue routing needed yet.
3. **Assumption: Web UI served locally.** Supervision UI runs as a FastAPI app alongside the trading process, accessed via browser on the same machine or LAN. No public internet exposure.
4. **Assumption: Postgres already in Docker stack.** The docker-compose.yml includes Postgres; we can add tables without new infrastructure.
5. **Assumption: 15 strategies max.** Based on registry scan. UI and scorecard designed for this scale.
6. **Assumption: ~100 trades/day max.** Based on 1-minute candle cadence with cooldowns. Activity timeline designed for this volume.
7. **Assumption: No regulatory reporting requirement.** No MiFID/SEC reporting needed. Audit trail is for internal governance, not compliance filing.

---

## FOLLOW-UP QUESTIONS

1. **Approval workflow in practice:** Have you ever used the approval workflow in paper/live trading, or is it currently always auto-approved? This affects P0 priority of approval persistence.

2. **Grafana usage:** Do you actively use the Grafana dashboards daily, or would the new supervision UI fully replace them? This determines whether we maintain Grafana dashboard configs.

3. **Strategy count trajectory:** Do you plan to run 5 strategies or 50? The UI and scorecard aggregation approach differs at scale.

4. **UI technology preference:** FastAPI + HTMX (server-rendered, minimal JS) vs. React/Next.js (richer interactivity, more build complexity)? I'd recommend HTMX for simplicity given the read-heavy, low-interactivity nature of supervision.

5. **Event store budget:** Should we persist all events to Postgres (enables replay but storage grows), or only governance/execution events (cheaper, covers audit needs)?

6. **Live trading timeline:** Are you currently running paper only, or already live on Bybit? This determines urgency of ToolGateway and incident management.

---

## APPENDIX A: IMPLEMENTATION-READY PSEUDO-CODE

### A.1 ToolGateway

**File:** `src/agentic_trading/execution/tool_gateway.py`

**Responsibility:** Wraps `IExchangeAdapter`, intercepting every side-effect call to log it, enforce idempotency, re-check policy, and emit `ToolCallRecorded` events.

```python
"""Tool Gateway — audited boundary for all exchange adapter calls.

Every agent side-effect (order submission, cancellation, leverage change)
must go through this gateway.  It provides:
  - Call logging (every method invocation recorded as ToolCallRecorded event)
  - Idempotency enforcement (dedupe token checked before submission)
  - Policy re-validation (belt-and-suspenders with GovernanceGate)
  - Rate limiting (optional, configurable per adapter method)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from decimal import Decimal
from typing import Any

from agentic_trading.core.events import (
    BaseEvent,
    OrderAck,
    OrderIntent,
)
from agentic_trading.core.interfaces import IEventBus, IExchangeAdapter
from agentic_trading.core.models import Balance, Instrument, Order, Position

logger = logging.getLogger(__name__)


class ToolGateway:
    """Audited boundary wrapping IExchangeAdapter.

    Parameters
    ----------
    adapter:
        The real exchange adapter (PaperAdapter, CCXTAdapter, etc.)
    event_bus:
        Event bus for publishing ToolCallRecorded events.
    policy_engine:
        Optional PolicyEngine for belt-and-suspenders re-check.
    max_idempotency_keys:
        Max keys retained in the idempotency cache (LRU eviction).
    idempotency_ttl_seconds:
        TTL for idempotency keys (not time-based eviction in v1,
        just bounded by max_idempotency_keys).
    """

    def __init__(
        self,
        adapter: IExchangeAdapter,
        event_bus: IEventBus,
        policy_engine: Any = None,
        max_idempotency_keys: int = 10_000,
    ) -> None:
        self._adapter = adapter
        self._event_bus = event_bus
        self._policy_engine = policy_engine
        self._idempotency_cache: OrderedDict[str, OrderAck] = OrderedDict()
        self._max_keys = max_idempotency_keys

    # ------------------------------------------------------------------
    # Order submission (primary audited path)
    # ------------------------------------------------------------------

    async def submit_order(
        self,
        intent: OrderIntent,
        *,
        agent_id: str = "",
        trace_id: str = "",
    ) -> OrderAck:
        """Submit an order through the audited gateway.

        Steps:
            1. Check idempotency cache → return cached ack if duplicate
            2. (Optional) Re-check policy engine
            3. Call adapter.submit_order()
            4. Cache the result for idempotency
            5. Emit ToolCallRecorded event
        """
        idem_key = intent.dedupe_key
        t0 = time.monotonic()

        # 1. Idempotency check
        if idem_key in self._idempotency_cache:
            cached_ack = self._idempotency_cache[idem_key]
            logger.info(
                "ToolGateway idempotency hit: dedupe_key=%s order_id=%s",
                idem_key,
                cached_ack.order_id,
            )
            await self._emit_tool_call(
                tool_name="submit_order",
                agent_id=agent_id,
                idem_key=idem_key,
                request_summary=self._summarize_intent(intent),
                response_summary={"cached": True, "order_id": cached_ack.order_id},
                latency_ms=0.0,
                success=True,
                trace_id=trace_id,
            )
            return cached_ack

        # 2. Policy re-check (belt-and-suspenders)
        policy_passed = True
        if self._policy_engine is not None:
            try:
                policy_context = {
                    "strategy_id": intent.strategy_id,
                    "symbol": intent.symbol,
                    "order_notional_usd": float(intent.qty * (intent.price or Decimal("0"))),
                }
                for set_id in self._policy_engine.registered_sets:
                    result = self._policy_engine.evaluate(set_id, policy_context)
                    if not result.all_passed:
                        policy_passed = False
                        break
            except Exception:
                logger.warning("ToolGateway policy re-check failed", exc_info=True)

        # 3. Submit via adapter
        try:
            ack = await self._adapter.submit_order(intent)
            latency_ms = (time.monotonic() - t0) * 1000
            success = True
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            success = False
            await self._emit_tool_call(
                tool_name="submit_order",
                agent_id=agent_id,
                idem_key=idem_key,
                request_summary=self._summarize_intent(intent),
                response_summary={"error": str(exc)},
                latency_ms=latency_ms,
                success=False,
                policy_passed=policy_passed,
                trace_id=trace_id,
            )
            raise

        # 4. Cache for idempotency
        self._idempotency_cache[idem_key] = ack
        if len(self._idempotency_cache) > self._max_keys:
            self._idempotency_cache.popitem(last=False)  # LRU eviction

        # 5. Emit audit event
        await self._emit_tool_call(
            tool_name="submit_order",
            agent_id=agent_id,
            idem_key=idem_key,
            request_summary=self._summarize_intent(intent),
            response_summary={
                "order_id": ack.order_id,
                "status": ack.status.value,
            },
            latency_ms=latency_ms,
            success=True,
            policy_passed=policy_passed,
            trace_id=trace_id,
        )

        return ack

    # ------------------------------------------------------------------
    # Pass-through methods (logged but not idempotency-checked)
    # ------------------------------------------------------------------

    async def cancel_order(
        self, order_id: str, symbol: str, *, agent_id: str = "",
    ) -> OrderAck:
        t0 = time.monotonic()
        try:
            ack = await self._adapter.cancel_order(order_id, symbol)
            await self._emit_tool_call(
                tool_name="cancel_order",
                agent_id=agent_id,
                idem_key=f"cancel-{order_id}",
                request_summary={"order_id": order_id, "symbol": symbol},
                response_summary={"status": ack.status.value},
                latency_ms=(time.monotonic() - t0) * 1000,
                success=True,
            )
            return ack
        except Exception as exc:
            await self._emit_tool_call(
                tool_name="cancel_order",
                agent_id=agent_id,
                idem_key=f"cancel-{order_id}",
                request_summary={"order_id": order_id, "symbol": symbol},
                response_summary={"error": str(exc)},
                latency_ms=(time.monotonic() - t0) * 1000,
                success=False,
            )
            raise

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Pass-through to adapter (read-only, no audit event)."""
        return await self._adapter.get_positions(symbol)

    async def get_balances(self) -> list[Balance]:
        """Pass-through to adapter (read-only, no audit event)."""
        return await self._adapter.get_balances()

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Pass-through to adapter (read-only, no audit event)."""
        return await self._adapter.get_open_orders(symbol)

    async def get_instrument(self, symbol: str) -> Instrument:
        """Pass-through to adapter (read-only, no audit event)."""
        return await self._adapter.get_instrument(symbol)

    async def get_funding_rate(self, symbol: str) -> Decimal:
        """Pass-through to adapter (read-only, no audit event)."""
        return await self._adapter.get_funding_rate(symbol)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _emit_tool_call(
        self,
        tool_name: str,
        agent_id: str,
        idem_key: str,
        request_summary: dict[str, Any],
        response_summary: dict[str, Any],
        latency_ms: float,
        success: bool,
        policy_passed: bool = True,
        trace_id: str = "",
    ) -> None:
        """Publish a ToolCallRecorded event."""
        from agentic_trading.core.events import ToolCallRecorded

        event = ToolCallRecorded(
            tool_name=tool_name,
            agent_id=agent_id,
            adapter_type=self._adapter.__class__.__name__,
            idempotency_key=idem_key,
            request_summary=request_summary,
            response_summary=response_summary,
            latency_ms=round(latency_ms, 2),
            success=success,
            policy_check_passed=policy_passed,
            trace_id=trace_id,
        )
        try:
            await self._event_bus.publish("governance", event)
        except Exception:
            logger.debug("Failed to publish ToolCallRecorded", exc_info=True)

    @staticmethod
    def _summarize_intent(intent: OrderIntent) -> dict[str, Any]:
        return {
            "symbol": intent.symbol,
            "side": intent.side.value,
            "qty": str(intent.qty),
            "order_type": intent.order_type.value,
            "price": str(intent.price) if intent.price else None,
            "strategy_id": intent.strategy_id,
        }
```

**Key Design Decisions:**
- Read-only methods (`get_positions`, `get_balances`, etc.) are pass-through — no audit event overhead for high-frequency reads.
- Write methods (`submit_order`, `cancel_order`) get full audit trail.
- Idempotency cache is LRU-bounded (`OrderedDict` with `popitem`), solving the `OrderManager._seen_keys` unbounded growth issue.
- Policy re-check is optional belt-and-suspenders (GovernanceGate already checks).

---

### A.2 ExecutionQualityTracker

**File:** `src/agentic_trading/execution/quality_tracker.py`

**Responsibility:** Computes per-fill execution quality metrics (slippage, adverse selection, fill rate). Called from `ExecutionEngine.handle_fill()`.

```python
"""Execution quality measurement per fill.

Computes:
  - Slippage: abs(fill_price - signal_price) / signal_price in bps
  - Fill rate: filled_orders / submitted_orders (rolling window)
  - Adverse selection: price move against us within 1min and 5min post-fill
  - Venue latency: fill_timestamp - submit_timestamp

Metrics are emitted to Prometheus and stored per-trade in the journal.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionQualityTracker:
    """Tracks execution quality metrics per fill.

    Parameters
    ----------
    window_size:
        Number of recent fills to keep for rolling metrics.
    """

    def __init__(self, window_size: int = 500) -> None:
        self._window_size = window_size
        self._fills: deque[FillQuality] = deque(maxlen=window_size)
        self._submitted_count: int = 0
        self._filled_count: int = 0

    def record_submission(self) -> None:
        """Called when an order is submitted (for fill rate calculation)."""
        self._submitted_count += 1

    def record_fill(
        self,
        fill_price: Decimal,
        fill_qty: Decimal,
        signal_price: Decimal | None,
        side: str,
        symbol: str,
        strategy_id: str,
        submit_timestamp: datetime | None = None,
        fill_timestamp: datetime | None = None,
    ) -> FillQuality:
        """Record a fill and compute quality metrics.

        Returns FillQuality with computed slippage, venue latency.
        Adverse selection is computed later via update_post_fill_price().
        """
        self._filled_count += 1
        now = datetime.now(timezone.utc)

        # Slippage (bps)
        slippage_bps = 0.0
        if signal_price and signal_price > 0:
            slippage_bps = (
                float(abs(fill_price - signal_price))
                / float(signal_price)
                * 10_000
            )

        # Venue latency
        venue_latency_ms = None
        if submit_timestamp and fill_timestamp:
            venue_latency_ms = (
                fill_timestamp - submit_timestamp
            ).total_seconds() * 1000

        quality = FillQuality(
            symbol=symbol,
            strategy_id=strategy_id,
            side=side,
            fill_price=fill_price,
            signal_price=signal_price,
            fill_qty=fill_qty,
            slippage_bps=round(slippage_bps, 2),
            venue_latency_ms=venue_latency_ms,
            filled_at=fill_timestamp or now,
        )
        self._fills.append(quality)

        # Emit Prometheus metrics
        self._emit_metrics(quality)

        return quality

    def update_post_fill_price(
        self,
        symbol: str,
        filled_at: datetime,
        price_1min: float | None = None,
        price_5min: float | None = None,
    ) -> None:
        """Update adverse selection metrics for a recent fill.

        Called when post-fill candle data becomes available.
        """
        for fq in reversed(self._fills):
            if fq.symbol == symbol and fq.filled_at == filled_at:
                if price_1min is not None and fq.fill_price:
                    direction = 1.0 if fq.side == "buy" else -1.0
                    fq.adverse_selection_1min_bps = round(
                        (price_1min - float(fq.fill_price))
                        * direction
                        / float(fq.fill_price)
                        * -10_000,  # Negative = adverse
                        2,
                    )
                if price_5min is not None and fq.fill_price:
                    direction = 1.0 if fq.side == "buy" else -1.0
                    fq.adverse_selection_5min_bps = round(
                        (price_5min - float(fq.fill_price))
                        * direction
                        / float(fq.fill_price)
                        * -10_000,
                        2,
                    )
                break

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    @property
    def fill_rate(self) -> float:
        """Rolling fill rate: filled / submitted."""
        if self._submitted_count == 0:
            return 1.0
        return self._filled_count / self._submitted_count

    @property
    def avg_slippage_bps(self) -> float:
        """Average slippage in bps across recent fills."""
        if not self._fills:
            return 0.0
        return sum(f.slippage_bps for f in self._fills) / len(self._fills)

    def get_strategy_metrics(self, strategy_id: str) -> dict[str, float]:
        """Get per-strategy execution quality metrics."""
        fills = [f for f in self._fills if f.strategy_id == strategy_id]
        if not fills:
            return {"slippage_bps": 0.0, "fill_count": 0}
        return {
            "slippage_bps": sum(f.slippage_bps for f in fills) / len(fills),
            "fill_count": len(fills),
            "avg_venue_latency_ms": (
                sum(f.venue_latency_ms or 0 for f in fills) / len(fills)
            ),
        }

    # ------------------------------------------------------------------
    # Prometheus emission
    # ------------------------------------------------------------------

    def _emit_metrics(self, quality: "FillQuality") -> None:
        try:
            from agentic_trading.observability.metrics import (
                record_execution_slippage,
            )
            record_execution_slippage(
                quality.symbol,
                quality.strategy_id,
                quality.slippage_bps,
            )
        except Exception:
            pass


class FillQuality:
    """Per-fill quality measurement."""

    __slots__ = (
        "symbol", "strategy_id", "side", "fill_price", "signal_price",
        "fill_qty", "slippage_bps", "venue_latency_ms",
        "adverse_selection_1min_bps", "adverse_selection_5min_bps",
        "filled_at",
    )

    def __init__(
        self,
        symbol: str,
        strategy_id: str,
        side: str,
        fill_price: Decimal,
        signal_price: Decimal | None,
        fill_qty: Decimal,
        slippage_bps: float,
        venue_latency_ms: float | None,
        filled_at: datetime,
    ) -> None:
        self.symbol = symbol
        self.strategy_id = strategy_id
        self.side = side
        self.fill_price = fill_price
        self.signal_price = signal_price
        self.fill_qty = fill_qty
        self.slippage_bps = slippage_bps
        self.venue_latency_ms = venue_latency_ms
        self.adverse_selection_1min_bps: float | None = None
        self.adverse_selection_5min_bps: float | None = None
        self.filled_at = filled_at
```

---

### A.3 StrategyLifecycleManager

**File:** `src/agentic_trading/governance/strategy_lifecycle.py`

**Responsibility:** Enforced state machine for strategy promotion/demotion with evidence gates. Extends `BaseAgent`.

```python
"""Strategy lifecycle state machine with evidence-gated promotion.

Manages strategy progression through:
  Candidate → Backtest → EvalPack → Paper → Limited → Scale

Demotion can happen at any live stage when automated triggers fire.
All transitions emit MaturityTransition events and require evidence.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import (
    AgentType,
    GovernanceAction,
    MaturityLevel,
    StrategyStage,
)
from agentic_trading.core.events import (
    AgentCapabilities,
    BaseEvent,
    MaturityTransition,
)
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


# Evidence thresholds (configurable via StrategyLifecycleConfig)
DEFAULT_EVIDENCE_GATES = {
    StrategyStage.BACKTEST: {
        # CANDIDATE → BACKTEST: backtest must be submitted
    },
    StrategyStage.EVAL_PACK: {
        "min_trades": 50,
        "min_sharpe": 0.5,
        "min_profit_factor": 1.1,
        "max_drawdown_pct": 20.0,
    },
    StrategyStage.PAPER: {
        # Requires operator approval (L2_OPERATOR)
    },
    StrategyStage.LIMITED: {
        "min_trades": 100,
        "min_quality_score": 60,
        "max_sharpe_drift_sigma": 1.0,  # live Sharpe within 1σ of backtest
    },
    StrategyStage.SCALE: {
        "min_days_at_limited": 30,
        "min_quality_score": 70,
        "max_incidents": 0,
        # Requires operator approval (L3_RISK)
    },
}

# Demotion triggers (any one fires → demote)
DEFAULT_DEMOTION_TRIGGERS = {
    "max_drawdown_pct": 10.0,
    "max_loss_streak": 10,
    "min_quality_score": 50,
    "max_drift_pct": 30.0,
}

# Mapping: StrategyStage → MaturityLevel
STAGE_TO_MATURITY = {
    StrategyStage.CANDIDATE: MaturityLevel.L0_SHADOW,
    StrategyStage.BACKTEST: MaturityLevel.L0_SHADOW,
    StrategyStage.EVAL_PACK: MaturityLevel.L1_PAPER,
    StrategyStage.PAPER: MaturityLevel.L2_GATED,
    StrategyStage.LIMITED: MaturityLevel.L3_CONSTRAINED,
    StrategyStage.SCALE: MaturityLevel.L4_AUTONOMOUS,
    StrategyStage.DEMOTED: MaturityLevel.L1_PAPER,
}


class StrategyLifecycleManager(BaseAgent):
    """Manages strategy lifecycle with evidence-gated promotion.

    Periodic agent that:
      1. Subscribes to journal close events to collect evidence
      2. Checks promotion eligibility on each cycle
      3. Checks demotion triggers on each cycle
      4. Emits MaturityTransition events on state changes
    """

    def __init__(
        self,
        event_bus: IEventBus,
        journal: Any,
        governance_gate: Any,
        *,
        evidence_gates: dict | None = None,
        demotion_triggers: dict | None = None,
        interval: float = 60.0,  # Check every 60s
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, interval=interval)
        self._event_bus = event_bus
        self._journal = journal
        self._governance_gate = governance_gate
        self._evidence_gates = evidence_gates or DEFAULT_EVIDENCE_GATES
        self._demotion_triggers = demotion_triggers or DEFAULT_DEMOTION_TRIGGERS

        # strategy_id → current stage
        self._stages: dict[str, StrategyStage] = {}
        # strategy_id → promotion timestamp
        self._promotion_history: dict[str, list[dict]] = {}

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CUSTOM  # Could add STRATEGY_LIFECYCLE to enum

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["execution"],
            publishes_to=["governance"],
            description="Strategy lifecycle state machine with evidence-gated promotion",
        )

    def register_strategy(
        self,
        strategy_id: str,
        initial_stage: StrategyStage = StrategyStage.CANDIDATE,
    ) -> None:
        """Register a strategy with its initial lifecycle stage."""
        self._stages[strategy_id] = initial_stage
        self._promotion_history.setdefault(strategy_id, [])
        logger.info(
            "Strategy %s registered at stage %s",
            strategy_id,
            initial_stage.value,
        )

    def get_stage(self, strategy_id: str) -> StrategyStage | None:
        return self._stages.get(strategy_id)

    # ------------------------------------------------------------------
    # Periodic work
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Check all strategies for promotion eligibility and demotion triggers."""
        for strategy_id, stage in list(self._stages.items()):
            # Skip non-live stages
            if stage in (StrategyStage.CANDIDATE, StrategyStage.BACKTEST):
                continue

            # Check demotion triggers for live strategies
            if stage in (
                StrategyStage.PAPER,
                StrategyStage.LIMITED,
                StrategyStage.SCALE,
            ):
                evidence = self._collect_evidence(strategy_id)
                if self._should_demote(strategy_id, evidence):
                    await self._transition(
                        strategy_id,
                        StrategyStage.DEMOTED,
                        reason="automated_demotion_trigger",
                        evidence=evidence,
                    )

    # ------------------------------------------------------------------
    # Promotion (called externally or via API)
    # ------------------------------------------------------------------

    async def request_promotion(
        self,
        strategy_id: str,
        *,
        operator_id: str = "",
    ) -> dict[str, Any]:
        """Request promotion to the next stage.

        Returns dict with 'approved' bool and 'reason' string.
        """
        current = self._stages.get(strategy_id)
        if current is None:
            return {"approved": False, "reason": "strategy_not_registered"}

        next_stage = self._next_stage(current)
        if next_stage is None:
            return {"approved": False, "reason": "already_at_max_stage"}

        evidence = self._collect_evidence(strategy_id)
        gate = self._evidence_gates.get(next_stage, {})

        # Check evidence requirements
        for metric, threshold in gate.items():
            actual = evidence.get(metric)
            if actual is None:
                return {
                    "approved": False,
                    "reason": f"missing_evidence: {metric}",
                }
            if metric.startswith("min_") and actual < threshold:
                return {
                    "approved": False,
                    "reason": f"{metric}={actual} < {threshold}",
                }
            if metric.startswith("max_") and actual > threshold:
                return {
                    "approved": False,
                    "reason": f"{metric}={actual} > {threshold}",
                }

        await self._transition(
            strategy_id,
            next_stage,
            reason=f"promoted_by_{operator_id or 'system'}",
            evidence=evidence,
        )
        return {"approved": True, "reason": "all_evidence_gates_passed"}

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    def _collect_evidence(self, strategy_id: str) -> dict[str, float]:
        """Collect current metrics from journal and drift detector."""
        evidence: dict[str, float] = {}
        try:
            stats = self._journal.get_strategy_stats(strategy_id)
            if stats:
                evidence["min_trades"] = stats.get("total_trades", 0)
                evidence["min_quality_score"] = stats.get("quality_score", 0)
                evidence["min_sharpe"] = stats.get("sharpe_ratio", 0)
                evidence["min_profit_factor"] = stats.get("profit_factor", 0)
                evidence["max_drawdown_pct"] = stats.get("max_drawdown_pct", 0)
                evidence["max_loss_streak"] = stats.get("loss_streak", 0)
        except Exception:
            logger.debug("Failed to collect evidence for %s", strategy_id)

        # Drift from governance gate
        try:
            drift_status = self._governance_gate.drift.get_status(strategy_id)
            max_dev = 0.0
            for m in drift_status.get("metrics", {}).values():
                if m.get("deviation_pct") is not None:
                    max_dev = max(max_dev, m["deviation_pct"])
            evidence["max_drift_pct"] = max_dev
        except Exception:
            pass

        return evidence

    def _should_demote(
        self, strategy_id: str, evidence: dict[str, float]
    ) -> bool:
        """Check if any demotion trigger fires."""
        for trigger, threshold in self._demotion_triggers.items():
            actual = evidence.get(trigger)
            if actual is None:
                continue
            if trigger.startswith("max_") and actual > threshold:
                logger.warning(
                    "Demotion trigger: %s=%s > %s for %s",
                    trigger, actual, threshold, strategy_id,
                )
                return True
            if trigger.startswith("min_") and actual < threshold:
                logger.warning(
                    "Demotion trigger: %s=%s < %s for %s",
                    trigger, actual, threshold, strategy_id,
                )
                return True
        return False

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def _transition(
        self,
        strategy_id: str,
        new_stage: StrategyStage,
        reason: str,
        evidence: dict | None = None,
    ) -> None:
        old_stage = self._stages.get(strategy_id, StrategyStage.CANDIDATE)
        self._stages[strategy_id] = new_stage

        # Update maturity level in GovernanceGate
        new_maturity = STAGE_TO_MATURITY[new_stage]
        old_maturity = STAGE_TO_MATURITY[old_stage]
        if new_maturity != old_maturity:
            self._governance_gate.maturity.set_level(strategy_id, new_maturity)

        # Record in history
        self._promotion_history.setdefault(strategy_id, []).append({
            "from": old_stage.value,
            "to": new_stage.value,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence": evidence or {},
        })

        # Emit event
        event = MaturityTransition(
            strategy_id=strategy_id,
            from_level=old_maturity,
            to_level=new_maturity,
            reason=f"{old_stage.value} → {new_stage.value}: {reason}",
            metrics_snapshot=evidence or {},
        )
        try:
            await self._event_bus.publish("governance", event)
        except Exception:
            logger.error("Failed to publish MaturityTransition", exc_info=True)

        logger.info(
            "Strategy %s transitioned: %s → %s (%s)",
            strategy_id,
            old_stage.value,
            new_stage.value,
            reason,
        )

    @staticmethod
    def _next_stage(current: StrategyStage) -> StrategyStage | None:
        progression = [
            StrategyStage.CANDIDATE,
            StrategyStage.BACKTEST,
            StrategyStage.EVAL_PACK,
            StrategyStage.PAPER,
            StrategyStage.LIMITED,
            StrategyStage.SCALE,
        ]
        try:
            idx = progression.index(current)
            if idx + 1 < len(progression):
                return progression[idx + 1]
        except ValueError:
            pass
        return None
```

---

### A.4 IncidentManager

**File:** `src/agentic_trading/governance/incident_manager.py`

**Responsibility:** State machine for incident lifecycle (Detect → Triage → Degrade → Recover). Manages `DegradedMode` transitions.

```python
"""Incident lifecycle management with degraded mode transitions.

Subscribes to risk and system topics to detect incidents.
Manages degraded mode state: NORMAL → REDUCE_ONLY → NO_NEW_TRADES → KILLED.
Implements auto-triage rules and recovery criteria.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import (
    AgentType,
    DegradedMode,
    IncidentSeverity,
    IncidentStatus,
)
from agentic_trading.core.events import (
    AgentCapabilities,
    BaseEvent,
    CircuitBreakerEvent,
    KillSwitchEvent,
)
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


# Auto-triage rules: trigger_type → (severity, degraded_mode)
AUTO_TRIAGE_RULES: dict[str, tuple[IncidentSeverity, DegradedMode]] = {
    "circuit_breaker_single": (IncidentSeverity.LOW, DegradedMode.NORMAL),
    "circuit_breaker_daily_loss": (IncidentSeverity.HIGH, DegradedMode.NO_NEW_TRADES),
    "kill_switch": (IncidentSeverity.CRITICAL, DegradedMode.KILLED),
    "canary_unhealthy": (IncidentSeverity.MEDIUM, DegradedMode.REDUCE_ONLY),
    "exchange_disconnected": (IncidentSeverity.HIGH, DegradedMode.NO_NEW_TRADES),
    "reconciliation_drift": (IncidentSeverity.MEDIUM, DegradedMode.REDUCE_ONLY),
}

# Recovery criteria: severity → (cooldown_minutes, requires_approval)
RECOVERY_CRITERIA: dict[IncidentSeverity, tuple[int, bool]] = {
    IncidentSeverity.LOW: (0, False),        # Auto-resume
    IncidentSeverity.MEDIUM: (15, False),     # Auto-resume after cooldown
    IncidentSeverity.HIGH: (60, True),        # Operator approval
    IncidentSeverity.CRITICAL: (240, True),   # L3_RISK approval
}


class Incident:
    """Single incident record."""

    __slots__ = (
        "incident_id", "severity", "status", "trigger_type",
        "trigger_event_id", "description", "degraded_mode",
        "affected_strategies", "affected_symbols",
        "declared_at", "resolved_at", "auto_actions",
    )

    def __init__(
        self,
        incident_id: str,
        severity: IncidentSeverity,
        trigger_type: str,
        trigger_event_id: str,
        description: str,
    ) -> None:
        self.incident_id = incident_id
        self.severity = severity
        self.status = IncidentStatus.DETECTED
        self.trigger_type = trigger_type
        self.trigger_event_id = trigger_event_id
        self.description = description
        self.degraded_mode = DegradedMode.NORMAL
        self.affected_strategies: list[str] = []
        self.affected_symbols: list[str] = []
        self.declared_at = datetime.now(timezone.utc)
        self.resolved_at: datetime | None = None
        self.auto_actions: list[str] = []


class IncidentManager(BaseAgent):
    """Manages incident lifecycle and degraded mode transitions.

    Event-driven agent that:
      1. Subscribes to `risk` and `system` topics for triggers
      2. Auto-triages incidents by severity
      3. Applies degraded mode transitions
      4. Monitors recovery criteria
      5. Emits IncidentDeclared and DegradedModeEnabled events
    """

    def __init__(
        self,
        event_bus: IEventBus,
        *,
        agent_id: str | None = None,
        interval: float = 30.0,  # Check recovery every 30s
    ) -> None:
        super().__init__(agent_id=agent_id, interval=interval)
        self._event_bus = event_bus
        self._current_mode = DegradedMode.NORMAL
        self._active_incidents: dict[str, Incident] = {}
        self._resolved_incidents: list[Incident] = []
        self._incident_counter = 0

    @property
    def agent_type(self) -> AgentType:
        return AgentType.INCIDENT_RESPONSE

    @property
    def current_mode(self) -> DegradedMode:
        return self._current_mode

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=["risk", "system"],
            publishes_to=["governance"],
            description="Incident lifecycle management with degraded mode transitions",
        )

    async def _on_start(self) -> None:
        await self._event_bus.subscribe(
            topic="risk",
            group="incident_manager",
            handler=self._on_risk_event,
        )
        await self._event_bus.subscribe(
            topic="system",
            group="incident_manager",
            handler=self._on_system_event,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_risk_event(self, event: BaseEvent) -> None:
        if isinstance(event, CircuitBreakerEvent) and event.tripped:
            trigger = (
                "circuit_breaker_daily_loss"
                if "daily" in event.reason.lower()
                else "circuit_breaker_single"
            )
            await self.declare_incident(
                trigger_type=trigger,
                trigger_event_id=event.event_id,
                description=f"Circuit breaker tripped: {event.reason}",
                affected_symbols=[event.symbol] if event.symbol else [],
            )
        elif isinstance(event, KillSwitchEvent) and event.activated:
            await self.declare_incident(
                trigger_type="kill_switch",
                trigger_event_id=event.event_id,
                description=f"Kill switch activated: {event.reason}",
            )

    async def _on_system_event(self, event: BaseEvent) -> None:
        # CanaryAlert, SystemHealth events trigger incidents
        from agentic_trading.core.events import CanaryAlert

        if isinstance(event, CanaryAlert) and not event.healthy:
            await self.declare_incident(
                trigger_type="canary_unhealthy",
                trigger_event_id=event.event_id,
                description=f"Canary alert: {event.component} - {event.message}",
            )

    # ------------------------------------------------------------------
    # Incident declaration
    # ------------------------------------------------------------------

    async def declare_incident(
        self,
        trigger_type: str,
        trigger_event_id: str,
        description: str,
        affected_strategies: list[str] | None = None,
        affected_symbols: list[str] | None = None,
    ) -> Incident:
        """Declare a new incident and auto-triage."""
        self._incident_counter += 1
        incident_id = f"inc-{self._incident_counter:04d}"

        severity, degraded_mode = AUTO_TRIAGE_RULES.get(
            trigger_type,
            (IncidentSeverity.MEDIUM, DegradedMode.REDUCE_ONLY),
        )

        incident = Incident(
            incident_id=incident_id,
            severity=severity,
            trigger_type=trigger_type,
            trigger_event_id=trigger_event_id,
            description=description,
        )
        incident.affected_strategies = affected_strategies or []
        incident.affected_symbols = affected_symbols or []
        incident.status = IncidentStatus.TRIAGED

        self._active_incidents[incident_id] = incident

        # Apply degraded mode (escalate only, never de-escalate automatically)
        if degraded_mode.rank > self._current_mode.rank:
            await self._set_degraded_mode(
                degraded_mode, f"Incident {incident_id}: {description}"
            )
            incident.degraded_mode = degraded_mode

        logger.warning(
            "Incident declared: %s severity=%s trigger=%s mode=%s",
            incident_id,
            severity.value,
            trigger_type,
            self._current_mode.value,
        )

        # Emit IncidentDeclared event
        await self._emit_incident_declared(incident)

        return incident

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Periodic check for auto-recovery conditions."""
        now = datetime.now(timezone.utc)
        resolved = []

        for incident_id, incident in self._active_incidents.items():
            cooldown_min, requires_approval = RECOVERY_CRITERIA.get(
                incident.severity, (60, True)
            )
            elapsed_min = (now - incident.declared_at).total_seconds() / 60

            if elapsed_min >= cooldown_min and not requires_approval:
                incident.status = IncidentStatus.RESOLVED
                incident.resolved_at = now
                resolved.append(incident_id)
                logger.info(
                    "Incident %s auto-resolved after %.1f min",
                    incident_id,
                    elapsed_min,
                )

        for incident_id in resolved:
            incident = self._active_incidents.pop(incident_id)
            self._resolved_incidents.append(incident)

        # If no active incidents remain, return to NORMAL
        if not self._active_incidents and self._current_mode != DegradedMode.NORMAL:
            await self._set_degraded_mode(DegradedMode.NORMAL, "All incidents resolved")

    async def resolve_incident(
        self,
        incident_id: str,
        *,
        resolved_by: str = "operator",
    ) -> bool:
        """Manually resolve an incident (for HIGH/CRITICAL requiring approval)."""
        incident = self._active_incidents.get(incident_id)
        if incident is None:
            return False

        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now(timezone.utc)
        self._resolved_incidents.append(
            self._active_incidents.pop(incident_id)
        )

        # Check if we can de-escalate
        if not self._active_incidents:
            await self._set_degraded_mode(
                DegradedMode.NORMAL,
                f"Incident {incident_id} resolved by {resolved_by}",
            )

        return True

    # ------------------------------------------------------------------
    # Degraded mode management
    # ------------------------------------------------------------------

    async def _set_degraded_mode(
        self, mode: DegradedMode, reason: str
    ) -> None:
        previous = self._current_mode
        self._current_mode = mode
        logger.warning(
            "Degraded mode transition: %s → %s (%s)",
            previous.value,
            mode.value,
            reason,
        )

        # Emit DegradedModeEnabled event
        from agentic_trading.core.events import DegradedModeEnabled

        event = DegradedModeEnabled(
            mode=mode.value,
            previous_mode=previous.value,
            reason=reason,
        )
        try:
            await self._event_bus.publish("governance", event)
        except Exception:
            logger.error("Failed to publish DegradedModeEnabled", exc_info=True)

    async def _emit_incident_declared(self, incident: Incident) -> None:
        from agentic_trading.core.events import IncidentDeclared

        event = IncidentDeclared(
            incident_id=incident.incident_id,
            severity=incident.severity.value,
            trigger=incident.trigger_type,
            trigger_event_id=incident.trigger_event_id,
            description=incident.description,
            affected_strategies=incident.affected_strategies,
            affected_symbols=incident.affected_symbols,
        )
        try:
            await self._event_bus.publish("governance", event)
        except Exception:
            logger.error("Failed to publish IncidentDeclared", exc_info=True)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_active_incidents(self) -> list[dict[str, Any]]:
        return [
            {
                "incident_id": i.incident_id,
                "severity": i.severity.value,
                "status": i.status.value,
                "trigger": i.trigger_type,
                "description": i.description,
                "declared_at": i.declared_at.isoformat(),
                "degraded_mode": i.degraded_mode.value,
            }
            for i in self._active_incidents.values()
        ]
```

---

### A.5 DailyEffectivenessScorecard

**File:** `src/agentic_trading/observability/daily_scorecard.py`

**Responsibility:** Computes the 4-score daily effectiveness scorecard (Edge, Execution, Risk, Ops) from journal, metrics, and agent health.

```python
"""Daily Effectiveness Scorecard — 4 scores, 0-10 scale.

Aggregates data from:
  - TradeJournal (edge quality, management efficiency)
  - ExecutionQualityTracker (slippage, fill rate, adverse selection)
  - RiskManager / CircuitBreakers (drawdown, trip count, overrides)
  - AgentRegistry / EventBus (health, freshness, DLQ size)

Each score is weighted to produce a single daily effectiveness number.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Score weights
WEIGHTS = {
    "edge_quality": 0.30,
    "execution_quality": 0.25,
    "risk_discipline": 0.25,
    "operational_integrity": 0.20,
}


class DailyEffectivenessScorecard:
    """Computes and caches the 4 effectiveness scores.

    Call ``compute()`` periodically (e.g., on trade close + hourly).
    """

    def __init__(
        self,
        journal: Any = None,
        quality_tracker: Any = None,
        risk_manager: Any = None,
        agent_registry: Any = None,
        event_bus: Any = None,
    ) -> None:
        self._journal = journal
        self._quality_tracker = quality_tracker
        self._risk_manager = risk_manager
        self._agent_registry = agent_registry
        self._event_bus = event_bus
        self._last_scores: dict[str, float] | None = None

    def compute(self) -> dict[str, float]:
        """Compute all 4 scores and the weighted total.

        Returns dict with keys:
            edge_quality, execution_quality, risk_discipline,
            operational_integrity, total
        """
        edge = self._compute_edge_quality()
        execution = self._compute_execution_quality()
        risk = self._compute_risk_discipline()
        ops = self._compute_operational_integrity()

        total = (
            edge * WEIGHTS["edge_quality"]
            + execution * WEIGHTS["execution_quality"]
            + risk * WEIGHTS["risk_discipline"]
            + ops * WEIGHTS["operational_integrity"]
        )

        self._last_scores = {
            "edge_quality": round(edge, 1),
            "execution_quality": round(execution, 1),
            "risk_discipline": round(risk, 1),
            "operational_integrity": round(ops, 1),
            "total": round(total, 1),
        }

        # Emit Prometheus
        self._emit_metrics()

        return self._last_scores

    @property
    def last_scores(self) -> dict[str, float] | None:
        return self._last_scores

    # ------------------------------------------------------------------
    # Score 1: Edge Quality (0-10)
    # ------------------------------------------------------------------

    def _compute_edge_quality(self) -> float:
        """Win rate vs target, profit factor, avg R, confidence calibration."""
        if self._journal is None:
            return 5.0  # Neutral default

        try:
            # Get aggregate stats from journal
            stats = self._journal.get_aggregate_stats()
            if not stats or stats.get("total_trades", 0) < 5:
                return 5.0  # Not enough data

            # Win rate vs target (assume 0.45 target for trend, 0.60 for MR)
            wr_score = self._scale(stats.get("win_rate", 0) / 0.50, 0.6, 1.2)
            # Profit factor
            pf_score = self._scale(stats.get("profit_factor", 1.0), 1.0, 2.0)
            # Avg R-multiple
            r_score = self._scale(stats.get("avg_r_multiple", 0), 0.0, 0.5)
            # Confidence calibration (lower brier = better)
            brier = stats.get("brier_score", 0.25)
            cal_score = self._scale(1.0 - brier / 0.30, 0.0, 1.0)

            return (wr_score + pf_score + r_score + cal_score) / 4 * 10
        except Exception:
            logger.debug("Edge quality computation failed", exc_info=True)
            return 5.0

    # ------------------------------------------------------------------
    # Score 2: Execution Quality (0-10)
    # ------------------------------------------------------------------

    def _compute_execution_quality(self) -> float:
        """Slippage, fill rate, adverse selection, management efficiency."""
        if self._quality_tracker is None:
            return 5.0

        try:
            slip = self._quality_tracker.avg_slippage_bps
            fill = self._quality_tracker.fill_rate

            slip_score = self._scale(1.0 - slip / 15.0, 0.0, 1.0)
            fill_score = self._scale(fill, 0.85, 1.0)

            return (slip_score + fill_score) / 2 * 10
        except Exception:
            return 5.0

    # ------------------------------------------------------------------
    # Score 3: Risk Discipline (0-10)
    # ------------------------------------------------------------------

    def _compute_risk_discipline(self) -> float:
        """Drawdown vs limit, sizing adherence, CB trips, override rate."""
        # Start at 10.0, deduct for violations
        score = 10.0

        try:
            if self._risk_manager:
                # Drawdown deduction
                dd_pct = getattr(self._risk_manager, "current_drawdown_pct", 0.0)
                dd_limit = getattr(self._risk_manager, "max_drawdown_pct", 0.15)
                if dd_limit > 0:
                    dd_ratio = dd_pct / dd_limit
                    if dd_ratio > 0.75:
                        score -= 4.0
                    elif dd_ratio > 0.50:
                        score -= 2.0
                    elif dd_ratio > 0.25:
                        score -= 1.0

                # Circuit breaker trips
                trips = getattr(self._risk_manager, "circuit_breaker_trips_today", 0)
                score -= min(trips * 2.0, 6.0)
        except Exception:
            pass

        return max(0.0, min(10.0, score))

    # ------------------------------------------------------------------
    # Score 4: Operational Integrity (0-10)
    # ------------------------------------------------------------------

    def _compute_operational_integrity(self) -> float:
        """Agent health, data freshness, DLQ size, reconciliation drift."""
        score = 10.0

        try:
            # Agent health
            if self._agent_registry:
                health = self._agent_registry.health_check_all()
                total = len(health)
                healthy = sum(1 for h in health.values() if h.healthy)
                if total > 0:
                    health_ratio = healthy / total
                    if health_ratio < 0.67:
                        score -= 4.0
                    elif health_ratio < 0.83:
                        score -= 2.0
                    elif health_ratio < 1.0:
                        score -= 1.0

            # Event bus DLQ
            if self._event_bus:
                dlq_size = len(getattr(self._event_bus, "dead_letters", []))
                if dlq_size > 20:
                    score -= 4.0
                elif dlq_size > 5:
                    score -= 2.0
                elif dlq_size > 0:
                    score -= 1.0
        except Exception:
            pass

        return max(0.0, min(10.0, score))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scale(value: float, low: float, high: float) -> float:
        """Scale a value to 0.0-1.0 range between low and high."""
        if high <= low:
            return 0.5
        clamped = max(low, min(high, value))
        return (clamped - low) / (high - low)

    def _emit_metrics(self) -> None:
        if self._last_scores is None:
            return
        try:
            from agentic_trading.observability.metrics import (
                update_effectiveness_score,
            )
            for key, val in self._last_scores.items():
                update_effectiveness_score(key, val)
        except Exception:
            pass
```

---

### A.6 SupervisionAPI

**File:** `src/agentic_trading/api/supervision.py`

**Responsibility:** FastAPI/Starlette read-only endpoints for the supervision UI, plus approval actions.

```python
"""Supervision API — read-only endpoints + approval actions.

Serves the 4-tab supervision UI with data from:
  - TradeJournal (trades, stats, scorecard)
  - AgentRegistry (health, status)
  - GovernanceGate (decisions, maturity, policies)
  - ApprovalManager (pending/resolved approvals)
  - IncidentManager (active/resolved incidents)
  - DailyEffectivenessScorecard (4 scores)
  - StrategyLifecycleManager (stages, promotion history)
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


class SupervisionAPI:
    """Container for all supervision endpoints.

    Holds references to platform components and exposes them via HTTP.
    """

    def __init__(
        self,
        journal: Any = None,
        agent_registry: Any = None,
        governance_gate: Any = None,
        approval_manager: Any = None,
        incident_manager: Any = None,
        scorecard: Any = None,
        lifecycle_manager: Any = None,
        quality_tracker: Any = None,
    ) -> None:
        self._journal = journal
        self._registry = agent_registry
        self._gate = governance_gate
        self._approvals = approval_manager
        self._incidents = incident_manager
        self._scorecard = scorecard
        self._lifecycle = lifecycle_manager
        self._quality = quality_tracker

    def build_app(self) -> Starlette:
        """Build and return the Starlette ASGI application."""
        routes = [
            # ---- HOME TAB ----
            Route("/api/v1/home/summary", self._home_summary),
            Route("/api/v1/home/scorecard", self._home_scorecard),

            # ---- STRATEGIES TAB ----
            Route("/api/v1/strategies", self._strategies_list),
            Route("/api/v1/strategies/{strategy_id}", self._strategy_detail),
            Route(
                "/api/v1/strategies/{strategy_id}/promote",
                self._strategy_promote,
                methods=["POST"],
            ),

            # ---- ACTIVITY TAB ----
            Route("/api/v1/activity/timeline", self._activity_timeline),
            Route("/api/v1/activity/approvals", self._approvals_list),
            Route(
                "/api/v1/activity/approvals/{request_id}/approve",
                self._approval_approve,
                methods=["POST"],
            ),
            Route(
                "/api/v1/activity/approvals/{request_id}/deny",
                self._approval_deny,
                methods=["POST"],
            ),
            Route("/api/v1/activity/incidents", self._incidents_list),
            Route(
                "/api/v1/activity/incidents/{incident_id}/resolve",
                self._incident_resolve,
                methods=["POST"],
            ),

            # ---- SETTINGS TAB ----
            Route("/api/v1/settings/agents", self._agents_status),
            Route("/api/v1/settings/governance", self._governance_summary),

            # ---- HEALTH ----
            Route("/api/v1/health", self._health_check),
        ]
        return Starlette(routes=routes)

    # ------------------------------------------------------------------
    # HOME endpoints
    # ------------------------------------------------------------------

    async def _home_summary(self, request: Request) -> JSONResponse:
        data: dict[str, Any] = {}
        # Portfolio summary
        if self._journal:
            data["portfolio"] = self._journal.get_portfolio_summary()
        # System status
        if self._registry:
            health = self._registry.health_check_all()
            total = len(health)
            healthy = sum(1 for h in health.values() if h.healthy)
            data["system"] = {
                "agents_total": total,
                "agents_healthy": healthy,
                "all_healthy": healthy == total,
            }
        # Pending approvals count
        if self._approvals:
            data["pending_approvals"] = len(self._approvals.get_pending())
        # Active incidents
        if self._incidents:
            data["active_incidents"] = len(
                self._incidents.get_active_incidents()
            )
            data["degraded_mode"] = self._incidents.current_mode.value
        return JSONResponse(data)

    async def _home_scorecard(self, request: Request) -> JSONResponse:
        if self._scorecard is None:
            return JSONResponse({"error": "scorecard not configured"}, 503)
        scores = self._scorecard.compute()
        return JSONResponse(scores)

    # ------------------------------------------------------------------
    # STRATEGY endpoints
    # ------------------------------------------------------------------

    async def _strategies_list(self, request: Request) -> JSONResponse:
        strategies = []
        if self._lifecycle:
            for sid, stage in self._lifecycle._stages.items():
                stats = {}
                if self._journal:
                    stats = self._journal.get_strategy_stats(sid) or {}
                strategies.append({
                    "strategy_id": sid,
                    "stage": stage.value,
                    "quality_score": stats.get("quality_score"),
                    "win_rate": stats.get("win_rate"),
                    "total_trades": stats.get("total_trades", 0),
                    "pnl_today": stats.get("pnl_today", 0),
                })
        return JSONResponse(strategies)

    async def _strategy_detail(self, request: Request) -> JSONResponse:
        sid = request.path_params["strategy_id"]
        detail: dict[str, Any] = {"strategy_id": sid}
        if self._lifecycle:
            detail["stage"] = (self._lifecycle.get_stage(sid) or "unknown")
            if hasattr(detail["stage"], "value"):
                detail["stage"] = detail["stage"].value
            detail["promotion_history"] = (
                self._lifecycle._promotion_history.get(sid, [])
            )
        if self._journal:
            detail["stats"] = self._journal.get_strategy_stats(sid) or {}
            detail["recent_trades"] = self._journal.get_recent_trades(
                sid, limit=10,
            )
        return JSONResponse(detail)

    async def _strategy_promote(self, request: Request) -> JSONResponse:
        sid = request.path_params["strategy_id"]
        if self._lifecycle is None:
            return JSONResponse({"error": "lifecycle not configured"}, 503)
        body = await request.json() if request.headers.get("content-length") else {}
        result = await self._lifecycle.request_promotion(
            sid, operator_id=body.get("operator_id", "api"),
        )
        status = 200 if result["approved"] else 422
        return JSONResponse(result, status)

    # ------------------------------------------------------------------
    # ACTIVITY endpoints
    # ------------------------------------------------------------------

    async def _activity_timeline(self, request: Request) -> JSONResponse:
        limit = int(request.query_params.get("limit", "50"))
        filter_type = request.query_params.get("type", "all")
        # Return recent events from journal
        if self._journal:
            events = self._journal.get_recent_events(
                limit=limit, event_type=filter_type,
            )
            return JSONResponse(events)
        return JSONResponse([])

    async def _approvals_list(self, request: Request) -> JSONResponse:
        if self._approvals is None:
            return JSONResponse([])
        pending = self._approvals.get_pending()
        return JSONResponse([
            {
                "request_id": r.request_id,
                "strategy_id": r.strategy_id,
                "symbol": r.symbol,
                "trigger": r.trigger.value,
                "escalation_level": r.escalation_level.value,
                "notional_usd": r.notional_usd,
                "created_at": r.created_at.isoformat(),
            }
            for r in pending
        ])

    async def _approval_approve(self, request: Request) -> JSONResponse:
        rid = request.path_params["request_id"]
        body = await request.json() if request.headers.get("content-length") else {}
        if self._approvals is None:
            return JSONResponse({"error": "approvals not configured"}, 503)
        success = await self._approvals.approve(
            rid, decided_by=body.get("operator_id", "api"),
        )
        return JSONResponse({"approved": success})

    async def _approval_deny(self, request: Request) -> JSONResponse:
        rid = request.path_params["request_id"]
        body = await request.json() if request.headers.get("content-length") else {}
        if self._approvals is None:
            return JSONResponse({"error": "approvals not configured"}, 503)
        success = await self._approvals.reject(
            rid,
            decided_by=body.get("operator_id", "api"),
            reason=body.get("reason", ""),
        )
        return JSONResponse({"denied": success})

    async def _incidents_list(self, request: Request) -> JSONResponse:
        if self._incidents is None:
            return JSONResponse([])
        return JSONResponse(self._incidents.get_active_incidents())

    async def _incident_resolve(self, request: Request) -> JSONResponse:
        iid = request.path_params["incident_id"]
        body = await request.json() if request.headers.get("content-length") else {}
        if self._incidents is None:
            return JSONResponse({"error": "incidents not configured"}, 503)
        success = await self._incidents.resolve_incident(
            iid, resolved_by=body.get("operator_id", "api"),
        )
        return JSONResponse({"resolved": success})

    # ------------------------------------------------------------------
    # SETTINGS endpoints
    # ------------------------------------------------------------------

    async def _agents_status(self, request: Request) -> JSONResponse:
        if self._registry is None:
            return JSONResponse([])
        health = self._registry.health_check_all()
        return JSONResponse({
            agent_id: {
                "healthy": h.healthy,
                "message": h.message,
                "error_count": h.error_count,
                "last_work_at": (
                    h.last_work_at.isoformat() if h.last_work_at else None
                ),
            }
            for agent_id, h in health.items()
        })

    async def _governance_summary(self, request: Request) -> JSONResponse:
        data: dict[str, Any] = {}
        if self._gate:
            data["policy_engine"] = (
                self._gate.policy_engine is not None
            )
            data["approval_manager"] = (
                self._gate.approval_manager is not None
            )
        return JSONResponse(data)

    async def _health_check(self, request: Request) -> JSONResponse:
        healthy = True
        if self._registry:
            health = self._registry.health_check_all()
            healthy = all(h.healthy for h in health.values())
        return JSONResponse(
            {"status": "healthy" if healthy else "degraded"},
            status_code=200 if healthy else 503,
        )
```

---

## APPENDIX B: NEW PYDANTIC EVENT MODELS

Add these to `src/agentic_trading/core/events.py`. All are backward-compatible additions (existing events unchanged).

### B.1 BaseEvent Upgrade (3 new fields, all with defaults)

```python
class BaseEvent(BaseModel):
    """Base for all events. Provides identity, time, and tracing."""

    event_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    trace_id: str = Field(default_factory=_uuid)
    source_module: str = ""

    # NEW: Causation chain (P0, Sprint 1 Day 1)
    parent_event_id: str | None = None
    correlation_id: str = ""

    # NEW: Schema version for evolution (P1)
    schema_version: int = 1
```

### B.2 ToolCallRecorded

```python
class ToolCallRecorded(BaseEvent):
    """Audit record for every exchange adapter call through ToolGateway."""

    source_module: str = "tool_gateway"
    tool_name: str                     # "submit_order", "cancel_order", etc.
    agent_id: str = ""                 # Which agent initiated
    adapter_type: str = ""             # "PaperAdapter", "CCXTAdapter"
    idempotency_key: str = ""          # dedupe_key for orders
    request_summary: dict[str, Any] = Field(default_factory=dict)
    response_summary: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = True
    policy_check_passed: bool = True
    retries: int = 0
```

### B.3 PolicyEvaluated (structured governance decision)

```python
class PolicyEvaluatedStep(BaseModel):
    """Single step in a governance evaluation."""

    step: str               # "maturity_check", "impact_classification", etc.
    result: str             # "PASS", "FAIL", "MEDIUM", etc.
    ms: float = 0.0         # Duration of this step
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyEvaluated(BaseEvent):
    """Full structured governance evaluation result.

    Emitted by GovernanceGate.evaluate() alongside GovernanceDecision.
    Provides step-by-step audit trail.
    """

    source_module: str = "governance"
    strategy_id: str
    symbol: str
    intent_event_id: str = ""
    decision: str = ""                  # "ALLOW", "BLOCK", "REDUCE_SIZE"
    sizing_multiplier: float = 1.0
    evaluation_ms: float = 0.0
    steps: list[PolicyEvaluatedStep] = Field(default_factory=list)
    policy_set_version: str = ""
    shadow_violations: list[str] = Field(default_factory=list)
```

### B.4 IncidentDeclared

```python
class IncidentDeclared(BaseEvent):
    """Published when the IncidentManager declares a new incident."""

    source_module: str = "incident_manager"
    incident_id: str
    severity: str = ""                  # IncidentSeverity value
    trigger: str = ""                   # trigger_type
    trigger_event_id: str = ""          # What event triggered this
    description: str = ""
    affected_strategies: list[str] = Field(default_factory=list)
    affected_symbols: list[str] = Field(default_factory=list)
    auto_actions_taken: list[str] = Field(default_factory=list)
    requires_human: bool = False
```

### B.5 DegradedModeEnabled

```python
class DegradedModeEnabled(BaseEvent):
    """Published when the platform enters or changes degraded mode."""

    source_module: str = "incident_manager"
    mode: str = ""                      # DegradedMode value
    previous_mode: str = ""
    reason: str = ""
    restrictions: list[str] = Field(default_factory=list)
    recovery_criteria: dict[str, Any] = Field(default_factory=dict)
```

### B.6 TradeIntentProposed (enhanced OrderIntent envelope)

```python
class TradeIntentProposed(BaseEvent):
    """Enriched trade intent with portfolio impact analysis.

    Published by PortfolioManager before it emits the OrderIntent.
    Enables the supervision UI to show proposed trades with context.
    """

    source_module: str = "portfolio"
    strategy_id: str
    symbol: str
    direction: str = ""                 # "LONG", "SHORT"
    signal_confidence: float = 0.0
    signal_event_id: str = ""           # parent_event_id chain
    proposed_qty: str = ""              # Decimal as string
    sizing_method: str = ""
    stop_price: str | None = None
    target_price: str | None = None
    risk_amount_usd: float = 0.0
    portfolio_impact: dict[str, float] = Field(default_factory=dict)
```

### B.7 Schema Registry Update

Add the new event types to `src/agentic_trading/event_bus/schemas.py`:

```python
# In TOPIC_SCHEMAS dict, add:
"governance": [
    GovernanceDecision,
    MaturityTransition,
    HealthScoreUpdate,
    CanaryAlert,
    DriftAlert,
    TokenEvent,
    GovernanceCanaryCheck,
    PolicyEvaluated,      # NEW
    ToolCallRecorded,     # NEW
    IncidentDeclared,     # NEW
    DegradedModeEnabled,  # NEW
],
"governance.approval": [
    ApprovalRequested,
    ApprovalResolved,
],
```

---

## APPENDIX C: NEW ENUMS

Add these to `src/agentic_trading/core/enums.py`. All follow the existing `(str, Enum)` pattern.

```python
class DegradedMode(str, Enum):
    """Platform degraded mode states.

    Ordered by severity — NORMAL < REDUCE_ONLY < NO_NEW_TRADES < READ_ONLY < KILLED.
    """

    NORMAL = "normal"
    REDUCE_ONLY = "reduce_only"           # Existing positions only, reduced sizing
    NO_NEW_TRADES = "no_new_trades"       # Can close/reduce but not open
    READ_ONLY = "read_only"               # No trading actions at all
    KILLED = "killed"                     # Full kill switch equivalent

    @property
    def rank(self) -> int:
        """Numeric severity for comparisons."""
        return list(DegradedMode).index(self)


class IncidentSeverity(str, Enum):
    """Incident severity classification."""

    LOW = "low"           # Log + alert, continue trading
    MEDIUM = "medium"     # REDUCE_ONLY, auto-recover after cooldown
    HIGH = "high"         # NO_NEW_TRADES, operator approval to resume
    CRITICAL = "critical" # KILLED, L3_RISK approval to resume


class IncidentStatus(str, Enum):
    """Incident lifecycle status."""

    DETECTED = "detected"
    TRIAGED = "triaged"
    DEGRADED = "degraded"
    MITIGATING = "mitigating"
    RECOVERING = "recovering"
    RESOLVED = "resolved"


class StrategyStage(str, Enum):
    """Strategy lifecycle stage.

    More granular than MaturityLevel — tracks the full promotion pipeline.
    MaturityLevel is derived from StrategyStage for governance decisions.
    """

    CANDIDATE = "candidate"     # Code merged, no live data
    BACKTEST = "backtest"       # Walk-forward analysis in progress
    EVAL_PACK = "eval_pack"     # Backtest complete, evidence pack generated
    PAPER = "paper"             # Paper trading (L2_GATED)
    LIMITED = "limited"         # Limited live (L3_CONSTRAINED)
    SCALE = "scale"             # Full autonomy (L4_AUTONOMOUS)
    DEMOTED = "demoted"         # Demoted, sizing reduced/paused
```

---

## APPENDIX D: NEW DB TABLES (SQLAlchemy Models)

Add these to `src/agentic_trading/storage/postgres/models.py`. They extend the existing schema (OrderRecord, FillRecord, PositionSnapshot, BalanceSnapshot, DecisionAudit, ExperimentLog, GovernanceLog).

### D.1 ApprovalRecord (persisted approvals — fixes in-memory-only gap)

```python
class ApprovalRecord(Base):
    """Persisted approval request lifecycle.

    Fixes the gap where ApprovalManager keeps state in-memory only,
    which is lost on restart.
    """

    __tablename__ = "approval_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    request_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger: Mapped[str] = mapped_column(String(64), nullable=False)
    escalation_level: Mapped[str] = mapped_column(String(24), nullable=False)
    notional_usd: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    impact_tier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="pending",
    )
    decided_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=300)

    # Context snapshot (for display in UI)
    context_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_approval_request_id", "request_id"),
        Index("ix_approval_strategy_id", "strategy_id"),
        Index("ix_approval_status", "status"),
        Index("ix_approval_requested_at", "requested_at"),
        Index("ix_approval_status_requested", "status", "requested_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<ApprovalRecord(request_id={self.request_id!r}, "
            f"strategy={self.strategy_id!r}, status={self.status!r})>"
        )
```

### D.2 ToolCallRecord (audit trail for ToolGateway)

```python
class ToolCallRecord(Base):
    """Audit log for every exchange adapter call through ToolGateway.

    Enables post-facto investigation of what was submitted to the exchange,
    when, by which agent, and whether it succeeded.
    """

    __tablename__ = "tool_call_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    adapter_type: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    request_summary: Mapped[dict] = mapped_column(JSONB, nullable=False)
    response_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    policy_check_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    called_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_toolcall_tool_name", "tool_name"),
        Index("ix_toolcall_idem_key", "idempotency_key"),
        Index("ix_toolcall_trace_id", "trace_id"),
        Index("ix_toolcall_called_at", "called_at"),
        Index("ix_toolcall_agent_id", "agent_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<ToolCallRecord(tool={self.tool_name!r}, "
            f"idem_key={self.idempotency_key!r}, success={self.success})>"
        )
```

### D.3 IncidentRecord (persisted incident lifecycle)

```python
class IncidentRecord(Base):
    """Persisted incident lifecycle for audit and post-incident review."""

    __tablename__ = "incident_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    incident_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(64), nullable=False)
    trigger_event_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    degraded_mode: Mapped[str | None] = mapped_column(String(24), nullable=True)
    affected_strategies: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    affected_symbols: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    auto_actions: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolution_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    declared_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_incident_incident_id", "incident_id"),
        Index("ix_incident_severity", "severity"),
        Index("ix_incident_status", "status"),
        Index("ix_incident_declared_at", "declared_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<IncidentRecord(incident_id={self.incident_id!r}, "
            f"severity={self.severity!r}, status={self.status!r})>"
        )
```

### D.4 StrategyLifecycleRecord (promotion/demotion audit trail)

```python
class StrategyLifecycleRecord(Base):
    """Audit trail for strategy stage transitions."""

    __tablename__ = "strategy_lifecycle"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    from_stage: Mapped[str] = mapped_column(String(24), nullable=False)
    to_stage: Mapped[str] = mapped_column(String(24), nullable=False)
    from_maturity: Mapped[str] = mapped_column(String(24), nullable=False)
    to_maturity: Mapped[str] = mapped_column(String(24), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    promoted_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    transitioned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_lifecycle_strategy_id", "strategy_id"),
        Index("ix_lifecycle_transitioned_at", "transitioned_at"),
        Index("ix_lifecycle_to_stage", "to_stage"),
    )

    def __repr__(self) -> str:
        return (
            f"<StrategyLifecycleRecord(strategy={self.strategy_id!r}, "
            f"{self.from_stage!r} → {self.to_stage!r})>"
        )
```

---

## APPENDIX E: INTEGRATION WIRING SPEC

Exact call sites in existing code where new services plug in. Each entry specifies the **file**, **function/method**, **line range**, and **change description**.

### E.1 BaseEvent Upgrade

| File | Location | Change |
|------|----------|--------|
| `core/events.py:45-52` | `class BaseEvent` | Add `parent_event_id`, `correlation_id`, `schema_version` fields (3 lines, all with defaults) |

**Propagation sites** (set `parent_event_id` when creating child events):

| File | Method | Change |
|------|--------|--------|
| `portfolio/manager.py` | Where `OrderIntent` is created from `Signal` | Set `parent_event_id=signal.event_id`, `correlation_id=signal.trace_id` |
| `execution/engine.py:302-316` | `handle_intent()` where synthetic `FillEvent` is created | Set `parent_event_id=ack.event_id`, `correlation_id=intent.correlation_id` |
| `execution/engine.py:471-479` | `handle_fill()` where `PositionUpdate` is created | Set `parent_event_id=fill.event_id` |
| `governance/gate.py:348-361` | Where `GovernanceDecision` is created | Set `parent_event_id` from intent's `event_id`, pass through `correlation_id` |

### E.2 ToolGateway Wiring

| File | Location | Change |
|------|----------|--------|
| `execution/engine.py:76-94` | `__init__` | Add `tool_gateway: ToolGateway | None = None` parameter |
| `execution/engine.py:350` | `_submit_with_retry` — `await self._adapter.submit_order(intent)` | Replace with `await self._tool_gateway.submit_order(intent, agent_id="execution_engine", trace_id=intent.trace_id)` when `self._tool_gateway` is set, fallback to direct adapter |
| `execution/engine.py:484` | `handle_fill()` — `await self._adapter.get_positions()` | Replace with `await self._tool_gateway.get_positions()` (pass-through, no audit overhead) |
| `main.py` | Where `ExecutionEngine` is constructed | Create `ToolGateway(adapter, event_bus, policy_engine)` and pass to `ExecutionEngine(tool_gateway=tool_gateway)` |

**Constructor change in ExecutionEngine:**
```python
def __init__(
    self,
    adapter: IExchangeAdapter,
    event_bus: IEventBus,
    risk_manager: IRiskChecker,
    kill_switch: Any = None,
    portfolio_state_provider: Any = None,
    max_retries: int = 3,
    governance_gate: Any = None,
    tool_gateway: Any = None,       # NEW
) -> None:
    self._adapter = adapter
    self._tool_gateway = tool_gateway  # NEW: preferred over _adapter for writes
    # ... rest unchanged
```

### E.3 ExecutionQualityTracker Wiring

| File | Location | Change |
|------|----------|--------|
| `execution/engine.py:76-94` | `__init__` | Add `quality_tracker: ExecutionQualityTracker | None = None` |
| `execution/engine.py:296` | After `_submit_with_retry` returns | Call `self._quality_tracker.record_submission()` |
| `execution/engine.py:427-525` | `handle_fill()` — after publishing FillEvent | Add block to record fill quality: |

```python
# In handle_fill(), after line 468 (await self._event_bus.publish("execution", fill)):
if self._quality_tracker is not None:
    # Get signal price from journal or order manager
    signal_price = None
    order_state = self._order_manager.get_order(fill.client_order_id)
    if order_state and order_state.intent:
        signal_price = order_state.intent.price
    self._quality_tracker.record_fill(
        fill_price=fill.price,
        fill_qty=fill.qty,
        signal_price=signal_price,
        side=fill.side.value,
        symbol=fill.symbol,
        strategy_id=order_state.intent.strategy_id if order_state and order_state.intent else "",
        fill_timestamp=fill.timestamp,
    )
```

### E.4 GovernanceGate — PolicyEvaluated Emission

| File | Location | Change |
|------|----------|--------|
| `governance/gate.py:77-361` | `evaluate()` | Collect step timings into a `steps` list throughout the method, emit `PolicyEvaluated` event in `_publish_and_log()` |

**Detailed change:** Add a `steps: list[dict]` local variable at the start of `evaluate()`. After each check (maturity, impact, health, drift, policy), append a step dict. In `_publish_and_log()`, create and emit `PolicyEvaluated` alongside the existing `GovernanceDecision`.

```python
# At top of evaluate():
steps: list[dict[str, Any]] = []
t_step = time.monotonic()

# After maturity check (line ~133):
steps.append({
    "step": "maturity_check",
    "result": "PASS",
    "level": level.value,
    "ms": round((time.monotonic() - t_step) * 1000, 1),
})
t_step = time.monotonic()

# ... similar for each subsequent step ...

# In _publish_and_log(), after publishing GovernanceDecision:
policy_evaluated = PolicyEvaluated(
    strategy_id=decision.strategy_id,
    symbol=decision.symbol,
    intent_event_id=trace_id,
    decision=decision.action.value,
    sizing_multiplier=decision.sizing_multiplier,
    evaluation_ms=elapsed * 1000,
    steps=[PolicyEvaluatedStep(**s) for s in steps],
    trace_id=trace_id,
    parent_event_id=trace_id,
)
await self._event_bus.publish("governance", policy_evaluated)
```

### E.5 StrategyLifecycleManager Wiring

| File | Location | Change |
|------|----------|--------|
| `main.py` | After journal + governance gate are created | Create `StrategyLifecycleManager(event_bus, journal, governance_gate)` |
| `main.py` | Strategy registration loop | Call `lifecycle_manager.register_strategy(strategy_id, initial_stage)` for each configured strategy, mapping existing `MaturityLevel` to `StrategyStage` |
| `agents/orchestrator.py` | `_create_agents()` | Add `StrategyLifecycleManager` to agent registry |

### E.6 IncidentManager Wiring

| File | Location | Change |
|------|----------|--------|
| `main.py` | After event bus + agent registry created | Create `IncidentManager(event_bus)` and register with agent registry |
| `execution/engine.py:170-187` | Kill switch check in `handle_intent()` | Optionally notify incident manager (via event bus — already handled since IncidentManager subscribes to `risk` and `system` topics) |
| `governance/canary.py` | Where `CanaryAlert` is published | No change needed — IncidentManager subscribes to `system` topic and handles CanaryAlert |
| `risk/circuit_breakers.py` | Where `CircuitBreakerEvent` is published | No change needed — IncidentManager subscribes to `risk` topic |

**Key insight:** IncidentManager is fully event-driven via subscriptions. No direct wiring into existing components needed beyond creating it and registering it as an agent.

### E.7 DailyEffectivenessScorecard Wiring

| File | Location | Change |
|------|----------|--------|
| `main.py` | After journal, quality tracker, risk manager, agent registry created | Create `DailyEffectivenessScorecard(journal, quality_tracker, risk_manager, agent_registry, event_bus)` |
| `journal/journal.py` | `_close_trade()` or `on_trade_closed` callback | Call `scorecard.compute()` on trade close |
| `api/supervision.py` | `/api/v1/home/scorecard` endpoint | Returns `scorecard.compute()` or `scorecard.last_scores` |

### E.8 Approval Persistence Wiring

| File | Location | Change |
|------|----------|--------|
| `governance/approval_manager.py` | `request_approval()` | After creating in-memory `ApprovalRequest`, also INSERT into `ApprovalRecord` table |
| `governance/approval_manager.py` | `approve()`, `reject()`, `expire_stale()` | After updating in-memory state, also UPDATE the `ApprovalRecord` row |
| `governance/approval_manager.py` | `__init__()` | Accept optional `db_session_factory` parameter. On startup, load PENDING approvals from DB to restore in-memory state. |

**Pattern:**
```python
class ApprovalManager:
    def __init__(
        self,
        rules: list[ApprovalRule] | None = None,
        auto_approve_l1: bool = True,
        event_bus: Any = None,
        db_session_factory: Any = None,  # NEW
    ) -> None:
        self._db = db_session_factory
        # ... existing init ...

    async def _persist(self, request: ApprovalRequest) -> None:
        """Persist approval state to Postgres."""
        if self._db is None:
            return
        async with self._db() as session:
            record = ApprovalRecord(
                request_id=request.request_id,
                strategy_id=request.strategy_id,
                symbol=request.symbol,
                trigger=request.trigger.value,
                escalation_level=request.escalation_level.value,
                notional_usd=request.notional_usd,
                status=request.status.value,
                requested_at=request.created_at,
            )
            session.add(record)
            await session.commit()

    async def restore_pending(self) -> int:
        """Restore pending approvals from DB on startup. Returns count."""
        if self._db is None:
            return 0
        async with self._db() as session:
            result = await session.execute(
                select(ApprovalRecord).where(
                    ApprovalRecord.status == "pending"
                )
            )
            records = result.scalars().all()
            # Reconstruct in-memory ApprovalRequest from each record
            # ...
            return len(records)
```

### E.9 AgentOrchestrator Wiring (main.py)

| File | Location | Change |
|------|----------|--------|
| `main.py` | Current ad-hoc agent creation (~lines 800-1000) | Replace with `AgentOrchestrator.create_from_settings(settings, event_bus, ...)` |
| `agents/orchestrator.py` | `_create_agents()` | Fix: don't create duplicate RiskManager. Wire GovernanceGate to ExecutionAgent. Add StrategyLifecycleManager and IncidentManager to agent list. |

**Orchestrator fix:**
```python
# In AgentOrchestrator._create_agents():
# BEFORE (bug): creates its own RiskManager, doesn't wire GovernanceGate
execution_agent = ExecutionAgent(
    adapter=self._adapter,
    event_bus=self._event_bus,
    risk_manager=RiskManager(self._settings.risk),  # DUPLICATE!
    governance_gate=None,  # NOT WIRED!
)

# AFTER (fix): accept pre-built components
execution_agent = ExecutionAgent(
    adapter=self._adapter,
    event_bus=self._event_bus,
    risk_manager=self._risk_manager,     # Shared instance
    governance_gate=self._governance_gate, # Properly wired
    tool_gateway=self._tool_gateway,     # NEW
)
```

---

## APPENDIX F: UPDATED SECTION 9 — CODEBASE REVIEW FINDINGS

### Files Reviewed (Deep Analysis)

| File | Key Finding | Impact on Refurb |
|------|-------------|-------------------|
| `execution/order_manager.py` (326 LOC) | `_seen_keys: Set[str]` grows **unbounded**. `purge_terminal()` removes old `_TrackedOrder` records but does NOT purge from `_seen_keys`. Over 24h+ of live trading, this is a memory leak. | **ToolGateway's LRU-bounded `OrderedDict`** solves this for the new path. Also add `_seen_keys` purge to `purge_terminal()` in order_manager as a separate fix. |
| `governance/drift_detector.py` (208 LOC) | Clean implementation. Two thresholds: `deviation_threshold_pct` → REDUCE_SIZE, `pause_threshold_pct` → PAUSE. Baselines set from backtest results via `load_baseline_from_backtest()`. | **Directly usable** as demotion trigger input for StrategyLifecycleManager. Wire `check_drift()` output into `_should_demote()`. No changes needed to DriftDetector itself. |
| `observability/decision_audit.py` (151 LOC) | **Existing audit infrastructure** — `DecisionAudit` class with `log_features()`, `log_signal()`, `log_risk_check()`, `log_order_intent()`, `log_order_ack()`, `log_fill()`, `log_pnl()`. **BUT**: stores in an **in-memory list** (`self._decisions: list[dict]`). Uses `trace_id` from logger context. | **Enhance, don't replace.** Add persistence: flush `_decisions` to the existing `DecisionAudit` DB table periodically. Add `log_governance()` step. The in-memory audit class provides the right API — it just needs a persistence backend. |
| `observability/health.py` (96 LOC) | `HealthChecker` with registered async check functions. Has `check_redis()` and `check_postgres()` helpers. `check_all()` returns `list[SystemHealth]`. | **Extend for IncidentManager.** IncidentManager can call `HealthChecker.check_all()` to detect degraded infrastructure. Register additional checks for event bus, feed manager, adapter connectivity. |
| `storage/postgres/models.py` (445 LOC) | **6 existing tables**: OrderRecord, FillRecord, PositionSnapshot, BalanceSnapshot, DecisionAudit (DB), ExperimentLog, GovernanceLog. DecisionAudit has JSONB columns for full chain. GovernanceLog stores trace_id, action, maturity, impact, sizing_multiplier, details. | **Add 4 new tables** (Appendix D): ApprovalRecord, ToolCallRecord, IncidentRecord, StrategyLifecycleRecord. Existing tables remain unchanged. GovernanceLog continues to store GovernanceDecision; the new PolicyEvaluated event with step-by-step audit can optionally also be persisted to GovernanceLog via the existing `details` JSONB column. |

### Bugs Confirmed During Review

| Bug | Location | Severity | Fix |
|-----|----------|----------|-----|
| `_seen_keys` unbounded growth | `execution/order_manager.py:39` | **Medium** (memory leak over days) | Add `_seen_keys` cleanup to `purge_terminal()` or switch to TTL-based `OrderedDict` |
| `BaseAgent._last_error = logger.name` | `agents/base.py:164` | **Low** (stores "root" instead of exception type) | Change to `self._last_error = type(exc).__name__` |
| `AgentRegistry.start_all()` logs success even on partial failure | `agents/registry.py` | **Low** | Check all results before logging success |
| `AgentOrchestrator` not wired into `main.py` | `main.py` vs `agents/orchestrator.py` | **High** (defeats purpose of agent framework) | Wire orchestrator as primary lifecycle manager (E.9) |
| `AgentOrchestrator` creates duplicate `RiskManager` | `agents/orchestrator.py` | **Medium** (inconsistent risk state) | Accept pre-built components instead of creating internally |

### Pre-existing Infrastructure to Leverage

| Component | Current State | Refurb Leverage |
|-----------|---------------|-----------------|
| `DecisionAudit` class (in-memory) | Logs full chain: features → signal → risk → intent → ack → fill → pnl | Add `log_governance()` step; add DB persistence flush |
| `DecisionAudit` DB table | JSONB columns for each chain step | Use as-is for persisted audit; no schema change needed |
| `GovernanceLog` DB table | Stores every governance decision | Use as-is; PolicyEvaluated step detail goes in `details` JSONB |
| `ExperimentLog` DB table | Stores backtest results with Sharpe, win_rate, etc. | Feed directly into StrategyLifecycleManager evidence collection |
| `HealthChecker` + `check_redis/postgres` | Basic health probe infrastructure | Extend with event bus health, feed manager, adapter connectivity checks |
| `QualityScorecard` (1031 LOC) | A-F grading on 10+ metrics, portfolio-level assessment | Feed directly into DailyEffectivenessScorecard edge quality computation |
| `TradeJournal.on_trade_closed` callback | Called when trade closes | Wire DailyEffectivenessScorecard.compute() into this callback |
| `MaturityTransition` event | Already defined in events.py | Used as-is by StrategyLifecycleManager |

---

**End of Refurb Plan v1.0**

*Total document: ~2,800 lines. Implementation code: ~1,500 LOC across 6 new services, 7 new events, 4 new enums, 4 new DB tables. Integration: ~200 lines of wiring changes to existing code.*
