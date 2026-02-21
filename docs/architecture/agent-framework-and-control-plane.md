# Agent Framework & Control Plane Architecture

## 1. Overview

The platform uses a multi-agent architecture where specialized agents coordinate via an event bus. All exchange side effects are routed through the institutional control plane (ToolGateway).

```
┌─────────────────────────────────────────────────────────────────┐
│                       AgentOrchestrator                         │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐ ┌────────┐ │
│  │ Market  │ │ Feature  │ │  Risk  │ │ Execution │ │ Report │ │
│  │ Intel   │ │ Compute  │ │  Gate  │ │  Agent    │ │ Agent  │ │
│  └────┬────┘ └────┬─────┘ └───┬────┘ └─────┬─────┘ └────────┘ │
│       │           │           │             │                   │
│       └───────────┴───────────┴─────────────┘                   │
│                         EventBus                                │
│  Topics: market.candle, feature.vector, strategy.signal,        │
│          execution, risk, governance, surveillance, reporting   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │    ToolGateway      │
                    │ Policy → Approval   │
                    │ → Audit → Adapter   │
                    └────────────────────┘
```

## 2. Agent Framework

### 2.1 BaseAgent Lifecycle

All agents extend `BaseAgent` (`agents/base.py`) which provides:

- **start/stop** with graceful shutdown via `asyncio.CancelledError` handling
- **Health reporting** via `AgentHealthReport` (running state, error count, uptime)
- **Periodic work loop** at configurable `interval` seconds (0 = event-driven only)
- **Lifecycle hooks**: `_on_start()`, `_on_stop()`, `_work()` for subclasses

Status progression: `CREATED → STARTING → RUNNING → STOPPING → STOPPED`

### 2.2 Agent Registry

`AgentRegistry` (`agents/registry.py`):
- `register(agent)` / `unregister(agent_id)`
- `start_all()` — starts in registration order
- `stop_all()` — stops in **reverse** order (safety: execution stops before risk)
- `health_check_all()` — returns list of `AgentHealthReport`

### 2.3 Agent Orchestrator

`AgentOrchestrator` (`agents/orchestrator.py`) creates agents from `Settings`:
- Reads `settings.mode` to determine which agents to create
- Paper/live: creates MarketIntelligence, RiskGate, Execution agents
- Backtest: skips Execution (no adapter needed)
- Optional: GovernanceCanary, OptimizerScheduler, Reporting, Surveillance

### 2.4 Agent Inventory

| Agent | Type | Subscribes | Publishes | Path |
|-------|------|-----------|-----------|------|
| MarketIntelligenceAgent | MARKET_INTELLIGENCE | — | market.candle | `agents/market_intelligence.py` |
| FeatureComputationAgent | FEATURE_COMPUTATION | market.candle | feature.vector | `agents/feature_computation.py` |
| RiskGateAgent | RISK_GATE | execution | risk, governance | `agents/risk_gate.py` |
| ExecutionAgent | EXECUTION | execution, system | execution | `agents/execution.py` |
| SurveillanceAgent | SURVEILLANCE | execution.* | surveillance | `agents/surveillance.py` |
| ReportingAgent | REPORTING | — (periodic) | reporting | `agents/reporting.py` |
| GovernanceCanary | GOVERNANCE_CANARY | governance | system | `agents/governance_canary.py` |
| OptimizerScheduler | OPTIMIZER | — (periodic) | strategy.signal | `agents/optimizer_scheduler.py` |
| CMTAnalystAgent | CMT_ANALYST | market.candle | — | `agents/cmt_analyst.py` |
| DataQualityAgent | DATA_QUALITY | market.candle | system | `agents/data_quality.py` |

## 3. Institutional Control Plane

### 3.1 Action Flow

```
ProposedAction
  → [1] Validate tool (ToolName enum allowlist)
  → [2] Idempotency check (cache)
  → [2.5] Information barrier (role check)
  → [3] CPPolicyEvaluator (deterministic policy rules)
  → [4] CPApprovalService (tiered approval workflow)
  → [5] AuditLog.append() (MANDATORY — fail-closed)
  → [6] Kill switch (final gate)
  → [7] Rate limit
  → [8] Dispatch to adapter
  → [9] Record result in AuditLog
  → [10] Cache for idempotency
→ ToolCallResult
```

Path: `control_plane/tool_gateway.py`

### 3.2 ToolGateway

The ONLY path for exchange side effects. No agent calls the adapter directly.

**14 allowlisted tools** (ToolName enum):
- Mutating (8): SUBMIT_ORDER, CANCEL_ORDER, CANCEL_ALL_ORDERS, AMEND_ORDER, BATCH_SUBMIT_ORDERS, SET_TRADING_STOP, SET_LEVERAGE, SET_POSITION_MODE
- Read-only (6): GET_POSITIONS, GET_BALANCES, GET_OPEN_ORDERS, GET_INSTRUMENT, GET_FUNDING_RATE, GET_CLOSED_PNL

### 3.3 Policy Evaluator

`CPPolicyEvaluator` (`control_plane/policy_evaluator.py`) wraps the PolicyEngine:

- Evaluates all registered PolicySets against the action context
- Determines approval tier from the most severe failed rule action
- Supports degraded modes: NORMAL → CAUTIOUS → STOP_NEW_ORDERS → RISK_OFF_ONLY → READ_ONLY → FULL_STOP
- Protective tools (cancel, TP/SL) get fast-path T0 via tier overrides
- **Fail-closed**: any evaluator error blocks the action

### 3.4 Approval Tiers

| Tier | Behavior | TTL |
|------|----------|-----|
| T0_AUTONOMOUS | No human needed | — |
| T1_NOTIFY | Execute + post-hoc notification | 60s |
| T2_APPROVE | Hold for 1 human approval | 5 min |
| T3_DUAL_APPROVE | Hold for 2 approvals | 10 min |

Path: `control_plane/approval_service.py`

Information barrier: proposer cannot self-approve at T2/T3.

### 3.5 Audit Log

Append-only, fail-closed contract (`control_plane/audit_log.py`):
- `append()` MUST succeed or raise — no silent drops
- If unavailable, ToolGateway rejects all mutating calls
- Correlation-based retrieval: `read(correlation_id)`
- Optional JSONL file persistence

### 3.6 Information Barriers

- `ProposedAction.required_role`: action requires specific role
- `ActionScope.actor_role`: role of the acting agent
- Mismatch → rejected at step 2.5 before policy evaluation
- Proposer ≠ approver enforced at T2/T3 approval tiers

## 4. Pre-Trade Safety

`PreTradeChecker` (`execution/risk/pre_trade.py`) runs 12 checks per OrderIntent:

| Check | Description |
|-------|-------------|
| position_direction_conflict | Blocks orders against existing position |
| max_concurrent_positions | Cap on simultaneous open positions |
| daily_entry_limit | Max entries per calendar day |
| portfolio_cooldown | Min time between entries |
| **price_collar** | BPS-based band around mark price (market orders exempt) |
| **self_match_prevention** | Cross-check against resting orders on same venue |
| **message_throttle** | Per-strategy and per-symbol 60s sliding window |
| max_position_size | Fraction of portfolio equity cap |
| max_notional | Absolute notional cap |
| max_leverage | Portfolio leverage limit |
| exposure_limits | Gross exposure multiple cap |
| instrument_limits | Per-instrument min qty/notional |

## 5. Surveillance & Compliance

### 5.1 SurveillanceAgent

`agents/surveillance.py` — subscribes to execution.intent, execution.fill, execution.ack, execution.update:

- **Wash trade detection**: same-symbol buy+sell from same strategy within `wash_trade_window_sec`
- **Spoofing detection**: order submitted then cancelled within `spoof_cancel_window_sec`
- Publishes `SurveillanceCaseEvent` on the `surveillance` topic
- Persists cases via CaseManager

### 5.2 CaseManager

`compliance/case_manager.py` — compliance case lifecycle:

```
open → investigating → escalated → closed
             ↘                ↗
              → closed (false_positive)
```

JSONL persistence, thread-safe. Dispositions: confirmed, false_positive, inconclusive.

## 6. Feature Pipeline

### 6.1 FeatureEngine

`intelligence/features/engine.py` — computes 60+ indicators per candle:

- **Core**: EMA, SMA, RSI, Bollinger Bands, MACD, ADX, ATR, Stochastic, OBV, VWAP, Donchian, Keltner, Ichimoku, HyperWave
- **Advanced**: ARIMA forecaster (statsmodels), FFT spectral analysis (numpy)
- **SMC**: Smart Money Concepts (order blocks, FVGs, BOS/CHoCH)

### 6.2 Feature Versioning

Deterministic config hash (`content_hash`) attached to every `FeatureVector`:
- Changes when indicator parameters change
- Enables downstream consumers to detect config drift

### 6.3 ARIMA Forecaster

`intelligence/features/arima.py`:
- One-step-ahead forecast with confidence intervals
- Requires `min_observations` (default 60) candles
- Statsmodels SARIMAX with exponential smoothing fallback

### 6.4 FFT Spectral Analysis

`intelligence/features/fourier.py`:
- Dominant cycle extraction from price series
- Top N frequency components by magnitude
- Spectral entropy for noise detection

## 7. Model Registry

`intelligence/model_registry.py` — versioned model lifecycle:

```
RESEARCH → PAPER → LIMITED → PRODUCTION
                                   ↓
                               RETIRED (terminal)
```

- Auto-incrementing version numbers per model name
- Training provenance: data hash, hyperparameters
- Promotion requires explicit approval with approver identity
- Performance metrics tracking via `update_metrics()`
- JSONL persistence

## 8. Observability & Reporting

### 8.1 DailyEffectivenessScorecard

`observability/daily_scorecard.py`:
- 4 scores: edge_quality, execution_quality, risk_discipline, operational_integrity
- Used by strategy lifecycle for automated promotion/demotion

### 8.2 ReportingAgent

`agents/reporting.py`:
- Periodic daily report generation
- Aggregates from journal, risk manager, case manager, agent registry
- Publishes `DailyReportEvent` on `reporting` topic

### 8.3 AuditBundleGenerator

`observability/audit_bundle.py`:
- Collects all audit entries for a trace_id
- Determines outcome: executed_success, blocked_policy, pending_approval
- For regulatory export
