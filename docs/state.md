# Governance Framework — State of the World

> Distilled from `docs/legacy_dump.md.rtf` (Soteria-Inspired Governance Framework).
> Last updated: 2026-02-14

## Origin

The governance layer adapts seven concepts from the Soteria Coach cybersecurity
governance platform to automated trading: strategy maturity levels, outcome-based
health scoring, infrastructure canary, trade impact classification, live-vs-backtest
drift detection, scoped execution tokens, and audit logging.

## Relationship to the Risk Layer

The platform already has a risk layer (kill switch, circuit breakers, drawdown
monitor, pre/post-trade risk checks). Governance sits **above** risk — it controls
*which strategies may trade* and *at what sizing*, rather than checking individual
order limits. Governance is gated behind `settings.governance.enabled` (default
`false`), preserving all existing behaviour when disabled.

## Architecture Overview

```
Signal → ExecutionEngine
            │
            ├─ 1. Deduplication
            ├─ 2. ── GovernanceGate ──  ← NEW
            │       ├─ MaturityManager
            │       ├─ HealthTracker
            │       ├─ ImpactClassifier
            │       ├─ DriftDetector
            │       └─ TokenManager
            ├─ 3. Pre-trade risk check
            ├─ 4. Order submission
            └─ 5. Post-trade risk check
```

GovernanceGate is a single orchestrator entry point that composes sub-component
verdicts into a final `GovernanceAction` (ALLOW / REDUCE_SIZE / BLOCK / DEMOTE /
PAUSE / KILL).

## Components

### 1. Strategy Maturity Manager (`governance/maturity.py`)

Five-level ladder controlling execution scope:

| Level | Label | Sizing Cap | Executes? |
|-------|-------|-----------|-----------|
| L0 | Shadow | 0 % | No — log only |
| L1 | Paper | 0 % | No — paper only |
| L2 | Gated | 10 % | Yes — every order needs approval |
| L3 | Constrained | 25 % | Yes — limited sizing |
| L4 | Autonomous | 100 % | Yes — full autonomy |

**Promotion** is slow: requires 50+ trades, win rate > 0.45, profit factor > 1.1,
one level at a time. **Demotion** is fast: triggered by drawdown > 10 % or loss
streak > 10, can skip levels.

See [ADR-001](adr/001-maturity-ladder.md).

### 2. Strategy Health Tracker (`governance/health_score.py`)

Maintains a rolling outcome window per strategy producing a 0.0–1.0 health score.
Uses an epistemic debt/credit model: losses accumulate debt, wins add credit
(clearing debt first). The score drives a sizing multiplier — degraded health
reduces position size toward a configurable floor.

### 3. Infrastructure Canary (`governance/canary.py`)

Independent periodic watchdog that registers health-check callbacks for critical
components (exchange connectivity, data feeds, etc.). Tracks consecutive failures
and auto-triggers the kill switch when a threshold is exceeded, providing a
governance-level circuit breaker separate from the risk layer's.

### 4. Trade Impact Classifier (`governance/impact_classifier.py`)

Scores pending orders on four dimensions — irreversibility, blast radius,
concentration, and notional size — producing a composite ImpactTier (LOW / MEDIUM /
HIGH / CRITICAL). Higher tiers require higher maturity levels and may trigger
additional approval requirements.

See [ADR-002](adr/002-impact-tiering.md).

### 5. Drift Detector (`governance/drift_detector.py`)

Compares live performance metrics against backtest baselines per strategy.
Deviations beyond 30 % trigger REDUCE_SIZE; beyond 50 % trigger PAUSE. This
catches regime changes or model degradation that the risk layer alone would not
flag until drawdown thresholds are hit.

See [ADR-003](adr/003-drift-thresholds.md).

### 6. Scoped Execution Tokens (`governance/tokens.py`)

Time-bounded, revocable tokens that scope what a strategy is allowed to do.
Tokens carry a TTL, are audit-bound (tied to a trace ID), and support bulk
revocation per strategy. Provides fine-grained, temporary authorization on top
of maturity levels.

See [ADR-004](adr/004-execution-tokens.md).

## Integration Points

| File | Integration |
|------|-------------|
| `execution/engine.py` | Gate inserted between deduplication and pre-trade risk; BLOCK/PAUSE/KILL rejects, REDUCE_SIZE scales quantity |
| `portfolio/manager.py` | Optional `governance_sizing_fn` callback applied after regime multiplier in `_compute_size()` |
| `observability/metrics.py` | ~10 Prometheus counters/gauges (decisions, blocks, maturity level, health score, drift deviation, canary status, latency, active tokens) |
| `storage/postgres/models.py` | `GovernanceLog` ORM model for audit trail (trace_id, strategy_id, action, scores, JSONB details) |
| `main.py` | Wires all components in `_run_live_or_paper()` when `governance.enabled=true`; starts canary periodic loop |

## Core Additions

| Layer | Additions |
|-------|-----------|
| Enums | `MaturityLevel`, `ImpactTier`, `GovernanceAction` |
| Errors | `GovernanceError`, `MaturityGateError`, `ExecutionTokenError`, `GovernanceCanaryFailure` |
| Events | 7 events under `governance` topic (decision, maturity transition, health update, canary alert, drift alert, token event, canary check) |
| Config | `GovernanceConfig` with 6 sub-configs added to `Settings` |

## Test Coverage

~113 unit tests across 10 test files covering all governance components,
events, and configuration. All 423+ pre-existing tests remain unaffected.

## Implementation Phasing

The implementation followed four phases:

1. **Core types** (parallel): enums, errors, events, config, DB model
2. **Components** (parallel, after phase 1): six governance modules
3. **Orchestrator** (after phase 2): `gate.py`
4. **Integration** (after phase 3): engine, manager, metrics, main wiring

## Key Design Principles

- **Opt-in**: entirely behind a feature flag; disabled = zero behavioural change
- **Asymmetric transitions**: promotion is slow and evidence-gated; demotion is fast and protective
- **Composable verdicts**: each sub-component produces an independent assessment; the gate merges them conservatively (most restrictive wins)
- **Separation from risk**: governance answers "should this strategy trade?" while risk answers "is this specific order safe?"
- **Auditability**: every decision is logged to Postgres with full context and published as an event
