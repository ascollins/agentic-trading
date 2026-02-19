# Architecture Changes

Tracks incremental refactoring toward the target architecture defined in
`docs/institutional-control-plane-spec.md`.

---

## PR 1 — Canonical Domain Events + Event Bus + Event Store

**Date:** 2025-02-17

### What changed

Added the foundational event model that all subsequent refactoring PRs
depend on.  **No existing code was modified** — the new modules sit
alongside the legacy `core/events.py` and `event_bus/` and will
gradually replace them.

### New files

| File | Purpose |
|------|---------|
| `src/agentic_trading/domain/__init__.py` | Domain layer package |
| `src/agentic_trading/domain/events.py` | 17 canonical domain events with `WRITE_OWNERSHIP` table |
| `src/agentic_trading/infrastructure/__init__.py` | Infrastructure layer package |
| `src/agentic_trading/infrastructure/event_bus.py` | `INewEventBus` protocol + `InMemoryEventBus` (type-routed, ownership-enforced) |
| `src/agentic_trading/infrastructure/event_store.py` | `IEventStore` protocol + `InMemoryEventStore` + `JsonFileEventStore` |
| `tests/unit/test_domain_events.py` | 60 tests — immutability, serialization, ownership, causality |
| `tests/unit/test_infrastructure_event_bus.py` | 17 tests — pub/sub, ownership enforcement, error handling |
| `tests/unit/test_infrastructure_event_store.py` | 24 tests — append, read, replay, idempotency, persistence |
| `docs/architecture/CHANGES.md` | This file |

### Canonical event types (17)

| Layer | Events | Writer |
|-------|--------|--------|
| Intelligence | `CandleReceived`, `FeatureComputed` | `intelligence` |
| Signal | `SignalCreated`, `DecisionProposed` | `signal` |
| Policy | `DecisionApproved`, `DecisionRejected`, `DecisionPending` | `policy_gate` |
| Execution | `OrderPlanned`, `OrderSubmitted`, `OrderAccepted`, `OrderRejected` | `execution` |
| Reconciliation | `FillReceived`, `PositionUpdated`, `PnLUpdated` | `reconciliation` |
| Incident | `IncidentRaised`, `TradingHalted` | `incident` |
| Audit | `AuditLogged` | `execution.gateway` |

### Key design decisions

1. **Frozen dataclasses** (not Pydantic) — immutability guaranteed by
   Python, zero serialization overhead, simpler than `model_config =
   {"frozen": True}`.
2. **Tuples for collections** — `features`, `checks_passed`,
   `affected_symbols` use `tuple[...]` instead of `list[...]` to
   preserve immutability.
3. **Write-ownership enforcement** — the `InMemoryEventBus` checks
   `event.source` against `WRITE_OWNERSHIP` and raises
   `WriteOwnershipError` on mismatch.  Opt-out via
   `enforce_ownership=False`.
4. **Event store idempotency** — `append()` is a no-op on duplicate
   `event_id`, enabling at-least-once delivery semantics.
5. **Coexistence** — existing `BaseEvent` / `MemoryEventBus` /
   `RedisStreamsBus` are untouched.  New and old run side by side.

### Test results

- 101 new tests passing (60 + 17 + 24).
- 1750 existing tests passing, 0 regressions.
- 1 pre-existing failure (`test_position_math.py` — large Decimal math).

### What's next (PR 2)

Wire the `InMemoryEventBus` write-ownership enforcement into the
bootstrap path so violations are caught at runtime during integration
tests.

---

## PR 2 — Bus/Store Integration + Write-Ownership Proof

**Date:** 2025-02-17

### What changed

Wired the event bus and event store together so that every published
domain event is automatically persisted to the store (idempotently).
Added property-based and exhaustive tests proving the write-ownership
invariant holds, plus static analysis tests guarding the read-only
consumer contract.

### Modified files

| File | Change |
|------|--------|
| `src/agentic_trading/infrastructure/event_bus.py` | Added optional `event_store` param to `InMemoryEventBus`; auto-appends on publish (best-effort — store failure does not block handlers) |

### New files

| File | Purpose |
|------|---------|
| `tests/unit/test_write_ownership.py` | 38 tests — Hypothesis property test (200 examples), exhaustive matrix (17 correct + 17 wrong), static analysis of read-only consumers |
| `tests/unit/test_bus_store_integration.py` | 8 tests — auto-append, idempotency, ownership blocks storage, store failure resilience, full lifecycle replay |

### Key design decisions

1. **Best-effort store persistence** — if the event store raises during
   `append()`, the exception is logged but handlers still run.  This
   ensures the trading pipeline is never blocked by a storage failure.
2. **Ownership blocks before storage** — if `WriteOwnershipError` is
   raised, the event never reaches the store.  The store only contains
   events from legitimate writers.
3. **Property-based proof** — Hypothesis generates 200 random
   (agent, event_type) pairs.  Every combination where agent != owner
   is proven to raise `WriteOwnershipError`.
4. **Static analysis guards** — `narration/` must not import the domain
   event bus.  `observability/metrics.py` must not import it either.
   These tests will catch accidental publish calls in future PRs.

### Test results

- 46 new tests passing (38 ownership + 8 integration).
- 1796 total tests passing (up from 1750), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 3)

Directory restructure: rename `event_bus/` → `bus/`, move `data/` →
`intelligence/`, move `features/` → `intelligence/features/`.  Pure
file moves with import rewrites, no behavior changes.

---

## PR 3 — Directory Restructure

**Date:** 2025-02-17

### What changed

Moved three legacy packages to their target architecture locations.
Real code now lives at the canonical paths; old locations contain thin
re-export shims that forward to the new packages.  Deleted two unused
scaffold files.  **Zero behavior changes.**

### Directory moves

| Old path | New canonical path |
|----------|--------------------|
| `event_bus/` | `bus/` |
| `data/` | `intelligence/` |
| `features/` | `intelligence/features/` |
| `features/smc/` | `intelligence/features/smc/` |

### New files (canonical locations)

| File | Purpose |
|------|---------|
| `src/agentic_trading/bus/__init__.py` | Package marker |
| `src/agentic_trading/bus/memory_bus.py` | `MemoryEventBus` (moved from `event_bus/`) |
| `src/agentic_trading/bus/redis_streams.py` | `RedisStreamsBus` (moved from `event_bus/`) |
| `src/agentic_trading/bus/bus.py` | `create_event_bus()` factory (moved from `event_bus/`) |
| `src/agentic_trading/bus/schemas.py` | Topic → schema registry (moved from `event_bus/`) |
| `src/agentic_trading/intelligence/__init__.py` | Intelligence layer package |
| `src/agentic_trading/intelligence/candle_builder.py` | Multi-timeframe candle aggregation (moved from `data/`) |
| `src/agentic_trading/intelligence/feed_manager.py` | Live market-data feed manager (moved from `data/`) |
| `src/agentic_trading/intelligence/normalizer.py` | Data normalization (moved from `data/`) |
| `src/agentic_trading/intelligence/historical.py` | Historical data loader (moved from `data/`) |
| `src/agentic_trading/intelligence/data_qa.py` | Data quality checker (moved from `data/`) |
| `src/agentic_trading/intelligence/private_streams.py` | Private WebSocket streams (moved from `data/`) |
| `src/agentic_trading/intelligence/features/__init__.py` | Features sub-package |
| `src/agentic_trading/intelligence/features/engine.py` | Feature engine (moved from `features/`) |
| `src/agentic_trading/intelligence/features/indicators.py` | Technical indicators (moved from `features/`) |
| `src/agentic_trading/intelligence/features/correlation.py` | Cross-asset correlation (moved from `features/`) |
| `src/agentic_trading/intelligence/features/funding_basis.py` | Funding rate features (moved from `features/`) |
| `src/agentic_trading/intelligence/features/multi_timeframe.py` | Multi-timeframe alignment (moved from `features/`) |
| `src/agentic_trading/intelligence/features/smc/` | SMC detection suite (moved from `features/smc/`) |

### Modified files (backward-compat re-export shims)

| File | Change |
|------|--------|
| `event_bus/__init__.py` | Shim docstring noting canonical location |
| `event_bus/memory_bus.py` | Re-exports from `bus.memory_bus` |
| `event_bus/redis_streams.py` | Re-exports from `bus.redis_streams` |
| `event_bus/bus.py` | Re-exports from `bus.bus` |
| `event_bus/schemas.py` | Re-exports from `bus.schemas` |
| `data/__init__.py` | Shim docstring noting canonical location |
| `data/candle_builder.py` | Re-exports from `intelligence.candle_builder` |
| `data/feed_manager.py` | Re-exports from `intelligence.feed_manager` |
| `data/normalizer.py` | Re-exports from `intelligence.normalizer` |
| `data/historical.py` | Re-exports from `intelligence.historical` |
| `data/data_qa.py` | Re-exports from `intelligence.data_qa` |
| `data/private_streams.py` | Re-exports from `intelligence.private_streams` |
| `features/__init__.py` | Shim docstring noting canonical location |
| `features/engine.py` | Re-exports from `intelligence.features.engine` |
| `features/indicators.py` | Re-exports from `intelligence.features.indicators` |
| `features/correlation.py` | Re-exports from `intelligence.features.correlation` |
| `features/funding_basis.py` | Re-exports from `intelligence.features.funding_basis` |
| `features/multi_timeframe.py` | Re-exports from `intelligence.features.multi_timeframe` |
| `features/smc/*` | Re-exports from `intelligence.features.smc.*` |

### Deleted files

| File | Reason |
|------|--------|
| `features/sentiment.py` | Scaffold — no tests, no imports, marked "Status: Scaffold" |
| `features/whale_monitor.py` | Scaffold — no tests, no imports, marked "Status: Scaffold" |

### Key design decisions

1. **Copy + shim, not git-mv** — canonical code copied to new paths;
   old files replaced with re-export shims.  This means `git blame`
   on new files starts fresh, but every old import path still works
   via the shims.  Shims will be removed in PR 16.
2. **Internal imports updated** — `intelligence/feed_manager.py` now
   imports from `intelligence.candle_builder` (not `data.candle_builder`).
   `intelligence/features/engine.py` lazy-imports from
   `intelligence.features.smc` (not `features.smc`).
3. **No test changes** — all 1796 existing tests still import from old
   paths and pass unchanged, proving the shims work.
4. **Scaffolds deleted** — `sentiment.py` and `whale_monitor.py` had no
   consumers, no tests, and were explicitly marked as scaffolds.  They
   can be re-created under `intelligence/features/` when implemented.

### Test results

- 0 new tests (pure structural move).
- 1796 existing tests passing, 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 4)

Move `analysis/` → `intelligence/analysis/`, move `strategies/` →
`signal/strategies/`.  Same copy + shim pattern.

---

## PR 4 — Analysis + Strategies Directory Restructure

**Date:** 2025-02-17

### What changed

Moved two more legacy packages to their target architecture locations.
Same copy + shim pattern as PR 3.  **Zero behavior changes.**

### Directory moves

| Old path | New canonical path |
|----------|--------------------|
| `analysis/` | `intelligence/analysis/` |
| `strategies/` | `signal/strategies/` |
| `strategies/regime/` | `signal/strategies/regime/` |
| `strategies/research/` | `signal/strategies/research/` |

### New files (canonical locations)

| File | Purpose |
|------|---------|
| `src/agentic_trading/intelligence/analysis/__init__.py` | Analysis sub-package under intelligence |
| `src/agentic_trading/intelligence/analysis/htf_analyzer.py` | HTF market structure analyzer |
| `src/agentic_trading/intelligence/analysis/market_context.py` | Macro regime assessment |
| `src/agentic_trading/intelligence/analysis/rr_calculator.py` | Risk-reward calculation |
| `src/agentic_trading/intelligence/analysis/smc_confluence.py` | Multi-TF SMC confluence scoring |
| `src/agentic_trading/intelligence/analysis/smc_trade_plan.py` | SMC trade plan generator |
| `src/agentic_trading/intelligence/analysis/trade_plan.py` | Structured trade plan model |
| `src/agentic_trading/signal/__init__.py` | Signal layer package |
| `src/agentic_trading/signal/strategies/__init__.py` | Strategies sub-package |
| `src/agentic_trading/signal/strategies/base.py` | `BaseStrategy` ABC |
| `src/agentic_trading/signal/strategies/registry.py` | Strategy registry + factory |
| `src/agentic_trading/signal/strategies/*.py` | 11 strategy implementations |
| `src/agentic_trading/signal/strategies/regime/` | Regime detection (4 files) |
| `src/agentic_trading/signal/strategies/research/` | Research tooling (3 files) |

### Modified files (backward-compat re-export shims)

| File | Change |
|------|--------|
| `analysis/__init__.py` | Shim docstring |
| `analysis/*.py` (6 files) | Re-exports from `intelligence.analysis.*` |
| `strategies/__init__.py` | Shim docstring |
| `strategies/*.py` (14 files) | Re-exports from `signal.strategies.*` |
| `strategies/regime/*.py` (5 files) | Re-exports from `signal.strategies.regime.*` |
| `strategies/research/*.py` (4 files) | Re-exports from `signal.strategies.research.*` |

### Key design decisions

1. **Internal imports updated** — `intelligence/analysis/smc_trade_plan.py`
   now imports from `intelligence.analysis.rr_calculator` and
   `intelligence.analysis.smc_confluence` (not `analysis.*`).
2. **Relative imports preserved** — all strategy files use `from .base
   import BaseStrategy` and `from .registry import register_strategy`.
   These relative imports work unchanged at the new `signal/strategies/`
   location.
3. **Strategy registry singleton** — the `_REGISTRY` dict in `registry.py`
   is shared via the re-export shim, so `import agentic_trading.strategies.trend_following`
   still registers to the same registry as
   `import agentic_trading.signal.strategies.trend_following`.

### Test results

- 0 new tests (pure structural move).
- 1796 existing tests passing, 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 5)

Move `portfolio/` → `signal/portfolio/`, create `signal/runner.py`
(extract strategy runner from `main.py`).

---

## PR 5 — Portfolio Move + Strategy Runner Extract

**Date:** 2025-02-17

### What changed

1. Moved `portfolio/` → `signal/portfolio/` (same copy + shim pattern).
2. Created `signal/runner.py` — a new `StrategyRunner` class that
   encapsulates the core strategy dispatch loop previously buried inside
   the 200-line `_run_live_or_paper.on_feature_vector` closure in
   `main.py`.  The runner is a clean, testable module with an `on_signal`
   callback hook for downstream wiring.

### Directory moves

| Old path | New canonical path |
|----------|--------------------|
| `portfolio/` | `signal/portfolio/` |

### New files

| File | Purpose |
|------|---------|
| `src/agentic_trading/signal/portfolio/__init__.py` | Package marker |
| `src/agentic_trading/signal/portfolio/manager.py` | `PortfolioManager` (moved) |
| `src/agentic_trading/signal/portfolio/sizing.py` | Position sizing methods (moved) |
| `src/agentic_trading/signal/portfolio/allocator.py` | `PortfolioAllocator` (moved) |
| `src/agentic_trading/signal/portfolio/intent_converter.py` | `build_order_intents()` (moved) |
| `src/agentic_trading/signal/portfolio/correlation_risk.py` | `CorrelationRiskAnalyzer` (moved) |
| `src/agentic_trading/signal/runner.py` | `StrategyRunner` — dispatches feature vectors to strategies, publishes signals |
| `tests/unit/test_signal_runner.py` | 12 tests — dispatch, aliasing, callbacks, edge cases |

### Modified files (backward-compat re-export shims)

| File | Change |
|------|--------|
| `portfolio/__init__.py` | Shim docstring |
| `portfolio/manager.py` | Re-exports from `signal.portfolio.manager` |
| `portfolio/sizing.py` | Re-exports from `signal.portfolio.sizing` |
| `portfolio/allocator.py` | Re-exports from `signal.portfolio.allocator` |
| `portfolio/intent_converter.py` | Re-exports from `signal.portfolio.intent_converter` |
| `portfolio/correlation_risk.py` | Re-exports from `signal.portfolio.correlation_risk` |

### Key design decisions

1. **`StrategyRunner` is a clean extract, not a rewrite** — it
   encapsulates exactly the dispatch logic from `main.py`:
   receive FeatureVector → alias indicators → call `on_candle()` →
   publish Signal.  The heavy downstream wiring (portfolio manager,
   execution, narration, journal) stays in `main.py` for now.
2. **`on_signal` callback** — instead of hard-wiring to portfolio/execution,
   the runner accepts an optional callback.  `main.py` can pass a closure
   that feeds the portfolio manager and execution pipeline.
3. **`alias_features()` extracted** — the indicator aliasing logic
   (`adx_14` → `adx`, etc.) is now a standalone function, testable
   independently.
4. **Duck-typed feature engine** — the runner uses a `Protocol` for the
   feature engine dependency (`get_buffer(symbol, tf)`) to avoid circular
   imports.

### Test results

- 12 new tests passing (StrategyRunner + alias_features).
- 1808 total tests passing (up from 1796), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 6)

Unified `PolicyGate`: merge governance gate, approval manager, and policy
engine into a single entry point under `policy/`.

---

## PR 6 — Governance → Policy Restructure + Unified PolicyGate

**Date:** 2025-02-17

### What changed

1. Moved all 15 `governance/` modules to the new canonical `policy/`
   package, with 4 renamed files for clarity.
2. Replaced all `governance/` modules with thin re-export shims.
3. Created a new `PolicyGate` facade that composes `GovernanceGate` +
   `PolicyEngine` + `PolicyStore` + `ApprovalManager` into a single
   high-level API.

### Directory moves

| Old path | New canonical path | Notes |
|----------|--------------------|-------|
| `governance/gate.py` | `policy/governance_gate.py` | Renamed |
| `governance/policy_engine.py` | `policy/engine.py` | Renamed |
| `governance/policy_models.py` | `policy/models.py` | Renamed |
| `governance/policy_store.py` | `policy/store.py` | Renamed |
| `governance/approval_manager.py` | `policy/approval_manager.py` | |
| `governance/approval_models.py` | `policy/approval_models.py` | |
| `governance/canary.py` | `policy/canary.py` | |
| `governance/default_policies.py` | `policy/default_policies.py` | |
| `governance/drift_detector.py` | `policy/drift_detector.py` | |
| `governance/health_score.py` | `policy/health_score.py` | |
| `governance/impact_classifier.py` | `policy/impact_classifier.py` | |
| `governance/incident_manager.py` | `policy/incident_manager.py` | |
| `governance/maturity.py` | `policy/maturity.py` | |
| `governance/strategy_lifecycle.py` | `policy/strategy_lifecycle.py` | |
| `governance/tokens.py` | `policy/tokens.py` | |

### New files

| File | Purpose |
|------|---------|
| `src/agentic_trading/policy/__init__.py` | Package init with all public re-exports |
| `src/agentic_trading/policy/gate.py` | **`PolicyGate`** — unified facade wrapping all governance subsystems |
| `tests/unit/test_policy_gate.py` | 34 tests — factory, evaluate, policy management, accessors, shim identity checks |

### Modified files (backward-compat re-export shims)

| File | Re-exports from |
|------|-----------------|
| `governance/__init__.py` | `agentic_trading.policy` (all public names) |
| `governance/gate.py` | `policy.governance_gate` |
| `governance/policy_engine.py` | `policy.engine` |
| `governance/policy_models.py` | `policy.models` |
| `governance/policy_store.py` | `policy.store` |
| `governance/approval_manager.py` | `policy.approval_manager` |
| `governance/approval_models.py` | `policy.approval_models` |
| `governance/canary.py` | `policy.canary` |
| `governance/default_policies.py` | `policy.default_policies` |
| `governance/drift_detector.py` | `policy.drift_detector` |
| `governance/health_score.py` | `policy.health_score` |
| `governance/impact_classifier.py` | `policy.impact_classifier` |
| `governance/incident_manager.py` | `policy.incident_manager` |
| `governance/maturity.py` | `policy.maturity` |
| `governance/strategy_lifecycle.py` | `policy.strategy_lifecycle` |
| `governance/tokens.py` | `policy.tokens` |

### `PolicyGate` API

```python
from agentic_trading.policy.gate import PolicyGate

# Factory: builds all sub-components from config
gate = PolicyGate.from_config(governance_config, risk_config)

# Main evaluation — delegates to GovernanceGate internally
decision = await gate.evaluate(
    strategy_id="trend_following",
    symbol="BTC/USDT",
    notional_usd=25_000,
)

# Policy management
gate.register_policy_set(custom_ps)
gate.set_policy_mode("pre_trade_risk", PolicyMode.SHADOW)
gate.rollback_policy("pre_trade_risk")

# Component access
gate.maturity.set_level("strat", MaturityLevel.L3_CONSTRAINED)
gate.health.get_score("strat")
gate.drift.check_drift("strat")
```

### Key design decisions

1. **4 renamed files** — `gate.py` → `governance_gate.py` (frees up
   `gate.py` for the unified facade), `policy_engine.py` → `engine.py`,
   `policy_models.py` → `models.py`, `policy_store.py` → `store.py`
   (removes redundant `policy_` prefix now that they live under `policy/`).
2. **`PolicyGate.from_config()` factory** — single line to build a
   fully-wired policy subsystem from configuration.  Builds maturity,
   health, impact, drift, tokens, policy engine, policy store, approval
   manager, and governance gate internally.
3. **Default policies auto-registered** — `from_config()` registers
   pre-trade, post-trade, and strategy constraint policy sets
   automatically from `RiskConfig` thresholds.
4. **Relative imports in canonical files** — all `policy/` modules use
   relative imports (`from .models import ...`) for within-package refs,
   and absolute imports for cross-package refs (`agentic_trading.core.*`).
5. **Shim identity preservation** — `from agentic_trading.governance.gate
   import GovernanceGate` returns the same class object as
   `from agentic_trading.policy.governance_gate import GovernanceGate`.
   34 tests verify this identity for all 15 modules.

### Test results

- 34 new tests passing (PolicyGate facade + shim identity checks).
- 1842 total tests passing (up from 1808), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 7)

Move `risk/` → `execution/risk/` and create the execution gateway.

---

## PR 7 — Risk → Execution/Risk + Execution Gateway

**Date:** 2025-02-17

### What changed

1. Moved all 9 `risk/` modules into `execution/risk/` — risk management
   now lives alongside the execution engine that consumes it.
2. Replaced all `risk/` modules with thin re-export shims.
3. Created `ExecutionGateway` facade that composes `ExecutionEngine` +
   `RiskManager` + `ExecutionQualityTracker` into a single high-level API.

### Directory moves

| Old path | New canonical path |
|----------|--------------------|
| `risk/__init__.py` | `execution/risk/__init__.py` |
| `risk/alerts.py` | `execution/risk/alerts.py` |
| `risk/circuit_breakers.py` | `execution/risk/circuit_breakers.py` |
| `risk/drawdown.py` | `execution/risk/drawdown.py` |
| `risk/exposure.py` | `execution/risk/exposure.py` |
| `risk/kill_switch.py` | `execution/risk/kill_switch.py` |
| `risk/manager.py` | `execution/risk/manager.py` |
| `risk/post_trade.py` | `execution/risk/post_trade.py` |
| `risk/pre_trade.py` | `execution/risk/pre_trade.py` |
| `risk/var_es.py` | `execution/risk/var_es.py` |

### New files

| File | Purpose |
|------|---------|
| `src/agentic_trading/execution/risk/__init__.py` | Package init with all 11 public re-exports |
| `src/agentic_trading/execution/gateway.py` | **`ExecutionGateway`** — unified facade wrapping execution engine + risk manager |
| `tests/unit/test_execution_gateway.py` | 20 tests — factory, lifecycle, risk passthrough, accessors, shim identity checks |

### Modified files (backward-compat re-export shims)

| File | Re-exports from |
|------|-----------------|
| `risk/__init__.py` | `execution.risk` (11 public names) |
| `risk/alerts.py` | `execution.risk.alerts` |
| `risk/circuit_breakers.py` | `execution.risk.circuit_breakers` |
| `risk/drawdown.py` | `execution.risk.drawdown` |
| `risk/exposure.py` | `execution.risk.exposure` |
| `risk/kill_switch.py` | `execution.risk.kill_switch` |
| `risk/manager.py` | `execution.risk.manager` |
| `risk/post_trade.py` | `execution.risk.post_trade` |
| `risk/pre_trade.py` | `execution.risk.pre_trade` |
| `risk/var_es.py` | `execution.risk.var_es` |

### `ExecutionGateway` API

```python
from agentic_trading.execution.gateway import ExecutionGateway

# Factory: builds RiskManager + ExecutionEngine from config
gateway = ExecutionGateway.from_config(
    adapter=paper_adapter,
    event_bus=event_bus,
    risk_config=risk_config,
)
await gateway.start()

# Engine handles order lifecycle with built-in risk checks
await gateway.engine.handle_intent(order_intent)

# Risk management
await gateway.activate_kill_switch("emergency")
await gateway.evaluate_circuit_breakers({"volatility": 0.8})
gateway.update_instruments(instruments)

# Component access
gateway.engine         # ExecutionEngine
gateway.risk_manager   # RiskManager
gateway.order_manager  # OrderManager
gateway.quality_tracker  # ExecutionQualityTracker
```

### Key design decisions

1. **Risk under execution** — risk management is the execution engine's
   primary dependency (pre-trade checks gate every order).  Colocating
   them in `execution/risk/` makes the dependency visible in the directory
   structure.
2. **No import changes needed** — all risk modules use relative imports
   for within-package references (e.g., `from .alerts import AlertEngine`)
   which work unchanged at the new location.  Cross-package imports to
   `agentic_trading.core.*` also work unchanged.
3. **`ExecutionGateway.from_config()` factory** — single line to build
   a fully-wired execution subsystem.  Constructs `RiskManager`,
   `ExecutionEngine`, and `ExecutionQualityTracker` internally.
4. **RiskManager is the single public API** — only 3 source files import
   `RiskManager` directly; 5 test files import sub-modules (all via shims).

### Test results

- 20 new tests passing (ExecutionGateway facade + shim identity checks).
- 1862 total tests passing (up from 1842), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 8)

Move `journal/` → `reconciliation/journal/` and extract the
reconciliation layer.

---

## PR 8 — Journal + Reconciliation → Reconciliation Layer

**Date:** 2025-02-17

### What changed

1. Moved all 15 `journal/` modules into `reconciliation/journal/` — trade
   journaling now lives under the reconciliation layer.
2. Moved `execution/reconciliation.py` → `reconciliation/loop.py` — the
   exchange reconciliation loop is colocated with the journal.
3. Replaced all `journal/` modules and `execution/reconciliation.py` with
   thin re-export shims.
4. Created `ReconciliationManager` facade that composes `TradeJournal` +
   `RollingTracker` + `QualityScorecard` + `ReconciliationLoop` into a
   single high-level API.

### Directory moves

| Old path | New canonical path |
|----------|--------------------|
| `journal/__init__.py` | `reconciliation/journal/__init__.py` |
| `journal/record.py` | `reconciliation/journal/record.py` |
| `journal/journal.py` | `reconciliation/journal/journal.py` |
| `journal/rolling_tracker.py` | `reconciliation/journal/rolling_tracker.py` |
| `journal/confidence.py` | `reconciliation/journal/confidence.py` |
| `journal/monte_carlo.py` | `reconciliation/journal/monte_carlo.py` |
| `journal/overtrading.py` | `reconciliation/journal/overtrading.py` |
| `journal/coin_flip.py` | `reconciliation/journal/coin_flip.py` |
| `journal/mistakes.py` | `reconciliation/journal/mistakes.py` |
| `journal/session_analysis.py` | `reconciliation/journal/session_analysis.py` |
| `journal/correlation.py` | `reconciliation/journal/correlation.py` |
| `journal/replay.py` | `reconciliation/journal/replay.py` |
| `journal/export.py` | `reconciliation/journal/export.py` |
| `journal/persistence.py` | `reconciliation/journal/persistence.py` |
| `journal/quality_scorecard.py` | `reconciliation/journal/quality_scorecard.py` |
| `execution/reconciliation.py` | `reconciliation/loop.py` |

### New files

| File | Purpose |
|------|---------|
| `src/agentic_trading/reconciliation/__init__.py` | Package init with all 23 public re-exports + `ReconciliationManager` |
| `src/agentic_trading/reconciliation/journal/__init__.py` | Journal sub-package (copied from `journal/__init__.py`) |
| `src/agentic_trading/reconciliation/manager.py` | **`ReconciliationManager`** — unified facade wrapping journal + analytics + recon loop |
| `tests/unit/test_reconciliation_manager.py` | 32 tests — factory, lifecycle, accessors, delegated ops, shim identity checks |

### Modified files (backward-compat re-export shims)

| File | Re-exports from |
|------|-----------------|
| `journal/__init__.py` | `reconciliation.journal` (21 public names) |
| `journal/record.py` | `reconciliation.journal.record` |
| `journal/journal.py` | `reconciliation.journal.journal` |
| `journal/rolling_tracker.py` | `reconciliation.journal.rolling_tracker` |
| `journal/confidence.py` | `reconciliation.journal.confidence` |
| `journal/monte_carlo.py` | `reconciliation.journal.monte_carlo` |
| `journal/overtrading.py` | `reconciliation.journal.overtrading` |
| `journal/coin_flip.py` | `reconciliation.journal.coin_flip` |
| `journal/mistakes.py` | `reconciliation.journal.mistakes` |
| `journal/session_analysis.py` | `reconciliation.journal.session_analysis` |
| `journal/correlation.py` | `reconciliation.journal.correlation` |
| `journal/replay.py` | `reconciliation.journal.replay` |
| `journal/export.py` | `reconciliation.journal.export` |
| `journal/persistence.py` | `reconciliation.journal.persistence` |
| `journal/quality_scorecard.py` | `reconciliation.journal.quality_scorecard` |
| `execution/reconciliation.py` | `reconciliation.loop` |

### Import fixes in canonical files

| File | Old import | New import |
|------|-----------|------------|
| `reconciliation/journal/persistence.py` | `from ..storage.postgres.models import ...` | `from agentic_trading.storage.postgres.models import ...` |
| `reconciliation/loop.py` | `from .order_manager import OrderManager` | `from agentic_trading.execution.order_manager import OrderManager` |

### `ReconciliationManager` API

```python
from agentic_trading.reconciliation.manager import ReconciliationManager

# Factory: builds TradeJournal + analytics + recon loop from config
mgr = ReconciliationManager.from_config(
    adapter=paper_adapter,
    event_bus=event_bus,
    order_manager=order_manager,
)
await mgr.start()

# Journal operations
mgr.journal.open_trade(trace_id, strategy_id, ...)
trades = mgr.get_open_trades()
closed = mgr.get_closed_trades()

# Analytics
mgr.rolling_tracker  # RollingTracker
mgr.quality_scorecard  # QualityScorecard

# Exchange reconciliation
await mgr.reconcile()

# Backtest mode (no adapter → no recon loop)
mgr = ReconciliationManager.from_config()  # journal-only
```

### Key design decisions

1. **Journal under reconciliation** — the trade journal is the data source
   that reconciliation acts on.  Colocating them under `reconciliation/`
   makes the data → verification relationship explicit.
2. **`ReconciliationLoop` extracted from execution** — the recon loop
   previously lived under `execution/` but its concern is state verification,
   not order execution.  Moving it to `reconciliation/` reflects its true
   responsibility.
3. **Absolute imports for cross-package** — `persistence.py` changed from
   relative `..storage.postgres.models` to absolute
   `agentic_trading.storage.postgres.models` since the package depth changed.
   `loop.py` changed from relative `.order_manager` to absolute
   `agentic_trading.execution.order_manager`.
4. **Backtest-safe factory** — `from_config()` creates journal + analytics
   even without an adapter.  The recon loop is only created when all three
   required components (adapter, event_bus, order_manager) are provided.
5. **Private function shim fix** — the `quality_scorecard` shim explicitly
   re-imports private `_grade_*` helper functions used by existing tests,
   since `import *` skips underscore-prefixed names.

### Test results

- 32 new tests passing (ReconciliationManager facade + shim identity checks).
- 1897 total tests passing (up from 1862), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 9)

Extract the `IntelligenceManager` facade that unifies data feeds, feature
engine, and analysis under the intelligence layer.

---

## PR 9 — IntelligenceManager Facade

**Date:** 2025-02-17

### What changed

Created a unified `IntelligenceManager` facade that composes all
intelligence layer components — data ingestion (FeedManager,
CandleBuilder), feature computation (FeatureEngine), historical data
loading, data quality validation, and analysis tools (HTFAnalyzer,
SMCConfluenceScorer, SMCTradePlanGenerator) — into a single entry
point with a `from_config()` factory.

**No file moves** — the intelligence layer was already restructured
in PRs 3 and 4.  This PR adds only the facade and its tests.

### New files

| File | Purpose |
|------|---------|
| `src/agentic_trading/intelligence/manager.py` | **`IntelligenceManager`** — unified facade for the intelligence layer |
| `tests/unit/test_intelligence_manager.py` | 31 tests — factory, lifecycle, accessors, delegated ops, package imports |

### Modified files

| File | Change |
|------|--------|
| `src/agentic_trading/intelligence/__init__.py` | Added `IntelligenceManager` export and `__all__` |

### `IntelligenceManager` API

```python
from agentic_trading.intelligence.manager import IntelligenceManager

# Live / paper mode
mgr = IntelligenceManager.from_config(
    event_bus=event_bus,
    exchange_configs=settings.exchanges,
    symbols=["BTC/USDT"],
)
await mgr.start()

# Backtest mode (no event bus, no feeds)
mgr = IntelligenceManager.from_config(data_dir="data/historical")
candles = mgr.load_candles(Exchange.BYBIT, "BTC/USDT", Timeframe.M1)
for c in candles:
    mgr.add_candle(c)
fv = mgr.compute_features("BTC/USDT", Timeframe.M1, candles)

# Analysis
assessment = mgr.analyze_htf("BTC/USDT", aligned_features)
confluence = mgr.score_smc_confluence("BTC/USDT", aligned_features)
report = mgr.generate_smc_report("BTC/USDT", price, aligned_features)
plan = mgr.generate_trade_plan(report)

# Data quality
issues = mgr.check_data_quality(candles, Timeframe.M1)

# Component access
mgr.feature_engine     # FeatureEngine
mgr.candle_builder     # CandleBuilder (None in backtest)
mgr.feed_manager       # FeedManager (None in backtest)
mgr.historical_loader  # HistoricalDataLoader
mgr.data_qa            # DataQualityChecker
mgr.htf_analyzer       # HTFAnalyzer
mgr.smc_scorer         # SMCConfluenceScorer
mgr.trade_plan_generator  # SMCTradePlanGenerator
```

### Key design decisions

1. **Facade-only PR** — no file moves, no shims.  The intelligence
   directory was already canonical from PRs 3-4.  This PR purely adds
   the composed entry point.
2. **Mode-aware factory** — `from_config()` adapts to available
   resources: with `event_bus` → creates CandleBuilder (and optionally
   FeedManager if exchange configs present).  Without `event_bus` →
   creates a backtest-mode manager with direct-compute methods.
3. **Lazy imports** — heavy dependencies (FeedManager, CandleBuilder)
   are imported inside the factory rather than at module level, keeping
   import time fast for backtest-only usage.
4. **DataQualityChecker has per-candle API** — the facade's
   `check_data_quality()` delegates only `check_gaps()` (which takes
   a candle series).  Per-candle checks (staleness, price sanity,
   volume anomaly) are accessible via the `data_qa` property.
5. **Analysis tools always available** — HTFAnalyzer,
   SMCConfluenceScorer, and SMCTradePlanGenerator are stateless and
   cheap to instantiate, so `from_config()` always creates them.

### Test results

- 31 new tests passing (IntelligenceManager facade).
- 1928 total tests passing (up from 1897), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 10)

Create the `SignalManager` facade that unifies strategy dispatch and
portfolio management under the signal layer.

---

## PR 10 — SignalManager Facade

**Date:** 2025-02-17

### What changed

Created a unified `SignalManager` facade that composes the strategy
dispatch pipeline — `StrategyRunner`, `PortfolioManager`,
`PortfolioAllocator`, `CorrelationRiskAnalyzer`, and
`build_order_intents()` — into a single entry point with a
`from_config()` factory.

**No file moves** — the signal layer was already restructured in
PRs 4 and 5.  This PR adds only the facade and its tests.

### New files

| File | Purpose |
|------|---------|
| `src/agentic_trading/signal/manager.py` | **`SignalManager`** — unified facade for the signal layer |
| `tests/unit/test_signal_manager.py` | 31 tests — factory, lifecycle, accessors, delegated ops, package imports |

### Modified files

| File | Change |
|------|--------|
| `src/agentic_trading/signal/__init__.py` | Added `SignalManager` export and `__all__` |

### `SignalManager` API

```python
from agentic_trading.signal.manager import SignalManager

# Live / paper mode (with strategies + event bus)
mgr = SignalManager.from_config(
    strategy_ids=["trend_following", "mean_reversion"],
    feature_engine=feature_engine,
    event_bus=event_bus,
)
await mgr.start(ctx)

# Backtest / manual mode (no event bus)
mgr = SignalManager.from_config(
    max_position_pct=0.05,
    sizing_multiplier=0.75,
)
mgr.on_signal(signal)
targets = mgr.generate_targets(ctx, capital=100_000)
allocated = mgr.allocate(targets, portfolio, capital=100_000)
intents = SignalManager.build_intents(allocated, Exchange.BYBIT, now)

# Correlation tracking
mgr.update_returns("BTC/USDT", 0.01)
clusters = mgr.find_correlation_clusters()

# Component access
mgr.runner              # StrategyRunner (None without bus)
mgr.portfolio_manager   # PortfolioManager
mgr.allocator           # PortfolioAllocator
mgr.correlation_analyzer  # CorrelationRiskAnalyzer
mgr.strategies          # list[BaseStrategy]
mgr.signal_count        # int
```

### Key design decisions

1. **Facade-only PR** — no file moves, no shims.  The signal directory
   was already canonical from PRs 4-5.  This PR purely adds the
   composed entry point.
2. **Mode-aware factory** — `from_config()` only creates the
   `StrategyRunner` when both `feature_engine` and `event_bus` are
   provided.  Portfolio management (sizing, allocation, correlation)
   is always available for direct use.
3. **Strategy resolution** — accepts either pre-built `strategies` list
   or `strategy_ids` that are resolved via the strategy registry.
4. **Allocation with correlation** — `allocate()` automatically
   incorporates correlation clusters from the `CorrelationRiskAnalyzer`
   when available, bridging the gap between position sizing and
   cross-asset risk management.
5. **Static intent builder** — `build_intents()` is a static method
   since it's a pure conversion function with no state dependency.
6. **Lazy imports** — heavy dependencies (StrategyRunner, registry)
   are imported inside the factory to keep module-level imports light.

### Test results

- 31 new tests passing (SignalManager facade).
- 1959 total tests passing (up from 1928), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 11)

Create the `BusManager` facade that unifies event bus creation,
topic registry, and bus observability under the bus layer.

---

## PR 11 — BusManager Facade

**Date:** 2025-02-17

### What changed

Created a unified `BusManager` facade that composes both event buses
— the legacy topic-routed bus (`MemoryEventBus` / `RedisStreamsBus`)
and the new type-routed domain event bus (`InMemoryEventBus`) — along
with the topic schema registry and bus observability, into a single
entry point with a `from_config()` factory.

**No file moves** — the bus layer was already restructured in PR 3.
This PR adds only the facade and its tests.

### New files

| File | Purpose |
|------|---------|
| `src/agentic_trading/bus/manager.py` | **`BusManager`** — unified facade for the bus layer (dual-bus) |
| `tests/unit/test_bus_manager.py` | 29 tests — factory, lifecycle, legacy pub/sub, domain pub/sub, observability, metrics, schema helpers, package imports |

### Modified files

| File | Change |
|------|--------|
| `src/agentic_trading/bus/__init__.py` | Added `BusManager` export and `__all__` |

### `BusManager` API

```python
from agentic_trading.bus.manager import BusManager

# Backtest mode (legacy bus only)
mgr = BusManager.from_config(mode=Mode.BACKTEST)
await mgr.start()

# Paper/live mode (with Redis + domain bus)
mgr = BusManager.from_config(
    mode=Mode.PAPER,
    redis_url="redis://localhost:6379/0",
    enable_domain_bus=True,
    enforce_ownership=True,
    event_store=event_store,
)
await mgr.start()

# Legacy bus — topic-routed publish/subscribe
await mgr.publish("strategy.signal", signal_event)
await mgr.subscribe("feature.vector", "runner", handler)

# Domain bus — type-routed publish/subscribe
await mgr.publish_domain(signal_created_event)
mgr.subscribe_domain(SignalCreated, handler)

# Observability — legacy bus
mgr.messages_processed          # int
mgr.get_error_counts()          # dict[str, int]
mgr.get_dead_letters()          # list
mgr.clear_dead_letters()        # list

# Observability — domain bus
mgr.domain_messages_processed   # int (0 if disabled)
mgr.get_domain_error_counts()   # dict[str, int] ({} if disabled)
mgr.get_domain_dead_letters()   # list ([] if disabled)

# Unified metrics (both buses combined)
metrics = mgr.get_metrics()

# Schema registry helpers
topic = BusManager.get_topic_for_event(signal)  # "strategy.signal"
cls = BusManager.get_event_class("Signal")       # Signal class
topics = BusManager.list_topics()                 # sorted list

# Component access
mgr.legacy_bus    # MemoryEventBus or RedisStreamsBus
mgr.domain_bus    # InMemoryEventBus or None
mgr.is_running    # bool

await mgr.stop()
```

### Key design decisions

1. **Facade-only PR** — no file moves, no shims.  The bus directory
   was already canonical from PR 3.  This PR purely adds the composed
   entry point.
2. **Dual-bus architecture** — the facade owns both the legacy
   topic-routed bus and the new type-routed domain event bus.  The
   domain bus is opt-in via `enable_domain_bus=True` to maintain
   backward compatibility.
3. **Mode-aware factory** — `from_config()` delegates to
   `create_event_bus()` for the legacy bus (which selects
   `MemoryEventBus` for backtest or `RedisStreamsBus` for
   paper/live).  The domain bus is independently configured.
4. **Graceful degradation** — domain bus methods return sensible
   defaults when disabled (`0`, `{}`, `[]`) rather than raising.
   Only `publish_domain()` and `subscribe_domain()` raise
   `RuntimeError` when the domain bus is not enabled, since silently
   dropping domain events would mask bugs.
5. **Schema registry as static methods** — `get_topic_for_event()`,
   `get_event_class()`, and `list_topics()` are static since they
   query the global schema registry and don't depend on instance state.
6. **Unified metrics** — `get_metrics()` combines stats from both
   buses into a single dict, with domain bus metrics included only
   when enabled.

### Test results

- 29 new tests passing (BusManager facade).
- 1988 total tests passing (up from 1959), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 12)

Create the top-level `Orchestrator` that wires the layer managers
(`IntelligenceManager`, `SignalManager`, `BusManager`,
`ExecutionGateway`, `PolicyGate`, `ReconciliationManager`) into
a single bootstrap entry point, replacing the monolithic `main.py`.

---

## PR 12 — Orchestrator (Top-Level Bootstrap Facade)

**Date:** 2025-02-17

### What changed

Created a top-level `Orchestrator` facade that composes all six layer
managers — `BusManager`, `IntelligenceManager`, `SignalManager`,
`ExecutionGateway`, `PolicyGate`, `ReconciliationManager` — along with
the clock and `TradingContext`, into a single `from_config()` factory
that replaces the procedural wiring in `main.py`.

**No file moves** — this PR adds only the facade and its tests.
The existing `main.py` is left untouched for backward compatibility;
callers can continue using it or switch to the Orchestrator.

### New files

| File | Purpose |
|------|---------|
| `src/agentic_trading/orchestrator.py` | **`Orchestrator`** — top-level facade composing all six layer managers |
| `tests/unit/test_orchestrator.py` | 37 tests — factory (backtest/paper/governance), accessors, lifecycle, metrics, layer names, strategies, imports |

### `Orchestrator` API

```python
from agentic_trading.orchestrator import Orchestrator
from agentic_trading.core.config import Settings, load_settings

settings = load_settings("configs/paper.toml")

# Build fully wired Orchestrator
orch = Orchestrator.from_config(
    settings,
    adapter=paper_adapter,       # optional pre-built adapter
    tool_gateway=tool_gateway,   # optional control-plane gateway
)

# Lifecycle
await orch.start()   # bus → intelligence → execution → recon → signal
await orch.stop()    # reverse order

# Layer accessors
orch.bus              # BusManager
orch.intelligence     # IntelligenceManager
orch.signal           # SignalManager
orch.execution        # ExecutionGateway (None in backtest)
orch.policy           # PolicyGate (None when governance disabled)
orch.reconciliation   # ReconciliationManager

# Context and config
orch.ctx              # TradingContext (clock + event bus + instruments)
orch.settings         # Settings
orch.mode             # Mode enum
orch.is_backtest      # bool

# Aggregated metrics from all layers
metrics = orch.get_metrics()
# {
#   "mode": "paper",
#   "bus": { ... },
#   "signal_count": 42,
#   "strategy_count": 3,
#   "open_trades": 2,
#   "closed_trades": 15,
# }
```

### Key design decisions

1. **Facade-only PR** — no changes to `main.py`.  The existing bootstrap
   still works.  The Orchestrator is an alternative entry point that
   composes the same layer managers through their `from_config()`
   factories.
2. **Mode-aware construction** — `from_config()` uses `settings.mode` to
   decide which components to create:
   - **Backtest**: SimClock, MemoryEventBus, no execution gateway, no
     feed manager, no reconciliation loop.
   - **Paper/Live**: WallClock, RedisStreamsBus (via BusManager), full
     execution gateway (when adapter provided), live feeds, recon loop.
3. **Dependency-ordered lifecycle** — `start()` brings layers up in
   dependency order: bus → intelligence → execution → reconciliation →
   signal.  `stop()` reverses the order.
4. **Optional layers** — execution is `None` when no adapter is provided
   or in backtest mode.  Policy is `None` when governance is disabled.
   All code guards on `is not None`.
5. **Governance cross-wiring** — when governance is enabled, the
   PolicyGate's health tracker and drift detector are passed to the
   ReconciliationManager's journal, and the governance gate is wired
   into the ExecutionGateway.
6. **Unified metrics** — `get_metrics()` collects bus metrics, signal
   count, strategy count, and trade counts into a single dict for
   dashboards and monitoring.

### Test results

- 37 new tests passing (Orchestrator facade).
- 2025 total tests passing (up from 1988), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 13)

Wire `main.py` to use the `Orchestrator`, simplifying the 2000+ line
bootstrap into a thin shell that delegates to
`Orchestrator.from_config()` / `start()` / `stop()`.

---

## PR 13 — Wire `main.py` to Orchestrator

**Date:** 2025-02-17

### What changed

Refactored `main.py` to delegate clock, event bus, `TradingContext`,
and layer manager construction to `Orchestrator.from_config()`.  The
old manual steps 4–7 (SimClock/WallClock, `create_event_bus()`,
`TradingContext()`, `event_bus.start()`) are replaced by three lines:

```python
orch = Orchestrator.from_config(settings)
ctx = orch.ctx
event_bus = orch.bus.legacy_bus
```

**Backward compatibility**: no behaviour changes.  The same objects
are constructed with the same parameters — just through the
Orchestrator factory instead of inline.  All mode-specific runtime
logic (`_run_backtest`, `_run_live_or_paper`, `run_walk_forward`,
`run_optimize`) is left untouched; it still receives `(settings, ctx)`
and operates on the same `TradingContext` and event bus.

### Modified files

| File | Change |
|------|--------|
| `src/agentic_trading/main.py` | `run()` steps 4-7 replaced by `Orchestrator.from_config()` + `orch.bus.start()/stop()`. Removed direct imports of `SimClock`, `WallClock`, `create_event_bus`, `PortfolioState`. |

### New files

| File | Purpose |
|------|---------|
| `tests/unit/test_main_orchestrator.py` | 29 tests — module import structure, Orchestrator construction from `run()`, object type validation, CLI preserved, lifecycle integration |

### What was removed from `main.py` imports

```python
# BEFORE (direct construction)
from .core.clock import SimClock, WallClock
from .core.interfaces import PortfolioState, TradingContext
from .event_bus.bus import create_event_bus

# AFTER (via Orchestrator)
from .orchestrator import Orchestrator
from .core.interfaces import PortfolioState, TradingContext  # still needed by _run_live_or_paper
```

### Key design decisions

1. **Minimal diff** — only the `run()` function was changed; none of
   the 1500+ line `_run_live_or_paper`, `_run_backtest`,
   `run_walk_forward`, or `run_optimize` functions were modified.
2. **No behavioural changes** — the Orchestrator constructs the same
   `SimClock`/`WallClock`, the same `MemoryEventBus`/`RedisStreamsBus`,
   and the same `TradingContext` that `main.py` previously built inline.
3. **Event bus alias preserved** — `event_bus = orch.bus.legacy_bus`
   keeps the `_run_live_or_paper` function working unchanged, since it
   references `ctx.event_bus` which is the same object.
4. **CLI unaffected** — `cli.py` still imports `from .main import run`
   and all click commands work identically.
5. **Future refactoring scope** — the remaining ~1500 lines of runtime
   glue in `_run_live_or_paper` (narration, analytics callbacks, fill
   handlers, UI, agents) can be progressively migrated to layer
   managers in future PRs.

### Test results

- 29 new tests passing (main ↔ Orchestrator integration).
- 2054 total tests passing (up from 2025), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 14)

Begin migrating runtime glue from `_run_live_or_paper` into layer
managers — starting with the fill handler and signal handler which
can be absorbed by `ReconciliationManager` and `SignalManager`
respectively.

---

## PR 14 — Extract Fill Handler + Signal Pipeline into Layer Managers

**Date:** 2025-02-17

### What changed

Migrated two large blocks of procedural glue from `_run_live_or_paper`
into their owning layer facades:

1. **Fill handler → `ReconciliationManager`**: The ~140-line
   `on_execution_event` fill-classification and journal-recording logic
   is now encapsulated in `ReconciliationManager.handle_fill()`, which
   returns a structured `FillResult` dataclass.  The caller (main.py)
   retains only narration, TP/SL placement, and Prometheus metrics.

2. **Position reconciliation → `ReconciliationManager`**: The
   `_reconcile_journal_positions` helper is replaced by
   `ReconciliationManager.reconcile_positions()`, which returns a list
   of force-closed trace IDs.

3. **Signal pipeline → `SignalManager`**: The ~60-line signal→portfolio→
   intent pipeline (both directional entries and FLAT exits) is
   encapsulated in `SignalManager.process_signal()`, which returns a
   structured `SignalResult` dataclass.

4. **Feature aliasing → `SignalManager`**: The inline indicator aliasing
   (adx_14→adx, atr_14→atr, etc.) is now a `SignalManager.alias_features()`
   static method.

5. **Main.py refactored**: `_run_live_or_paper` now constructs local
   `_signal_mgr` and `_recon_mgr` facades and delegates to their new
   methods.  Narration, TP/SL, and metrics remain inline.

### Modified files

| File | Change |
|------|--------|
| `src/agentic_trading/reconciliation/manager.py` | Added `FillResult` dataclass, `handle_fill()`, `reconcile_positions()` |
| `src/agentic_trading/signal/manager.py` | Added `SignalResult` dataclass, `process_signal()`, `alias_features()`, `_FEATURE_ALIASES` |
| `src/agentic_trading/main.py` | `on_execution_event` delegates to `_recon_mgr.handle_fill()` + `reconcile_positions()`. `on_feature_vector` delegates to `_signal_mgr.process_signal()` + `alias_features()`. Removed inline `_reconcile_journal_positions` function. Constructs local `_signal_mgr` and `_recon_mgr` from existing components. |

### New files

| File | Purpose |
|------|---------|
| `tests/unit/test_recon_handle_fill.py` | 16 tests — entry/exit fill classification, fallback strategy IDs, FillResult dataclass, reconcile_positions (orphan detection, CCXT format, trace IDs) |
| `tests/unit/test_signal_process.py` | 21 tests — alias_features (all aliases, non-mutation, no-overwrite), process_signal entries (targets, cache population), process_signal exits (FLAT, exit_map, zero-qty), SignalResult dataclass |

### New API surface

```python
# ReconciliationManager
@dataclass
class FillResult:
    is_exit: bool
    strategy_id: str
    entry_trace_id: str | None = None
    direction: str = ""

def handle_fill(fill_event, signal_cache, exit_map,
                *, fallback_strategy_ids=None) -> FillResult: ...
def reconcile_positions(exchange_positions) -> list[str]: ...

# SignalManager
@dataclass
class SignalResult:
    intents: list = field(default_factory=list)
    is_exit: bool = False
    exit_trace_id: str | None = None
    entry_trace_id: str | None = None

def process_signal(signal, journal, ctx, exchange, capital,
                   *, signal_cache=None, exit_map=None) -> SignalResult: ...

@staticmethod
def alias_features(features: dict) -> dict: ...

_FEATURE_ALIASES = {"adx_14": "adx", "atr_14": "atr", "rsi_14": "rsi",
                    "donchian_upper_20": "donchian_upper",
                    "donchian_lower_20": "donchian_lower"}
```

### Key design decisions

1. **Structured result types** — `FillResult` and `SignalResult`
   dataclasses carry enough context for callers to perform follow-up
   actions (narration, TP/SL, metrics) without re-deriving the
   entry-vs-exit classification.
2. **Narration/TP/SL remain in main.py** — these have deep
   dependencies on narration services, exchange adapters, tool gateways,
   and settings that don't belong in the data-layer facades.  They will
   be migrated to dedicated services in a later PR.
3. **Local facade instances** — `_run_live_or_paper` constructs
   `_signal_mgr` and `_recon_mgr` from its locally-created
   `portfolio_manager` and `journal`.  In a future PR, these will be
   replaced by the Orchestrator-owned managers.
4. **Exit map consumed on read** — `handle_fill()` pops the exit_map
   entry so it's consumed once, matching the original inline behaviour.
5. **Feature aliases are data, not code** — `_FEATURE_ALIASES` dict
   makes it trivial to add new aliases without modifying control flow.
6. **No behavioural changes** — the same fills, the same intents,
   the same journal entries are produced.  The refactoring is purely
   structural.

### Test results

- 37 new tests passing (16 recon + 21 signal).
- 2091 total tests passing (up from 2054), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 15)

Wire `_run_live_or_paper` to use the Orchestrator-owned layer managers
directly (instead of constructing its own components), and begin
migrating remaining glue (narration, analytics callbacks, UI) into
dedicated service objects.

---

## PR 15 — Wire _run_live_or_paper to Orchestrator-Owned Components

**Date:** 2025-02-17

### What changed

Replaced five locally-constructed components inside `_run_live_or_paper`
with their Orchestrator-owned equivalents, eliminating duplicate
construction and ensuring a single source of truth for each layer.

1. **FeatureEngine**: `feature_engine = orch.intelligence.feature_engine`
   replaces `FeatureEngine(event_bus=ctx.event_bus)`.  The Orchestrator's
   IntelligenceManager builds the feature engine with the correct event
   bus and exchange configs from settings.

2. **Strategies**: `strategies = orch.signal.strategies` replaces the
   manual strategy module import + `create_strategy()` loop.  A fallback
   to `TrendFollowingStrategy()` is retained for when no strategies are
   configured.  Twelve `import agentic_trading.strategies.*` lines
   removed.

3. **PortfolioManager**: The local `PortfolioManager(...)` construction
   (with safe_mode sizing, governance sizing fn) is eliminated.  The
   Orchestrator's SignalManager already builds and owns the correctly-
   configured PortfolioManager.

4. **SignalManager**: `_signal_mgr = orch.signal` replaces the local
   `_SignalMgr(runner=None, portfolio_manager=portfolio_manager)`.

5. **ReconciliationManager**: `_recon_mgr = orch.reconciliation` replaces
   the local `_ReconMgr(journal=journal)`.  The local journal (which
   carries the analytics `_on_trade_closed` callback and DB persistence
   wrapper) is injected via `_recon_mgr._journal = journal` so fills
   and reconciliation drive the correct Prometheus metrics pipeline.

### _run_live_or_paper signature

```python
async def _run_live_or_paper(
    settings: Settings,
    ctx: TradingContext,
    orch: Orchestrator,          # ← new parameter
) -> None:
```

`run()` passes the Orchestrator:
```python
await _run_live_or_paper(settings, ctx, orch)
```

### Modified files

| File | Change |
|------|--------|
| `src/agentic_trading/main.py` | `_run_live_or_paper` signature takes `orch: Orchestrator`. Feature engine, strategies, portfolio manager, `_signal_mgr`, `_recon_mgr` sourced from orch. Removed `PortfolioManager`, `build_order_intents` imports. Removed 12 strategy module imports. Journal injected into orch.reconciliation. |

### New files

| File | Purpose |
|------|---------|
| `tests/unit/test_main_orch_wiring.py` | 31 tests — source-level assertions (no local construction, orch references present, signature correct), runtime wiring (portfolio_manager, process_signal, handle_fill, feature_engine, strategies), sizing propagation, journal injection, backtest path unchanged |

### Removed local construction

| Component | Old (PR 14) | New (PR 15) |
|-----------|-------------|-------------|
| `feature_engine` | `FeatureEngine(event_bus=ctx.event_bus)` | `orch.intelligence.feature_engine` |
| `strategies` | 12 imports + `create_strategy()` loop | `orch.signal.strategies` |
| `portfolio_manager` | `PortfolioManager(max_position_pct=..., sizing_multiplier=..., governance_sizing_fn=...)` | `orch.signal.portfolio_manager` |
| `_signal_mgr` | `SignalManager(runner=None, portfolio_manager=portfolio_manager)` | `orch.signal` |
| `_recon_mgr` | `ReconciliationManager(journal=journal)` | `orch.reconciliation` (journal injected) |

### Key design decisions

1. **Journal injection** — `_recon_mgr._journal = journal` is used to
   inject the locally-created journal (with its analytics callback chain
   and DB persistence wrapper) into the Orchestrator's
   ReconciliationManager.  This is a pragmatic bridge: the journal's
   `_on_trade_closed` callback wires 9 analytics components and
   Prometheus metrics, making it impractical to move wholesale.
2. **Fallback strategy preserved** — When `orch.signal.strategies` is
   empty (no strategy configs), `_run_live_or_paper` still falls back to
   creating a `TrendFollowingStrategy`.  This preserves backward
   compatibility for users who run without strategy configuration.
3. **Backtest path unchanged** — `_run_backtest` still constructs its own
   `FeatureEngine` and strategies.  Backtest mode uses different
   components (SimClock, BacktestEngine) and doesn't benefit from
   Orchestrator wiring yet.
4. **No behavioural changes** — All runtime behaviour is identical.  The
   same signals, fills, journal entries, TP/SL orders, and metrics are
   produced.

### Test results

- 31 new tests passing.
- 2122 total tests passing (up from 2091), 0 regressions.
- 1 pre-existing failure (unchanged).

### What's next (PR 16)

Migrate remaining inline glue from `_run_live_or_paper` — narration
service wiring, governance setup, startup reconciliation, and analytics
callback chains — into dedicated service objects or layer methods,
completing the extraction of `main.py` into a thin orchestration shell.
