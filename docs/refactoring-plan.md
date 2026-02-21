# Refactoring Plan: Alignment with Institutional Design Specification

## Design Spec Summary

The design describes an institutional-grade agentic trading platform with:
- A **deterministic control plane** (action boundary → policy evaluation → approval → idempotent tool call → audit) as the single path for all side effects.
- **Rich feature engineering**: ARIMA forecasts, Fourier/FFT components, MBO microstructure features, and technical indicators — all versioned and snapshot-persisted for replay.
- **Predictive modelling lifecycle**: ML models (LSTM/dense) with a ModelRegistry, walk-forward validation, and a research → paper → limited → production promotion pipeline.
- **Granular pre-trade controls**: price collars, message throttles, self-match prevention alongside existing order/position/leverage limits.
- **Surveillance & compliance**: a SurveillanceAgent for spoofing/layering/wash-trade detection, case management, regulatory mapping, information barriers.
- **Measurement system**: daily effectiveness scorecard (edge quality, execution quality, risk discipline, operational integrity) driving automatic promotions/demotions.
- **Minimal agent count**: core agents only (MarketIntelligence, FeatureComputation, Signal, Portfolio, ExecutionPlanner, PolicyGate, Approval, Execution, Reconciliation, IncidentResponse, GovernanceCanary, Surveillance, Reporting).

---

## Gap Summary

### Already Aligned (no changes needed)
- **Control plane pipeline**: `ToolGateway` → `CPPolicyEvaluator` → `CPApprovalService` → `AuditLog` — matches the ProposedAction → PolicyEvaluated → Approval → ToolCallRecorded flow.
- **OrderLifecycle FSM**: deterministic state machine with timeouts in `control_plane/state_machine.py`.
- **Approval tiers**: T0–T3 with auto-approve and escalation — aligns with design's Tier 0–3.
- **Fail-closed contract**: policy errors → BLOCK, audit unavailable → BLOCK, approval errors → BLOCK.
- **Idempotency enforcement**: ToolGateway caches by `idempotency_key`.
- **Kill switch**: Redis-backed with system-topic event publishing.
- **Policy engine**: declarative rules, versioned sets, shadow mode, scoping — matches design's PolicyGateAgent.
- **GovernanceCanary**: already monitoring governance health.
- **DailyEffectivenessScorecard**: 4-score structure matches design; formulas are close.
- **Reasoning/audit trace**: `SoteriaTrace`, `AgentConversation`, `PipelineLog` provide deep audit trails.

### Gaps

| # | Gap | Severity | Current State |
|---|-----|----------|---------------|
| G1 | **FeatureSnapshot persistence** — no persisted snapshot of feature vector + signal + model version at decision time | P0 | Feature vectors computed and published but not persisted with DecisionAudit |
| G2 | **ARIMA forecasts** — no time-series forecasting features | P1 | FeatureEngine has ~80+ indicators but no ARIMA |
| G3 | **Fourier/FFT components** — no spectral analysis features | P1 | Not implemented |
| G4 | **MBO microstructure features** — no order-book imbalance, depth, trade intensity | P1 | OrderbookEngine exists but limited; no MBO feed |
| G5 | **Price collars** — no limit-price band check vs reference price | P0 | PreTradeChecker has 9 checks but no price collar |
| G6 | **Message throttles (policy-level)** — no per-strategy/instrument/venue rate limits as policy rules | P0 | ToolGateway has per-tool rate limiting but not policy-declared |
| G7 | **Self-match prevention** — no check against resting orders on same venue | P0 | Not implemented |
| G8 | **SurveillanceAgent** — no spoofing/layering/wash-trade detection | P1 | Not implemented |
| G9 | **Case management** — no compliance case lifecycle, evidence bundling, escalation | P1 | Not implemented |
| G10 | **Regulatory mapping** — policies lack references to external regulations | P2 | PolicyRule/PolicySet have no regulatory fields |
| G11 | **Information barriers** — no role-based data segregation | P2 | Not implemented |
| G12 | **ExecutionPlannerAgent** — no order slicing, venue selection, contingencies | P1 | Orders submitted as single slices |
| G13 | **ModelRegistry** — no formal model versioning, training hash, approval status | P1 | No registry; strategies are code-defined |
| G14 | **Degraded modes incomplete** — missing CAUTIOUS and STOP_NEW_ORDERS | P1 | Has NORMAL, RISK_OFF_ONLY, READ_ONLY, FULL_STOP |
| G15 | **ReportingAgent** — no agent wrapper producing daily reports | P2 | `DailyEffectivenessScorecard` class exists but not agent-wrapped |
| G16 | **Scorecard formula gaps** — missing information ratio, Sharpe, participation rate, VaR coverage, recon break count, incident resolution time | P1 | Scorecard exists but uses simpler proxies |
| G17 | **Audit bundle generation** — no composite audit export for external inquiries | P2 | Individual traces exist but no bundler |
| G18 | **Feature versioning** — no `feature_version` hash stored with computed vectors | P1 | Features computed without version provenance |
| G19 | **FeatureComputationAgent** — design separates feature computation from market data ingestion | P2 | Both in MarketIntelligenceAgent today |
| G20 | **Execution quality tracker** — participation rate, opportunity cost, latency metrics incomplete | P1 | Prometheus histograms exist; no per-plan tracker |

---

## Refactor Actions

### 1. Control Changes (Action Boundary & Policy)

#### P0: Price Collars
- **File**: `src/agentic_trading/execution/risk/pre_trade.py`
- **Action**: Add `_check_price_collar()` to `PreTradeChecker`. Reject limit orders whose price exceeds a configurable band (e.g., ±2%) around reference price (mid or last trade).
- **Also**: Add `PRICE_COLLAR` to `PolicyType` enum; add default rule in `policy/default_policies.py::build_pre_trade_policies()`.
- **Schema**: Add `reference_price` and `price_collar_bps` fields to `RiskConfig` in `core/config.py`.

#### P0: Self-Match Prevention
- **File**: `src/agentic_trading/execution/risk/pre_trade.py`
- **Action**: Add `_check_self_match(intent, open_orders)` — compare intent side+price against resting orders on the same symbol/venue. Requires `open_orders` passed from `ExecutionEngine` (already available via adapter or ToolGateway read).
- **Signature change**: `PreTradeChecker.check()` needs an `open_orders: list[Order]` parameter.

#### P0: Message Throttles as Policy Rules
- **File**: `src/agentic_trading/policy/default_policies.py`
- **Action**: Add `build_message_throttle_policies()` factory — rules keyed on `messages_this_minute` per strategy, per symbol, per venue.
- **File**: `src/agentic_trading/execution/risk/pre_trade.py`
- **Action**: Add `_check_message_throttle()` with stateful per-(strategy, symbol) counters and sliding window.
- **Config**: Add `max_messages_per_minute_per_strategy` and `max_messages_per_minute_per_symbol` to `RiskConfig`.

#### P1: Degraded Mode Expansion
- **File**: `src/agentic_trading/control_plane/action_types.py`
- **Action**: Add `CAUTIOUS` and `STOP_NEW_ORDERS` to `DegradedMode` enum.
- **File**: `src/agentic_trading/control_plane/policy_evaluator.py`
- **Action**: Add mode-specific behaviour:
  - `CAUTIOUS` → tightened limits (sizing multiplier = 0.5, no new symbols).
  - `STOP_NEW_ORDERS` → block `SUBMIT_ORDER`/`BATCH_SUBMIT_ORDERS`; allow cancels.
- **File**: `src/agentic_trading/agents/incident_response.py`
- **Action**: Map new severity triggers to CAUTIOUS and STOP_NEW_ORDERS.

#### P2: Regulatory Mapping on PolicyRule
- **File**: `src/agentic_trading/policy/models.py` (or `governance/policy_models.py`)
- **Action**: Add optional `regulatory_refs: list[str] = []` field to `PolicyRule` and `PolicySet` (e.g., `["SEC-15c3-5", "FINRA-5210"]`).

### 2. Data & Storage Changes

#### P0: FeatureSnapshot Persistence
- **New file**: `src/agentic_trading/intelligence/feature_snapshot.py`
- **Model**: `FeatureSnapshot(BaseModel)` — `snapshot_id`, `symbol`, `timestamp`, `feature_vector` (dict), `feature_version_hash`, `model_id`, `model_version`, `signal_value`, `signal_confidence`, `rationale`.
- **Storage**: Append to JSONL file (`data/feature_snapshots.jsonl`) in paper/live; in-memory ring buffer for backtest.
- **Integration**: `ExecutionEngine._handle_intent_cp()` and `_handle_intent_legacy()` — create snapshot before order submission; reference `snapshot_id` in `DecisionAudit`.
- **Schema**: Add `snapshot_id: str = ""` to `DecisionAudit` model (or whichever audit model is used in `AuditEntry`).

#### P1: Feature Version Hashing
- **File**: `src/agentic_trading/intelligence/features/engine.py`
- **Action**: Compute `feature_version = content_hash(indicator_list_sorted + param_hash)` on `FeatureEngine.__init__()`. Embed in every `FeatureVector` event.
- **Schema**: Add `feature_version: str = ""` to `FeatureVector` model.

#### P1: ARIMA Forecast Features
- **New file**: `src/agentic_trading/intelligence/features/arima.py`
- **Action**: Implement `ARIMAForecaster` using `pmdarima.auto_arima`. Fits per instrument on rolling window. Returns one-step-ahead forecast, confidence interval, fitted order.
- **Integration**: Call from `FeatureEngine.compute_features()` when candle buffer ≥ 60 points. Add `arima_forecast`, `arima_lower`, `arima_upper`, `arima_order` to feature dict.
- **Dependency**: Add `pmdarima` to `pyproject.toml`.

#### P1: Fourier/FFT Features
- **New file**: `src/agentic_trading/intelligence/features/fourier.py`
- **Action**: Implement `FourierExtractor` — DFT on rolling window (128 points). Extract magnitude and phase of lowest 3–9 frequency components.
- **Integration**: Call from `FeatureEngine.compute_features()`. Add `fft_mag_1..9`, `fft_phase_1..9` to feature dict.

#### P1: MBO Microstructure Features (partial)
- **File**: `src/agentic_trading/intelligence/features/orderbook.py`
- **Action**: Extend `OrderbookEngine` to compute: order-book imbalance (bid vs ask depth ratio), spread, depth at N price levels, trade intensity (trades/sec), waiting times.
- **Schema**: Add microstructure fields to `FeatureVector`.

### 3. Risk & Compliance Updates

#### P1: SurveillanceAgent
- **New file**: `src/agentic_trading/agents/surveillance.py`
- **Agent**: `SurveillanceAgent(BaseAgent)`, `AgentType.SURVEILLANCE`.
- **Subscribes to**: `execution` (orders + fills), `strategy.signal`, `state` topics.
- **Detection rules** (initial set):
  - **Wash trading**: same-symbol buy+sell within N seconds from same strategy.
  - **Spoofing/layering**: large order submitted then cancelled within M seconds.
  - **Self-crossing**: fills where both sides are internal (already prevented by self-match control, this is post-hoc).
- **Output**: Publishes `SurveillanceCase` event on `surveillance` topic. Case model includes `case_id`, `case_type`, `severity`, `evidence`, `status`, `disposition`.

#### P1: Case Management
- **New file**: `src/agentic_trading/compliance/case_manager.py`
- **Model**: `ComplianceCase(BaseModel)` — lifecycle (open → investigating → escalated → closed), evidence list, timeline, assigned officer, disposition.
- **Storage**: JSONL persistence (`data/surveillance_cases.jsonl`).
- **Integration**: SurveillanceAgent creates cases; UI surfaces open cases.

#### P2: Information Barriers
- **File**: `src/agentic_trading/control_plane/action_types.py`
- **Action**: Add `required_role: str | None = None` to `ProposedAction`. ToolGateway validates caller role against action.
- **File**: `src/agentic_trading/control_plane/approval_service.py`
- **Action**: Enforce proposer ≠ approver constraint.

#### P2: Audit Bundle Generation
- **New file**: `src/agentic_trading/observability/audit_bundle.py`
- **Action**: `AuditBundleGenerator.generate(trace_id)` — collects `DecisionAudit`, `FeatureSnapshot`, `PolicyEvaluated` events, `ApprovalRecord`, and any `SurveillanceCase` linked to the trace. Outputs a self-contained JSON/PDF bundle.

### 4. Execution Enhancements

#### P1: ExecutionPlannerAgent
- **New file**: `src/agentic_trading/agents/execution_planner.py`
- **Agent**: `ExecutionPlannerAgent(BaseAgent)`, `AgentType.EXECUTION_PLANNER`.
- **Model**: `ExecutionPlan(BaseModel)` — `plan_id`, `intent_id`, `slices: list[OrderSlice]`, `venue`, `order_type`, `urgency`, `expected_slippage_bps`, `max_participation_rate`, `contingencies`.
- **Flow change**: `PortfolioManager` emits `TradeIntent` → `ExecutionPlannerAgent` creates `ExecutionPlan` → publishes on `execution.plan` → `ExecutionEngine` executes slices.
- **Initial implementation**: single-slice pass-through (preserves current behaviour); multi-slice and multi-venue logic added later.

#### P1: Execution Quality Tracker
- **New file**: `src/agentic_trading/observability/execution_quality.py`
- **Class**: `ExecutionQualityTracker` — per-plan and daily aggregate metrics:
  - Slippage vs mid price at intent time.
  - Participation rate (fill volume / market volume in window).
  - Latency: intent → submission → ack → fill.
  - Opportunity cost: intended price vs last price if unfilled.
- **Integration**: `ExecutionEngine.handle_fill()` feeds data; `DailyEffectivenessScorecard` reads tracker.

### 5. Modelling & Lifecycle

#### P1: ModelRegistry
- **New file**: `src/agentic_trading/intelligence/model_registry.py`
- **Model**: `ModelRecord(BaseModel)` — `model_id`, `version`, `training_data_hash`, `hyperparameters`, `metrics` (MSE, Sharpe, etc.), `stage` (RESEARCH/PAPER/LIMITED/PRODUCTION), `approved_by`, `approved_at`.
- **Storage**: JSONL persistence.
- **Integration**: `StrategyRunner` and `CMTAnalystAgent` reference `model_id` when generating signals. `FeatureSnapshot` includes `model_id`.

#### P2: Strategy Lifecycle Automation
- **File**: `src/agentic_trading/governance/strategy_lifecycle.py`
- **Action**: Connect scorecard metrics to automatic promotion/demotion triggers. E.g., sustained edge_quality < 3.0 for 5 days → auto-demote from PRODUCTION to LIMITED.

### 6. Measurement & UI Adjustments

#### P1: Scorecard Formula Enhancements
- **File**: `src/agentic_trading/observability/daily_scorecard.py`
- **Edge quality**: Add information ratio (`mean_return / std_return`) and Sharpe ratio from journal stats. Align with design's `0.5 * IR/2 + 0.3 * hit_rate*10 + 0.2 * Sharpe` formula.
- **Execution quality**: Add participation rate score and latency score (requires `ExecutionQualityTracker` from action 4b). Align with design's 3-component formula.
- **Risk discipline**: Add VaR coverage score (`1 - max(0, realised_loss - var_limit) / var_limit`). Replace raw drawdown deduction with utilisation ratio.
- **Operational integrity**: Add recon break count, incident resolution time, canary health score. Align with design's 4-component average.

#### P2: ReportingAgent Wrapper
- **New file**: `src/agentic_trading/agents/reporting.py`
- **Agent**: `ReportingAgent(BaseAgent)` — periodic (runs daily or on-demand). Calls `DailyEffectivenessScorecard.compute()`, compiles PnL summary, risk summary, recon status, incident log, surveillance cases. Publishes `DailyReport` event and writes to storage.

#### P2: FeatureComputationAgent (Separation)
- **New file**: `src/agentic_trading/agents/feature_computation.py`
- **Action**: Extract feature computation from `MarketIntelligenceAgent` into a dedicated `FeatureComputationAgent`. MarketIntelligence focuses on data ingestion; FeatureComputation subscribes to `market.candle` and publishes `feature.vector`.
- **Low risk**: current `MarketIntelligenceAgent` already delegates to `FeatureEngine`; this is a clean extraction.

### 7. Agent Rationalisation

The design emphasises minimal agents. Current agent set vs design recommendation:

| Current Agent | Design Equivalent | Action |
|---|---|---|
| MarketIntelligenceAgent | MarketIntelligenceAgent | Keep |
| (embedded in above) | FeatureComputationAgent | Extract (P2, see G19) |
| SignalAgent | SignalAgent | Keep |
| (PortfolioManager) | PortfolioAgent | Already exists as component; no agent wrapper needed |
| — | ExecutionPlannerAgent | **Add** (P1) |
| ExecutionAgent | ExecutionAgent | Keep |
| RiskGateAgent | (merged into PolicyGateAgent) | Keep; rename consideration only |
| GovernanceCanary | GovernanceCanary | Keep |
| DataQualityAgent | (part of FeatureComputationAgent) | Keep separate — adds value |
| IncidentResponseAgent | IncidentResponseAgent | Keep |
| CMTAnalystAgent | (not in design) | Keep — provides LLM-based analysis, non-core |
| PredictionMarketAgent | (not in design) | Keep — alpha source, non-core |
| EfficacyAgent | (not in design) | Keep — diagnostics, non-core |
| — | SurveillanceAgent | **Add** (P1) |
| — | ReportingAgent | **Add** (P2) |

**No agents need removal.** The design's "minimal" guidance is satisfied because all current agents serve distinct purposes. The non-design agents (CMT, PredictionMarket, Efficacy) are clearly alpha/diagnostics-related and don't fragment the control boundary.

---

## Prioritisation Summary

### P0 — Must-Fix (enforce safety invariants)

| Action | Effort | Files |
|---|---|---|
| FeatureSnapshot persistence | Medium | New: `intelligence/feature_snapshot.py`; Modify: `execution/engine.py`, audit models |
| Price collars | Small | Modify: `execution/risk/pre_trade.py`, `policy/default_policies.py`, `core/config.py` |
| Self-match prevention | Small | Modify: `execution/risk/pre_trade.py`, `execution/engine.py` |
| Message throttles (policy-level) | Small | Modify: `execution/risk/pre_trade.py`, `policy/default_policies.py`, `core/config.py` |

### P1 — Follow-Up (institutional features)

| Action | Effort | Files |
|---|---|---|
| ARIMA forecasts | Medium | New: `intelligence/features/arima.py`; Modify: `intelligence/features/engine.py` |
| Fourier/FFT features | Small | New: `intelligence/features/fourier.py`; Modify: `intelligence/features/engine.py` |
| Feature version hashing | Small | Modify: `intelligence/features/engine.py`, FeatureVector schema |
| SurveillanceAgent | Large | New: `agents/surveillance.py`, `compliance/case_manager.py` |
| ExecutionPlannerAgent | Medium | New: `agents/execution_planner.py`, `execution/plan.py` |
| Execution quality tracker | Medium | New: `observability/execution_quality.py` |
| ModelRegistry | Medium | New: `intelligence/model_registry.py` |
| Scorecard formula updates | Small | Modify: `observability/daily_scorecard.py` |
| Degraded mode expansion | Small | Modify: `control_plane/action_types.py`, `control_plane/policy_evaluator.py` |
| MBO microstructure features | Medium | Modify: `intelligence/features/orderbook.py` |

### P2 — Later (polish & compliance)

| Action | Effort | Files |
|---|---|---|
| ReportingAgent wrapper | Small | New: `agents/reporting.py` |
| FeatureComputationAgent extraction | Small | New: `agents/feature_computation.py`; Modify: `agents/market_intelligence.py` |
| Regulatory mapping on PolicyRule | Small | Modify: `policy/models.py` |
| Information barriers | Medium | Modify: `control_plane/action_types.py`, `control_plane/approval_service.py` |
| Audit bundle generation | Medium | New: `observability/audit_bundle.py` |
| Strategy lifecycle automation | Medium | Modify: `governance/strategy_lifecycle.py` |

---

## Recommended Implementation Order

1. **P0 sprint** (1–2 weeks): FeatureSnapshot persistence → price collars → self-match prevention → message throttles. These close safety gaps and are prerequisites for institutional deployment.
2. **P1a sprint** (2–3 weeks): ExecutionPlannerAgent → execution quality tracker → scorecard formula updates → degraded mode expansion. These improve execution measurement and operational control.
3. **P1b sprint** (2–3 weeks): ARIMA → FFT → feature version hashing → MBO microstructure. These enrich the signal pipeline.
4. **P1c sprint** (2–3 weeks): SurveillanceAgent → case management → ModelRegistry. These add compliance and modelling infrastructure.
5. **P2 sprint** (ongoing): ReportingAgent → regulatory mapping → information barriers → audit bundles → strategy lifecycle automation. These are polish items.

---

## Key Architectural Notes

- **Dual execution paths remain valid**: CP mode (ToolGateway) is the institutional path; legacy (GovernanceGate) serves backward compatibility. All P0 controls must work in BOTH paths.
- **No agent should bypass the action boundary**: the design's central invariant. Already enforced — verify during P0 that new pre-trade checks (price collars, self-match, throttles) are called in both CP and legacy flows.
- **Feature snapshots are the highest-priority data gap**: without them, no historical decision can be fully replayed. This blocks auditability claims.
- **SurveillanceAgent is large but not urgent**: it depends on having good execution data (fills, orders) which already exists. Scope the initial version to wash-trade detection only and expand iteratively.
