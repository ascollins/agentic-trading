# Knowledge Taxonomy

## Overview

The knowledge taxonomy organises every piece of retrievable information on the
platform into a 20-node tree across 5 tiers. Each node has a stable identifier,
a definition, tagging rules, and an ingestion point. Retrieval policy
(top-k limits, freshness bias, trust weighting) is configured separately in
`configs/retrieval_policy.yaml`.

## Tree

```
knowledge/
├── market/                        # What the market is doing
│   ├── market.price_action        # OHLCV candles, ticks, live prices
│   ├── market.regime              # Regime detection & volatility/liquidity state
│   ├── market.structure           # Market structure: BOS, CHoCH, swing points
│   └── market.external            # News, sentiment, whale alerts
│
├── analysis/                      # What agents conclude
│   ├── analysis.htf_assessment    # Higher-timeframe multi-TF analysis
│   ├── analysis.smc_report        # Smart Money Concepts confluence
│   ├── analysis.cmt_assessment    # CMT 9-layer LLM analysis
│   ├── analysis.market_context    # Synthesized market context snapshot
│   └── analysis.data_quality      # Data quality checks & issues
│
├── decision/                      # What the platform decides
│   ├── decision.trade_plan        # Entry / stop / target plan
│   ├── decision.signal            # Strategy signal (direction + confidence)
│   ├── decision.risk_event        # Risk alerts, circuit breakers, kill switch
│   ├── decision.governance        # Policy evaluations, approval decisions
│   └── decision.reasoning_trace   # Agent reasoning chain (step-by-step)
│
├── execution/                     # What happened
│   ├── execution.order_lifecycle  # Order submitted → filled / cancelled
│   ├── execution.fill_record      # Actual fills with prices & fees
│   └── execution.position_state   # Open position snapshots
│
└── reference/                     # Standing knowledge (rarely changes)
    ├── reference.policy_rule      # Governance rules & policy sets
    ├── reference.strategy_config  # Strategy parameters & maturity level
    └── reference.instrument_spec  # Instrument definitions & constraints
```

## Node Definitions

### Tier 1 — Market

| Node | Definition | `MemoryEntryType` | Primary Source |
|------|-----------|-------------------|----------------|
| `market.price_action` | OHLCV candle data, tick events, and live bid/ask/last prices. The raw material for all analysis. | *(not stored in memory store — lives in FactTable and candle buffers)* | `FeedManager`, `CandleBuilder`, `FactTable` |
| `market.regime` | Regime classification (trend/range), volatility state (high/low), and liquidity state. Detected by the regime detector and stored as structured snapshots. | *(FactTable.regimes)* | `RegimeDetector`, event topic `strategy.signal` |
| `market.structure` | Swing highs/lows, break-of-structure (BOS), change-of-character (CHoCH), order blocks, and fair value gaps. The structural backbone for SMC analysis. | *(embedded in SMC_REPORT content)* | `SMCConfluenceEngine` |
| `market.external` | Third-party signals: news events, sentiment scores, whale transaction alerts, open interest changes. | *(not yet stored — future)* | Event topics `feature.vector` (`NewsEvent`, `WhaleEvent`) |

### Tier 2 — Analysis

| Node | Definition | `MemoryEntryType` | Primary Source |
|------|-----------|-------------------|----------------|
| `analysis.htf_assessment` | Multi-timeframe structural analysis: overall bias, alignment score, per-timeframe summaries with momentum and trend strength, confluences and conflicts. | `HTF_ASSESSMENT` | `HTFAnalyzer` |
| `analysis.smc_report` | Smart Money Concepts confluence report: order blocks, FVGs, liquidity sweeps, BOS/CHoCH counts, confluence scores, and key observations per timeframe. | `SMC_REPORT` | `SMCConfluenceEngine` |
| `analysis.cmt_assessment` | CMT 9-layer analysis produced by Claude: layer scores (1-9), confluence dimensions (7), trade plan, thesis, and system health. The richest single analysis artifact. | `CMT_ASSESSMENT` | `CMTAnalysisEngine` |
| `analysis.market_context` | Synthesized market context snapshot: regime, volatility, liquidity, recent swings, key levels, bias, and confidence. A pre-digested summary for downstream agents. | *(embedded in AgentContext)* | `MarketContextBuilder` |
| `analysis.data_quality` | Data quality issues detected by QA checks: gaps, stale feeds, anomalous candles, missing fields. Severity-graded (INFO/WARNING/CRITICAL). | *(logged, not yet in memory store)* | `DataQualityChecker` |

### Tier 3 — Decision

| Node | Definition | `MemoryEntryType` | Primary Source |
|------|-----------|-------------------|----------------|
| `decision.trade_plan` | Complete trade setup: direction, conviction, entry zone, stop loss, targets with R:R ratios, position sizing, invalidation criteria, and rationale. | `TRADE_PLAN` | `CMTAnalysisEngine`, `TradePlanBuilder` |
| `decision.signal` | Strategy signal with direction (LONG/SHORT/FLAT), confidence score, and associated metadata. The trigger for order intent creation. | `SIGNAL` | `Strategy`, event topic `strategy.signal` |
| `decision.risk_event` | Risk alerts (severity-graded), circuit breaker trips, kill switch activations, drawdown warnings, VaR breaches. Critical for understanding why actions were blocked. | `RISK_EVENT` | `RiskManager`, `CircuitBreakerMonitor` |
| `decision.governance` | Policy evaluation results (pass/fail per rule, sizing multipliers), approval request lifecycle (pending/approved/rejected/expired), maturity gate decisions. | *(PostgreSQL decision_audits, governance_logs)* | `GovernanceGate`, `PolicyEngine`, `ApprovalManager` |
| `decision.reasoning_trace` | Step-by-step agent reasoning: perception → hypothesis → evaluation → decision → action phases with evidence and confidence at each step. Includes raw Claude thinking. | `REASONING_TRACE` | `ReasoningTrace`, `PipelineResult` |

### Tier 4 — Execution

| Node | Definition | `MemoryEntryType` | Primary Source |
|------|-----------|-------------------|----------------|
| `execution.order_lifecycle` | Order state transitions: PENDING → SUBMITTED → FILLED/CANCELLED/REJECTED/EXPIRED. Includes order type, side, quantity, and exchange acknowledgement. | *(PostgreSQL orders table)* | `ExecutionEngine`, event topic `execution` |
| `execution.fill_record` | Actual execution fills: fill price, quantity, fees, slippage, and timestamp. The ground truth of what was traded. | *(PostgreSQL fills table)* | `PaperAdapter`/`CCXTAdapter`, `FillEvent` |
| `execution.position_state` | Current open positions: entry price, quantity, unrealized PnL, margin, and liquidation price. Snapshotted periodically. | *(PostgreSQL position_snapshots, Redis hot cache)* | `PortfolioManager`, `ReconciliationLoop` |

### Tier 5 — Reference

| Node | Definition | `MemoryEntryType` | Primary Source |
|------|-----------|-------------------|----------------|
| `reference.policy_rule` | Governance rules (pre-trade risk limits, execution constraints, compliance rules) and versioned policy sets. Declarative, rarely changes. | *(JSON files via PolicyStore)* | `PolicyStore`, `default_policies` |
| `reference.strategy_config` | Strategy parameters (EMA periods, ATR multipliers), maturity level (L0-L4), and optimization history. The strategy's identity. | *(TOML configs, PostgreSQL)* | `configs/strategies.toml`, `OptimizerScheduler` |
| `reference.instrument_spec` | Exchange-normalised instrument definitions: precision, tick size, min/max quantity, fees, leverage limits, trading sessions. | *(TOML configs, in-memory)* | `configs/instruments.toml`, `InstrumentManager` |

## Tagging Rules

Tags are the primary mechanism for semantic retrieval within the memory store.
Every `MemoryEntry` carries a `tags: list[str]` field. These rules govern
what tags are applied, by whom, and when.

### Mandatory Tags

Every memory entry **must** carry at least one tag from each applicable
category:

| Category | Tag Format | Examples | Applied By |
|----------|-----------|----------|------------|
| **Taxonomy node** | `node:<node_id>` | `node:analysis.cmt_assessment` | Ingestion point (automatic) |
| **Direction** | `long`, `short`, `flat`, `neutral` | `long` | Analysis/signal producer |
| **Confidence** | `high_conf`, `med_conf`, `low_conf` | `high_conf` | Analysis/signal producer |

### Recommended Tags

| Category | Tag Format | Examples | Applied By |
|----------|-----------|----------|------------|
| **Conviction** | `conviction:<level>` | `conviction:high`, `conviction:moderate` | `TradePlanBuilder` |
| **Setup grade** | `grade:<grade>` | `grade:A+`, `grade:B` | `TradePlanBuilder` |
| **Regime** | `regime:<type>` | `regime:trend`, `regime:range` | `RegimeDetector` |
| **Severity** | `severity:<level>` | `severity:critical`, `severity:warning` | Risk/data-quality producers |
| **Impact** | `impact:<tier>` | `impact:high`, `impact:critical` | `GovernanceGate` |
| **Veto** | `veto` | `veto` | CMT analysis (when confluence veto triggered) |
| **Confluence** | `confluence_met`, `confluence_missed` | `confluence_met` | CMT analysis |

### Tag Application Rules

1. **Taxonomy node tag is mandatory.** The ingestion point (the component that
   calls `ContextManager.write_analysis()`) must include `node:<node_id>`.
   This is non-negotiable — entries without a node tag are unclassified and
   may be deprioritised in retrieval.

2. **Direction and confidence tags** are required for analysis and decision
   nodes. Market and reference nodes may omit them.

3. **Tags are additive.** A single entry can carry tags from multiple
   categories (e.g., `["node:analysis.cmt_assessment", "long", "high_conf",
   "confluence_met", "grade:A"]`).

4. **Tags are lowercase, snake_case.** No spaces, no uppercase.

5. **Free-form tags are allowed** beyond the mandatory and recommended
   categories. Use them for strategy-specific semantics (e.g., `entry_zone`,
   `breakout`, `pullback`).

## Ingestion Rules

Ingestion rules define where new knowledge lands and what metadata is
attached at creation time.

### Ingestion Matrix

| Node | Ingestion Point | Storage | TTL | Metadata |
|------|----------------|---------|-----|----------|
| `market.price_action` | `CandleBuilder` → FactTable | In-memory ring buffer + Parquet | Real-time (no decay) | `source_type=MARKET_DATA`, `trust=HIGH` |
| `market.regime` | `RegimeDetector` → FactTable | In-memory | Real-time | `source_type=DERIVED`, `trust=HIGH` |
| `market.structure` | `SMCConfluenceEngine` → memory store | JSONL | 24h | `source_type=DERIVED`, `trust=MEDIUM` |
| `market.external` | Event handlers → memory store | JSONL | 6h | `source_type=EXTERNAL`, `trust=LOW` |
| `analysis.htf_assessment` | `HTFAnalyzer` → `write_analysis()` | JSONL | 24h | `source_type=AGENT`, `trust=MEDIUM` |
| `analysis.smc_report` | `SMCConfluenceEngine` → `write_analysis()` | JSONL | 24h | `source_type=AGENT`, `trust=MEDIUM` |
| `analysis.cmt_assessment` | `CMTAnalysisEngine` → `write_analysis()` | JSONL | 24h | `source_type=AGENT`, `trust=MEDIUM` |
| `analysis.market_context` | `MarketContextBuilder` → `write_analysis()` | JSONL | 12h | `source_type=DERIVED`, `trust=HIGH` |
| `analysis.data_quality` | `DataQualityChecker` → `write_analysis()` | JSONL | 48h | `source_type=SYSTEM`, `trust=VERIFIED` |
| `decision.trade_plan` | `TradePlanBuilder` → `write_analysis()` | JSONL | 24h | `source_type=AGENT`, `trust=MEDIUM` |
| `decision.signal` | `Strategy` → `write_analysis()` | JSONL | 12h | `source_type=STRATEGY`, `trust=HIGH` |
| `decision.risk_event` | `RiskManager` → `write_analysis()` | JSONL | 48h | `source_type=SYSTEM`, `trust=VERIFIED` |
| `decision.governance` | `GovernanceGate` → PostgreSQL | PostgreSQL | Permanent | `source_type=SYSTEM`, `trust=VERIFIED` |
| `decision.reasoning_trace` | Pipeline → `write_analysis()` | JSONL | 72h | `source_type=AGENT`, `trust=MEDIUM` |
| `execution.order_lifecycle` | `ExecutionEngine` → PostgreSQL | PostgreSQL | Permanent | `source_type=SYSTEM`, `trust=VERIFIED` |
| `execution.fill_record` | Adapter → PostgreSQL | PostgreSQL | Permanent | `source_type=SYSTEM`, `trust=VERIFIED` |
| `execution.position_state` | `ReconciliationLoop` → PostgreSQL + Redis | PostgreSQL + Redis | Permanent | `source_type=SYSTEM`, `trust=VERIFIED` |
| `reference.policy_rule` | `PolicyStore` → JSON files | JSON files | Permanent | `source_type=POLICY`, `trust=VERIFIED` |
| `reference.strategy_config` | Config loader / Optimizer → TOML | TOML files | Permanent | `source_type=SYSTEM`, `trust=VERIFIED` |
| `reference.instrument_spec` | Config loader → TOML | TOML files | Permanent | `source_type=SYSTEM`, `trust=VERIFIED` |

### Ingestion Flow

```
Producer (Agent / Engine / Detector)
  │
  ├─ 1. Classify → pick taxonomy node
  ├─ 2. Tag → apply mandatory + recommended tags
  ├─ 3. Metadata → set source_type, trust_level, quality_score
  ├─ 4. TTL → set ttl_hours per retrieval_policy.yaml
  │
  └─ 5. Store
       ├─ Memory store → ContextManager.write_analysis()
       ├─ Fact table → FactTable.update_*()
       ├─ PostgreSQL → via ORM models
       └─ Config files → PolicyStore / file I/O
```

### Ingestion Validation

Before storing, the ingestion point should verify:

1. **Node tag present** in `tags[]` (e.g., `node:analysis.cmt_assessment`)
2. **Symbol set** if the node is symbol-scoped (all market, analysis,
   decision nodes)
3. **TTL matches policy** — use the value from `retrieval_policy.yaml`,
   not a hard-coded default
4. **Content is non-empty** — reject empty `content: {}` entries

## Mapping: MemoryEntryType → Taxonomy Node

| `MemoryEntryType` | Taxonomy Node |
|--------------------|---------------|
| `HTF_ASSESSMENT` | `analysis.htf_assessment` |
| `SMC_REPORT` | `analysis.smc_report` |
| `CMT_ASSESSMENT` | `analysis.cmt_assessment` |
| `TRADE_PLAN` | `decision.trade_plan` |
| `SIGNAL` | `decision.signal` |
| `RISK_EVENT` | `decision.risk_event` |
| `REASONING_TRACE` | `decision.reasoning_trace` |

Nodes without a `MemoryEntryType` mapping are stored in other backends
(FactTable, PostgreSQL, config files) and are retrieved through dedicated
APIs rather than the memory store.

## Retrieval Overview

Retrieval policy is configured per node in `configs/retrieval_policy.yaml`.
Each node specifies:

| Parameter | Description |
|-----------|-------------|
| `top_k` | Maximum items returned per query |
| `ttl_hours` | Time-to-live for exponential decay (0 = no decay) |
| `freshness_bias` | Multiplier on time-decay lambda (>1 = faster decay, <1 = slower) |
| `min_trust` | Minimum trust level (entries below this are filtered) |
| `min_relevance` | Minimum effective relevance score after decay + trust weighting |
| `prefer_recent` | Whether to sort by recency before relevance (for fast-moving nodes) |

### Retrieval Formula

```
effective_relevance =
    base_relevance
    × exp(-lambda × freshness_bias × age_hours)
    × trust_weight
    × quality_score

where:
    lambda = ln(10) / ttl_hours
    trust_weight = TRUST_WEIGHTS[trust_level]
    quality_score = metadata.quality_score (default 1.0)
```

Items with `effective_relevance < min_relevance` are dropped.
Results are sorted by `effective_relevance` descending, truncated to `top_k`.

### Node-Specific Retrieval Behavior

| Tier | Freshness | Trust Floor | Top-K | Rationale |
|------|-----------|-------------|-------|-----------|
| **Market** | Very high (1h-6h TTL) | `medium` | 5 | Stale prices are dangerous |
| **Analysis** | High (12h-24h TTL) | `low` | 3 | Yesterday's analysis is rarely relevant |
| **Decision** | Medium (12h-48h TTL) | `medium` | 5 | Trade plans and signals age moderately |
| **Execution** | Low (no decay) | `verified` | 10 | Historical fills are ground truth |
| **Reference** | None (permanent) | `verified` | 3 | Policies and configs are stable |
