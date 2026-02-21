# LLM Interaction Envelope — Design Document

## Overview

The **LLM Interaction Envelope** is the mandatory contract for every LLM call
in the platform. It captures what goes into a call (instructions, context,
evidence, budget, safety constraints) and what comes out (raw output, parsed
output, validation results) in one auditable, provider-agnostic unit.

The envelope **does not execute calls**. Execution remains with the caller
(today: `CMTAnalysisEngine`; future: any agent). The envelope defines the
contract.

## Architecture

```
┌────────────────────────────────────────────────────┐
│                  Calling Agent                      │
│  (CMTAnalysisEngine, PlanningAgent, ExecAgent...)   │
└──────────────┬──────────────────────┬───────────────┘
               │                      │
        1. build()              4. store()
               │                      │
               ▼                      ▼
    ┌──────────────────┐   ┌───────────────────────┐
    │  EnvelopeBuilder │   │  IInteractionStore    │
    │  (fluent API)    │   │  Memory / JSONL       │
    └────────┬─────────┘   └───────────────────────┘
             │
      2. LLMEnvelope
             │
             ▼
    ┌──────────────────┐
    │  Provider SDK    │    3. Returns raw response
    │  (Anthropic,     │──────► LLMResult
    │   OpenAI, etc.)  │
    └──────────────────┘

    LLMInteraction = Envelope + Result → persisted
```

## Field Reference

### LLMEnvelope

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `envelope_id` | `str` | UUID v4 | Unique identifier |
| `trace_id` | `str` | UUID v4 | Correlation ID for related interactions |
| `causation_id` | `str` | `""` | ID of the direct cause event |
| `tenant_id` | `str` | `"default"` | Multi-tenant isolation key |
| `created_at` | `datetime` | UTC now | Envelope creation timestamp |
| `workflow` | `EnvelopeWorkflow` | `GENERAL` | Purpose classification |
| `agent_id` | `str` | `""` | Calling agent identifier |
| `agent_type` | `str` | `""` | Calling agent type |
| `instructions` | `str` | *(required)* | System prompt / task description |
| `context` | `dict` | `{}` | Platform state snapshot |
| `retrieved_evidence` | `list[EvidenceItem]` | `[]` | Evidence for the LLM |
| `tools_allowed` | `list[str]` | `[]` | Tool-use allowlist |
| `budget` | `LLMBudget` | defaults | Token and cost budget |
| `expected_output_schema` | `dict` | `{}` | JSON Schema for output |
| `safety_constraints` | `SafetyConstraints` | defaults | Safety guardrails |
| `response_format` | `ResponseFormat` | `JSON` | Expected format |
| `provider` | `LLMProvider` | `ANTHROPIC` | Target provider |
| `model` | `str` | `""` | Model identifier |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `retry_policy` | `RetryPolicy` | defaults | Retry configuration |
| `envelope_hash` | `str` | computed | SHA256[:16] integrity hash |

### LLMResult

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `result_id` | `str` | UUID v4 | Unique identifier |
| `envelope_id` | `str` | *(required)* | References the envelope |
| `timestamp` | `datetime` | UTC now | Capture timestamp |
| `raw_output` | `str` | `""` | Raw LLM text output |
| `parsed_output` | `dict` | `{}` | Parsed structured output |
| `validation_passed` | `bool` | `False` | Schema validation result |
| `validation_errors` | `list[str]` | `[]` | Validation error messages |
| `latency_ms` | `float` | `0.0` | Round-trip latency |
| `input_tokens` | `int` | `0` | Input tokens consumed |
| `output_tokens` | `int` | `0` | Output tokens generated |
| `thinking_tokens` | `int` | `0` | Extended thinking tokens |
| `cost_usd` | `float` | `0.0` | Estimated cost |
| `provider` | `LLMProvider` | `ANTHROPIC` | Serving provider |
| `model` | `str` | `""` | Serving model |
| `attempt_number` | `int` | `1` | Retry attempt (1-based) |
| `error` | `str \| None` | `None` | Error message if failed |
| `success` | `bool` | `True` | Whether call succeeded |
| `output_hash` | `str` | computed | SHA256[:16] of raw_output |

## Retry Rules

- **Default**: 3 retries with exponential backoff (1s base, 30s max)
- **Retryable errors**: `rate_limit`, `timeout`, `server_error`
- **Non-retryable**: `auth_error`, `invalid_request`, `budget_exhausted`
- **On retry**: increment `LLMResult.attempt_number`; persist each attempt
  as a separate `LLMInteraction`
- **Final failure**: last `LLMResult` has `success=False` and `error` set

## Temperature Rules

| Workflow | Default | Override Allowed? | Notes |
|----------|---------|-------------------|-------|
| Analysis | `0.0` | Yes | Quantitative analysis should be reproducible |
| Planning | `0.0` | Yes | Parameter selection must be deterministic |
| Execution | `0.0` | **No** | `require_deterministic=True` enforced |
| General | `0.0` | Yes | Caller decides |

## Deterministic Mode

When `SafetyConstraints.require_deterministic=True`:

1. `temperature` **must** be `0.0`
2. Builder validates at `build()` time
3. Raises `EnvelopeValidationError` if violated
4. Execution workflow enables this by default

## Fallback Rules

- **Primary**: use configured `provider` + `model`
- **Fallback**: caller responsibility (e.g. Anthropic SDK -> httpx)
- **Envelope captures**: which attempt succeeded via `attempt_number`
- **Cross-provider fallback**: create a new envelope with different
  `provider`/`model` — each attempt is its own `LLMInteraction`

## Storage Format

Interactions are persisted as JSONL (one JSON object per line):

```
{"interaction_id":"abc...","envelope":{...},"result":{...},"stored_at":"2025-..."}
{"interaction_id":"def...","envelope":{...},"result":{...},"stored_at":"2025-..."}
```

Default path: `data/llm_interactions.jsonl`

Two implementations:

| Store | Use Case | Persistence |
|-------|----------|-------------|
| `MemoryInteractionStore` | Backtest + unit tests | None (in-memory) |
| `JsonFileInteractionStore` | Paper + live trading | JSONL file |

## Example Envelopes

### (a) Analysis — CMT 9-Layer Assessment

```python
from agentic_trading.llm import EnvelopeBuilder, LLMProvider

envelope = (
    EnvelopeBuilder()
    .for_analysis()
    .with_instructions(
        "Apply the full 9-layer CMT framework to this market data. "
        "Return a structured assessment with confidence scores."
    )
    .with_context({
        "symbol": "BTCUSDT",
        "timeframes": ["5m", "15m", "1h", "4h", "1d"],
        "regime": "trending_up",
        "portfolio_exposure_pct": 0.15,
    })
    .add_evidence("candle_history", {
        "1h": {"open": 42500, "high": 42800, "low": 42400, "close": 42750},
    })
    .add_evidence("indicator_values", {
        "rsi_14": 62.3, "atr_14": 450.2, "ema_12": 42600,
    })
    .add_evidence("smc_confluence", {
        "order_blocks": [{"price": 42200, "type": "bullish"}],
        "fvg": [{"start": 42300, "end": 42500}],
    })
    .with_budget(max_output_tokens=4096, thinking_budget=8000)
    .with_output_schema(CMT_RESPONSE_SCHEMA)
    .with_provider(LLMProvider.ANTHROPIC, "claude-sonnet-4-5-20250929")
    .with_trace(trace_id="signal-trace-abc")
    .with_agent("cmt-analyst-01", "cmt_analyst")
    .build()
)
```

**Preset values**: temperature=0.0, require_json=True, thinking_budget=8000

### (b) Planning — Strategy Parameter Optimization

```python
envelope = (
    EnvelopeBuilder()
    .for_planning()
    .with_instructions(
        "Recommend optimal parameters for the SMC strategy "
        "based on backtest and walk-forward results."
    )
    .with_context({
        "strategy_id": "smc_btcusdt",
        "current_params": {"ema_fast": 12, "ema_slow": 26, "atr_mult": 2.0},
        "backtest_window": "2024-01-01 to 2024-12-31",
    })
    .add_evidence("backtest_results", {
        "sharpe": 1.8, "max_dd": -12.5, "win_rate": 0.58,
    })
    .add_evidence("walk_forward", {
        "oos_sharpe": 1.2, "is_overfit": False,
    })
    .with_budget(max_output_tokens=8192)
    .with_output_schema(PARAMETER_RECOMMENDATION_SCHEMA)
    .with_provider(LLMProvider.ANTHROPIC, "claude-sonnet-4-5-20250929")
    .build()
)
```

**Preset values**: temperature=0.0, require_json=True, max_output_tokens=8192

### (c) Execution — Fill Strategy Decision

```python
envelope = (
    EnvelopeBuilder()
    .for_execution()
    .with_instructions(
        "Select optimal fill strategy for this order intent."
    )
    .with_context({
        "symbol": "BTCUSDT",
        "side": "buy",
        "qty": "0.5",
        "urgency": 0.8,
        "spread_bps": 2.1,
        "book_depth_usd": 500000,
    })
    .add_evidence("orderbook_snapshot", {
        "bids": [["42500", "1.2"], ["42499", "0.8"]],
        "asks": [["42501", "0.5"], ["42502", "1.0"]],
    })
    .with_budget(max_output_tokens=2048)
    .with_output_schema(FILL_STRATEGY_SCHEMA)
    .with_provider(LLMProvider.ANTHROPIC, "claude-sonnet-4-5-20250929")
    .with_trace(trace_id="order-trace-xyz", causation_id="signal-event-123")
    .build()
)
```

**Preset values**: temperature=0.0, require_deterministic=True,
require_json=True, max_output_tokens=2048

## Integration Guide

### 1. Build an Envelope

```python
from agentic_trading.llm import EnvelopeBuilder, LLMProvider

envelope = (
    EnvelopeBuilder()
    .for_analysis()
    .with_instructions("Your prompt here")
    .with_context({"key": "value"})
    .with_provider(LLMProvider.ANTHROPIC, "claude-sonnet-4-5-20250929")
    .build()
)
```

### 2. Execute the Call (caller responsibility)

```python
import time
import anthropic

client = anthropic.AsyncAnthropic()
start = time.monotonic()

response = await client.messages.create(
    model=envelope.model,
    max_tokens=envelope.budget.max_output_tokens,
    temperature=envelope.temperature,
    messages=[{"role": "user", "content": envelope.instructions}],
)

latency_ms = (time.monotonic() - start) * 1000
```

### 3. Capture the Result

```python
from agentic_trading.llm import LLMResult

result = LLMResult(
    envelope_id=envelope.envelope_id,
    raw_output=response.content[0].text,
    parsed_output=json.loads(response.content[0].text),
    validation_passed=True,
    latency_ms=latency_ms,
    input_tokens=response.usage.input_tokens,
    output_tokens=response.usage.output_tokens,
    provider=envelope.provider,
    model=envelope.model,
)
```

### 4. Store the Interaction

```python
from agentic_trading.llm import LLMInteraction, MemoryInteractionStore

store = MemoryInteractionStore()  # or JsonFileInteractionStore()

interaction = LLMInteraction(envelope=envelope, result=result)
await store.store(interaction)
```

### 5. Query Later

```python
# By envelope ID
interaction = await store.get_by_envelope_id(envelope.envelope_id)

# By trace ID (all interactions in a decision chain)
chain = await store.get_by_trace_id(trace_id="signal-trace-abc")

# Most recent
recent = await store.recent(limit=10)
```

## Error Types

| Error | When |
|-------|------|
| `EnvelopeValidationError` | Missing instructions, temperature mismatch |
| `LLMBudgetExhaustedError` | Daily call limit or cost ceiling exceeded |
| `LLMResponseValidationError` | Output fails expected_output_schema |

All inherit from `LLMError` which inherits from `TradingError`.
