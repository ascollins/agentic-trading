# Validation Framework

## Overview

The Validation Framework detects and contains LLM hallucinations by running
every agent output through a multi-layer validation pipeline.  The premise:
you cannot prevent hallucinations, but you can detect fabricated price levels,
unjustified confidence scores, and uncited claims before they reach execution.

The pipeline produces a `ValidationResult` with a quality score (0.0-1.0) that
feeds directly into `MetadataStandard.quality_score`, automatically
down-weighting low-quality outputs in future retrieval.

## Architecture

```
LLM raw output (parsed_output: dict)
  |
  +-- LLMEnvelope (retrieved_evidence, expected_output_schema)
  +-- LLMResult   (validation_passed, validation_errors)
  |
  v
+---------------------------------------------------------------+
| ValidationPipeline.run(parsed_output, envelope, result)       |
|                                                               |
|  [1] SchemaValidator --------- hard fail gate ----+           |
|       JSON Schema + Pydantic model                | STOP      |
|                                                   | if fail   |
|  [2] EvidenceValidator ------+                    |           |
|       claim extraction +     |                    |           |
|       citation check         |                    |           |
|                              |                    |           |
|  [3] BusinessRuleValidator --+                    |           |
|       declarative rules      |                    |           |
|       (Operator enum,        |                    |           |
|       dot-path resolution)   |                    |           |
|                              |                    |           |
|  [4] CritiqueValidator ------+ (conditional)      |           |
|       cost-gated second LLM  |                    |           |
|                              v                    v           |
|  +-- ValidationResult ----------------------------+           |
|  |  overall_passed, quality_score (0-1),                      |
|  |  issues[], claims[], critique, remediation                 |
|  +------------------------------------------------------------+
|                     |                                         |
|  RemediationEngine.decide(result) -->                         |
|    RETRY | RE_RETRIEVE | ESCALATE |                           |
|    INSUFFICIENT_EVIDENCE | ACCEPT_WITH_WARNINGS               |
+---------------------------------------------------------------+
         |
         v
  llm_result.validation_passed = overall_passed   (backward compat)
  llm_result.validation_errors = [formatted issues]
  metadata.quality_score = result.quality_score    (at call site)
```

## Layer 1: Schema Validation (Hard Fail)

Structural validation of the parsed LLM output.  If this fails, all
subsequent layers are skipped — the output is structurally unusable.

**Modes:**

| Mode | Source | How |
|------|--------|-----|
| JSON Schema | `envelope.expected_output_schema` | `jsonschema.Draft202012Validator` |
| Pydantic Model | `SchemaValidator.register_model()` | `model_class.model_validate(data)` |

**Example:**

```python
from agentic_trading.validation import SchemaValidator

validator = SchemaValidator()
validator.register_model("CMTAssessmentResponse", CMTAssessmentResponse)
# or: envelope.expected_output_schema = {...}
```

## Layer 2: Evidence Validation

The hallucination-detection layer.  Every factual claim in the output
must be traceable to retrieved evidence or explicitly marked as an
assumption.

**Claim classification:**

| Type | Meaning | Detection |
|------|---------|-----------|
| `CITED` | Backed by evidence source | Output has `evidence_refs` field mapping field -> source_id |
| `ASSUMPTION` | Explicitly marked | Field text contains "assumption" or listed in `assumptions[]` |
| `DERIVED` | Computed from cited data | Numeric field (e.g. computed indicator) |
| `UNCITED` | No evidence link | String content > 10 chars with no citation (violation) |

**Threshold:** If the fraction of UNCITED claims exceeds `max_uncited_ratio`
(default 0.3 = 30%), the layer produces an ERROR issue.

**Unused evidence:** Evidence provided but never cited produces an INFO issue.

**How agents cite evidence:**

```json
{
  "_output_type": "Signal",
  "rationale": "BTC breaking above 200 EMA with volume confirmation",
  "evidence_refs": {
    "rationale": ["candle_history", "indicator_values"]
  },
  "assumptions": ["stop_loss"],
  "confidence": 0.85
}
```

## Layer 3: Business Rule Validation

Declarative rules evaluated against the parsed output, following the
same pattern as the governance `PolicyEngine`.

**Rule types:**

| Type | Description | Example |
|------|-------------|---------|
| `REQUIRED` | Field must be present and non-empty | thesis must exist |
| `RANGE` | Field must be within bounds | confidence in [0, 1] |
| `INVARIANT` | Cross-field consistency | stop_loss < entry_price for longs |
| `CROSS_CHECK` | Field matches another field | direction consistent with thesis |

**Reuses:**

- `Operator` enum from `policy/models.py` (9 operators)
- `PolicyEngine._resolve_field()` for dot-path resolution
- `PolicyEngine._check_condition()` for operator evaluation

**Built-in rule factories:**

```python
from agentic_trading.validation import build_signal_rules, build_cmt_rules

signal_rules = build_signal_rules()   # confidence range, direction valid, rationale required
cmt_rules = build_cmt_rules()         # thesis required, system_health valid, layers non-empty
```

**Custom rules:**

```python
from agentic_trading.validation import BusinessRule, BusinessRuleSet, BusinessRuleType
from agentic_trading.policy.models import Operator

custom = BusinessRuleSet(
    set_id="custom_v1",
    name="Custom Rules",
    rules=[
        BusinessRule(
            rule_id="rr_ratio_min",
            name="R:R ratio minimum",
            rule_type=BusinessRuleType.RANGE,
            field="trade_plan.rr_ratio",
            operator=Operator.GE,
            threshold=1.5,
            output_types=["CMTAssessmentResponse"],
        ),
    ],
)
```

## Layer 4: Critique Validation (Conditional)

A second LLM reviews the output for logical errors, unsupported claims,
and unjustified confidence.  Only triggered when cost is justified.

**Trigger conditions:**

| Trigger | Default | Setting |
|---------|---------|---------|
| Output type in always-critique list | CMTAssessmentResponse | `always_critique_types` |
| Notional USD exceeds threshold | >= $50,000 | `notional_usd_threshold` |
| Prior quality_score below floor | < 0.5 | `confidence_floor` |

**Cost gating:**

- Default model: `claude-haiku-4-5-20250929` (cheap, fast)
- Budget cap: `max_cost_usd = $0.10` per critique
- LLM callable is injected (decoupled from client)

**Acceptance:** Critique score < 0.6 produces an ERROR issue.

## Quality Score

The pipeline computes a 0.0-1.0 quality score from validation issues:

```
Start at 1.0
  Schema failure     -> 0.0 (immediate)
  Each CRITICAL      -> -0.3
  Each ERROR         -> -0.15
  Each WARNING       -> -0.05
  Floor at 0.0
```

This maps directly to `MetadataStandard.quality_score`, which feeds
into `compute_effective_relevance()` — low-quality outputs are
automatically down-weighted in future retrieval.

## Remediation

When validation fails, the `RemediationEngine` recommends next actions.
It is a pure decision function — it does not execute side effects.

**State machine:**

```
PENDING -> RETRYING -> (passed: RESOLVED)
                    -> (failed, retries < max: RETRYING)
                    -> (retries exhausted: RE_RETRIEVING)
                       -> RETRYING -> RESOLVED
                                   -> ESCALATED
                                   -> EXHAUSTED
```

**Actions:**

| Action | When | Effect |
|--------|------|--------|
| `ACCEPT_WITH_WARNINGS` | Passed with warnings | Continue with degraded quality_score |
| `RETRY` | Retries remaining | Re-run LLM call (same envelope) |
| `RE_RETRIEVE` | Retries exhausted | Fetch fresh evidence, then retry |
| `ESCALATE` | Critical issues or exhausted | Hand to human operator |
| `INSUFFICIENT_EVIDENCE` | Evidence issues, all exhausted | Return "cannot answer" |

**Config per output type:**

```python
from agentic_trading.validation import RemediationPolicy

policy = RemediationPolicy(
    output_type="CMTAssessmentResponse",
    max_retries=3,
    max_re_retrievals=2,
    auto_escalate_on_critical=True,
)
```

## Errors

```
TradingError
  ValidationError
    SchemaValidationError       # Structural failure
    EvidenceValidationError     # Uncited threshold exceeded
    BusinessRuleValidationError # Invariant violated
    CritiqueValidationError     # Critique score too low
    RemediationExhaustedError   # All remediation attempts failed
```

## Usage

### Basic Pipeline

```python
from agentic_trading.validation import (
    ValidationPipeline,
    SchemaValidator,
    EvidenceValidator,
    BusinessRuleValidator,
    RemediationEngine,
    build_signal_rules,
)

# Setup
schema_v = SchemaValidator()
evidence_v = EvidenceValidator(max_uncited_ratio=0.3)

biz_v = BusinessRuleValidator()
biz_v.register(build_signal_rules())

pipeline = ValidationPipeline(
    validators=[schema_v, evidence_v, biz_v],
    remediation_engine=RemediationEngine(),
)

# Run
result = pipeline.run(parsed_output, envelope, llm_result)

# Use
if result.overall_passed:
    metadata.quality_score = result.quality_score
else:
    if result.recommended_action == RemediationAction.RETRY:
        # Re-invoke the LLM
        ...
    elif result.recommended_action == RemediationAction.ESCALATE:
        # Alert human operator
        ...
```

### With Critique

```python
from agentic_trading.validation import (
    CritiqueValidator,
    CritiqueTriggerConfig,
)

critique_v = CritiqueValidator(
    trigger_config=CritiqueTriggerConfig(
        always_critique_types=["CMTAssessmentResponse"],
        notional_usd_threshold=50_000,
    ),
    call_llm=my_llm_callable,  # injected
)

pipeline = ValidationPipeline(
    validators=[schema_v, evidence_v, biz_v],
    critique_validator=critique_v,
)
```

### IValidator Protocol

Custom validators implement `IValidator`:

```python
from agentic_trading.validation import IValidator, ValidationIssue

class CustomValidator:
    @property
    def layer_name(self) -> str:
        return "custom"

    def validate(self, parsed_output, envelope, result):
        issues = []
        # ... custom checks ...
        return issues
```

## File Reference

| File | What |
|------|------|
| `src/agentic_trading/validation/models.py` | Core models: enums, ValidationResult, ValidationIssue, ClaimAnnotation, CritiqueResult, RemediationRecord |
| `src/agentic_trading/validation/errors.py` | Error hierarchy (5 error types) |
| `src/agentic_trading/validation/protocol.py` | IValidator protocol |
| `src/agentic_trading/validation/schema_validator.py` | Layer 1: JSON Schema + Pydantic |
| `src/agentic_trading/validation/evidence_validator.py` | Layer 2: claim-to-evidence linking |
| `src/agentic_trading/validation/business_rules.py` | Layer 3: declarative rules + built-in factories |
| `src/agentic_trading/validation/critique_validator.py` | Layer 4: cost-gated second-model critique |
| `src/agentic_trading/validation/remediation.py` | Remediation state machine |
| `src/agentic_trading/validation/pipeline.py` | ValidationPipeline orchestrator |
| `src/agentic_trading/validation/config.py` | ValidationConfig top-level config |
| `src/agentic_trading/validation/__init__.py` | Public API (30+ exports) |
| `schemas/validation/validation-result.json` | JSON Schema for ValidationResult |
| `schemas/validation/business-rule.json` | JSON Schema for BusinessRule |
| `schemas/validation/claim-annotation.json` | JSON Schema for ClaimAnnotation |

## Integration Points

1. **LLMResult backward compat**: Pipeline writes `llm_result.validation_passed`
   and `validation_errors` after running.

2. **MetadataStandard.quality_score**: `ValidationResult.quality_score` maps
   directly (set at the call site, not inside the framework).

3. **PolicyEngine reuse**: `_resolve_field()` and `_check_condition()` static
   methods are called directly from BusinessRuleValidator.

4. **Operator enum**: Imported from `policy/models.py`, not duplicated.

5. **EvidenceItem**: Referenced from `llm/envelope.py` for source matching.
