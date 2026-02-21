# Platform-Wide Metadata Standard

## Overview

The **MetadataStandard** is a composable Pydantic model that any platform entity
can embed via an optional field:

```python
class MyEntity(BaseModel):
    ...
    metadata: MetadataStandard | None = None
```

It provides 12 canonical fields covering identity, provenance, trust,
classification, lifecycle, and versioning. It does **not** replace existing
model-specific metadata — it augments it with a shared contract.

## Architecture

```
┌──────────────────────┐   ┌──────────────────────┐
│    MemoryEntry        │   │    EvidenceItem       │
│  entry_id, tags, ...  │   │  source, content, ... │
│  metadata: Standard?  │   │  metadata: Standard?  │
└──────────┬───────────┘   └──────────┬───────────┘
           │                           │
           └─────────┬─────────────────┘
                     ▼
          ┌─────────────────────┐
          │  MetadataStandard   │
          │  source_id          │
          │  source_type        │
          │  author             │
          │  trust_level        │
          │  quality_score      │
          │  status             │
          │  lineage            │
          │  scope              │
          │  ...                │
          └─────────────────────┘
```

## Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_id` | `str` | UUID v4 | Unique identifier for this entity |
| `source_type` | `SourceType` | `SYSTEM` | What produced this data |
| `author` | `str` | `""` | Creator: agent_id, `human:<user>`, component |
| `created_at` | `datetime` | UTC now | Creation timestamp |
| `updated_at` | `datetime` | UTC now | Last modification timestamp |
| `scope` | `MetadataScope` | empty | Jurisdiction / applicability |
| `classification` | `Classification` | `INTERNAL` | Sensitivity level |
| `quality_score` | `float` | `1.0` | Application-defined quality (0.0–1.0) |
| `trust_level` | `TrustLevel` | `HIGH` | Source trust level |
| `status` | `MetadataStatus` | `ACTIVE` | Lifecycle status |
| `expiry` | `datetime \| None` | `None` | Hard expiry (cut-off) |
| `deprecated_at` | `datetime \| None` | `None` | When deprecated |
| `deprecated_reason` | `str` | `""` | Why deprecated |
| `superseded_by` | `str` | `""` | source_id of replacement |
| `version` | `int` | `1` | Version number |
| `lineage` | `Lineage` | empty | Provenance chain |
| `content_hash` | `str` | `""` | SHA256[:16] integrity hash |

## Enums

### SourceType

| Value | Description |
|-------|-------------|
| `agent` | LLM agent output |
| `strategy` | Strategy signal/decision |
| `market_data` | Exchange feed data |
| `human` | Human-authored content |
| `system` | Platform-generated |
| `derived` | Computed from other sources |
| `external` | Third-party API |
| `policy` | Governance rule |

### Classification

| Value | Description |
|-------|-------------|
| `public` | Safe for any consumer |
| `internal` | Platform-internal only |
| `confidential` | Restricted to authorized agents |
| `restricted` | Requires explicit clearance |

### TrustLevel

| Value | Weight | Description |
|-------|--------|-------------|
| `verified` | 1.0 | Cryptographically signed or reconciled |
| `high` | 0.95 | Trusted source, no anomalies |
| `medium` | 0.8 | Standard trust |
| `low` | 0.5 | Unverified or degraded |
| `untrusted` | 0.0 | Flagged — filtered from decisions |

### MetadataStatus

| Value | Description |
|-------|-------------|
| `active` | Normal, in-use |
| `deprecated` | Still works, but warns on use |
| `do_not_use` | Filtered from retrieval results |
| `archived` | Read-only historical record |
| `expired` | Past expiry time, filtered |

## Required Metadata by Entity Type

### Knowledge Documents

Policy rules, strategy descriptions, config docs.

| Field | Required | Typical Value |
|-------|----------|---------------|
| `source_id` | Yes | UUID |
| `source_type` | Yes | `HUMAN` or `SYSTEM` |
| `author` | Yes | `human:<user>` or component |
| `created_at` | Yes | auto |
| `classification` | Yes | `INTERNAL` |
| `version` | Yes | 1+ |
| `status` | Yes | `ACTIVE` |
| trust_level | Recommended | `HIGH` / `VERIFIED` |

### Chunks (Document Subdivisions)

| Field | Required | Typical Value |
|-------|----------|---------------|
| `source_id` | Yes | UUID |
| `source_type` | Yes | `DERIVED` |
| `lineage` | Yes | parent_id → doc source_id |
| `version` | Yes | inherits parent |
| trust_level | Recommended | inherits parent |

### Vector Entries (Future)

| Field | Required | Typical Value |
|-------|----------|---------------|
| `source_id` | Yes | UUID |
| `source_type` | Yes | `DERIVED` |
| `lineage` | Yes | parent_id → source chunk |
| `content_hash` | Yes | hash of embedding input |
| `expiry` | Recommended | stale after N hours |
| `quality_score` | Recommended | embedding quality |
| trust_level | Recommended | `MEDIUM` |

### Decisions

Governance, risk, policy evaluation results.

| Field | Required | Typical Value |
|-------|----------|---------------|
| `source_id` | Yes | UUID |
| `source_type` | Yes | `SYSTEM` or `AGENT` |
| `author` | Yes | agent/component that decided |
| `scope` | Yes | strategy, symbol scoping |
| `lineage` | Recommended | trace_id / causation chain |
| trust_level | Yes | `VERIFIED` (rules) / `HIGH` (agent) |

### Generated Artifacts

CMT assessments, narrations, reasoning traces, LLM outputs.

| Field | Required | Typical Value |
|-------|----------|---------------|
| `source_id` | Yes | UUID |
| `source_type` | Yes | `AGENT` |
| `author` | Yes | agent_id |
| `quality_score` | Yes | from LLM confidence / validation |
| `trust_level` | Yes | `MEDIUM` (LLM output) |
| `lineage` | Yes | envelope_id / trace_id chain |
| `expiry` | Recommended | artifacts age out |

## Metadata Flow: Retrieval → Evidence → LLM Citation

```
1. STORAGE
   MemoryEntry stored with metadata: MetadataStandard
   Fields populated: source_type, author, trust_level, quality_score,
                     status, lineage, expiry

2. RETRIEVAL (query-time filtering)
   IMemoryStore.query():
     a. Existing filters: symbol, entry_type, tags, since
     b. metadata.is_usable() → filter out DO_NOT_USE, EXPIRED, UNTRUSTED
     c. Time-decay scoring (existing ttl_hours)
     d. Trust-weighted relevance:
        effective = decayed × trust_weight × quality_score
        where trust_weight = TRUST_WEIGHTS[trust_level]

3. ASSEMBLY (EnvelopeBuilder)
   .add_evidence(source, content, relevance, metadata_std=entry.metadata)
   → EvidenceItem carries metadata_std alongside source/content/relevance

4. LLM PROMPT CONSTRUCTION
   Evidence sorted by trust_level then relevance.
   DEPRECATED items prefixed: "[DEPRECATED: {reason}]"
   Lineage summarized for audit context.

5. CITATION IN LLM OUTPUT
   LLM references evidence by source_id.
   parsed_output.citations[].source_id traces back:
     source_id → metadata_std → lineage → root_id

6. AUDIT TRAIL
   trace_id → LLMInteraction → envelope → evidence[].metadata_std
                                              → lineage → root source
```

### Trust-Weighted Relevance Formula

```python
from agentic_trading.meta import compute_effective_relevance, TrustLevel

effective = compute_effective_relevance(
    base_relevance=0.85,
    trust_level=TrustLevel.MEDIUM,
    quality_score=0.9,
)
# Result: 0.85 * 0.8 * 0.9 = 0.612
```

## Deprecated / Do-Not-Use Handling

### Status Transitions

```
              ACTIVE
             /  |   \
deprecate() /   |    \ (time-based)
           v    |     v
     DEPRECATED |   EXPIRED
          |     |
mark_do_not_use()
          v     v
     DO_NOT_USE
          |
    (admin archive)
          v
       ARCHIVED
```

### Behavioral Rules

| Status | Retrieval | LLM Evidence | Logging |
|--------|-----------|--------------|---------|
| `ACTIVE` | Normal | Normal relevance | None |
| `DEPRECATED` | Included + warning | Included, 0.7× weight | `logger.warning(...)` |
| `DO_NOT_USE` | Filtered at query | Never included | `logger.info(...)` |
| `EXPIRED` | Filtered at query | Never included | `logger.debug(...)` |
| `ARCHIVED` | Explicit archive query only | Never included | None |

### Expiry Mechanics

Two expiry mechanisms coexist:

1. **Soft expiry** (existing `MemoryEntry.ttl_hours`): Exponential decay
   reduces relevance over time. Entry remains retrievable but diminishes.

2. **Hard expiry** (`MetadataStandard.expiry`): After this datetime,
   `is_usable()` returns `False` and the item is filtered from normal
   retrieval. This is a hard cut-off.

When both present, hard expiry takes precedence.

### Supersession Chain

When deprecated with `superseded_by`:
- Retrieval code should follow the chain to find the active replacement
- Max depth: 5 hops (prevents cycles)
- If replacement is also deprecated/expired, fall back to most recent
  ACTIVE ancestor

### Example: Deprecating a Source

```python
from agentic_trading.meta import MetadataStandard

# Original metadata
meta = MetadataStandard(
    source_type=SourceType.HUMAN,
    author="human:analyst-1",
)

# Deprecate with reason
deprecated = meta.deprecate(
    reason="Replaced by automated CMT assessment",
    superseded_by="new-source-id-abc",
)
assert deprecated.status == MetadataStatus.DEPRECATED
assert deprecated.deprecated_reason == "Replaced by automated CMT assessment"
assert deprecated.superseded_by == "new-source-id-abc"
assert deprecated.is_usable()  # Still usable, but warns

# Hard block
blocked = deprecated.mark_do_not_use(reason="Contains stale data")
assert not blocked.is_usable()  # Filtered from retrieval
```

## Integration Guide

### Composing with Existing Models

The standard is adopted incrementally. Add an optional field — no breaking
changes to serialization:

```python
# Future: EvidenceItem with metadata
class EvidenceItem(BaseModel):
    source: str
    content: dict[str, Any] = Field(default_factory=dict)
    relevance: float = 1.0
    retrieved_at: datetime = Field(default_factory=_now)
    metadata: MetadataStandard | None = None  # NEW — optional

# Future: MemoryEntry with metadata
class MemoryEntry(BaseModel):
    entry_id: str = Field(default_factory=_uuid)
    entry_type: MemoryEntryType
    ...
    metadata: MetadataStandard | None = None  # NEW — optional
```

Because the field defaults to `None`, existing JSONL and Postgres data
deserializes without error.

### Creating Metadata with Factories

```python
from agentic_trading.meta import (
    metadata_for_agent,
    metadata_for_market_data,
    metadata_for_derived,
)

# Agent-generated artifact
meta = metadata_for_agent("cmt-analyst-01", "cmt_analyst")
# → source_type=AGENT, trust_level=MEDIUM

# Market data
meta = metadata_for_market_data("bybit", "BTCUSDT")
# → trust_level=HIGH, scope.exchanges=["bybit"], scope.symbols=["BTCUSDT"]

# Derived from another entity
child = metadata_for_derived(meta, derivation="aggregated")
# → source_type=DERIVED, lineage.parent_id=meta.source_id, chain_depth=1
```

### Filtering with is_usable()

```python
entries = memory_store.query(symbol="BTCUSDT", limit=20)

usable = [
    e for e in entries
    if e.metadata is None or e.metadata.is_usable()
]
```

## Governance Rules

1. **All new entities** should populate `source_type` and `author`.
2. **LLM-generated artifacts** must set `trust_level=MEDIUM` and provide
   `quality_score` from validation results.
3. **Derived entities** must populate `lineage` to enable provenance tracing.
4. **Deprecation** requires a `reason`. Setting `superseded_by` is recommended
   but not mandatory.
5. **DO_NOT_USE** is a hard filter — use only when data is known to be
   incorrect or harmful.
6. **ARCHIVED** is for permanent read-only storage — the entity remains
   queryable via explicit archive queries but is excluded from normal
   retrieval and LLM evidence.
