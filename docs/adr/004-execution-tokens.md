# ADR-004: Scoped Execution Tokens

**Status:** Accepted
**Date:** 2026-02-14
**Context:** Governance Framework (Soteria-inspired)

## Context

Maturity levels provide coarse-grained, persistent authorization. Some scenarios
require finer-grained, temporary permissions — e.g., allowing a gated (L2)
strategy to execute a specific trade window, or revoking all execution rights
for a strategy during maintenance.

## Decision

Introduce scoped, time-bounded execution tokens with the following properties:

- **TTL**: tokens expire automatically after a configurable duration.
- **Revocable**: individual tokens or all tokens for a strategy can be revoked
  instantly.
- **Audit-bound**: each token is tied to a trace ID for end-to-end traceability.
- **Single-use or multi-use**: configurable consumption semantics.

When token enforcement is enabled for a maturity level, the governance gate
requires a valid token in addition to passing other checks.

## Alternatives Considered

1. **Time-windowed maturity overrides** — conflates the maturity model with
   temporary permissions; harder to audit.
2. **Manual approval queue** — adds human latency; doesn't scale for high-frequency
   strategy execution.
3. **No token layer (maturity only)** — insufficient for scenarios requiring
   temporary or scoped authorization.

## Consequences

- Adds a token validation step in the governance gate hot path.
- Token cleanup (expiry) must run periodically to avoid unbounded memory growth.
- Provides an emergency kill mechanism: revoking all tokens for a strategy
  immediately halts its execution without changing maturity level.
