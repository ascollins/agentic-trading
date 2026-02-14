# ADR-002: Four-Dimensional Trade Impact Classification

**Status:** Accepted
**Date:** 2026-02-14
**Context:** Governance Framework (Soteria-inspired)

## Context

Not all trades carry equal risk. A small reduce-only order is fundamentally
different from a large new position in an illiquid market. The risk layer checks
order-level limits but does not assess the systemic impact of a trade relative
to the overall portfolio.

## Decision

Classify each pending order on four dimensions before execution:

1. **Irreversibility** — how hard is it to unwind (e.g., illiquid markets, options)?
2. **Blast radius** — what fraction of portfolio equity is affected?
3. **Concentration** — does this increase single-asset concentration?
4. **Notional size** — raw dollar exposure.

A composite score maps to four tiers: LOW, MEDIUM, HIGH, CRITICAL. Higher tiers
require higher maturity levels and may trigger additional approval requirements.

## Alternatives Considered

1. **Notional-only thresholds** — misses concentration and irreversibility risk.
2. **Per-exchange classification** — too fragmented; doesn't account for
   cross-exchange portfolio effects.

## Consequences

- Impact tier is factored into governance gate decisions alongside maturity and
  health score.
- CRITICAL-tier trades are blocked unless the strategy is at L4 Autonomous.
- Adds a small computational overhead per order (four dimension calculations).
