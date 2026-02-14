# ADR-005: Governance Layer Above Risk Layer

**Status:** Accepted
**Date:** 2026-02-14
**Context:** Governance Framework (Soteria-inspired)

## Context

The platform has an existing risk layer (kill switch, circuit breakers, drawdown
monitor, pre/post-trade risk checks) that validates individual orders. A separate
concern — whether a *strategy* should be allowed to trade at all, and at what
sizing — was previously unaddressed.

## Decision

Place the governance layer **above** the risk layer in the execution pipeline:

1. Governance gate runs between deduplication and pre-trade risk check.
2. Governance answers: "should this strategy trade, and at what scale?"
3. Risk answers: "is this specific order within safe limits?"

The governance layer is entirely opt-in via `settings.governance.enabled`
(default `false`). When disabled, the execution pipeline is unchanged.

## Alternatives Considered

1. **Merge governance into the risk layer** — muddies the separation of concerns;
   strategy-level policy and order-level safety serve different purposes.
2. **Governance after risk checks** — wastes risk-check computation on orders
   that governance would reject.
3. **Separate pre-processing service** — over-engineered for an in-process
   concern; adds network latency and deployment complexity.

## Consequences

- Clear separation: governance = strategy policy, risk = order safety.
- Feature-flagged, so zero impact on existing tests and behaviour.
- Governance decisions are logged and metricked independently from risk events.
- Both layers must pass for an order to execute, providing defense in depth.
