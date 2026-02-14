# ADR-001: Five-Level Strategy Maturity Ladder

**Status:** Accepted
**Date:** 2026-02-14
**Context:** Governance Framework (Soteria-inspired)

## Context

Strategies range from untested ideas to battle-proven production systems. Without
a formal maturity model, new or degraded strategies can execute with the same
authority as proven ones, creating outsized risk.

## Decision

Adopt a five-level maturity ladder (L0 Shadow → L4 Autonomous) with asymmetric
transition rules:

- **Promotion** requires evidence: 50+ trades, win rate > 0.45, profit factor > 1.1.
  Only one level at a time.
- **Demotion** is protective and fast: drawdown > 10 % or loss streak > 10 triggers
  immediate demotion, which can skip multiple levels.

Each level carries a hard sizing cap (L0/L1 = 0 %, L2 = 10 %, L3 = 25 %,
L4 = 100 %).

## Alternatives Considered

1. **Binary on/off per strategy** — too coarse; no graduated trust building.
2. **Three-level model (shadow / paper / live)** — lacks the constrained-sizing
   middle ground that limits blast radius for newly-promoted strategies.
3. **Continuous score without discrete levels** — harder to reason about and
   communicate to operators.

## Consequences

- New strategies must earn execution privileges through demonstrated performance.
- A single bad drawdown event can immediately restrict a strategy, before the
  risk layer's circuit breakers fire.
- Operators can manually set maturity levels as an override.
