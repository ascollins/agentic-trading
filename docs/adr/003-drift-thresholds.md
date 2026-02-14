# ADR-003: Live-vs-Backtest Drift Detection Thresholds

**Status:** Accepted
**Date:** 2026-02-14
**Context:** Governance Framework (Soteria-inspired)

## Context

Strategies are developed and validated against historical data. When deployed
live, regime changes, data quality issues, or model degradation can cause
performance to diverge from expectations. The risk layer only intervenes once
drawdown limits are breached — which may be too late.

## Decision

Compare live rolling metrics against registered backtest baselines per strategy.
Two action thresholds:

- **30 % deviation** → REDUCE_SIZE (scale down position sizing)
- **50 % deviation** → PAUSE (halt strategy execution entirely)

Baselines are registered at deployment time. Tracked metrics include win rate,
average R-multiple, profit factor, and trade frequency.

## Alternatives Considered

1. **Statistical tests (e.g., KS test, t-test)** — more rigorous but requires
   larger sample sizes and adds latency; percentage deviation is simpler and
   sufficient for early detection.
2. **Single threshold with binary halt** — too aggressive; the two-tier model
   allows graceful degradation before full pause.
3. **No drift detection (rely on drawdown monitor)** — drawdown is a lagging
   indicator; drift detection catches problems earlier.

## Consequences

- Strategies that deviate significantly from backtested performance are
  automatically throttled or paused.
- Operators must set meaningful baselines; poor baselines produce false positives.
- Threshold values (30 % / 50 %) are configurable via `DriftDetectorConfig`.
