"""Deterministic reflection agent — pattern-based insights from trade history.

After every N closed trades, this agent queries recent TRADE_OUTCOME entries
from the memory store, groups them by strategy, computes aggregate stats
(win rate, avg R, consecutive losses, losing symbols), and stores a
REFLECTION memory entry with actionable warnings.

No LLM is used — all analysis is pure arithmetic and pattern matching.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import timedelta
from typing import Any

from agentic_trading.context.memory_store import MemoryEntry
from agentic_trading.core.enums import MemoryEntryType
from agentic_trading.core.ids import utc_now as _now

logger = logging.getLogger(__name__)

# Default TTL for reflection entries: 48 hours
_REFLECTION_TTL_HOURS = 48.0

# How many recent trade outcomes to analyze
_LOOKBACK_HOURS = 168  # 7 days


class ReflectionAgent:
    """Deterministic pattern-based reflection on recent trades.

    Parameters
    ----------
    memory_store:
        Any object implementing ``IMemoryStore`` (``store()``, ``query()``).
    trigger_every_n:
        Generate a reflection after every N closed trades (default 5).
    ttl_hours:
        TTL for reflection entries (default 48).
    """

    def __init__(
        self,
        memory_store: Any,
        trigger_every_n: int = 5,
        ttl_hours: float = _REFLECTION_TTL_HOURS,
    ) -> None:
        self._store = memory_store
        self._trigger_n = max(1, trigger_every_n)
        self._ttl = ttl_hours
        self._trade_count = 0
        self._reflection_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_trade_closed(self, trade: Any) -> None:
        """Increment trade counter and trigger reflection if threshold met."""
        self._trade_count += 1
        if self._trade_count % self._trigger_n == 0:
            self._run_reflection()

    @property
    def reflection_count(self) -> int:
        """Number of reflections generated since init."""
        return self._reflection_count

    # ------------------------------------------------------------------
    # Core reflection logic
    # ------------------------------------------------------------------

    def _run_reflection(self) -> None:
        """Query recent outcomes and generate a REFLECTION entry."""
        try:
            since = _now() - timedelta(hours=_LOOKBACK_HOURS)
            outcomes = self._store.query(
                entry_type=MemoryEntryType.TRADE_OUTCOME,
                since=since,
                limit=100,
                min_relevance=0.0,
            )
            if not outcomes:
                return

            insights = self._analyze(outcomes)
            if not insights:
                return

            summary = " | ".join(insights)
            content = {
                "trade_count": len(outcomes),
                "insights": insights,
                "stats": self._compute_stats(outcomes),
            }

            entry = MemoryEntry(
                entry_type=MemoryEntryType.REFLECTION,
                tags=["reflection", "auto"],
                content=content,
                summary=summary,
                relevance_score=1.0,
                ttl_hours=self._ttl,
            )
            self._store.store(entry)
            self._reflection_count += 1
            logger.info("Reflection generated: %s", summary[:120])
        except Exception:
            logger.warning("Reflection failed", exc_info=True)

    def _analyze(self, outcomes: list[MemoryEntry]) -> list[str]:
        """Extract actionable insights from recent trade outcomes."""
        insights: list[str] = []

        # Group by strategy
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for entry in outcomes:
            sid = entry.content.get("strategy_id", "unknown")
            by_strategy[sid].append(entry.content)

        for strategy_id, trades in by_strategy.items():
            n = len(trades)
            if n < 2:
                continue

            wins = sum(1 for t in trades if t.get("outcome") == "win")
            win_rate = wins / n
            r_values = [t.get("r_multiple", 0.0) for t in trades]
            avg_r = sum(r_values) / n if n else 0.0

            # Consecutive losses (most recent first)
            sorted_trades = sorted(
                trades,
                key=lambda t: t.get("hold_hours", 0),
                reverse=True,
            )
            consec_losses = 0
            for t in sorted_trades:
                if t.get("outcome") == "loss":
                    consec_losses += 1
                else:
                    break

            # Low win rate warning
            if n >= 3 and win_rate < 0.30:
                insights.append(
                    f"WARNING: {strategy_id} win_rate={win_rate:.0%} "
                    f"over {n} trades"
                )

            # Consecutive loss streak
            if consec_losses >= 3:
                insights.append(
                    f"CAUTION: {strategy_id} has {consec_losses} "
                    f"consecutive losses"
                )

            # Negative average R
            if n >= 3 and avg_r < -0.5:
                insights.append(
                    f"ALERT: {strategy_id} avg_R={avg_r:+.2f} "
                    f"over {n} trades"
                )

            # Big winners worth noting
            big_winners = [t for t in trades if t.get("r_multiple", 0) >= 2.0]
            if big_winners:
                symbols = set(t.get("symbol", "?") for t in big_winners)
                insights.append(
                    f"STRONG: {strategy_id} {len(big_winners)} big "
                    f"winners (R>=2) in {', '.join(symbols)}"
                )

        # Cross-strategy: losing symbols
        by_symbol: dict[str, list[dict]] = defaultdict(list)
        for entry in outcomes:
            sym = entry.content.get("symbol", "")
            if sym:
                by_symbol[sym].append(entry.content)

        for sym, trades in by_symbol.items():
            losses = [t for t in trades if t.get("outcome") == "loss"]
            if len(losses) >= 3 and len(losses) / len(trades) > 0.7:
                insights.append(
                    f"AVOID: {sym} has {len(losses)}/{len(trades)} "
                    f"losses across strategies"
                )

        return insights

    def _compute_stats(self, outcomes: list[MemoryEntry]) -> dict:
        """Compute aggregate stats from outcomes for the content dict."""
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for entry in outcomes:
            sid = entry.content.get("strategy_id", "unknown")
            by_strategy[sid].append(entry.content)

        stats: dict[str, Any] = {}
        for sid, trades in by_strategy.items():
            n = len(trades)
            wins = sum(1 for t in trades if t.get("outcome") == "win")
            r_values = [t.get("r_multiple", 0.0) for t in trades]
            stats[sid] = {
                "total": n,
                "wins": wins,
                "win_rate": round(wins / n, 2) if n else 0,
                "avg_r": round(sum(r_values) / n, 2) if n else 0,
                "sum_r": round(sum(r_values), 2),
            }
        return stats
