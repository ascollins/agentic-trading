"""Strategy maturity level manager.

Implements a five-level maturity progression inspired by the Soteria
Agent Maturity Manager:

- **L0 (Shadow)**: Log-only, no execution permitted.
- **L1 (Paper)**: Paper trading only, no real orders.
- **L2 (Gated)**: Live execution with strict oversight (10% sizing cap).
- **L3 (Constrained)**: Live with a configurable sizing cap (default 25%).
- **L4 (Autonomous)**: Full autonomy, no sizing restrictions.

**Promotion** is slow and requires sustained good performance.
**Demotion** is fast and can skip levels on severe underperformance.
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_trading.core.config import MaturityConfig
from agentic_trading.core.enums import MaturityLevel
from agentic_trading.core.events import MaturityTransition

logger = logging.getLogger(__name__)

# Ordered list for rank-based operations
_LEVELS = list(MaturityLevel)

# Sizing caps per maturity level
_SIZING_CAPS: dict[MaturityLevel, float] = {
    MaturityLevel.L0_SHADOW: 0.0,
    MaturityLevel.L1_PAPER: 0.0,
    MaturityLevel.L2_GATED: 0.10,
    MaturityLevel.L3_CONSTRAINED: 0.25,  # overridden by config
    MaturityLevel.L4_AUTONOMOUS: 1.0,
}


class MaturityManager:
    """Tracks and manages per-strategy maturity levels.

    Usage::

        mgr = MaturityManager(config)
        mgr.get_level("trend_following")  # L1_paper (default)
        transition = mgr.evaluate_promotion("trend_following", metrics)
    """

    def __init__(self, config: MaturityConfig) -> None:
        self._config = config
        self._levels: dict[str, MaturityLevel] = {}
        # Override L3 cap from config
        _SIZING_CAPS[MaturityLevel.L3_CONSTRAINED] = config.l3_sizing_cap

    # ------------------------------------------------------------------
    # Level queries
    # ------------------------------------------------------------------

    def get_level(self, strategy_id: str) -> MaturityLevel:
        """Return current maturity level, defaulting from config."""
        if strategy_id not in self._levels:
            default = MaturityLevel(self._config.default_level)
            self._levels[strategy_id] = default
        return self._levels[strategy_id]

    def set_level(
        self, strategy_id: str, level: MaturityLevel, reason: str = "admin_override"
    ) -> MaturityTransition | None:
        """Directly set maturity level (admin override)."""
        old = self.get_level(strategy_id)
        if old == level:
            return None
        self._levels[strategy_id] = level
        logger.info(
            "Maturity override: %s %s → %s (%s)",
            strategy_id, old.value, level.value, reason,
        )
        return MaturityTransition(
            strategy_id=strategy_id,
            from_level=old,
            to_level=level,
            reason=reason,
        )

    def can_execute(self, strategy_id: str) -> bool:
        """Whether the strategy is permitted to submit live orders."""
        level = self.get_level(strategy_id)
        return level.rank >= MaturityLevel.L2_GATED.rank

    def get_sizing_cap(self, strategy_id: str) -> float:
        """Maximum sizing multiplier for the strategy's maturity level."""
        return _SIZING_CAPS[self.get_level(strategy_id)]

    # ------------------------------------------------------------------
    # Promotion (slow)
    # ------------------------------------------------------------------

    def evaluate_promotion(
        self,
        strategy_id: str,
        metrics: dict[str, Any],
    ) -> MaturityTransition | None:
        """Evaluate whether a strategy qualifies for promotion.

        Promotion requires ALL of:
        - ``total_trades`` >= ``promotion_min_trades``
        - ``win_rate`` >= ``promotion_min_win_rate``
        - ``profit_factor`` >= ``promotion_min_profit_factor``

        Only promotes one level at a time.
        """
        level = self.get_level(strategy_id)
        if level == MaturityLevel.L4_AUTONOMOUS:
            return None  # Already at max

        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0.0)
        profit_factor = metrics.get("profit_factor", 0.0)

        cfg = self._config
        if (
            total_trades >= cfg.promotion_min_trades
            and win_rate >= cfg.promotion_min_win_rate
            and profit_factor >= cfg.promotion_min_profit_factor
        ):
            new_level = _LEVELS[level.rank + 1]
            self._levels[strategy_id] = new_level
            logger.info(
                "Strategy promoted: %s %s → %s (trades=%d win_rate=%.2f pf=%.2f)",
                strategy_id,
                level.value,
                new_level.value,
                total_trades,
                win_rate,
                profit_factor,
            )
            return MaturityTransition(
                strategy_id=strategy_id,
                from_level=level,
                to_level=new_level,
                reason="performance_criteria_met",
                metrics_snapshot={
                    "total_trades": float(total_trades),
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                },
            )
        return None

    # ------------------------------------------------------------------
    # Demotion (fast)
    # ------------------------------------------------------------------

    def evaluate_demotion(
        self,
        strategy_id: str,
        metrics: dict[str, Any],
    ) -> MaturityTransition | None:
        """Evaluate whether a strategy should be demoted.

        Triggers:
        - ``drawdown_pct`` > ``demotion_drawdown_pct``: demote to L1
        - ``loss_streak`` > ``demotion_loss_streak``: demote one level

        Can skip levels on severe drawdown.
        """
        level = self.get_level(strategy_id)
        if level == MaturityLevel.L0_SHADOW:
            return None  # Already at minimum

        drawdown = metrics.get("drawdown_pct", 0.0)
        loss_streak = metrics.get("loss_streak", 0)

        cfg = self._config

        # Severe drawdown → drop to L1 (paper)
        if drawdown > cfg.demotion_drawdown_pct:
            new_level = MaturityLevel.L1_PAPER
            self._levels[strategy_id] = new_level
            logger.warning(
                "Strategy demoted (drawdown): %s %s → %s (dd=%.2f%%)",
                strategy_id,
                level.value,
                new_level.value,
                drawdown * 100,
            )
            return MaturityTransition(
                strategy_id=strategy_id,
                from_level=level,
                to_level=new_level,
                reason=f"drawdown_{drawdown:.2%}_exceeded_threshold",
                metrics_snapshot={"drawdown_pct": drawdown, "loss_streak": float(loss_streak)},
            )

        # Extended loss streak → demote one level
        if loss_streak >= cfg.demotion_loss_streak:
            new_level = _LEVELS[max(0, level.rank - 1)]
            if new_level != level:
                self._levels[strategy_id] = new_level
                logger.warning(
                    "Strategy demoted (loss streak): %s %s → %s (streak=%d)",
                    strategy_id,
                    level.value,
                    new_level.value,
                    loss_streak,
                )
                return MaturityTransition(
                    strategy_id=strategy_id,
                    from_level=level,
                    to_level=new_level,
                    reason=f"loss_streak_{loss_streak}_exceeded_threshold",
                    metrics_snapshot={"drawdown_pct": drawdown, "loss_streak": float(loss_streak)},
                )
        return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def all_levels(self) -> dict[str, MaturityLevel]:
        """Return a copy of all tracked strategy levels."""
        return dict(self._levels)
