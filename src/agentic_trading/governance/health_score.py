"""Strategy health scoring via epistemic debt/credit model.

Inspired by Soteria's Outcome Validation component: each trade outcome
shifts a running debt/credit balance.  Losses accumulate *debt* (reducing
sizing multiplier), wins accumulate *credit* (slowly restoring it).

The score is ``1.0 - (debt / max_debt)``, clamped to ``[0.0, 1.0]``.
The sizing multiplier is ``max(min_sizing_multiplier, score)``.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from agentic_trading.core.config import HealthScoreConfig
from agentic_trading.core.events import HealthScoreUpdate

logger = logging.getLogger(__name__)


@dataclass
class _StrategyHealth:
    """Internal mutable state for one strategy."""

    debt: float = 0.0
    credit: float = 0.0
    outcomes: deque = field(default_factory=lambda: deque())


class HealthTracker:
    """Rolling outcome tracker with debt/credit sizing adjustment.

    Usage::

        tracker = HealthTracker(config)
        tracker.record_outcome("trend_following", won=False, r_multiple=-1.0)
        mult = tracker.get_sizing_multiplier("trend_following")
    """

    def __init__(self, config: HealthScoreConfig) -> None:
        self._config = config
        self._state: dict[str, _StrategyHealth] = defaultdict(
            _StrategyHealth
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        strategy_id: str,
        won: bool,
        r_multiple: float = 0.0,
    ) -> HealthScoreUpdate:
        """Record a trade outcome and return the updated health event.

        Args:
            strategy_id: Which strategy produced this trade.
            won: Whether the trade was profitable.
            r_multiple: R-multiple of the trade (positive = profit R).
        """
        state = self._state[strategy_id]
        cfg = self._config

        # Maintain rolling window
        state.outcomes.append((won, r_multiple))
        if len(state.outcomes) > cfg.window_size:
            state.outcomes.popleft()

        if won:
            # Credit reduces debt first, then adds surplus
            credit_amount = cfg.credit_per_win * cfg.recovery_rate
            if state.debt > 0:
                reduction = min(credit_amount, state.debt)
                state.debt -= reduction
                credit_amount -= reduction
            state.credit += credit_amount
        else:
            # Debt accumulates (scaled by loss magnitude)
            magnitude = max(1.0, abs(r_multiple))
            state.debt = min(
                cfg.max_debt,
                state.debt + cfg.debt_per_loss * magnitude,
            )

        score = self.get_score(strategy_id)
        mult = self.get_sizing_multiplier(strategy_id)

        logger.debug(
            "Health update: %s won=%s r=%.2f debt=%.2f credit=%.2f score=%.2f mult=%.2f",
            strategy_id,
            won,
            r_multiple,
            state.debt,
            state.credit,
            score,
            mult,
        )

        return HealthScoreUpdate(
            strategy_id=strategy_id,
            score=round(score, 4),
            debt=round(state.debt, 4),
            credit=round(state.credit, 4),
            sizing_multiplier=round(mult, 4),
            window_trades=len(state.outcomes),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_score(self, strategy_id: str) -> float:
        """Health score in [0.0, 1.0]. 1.0 = pristine, 0.0 = maximum debt."""
        state = self._state[strategy_id]
        if self._config.max_debt <= 0:
            return 1.0
        score = 1.0 - (state.debt / self._config.max_debt)
        return max(0.0, min(1.0, score))

    def get_sizing_multiplier(self, strategy_id: str) -> float:
        """Sizing multiplier derived from health score.

        Returns a value in [min_sizing_multiplier, 1.0].
        """
        score = self.get_score(strategy_id)
        return max(self._config.min_sizing_multiplier, score)

    def get_debt(self, strategy_id: str) -> float:
        """Current debt value for a strategy."""
        return self._state[strategy_id].debt

    def get_credit(self, strategy_id: str) -> float:
        """Current credit value for a strategy."""
        return self._state[strategy_id].credit

    def get_window_trades(self, strategy_id: str) -> int:
        """Number of trades in the rolling window."""
        return len(self._state[strategy_id].outcomes)

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def reset(self, strategy_id: str) -> None:
        """Reset a strategy's health state to pristine."""
        self._state[strategy_id] = _StrategyHealth()
        logger.info("Health reset: %s", strategy_id)
