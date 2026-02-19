"""Trade impact classification.

Scores each order across four dimensions and maps the composite score
to an :class:`ImpactTier`:

1. **Notional size** — absolute USD value of the order.
2. **Concentration** — order size as a percentage of portfolio equity.
3. **Blast radius** — leverage and margin mode exposure.
4. **Irreversibility** — reduce-only orders are less risky.

Inspired by Soteria's Epistemic Transition Detector (C1) which
classifies actions by their potential impact before allowing execution.
"""

from __future__ import annotations

import logging

from agentic_trading.core.config import ImpactClassifierConfig
from agentic_trading.core.enums import ImpactTier

logger = logging.getLogger(__name__)


class ImpactClassifier:
    """Classifies trade orders by impact tier.

    Usage::

        clf = ImpactClassifier(config)
        tier = clf.classify(
            symbol="BTC/USDT",
            notional_usd=100_000,
            portfolio_pct=0.05,
            is_reduce_only=False,
            leverage=3,
            existing_positions=5,
        )
    """

    def __init__(self, config: ImpactClassifierConfig) -> None:
        self._config = config

    def classify(
        self,
        symbol: str,
        notional_usd: float,
        portfolio_pct: float,
        is_reduce_only: bool = False,
        leverage: int = 1,
        existing_positions: int = 0,
    ) -> ImpactTier:
        """Classify an order's impact tier.

        Args:
            symbol: Trading pair.
            notional_usd: USD notional value of the order.
            portfolio_pct: Order size as a fraction of portfolio equity.
            is_reduce_only: Whether the order reduces an existing position.
            leverage: Leverage applied to the order.
            existing_positions: Number of existing open positions.

        Returns:
            :class:`ImpactTier` classification.
        """
        score = self._compute_score(
            notional_usd=notional_usd,
            portfolio_pct=portfolio_pct,
            is_reduce_only=is_reduce_only,
            leverage=leverage,
            existing_positions=existing_positions,
        )

        tier = self._score_to_tier(score)
        logger.debug(
            "Impact classification: %s notional=$%.0f pf_pct=%.2f%% "
            "leverage=%d score=%.2f → %s",
            symbol,
            notional_usd,
            portfolio_pct * 100,
            leverage,
            score,
            tier.value,
        )
        return tier

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        notional_usd: float,
        portfolio_pct: float,
        is_reduce_only: bool,
        leverage: int,
        existing_positions: int,
    ) -> float:
        """Compute composite impact score in [0.0, 1.0].

        Each dimension contributes 0.0–0.25, summing to a max of 1.0.
        """
        cfg = self._config

        # 1. Notional size (0–0.25)
        if notional_usd >= cfg.critical_notional_usd:
            notional_score = 0.25
        elif notional_usd >= cfg.high_notional_usd:
            notional_score = 0.20
        elif notional_usd >= cfg.high_notional_usd * 0.5:
            notional_score = 0.10
        else:
            notional_score = 0.05

        # 2. Concentration (0–0.25)
        if portfolio_pct >= cfg.concentration_threshold_pct:
            concentration_score = 0.25
        elif portfolio_pct >= cfg.concentration_threshold_pct * 0.5:
            concentration_score = 0.15
        else:
            concentration_score = portfolio_pct / cfg.concentration_threshold_pct * 0.25

        # 3. Blast radius — leverage + position count (0–0.25)
        leverage_score = min(0.15, (leverage - 1) * 0.03)
        position_count_score = min(0.10, existing_positions * 0.02)
        blast_score = leverage_score + position_count_score

        # 4. Irreversibility (0–0.25)
        # Reduce-only is safer, new positions are more impactful
        if is_reduce_only:
            irreversibility_score = 0.05
        else:
            irreversibility_score = 0.15
            # Higher leverage increases irreversibility
            if leverage > 3:
                irreversibility_score = 0.25

        total = notional_score + concentration_score + blast_score + irreversibility_score
        return min(1.0, total)

    @staticmethod
    def _score_to_tier(score: float) -> ImpactTier:
        """Map composite score to impact tier."""
        if score >= 0.75:
            return ImpactTier.CRITICAL
        if score >= 0.50:
            return ImpactTier.HIGH
        if score >= 0.25:
            return ImpactTier.MEDIUM
        return ImpactTier.LOW
