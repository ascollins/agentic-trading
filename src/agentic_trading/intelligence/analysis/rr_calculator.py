"""Risk-reward calculation and PnL projection.

Computes multi-target R:R ratios, probability-weighted expected R,
blended R:R, and per-scenario PnL projections for pre-trade assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentic_trading.core.enums import SetupGrade, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class TargetLevel:
    """A single take-profit target with R:R and scale-out percentage."""

    price: float
    rr_ratio: float  # R:R ratio for this target
    scale_out_pct: float  # Fraction of position to close (0.0–1.0)
    probability: float = 0.5  # Estimated hit probability


@dataclass
class RRResult:
    """Complete R:R analysis result."""

    entry: float
    stop_loss: float
    risk_per_unit: float  # |entry - stop_loss|
    direction: SignalDirection
    targets: list[TargetLevel] = field(default_factory=list)
    blended_rr: float = 0.0  # Scale-out-weighted average R:R
    expected_r: float = 0.0  # Probability-weighted expected R
    setup_grade: SetupGrade = SetupGrade.C
    is_valid: bool = True
    invalidation_reason: str = ""


def calculate_rr(
    entry: float,
    stop_loss: float,
    targets: list[float],
    scale_out_pcts: list[float] | None = None,
    direction: SignalDirection | None = None,
) -> RRResult:
    """Compute multi-target R:R ratios.

    Args:
        entry: Entry price.
        stop_loss: Stop-loss price.
        targets: List of take-profit price levels, ordered by proximity.
        scale_out_pcts: Optional per-target scale-out fractions.
            If *None*, distributes equally.  Values are normalised
            so they need not sum to 1.0.
        direction: Trade direction.  If *None*, inferred from
            ``entry`` vs ``stop_loss``.

    Returns:
        :class:`RRResult` with per-target R:R and blended metrics.
    """
    if entry <= 0 or stop_loss <= 0:
        return RRResult(
            entry=entry,
            stop_loss=stop_loss,
            risk_per_unit=0,
            direction=SignalDirection.FLAT,
            is_valid=False,
            invalidation_reason="Invalid prices",
        )

    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit == 0:
        return RRResult(
            entry=entry,
            stop_loss=stop_loss,
            risk_per_unit=0,
            direction=SignalDirection.FLAT,
            is_valid=False,
            invalidation_reason="Entry equals stop loss",
        )

    # Infer direction
    if direction is None:
        direction = (
            SignalDirection.LONG if entry > stop_loss else SignalDirection.SHORT
        )

    if not targets:
        return RRResult(
            entry=entry,
            stop_loss=stop_loss,
            risk_per_unit=round(risk_per_unit, 8),
            direction=direction,
            is_valid=False,
            invalidation_reason="No targets provided",
        )

    # Default equal scale-out
    n = len(targets)
    if scale_out_pcts is None:
        scale_out_pcts = [1.0 / n] * n

    # Normalise scale-out percentages
    total_pct = sum(scale_out_pcts)
    if total_pct > 0:
        scale_out_pcts = [p / total_pct for p in scale_out_pcts]
    else:
        scale_out_pcts = [1.0 / n] * n

    target_levels: list[TargetLevel] = []
    for i, tp in enumerate(targets):
        if direction == SignalDirection.LONG:
            reward_per_unit = tp - entry
        else:
            reward_per_unit = entry - tp

        rr = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
        pct = scale_out_pcts[i] if i < len(scale_out_pcts) else 0

        # Simple probability heuristic: closer targets more likely
        prob = max(0.1, min(0.9, 1.0 - (abs(rr) / 10.0)))

        target_levels.append(
            TargetLevel(
                price=tp,
                rr_ratio=round(rr, 2),
                scale_out_pct=round(pct, 4),
                probability=round(prob, 2),
            )
        )

    # Blended R:R (weighted by scale-out percentage)
    blended = sum(t.rr_ratio * t.scale_out_pct for t in target_levels)

    # Expected R (probability-weighted)
    expected = sum(
        t.rr_ratio * t.scale_out_pct * t.probability for t in target_levels
    )
    # Subtract probability-weighted stop-loss cost
    avg_stop_prob = 1.0 - sum(
        t.probability * t.scale_out_pct for t in target_levels
    )
    expected -= max(0, avg_stop_prob)

    grade = _assess_setup(blended, expected)

    return RRResult(
        entry=entry,
        stop_loss=stop_loss,
        risk_per_unit=round(risk_per_unit, 8),
        direction=direction,
        targets=target_levels,
        blended_rr=round(blended, 2),
        expected_r=round(expected, 2),
        setup_grade=grade,
    )


def project_pnl(
    account_size: float,
    risk_pct: float,
    rr_result: RRResult,
) -> dict[str, Any]:
    """Project PnL scenarios from an R:R result.

    Args:
        account_size: Total account value.
        risk_pct: Risk per trade as a decimal (e.g. 0.01 for 1%).
        rr_result: Precomputed :class:`RRResult`.

    Returns:
        Dict with per-target PnL projections and loss scenario.
    """
    risk_amount = account_size * risk_pct
    if risk_amount <= 0:
        return {
            "risk_amount": 0,
            "scenarios": [],
            "max_loss": 0,
            "full_target_profit": 0,
            "expected_value": 0,
            "blended_rr": rr_result.blended_rr,
            "setup_grade": rr_result.setup_grade.value,
        }

    scenarios = []
    for target in rr_result.targets:
        profit = risk_amount * target.rr_ratio * target.scale_out_pct
        scenarios.append(
            {
                "target_price": target.price,
                "rr_ratio": target.rr_ratio,
                "scale_out_pct": target.scale_out_pct,
                "profit_usd": round(profit, 2),
                "profit_pct": round(profit / account_size * 100, 4)
                if account_size > 0
                else 0,
                "probability": target.probability,
            }
        )

    total_win = sum(s["profit_usd"] for s in scenarios)
    prob_weighted_win = sum(
        s["profit_usd"] * s["probability"] for s in scenarios
    )
    avg_stop_prob = 1.0 - sum(
        t.probability * t.scale_out_pct for t in rr_result.targets
    )
    expected_value = prob_weighted_win - risk_amount * max(0, avg_stop_prob)

    return {
        "risk_amount": round(risk_amount, 2),
        "risk_pct": risk_pct,
        "max_loss": round(-risk_amount, 2),
        "full_target_profit": round(total_win, 2),
        "expected_value": round(expected_value, 2),
        "blended_rr": rr_result.blended_rr,
        "setup_grade": rr_result.setup_grade.value,
        "scenarios": scenarios,
    }


def _assess_setup(blended_rr: float, expected_r: float) -> SetupGrade:
    """Qualitative setup grading based on R:R metrics."""
    if blended_rr >= 4.0 and expected_r >= 1.5:
        return SetupGrade.A_PLUS
    if blended_rr >= 3.0 and expected_r >= 1.0:
        return SetupGrade.A
    if blended_rr >= 2.0 and expected_r >= 0.5:
        return SetupGrade.B
    if blended_rr >= 1.5 and expected_r >= 0.0:
        return SetupGrade.C
    if blended_rr >= 1.0:
        return SetupGrade.D
    return SetupGrade.F
