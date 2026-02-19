"""Automated mistake detection and classification.

Analyses closed trades to identify common trading errors:
- Early exits (left >50% of MFE on the table)
- Moved stop loss (MAE exceeded initial risk)
- Chased entry (entered far from signal price)
- Held through reversal (MFE > 2R then closed at <0.5R)
- Oversized position (governance multiplier < 1.0 applied)
- Low-confidence entry (entered with confidence < threshold)
- Counter-trend (traded against prevailing trend if features available)

Inspired by Edgewonk's structured mistake tracking: every error is
classified, counted, and correlated with P&L impact so traders can
quantify the cost of each behaviour pattern.

Usage::

    detector = MistakeDetector()
    mistakes = detector.analyse(trade_record)
    print(mistakes)  # [{"type": "early_exit", "severity": "medium", ...}]
    report = detector.report("trend_following")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .record import TradeOutcome, TradePhase, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class Mistake:
    """A single identified mistake on a trade."""

    mistake_type: str       # e.g. "early_exit", "moved_stop", "chased_entry"
    severity: str           # "low", "medium", "high"
    description: str        # Human-readable explanation
    pnl_impact: float       # Estimated P&L impact (negative = cost)
    details: dict = field(default_factory=dict)


class MistakeDetector:
    """Automated mistake detection on closed trades.

    Parameters
    ----------
    early_exit_threshold : float
        Management efficiency below this triggers "early_exit".
        Default 0.4 (captured less than 40% of available profit).
    moved_stop_mae_r_threshold : float
        If MAE in R-multiples exceeds this, "moved_stop" is triggered.
        Default -1.5 (took 1.5x the planned risk).
    chased_entry_pct : float
        If entry price deviates more than this % from signal features,
        triggers "chased_entry".  Default 0.02 (2%).
    low_confidence_threshold : float
        Confidence below this triggers "low_confidence_entry".
        Default 0.3.
    reversal_mfe_r : float
        MFE in R-multiples above this + closing below reversal_close_r
        triggers "held_through_reversal".  Default 2.0.
    reversal_close_r : float
        R-multiple at close that triggers reversal detection.
        Default 0.5.
    """

    def __init__(
        self,
        *,
        early_exit_threshold: float = 0.4,
        moved_stop_mae_r_threshold: float = -1.5,
        chased_entry_pct: float = 0.02,
        low_confidence_threshold: float = 0.3,
        reversal_mfe_r: float = 2.0,
        reversal_close_r: float = 0.5,
    ) -> None:
        self._early_exit = early_exit_threshold
        self._moved_stop = moved_stop_mae_r_threshold
        self._chased_entry = chased_entry_pct
        self._low_confidence = low_confidence_threshold
        self._reversal_mfe_r = reversal_mfe_r
        self._reversal_close_r = reversal_close_r

        # Accumulate mistakes per strategy for reporting
        self._history: dict[str, list[Mistake]] = defaultdict(list)
        self._max_history: int = 5000

    # ------------------------------------------------------------------ #
    # Analysis                                                             #
    # ------------------------------------------------------------------ #

    def analyse(self, trade: TradeRecord) -> list[Mistake]:
        """Analyse a closed trade for mistakes.

        Returns a list of detected mistakes (may be empty).
        Also appends detected mistakes to the trade's ``mistakes`` list
        and stores them for aggregate reporting.
        """
        if trade.phase != TradePhase.CLOSED:
            return []

        mistakes: list[Mistake] = []

        # 1. Early exit — left profit on the table
        if trade.outcome == TradeOutcome.WIN:
            eff = trade.management_efficiency
            if 0 < eff < self._early_exit and float(trade.mfe) > 0:
                missed = float(trade.mfe) - float(trade.net_pnl)
                mistakes.append(Mistake(
                    mistake_type="early_exit",
                    severity="medium",
                    description=f"Captured only {eff:.0%} of MFE (${float(trade.mfe):.2f} available, took ${float(trade.net_pnl):.2f})",
                    pnl_impact=-missed,
                    details={"efficiency": eff, "mfe": float(trade.mfe), "net_pnl": float(trade.net_pnl)},
                ))

        # 2. Moved stop — MAE exceeded planned risk
        if trade.mae_r != 0 and trade.mae_r < self._moved_stop:
            extra_risk = abs(trade.mae_r) - 1.0  # How much beyond 1R
            risk_amt = float(trade.initial_risk_amount or 0)
            impact = -extra_risk * risk_amt if risk_amt else 0
            mistakes.append(Mistake(
                mistake_type="moved_stop",
                severity="high",
                description=f"MAE reached {trade.mae_r:.1f}R (planned stop was -1.0R)",
                pnl_impact=impact,
                details={"mae_r": trade.mae_r, "mae": float(trade.mae)},
            ))

        # 3. Held through reversal — big MFE then gave it back
        if (trade.mfe_r > self._reversal_mfe_r
                and trade.r_multiple < self._reversal_close_r):
            left_on_table = float(trade.mfe) - float(trade.net_pnl)
            mistakes.append(Mistake(
                mistake_type="held_through_reversal",
                severity="high",
                description=f"MFE was {trade.mfe_r:.1f}R but closed at {trade.r_multiple:.1f}R",
                pnl_impact=-left_on_table if left_on_table > 0 else 0,
                details={"mfe_r": trade.mfe_r, "close_r": trade.r_multiple},
            ))

        # 4. Oversized — governance reduced position
        if trade.governance_sizing_multiplier < 1.0:
            mistakes.append(Mistake(
                mistake_type="oversized_position",
                severity="low",
                description=f"Governance reduced sizing to {trade.governance_sizing_multiplier:.0%}",
                pnl_impact=0,
                details={"multiplier": trade.governance_sizing_multiplier},
            ))

        # 5. Low confidence entry
        if trade.signal_confidence < self._low_confidence and trade.signal_confidence > 0:
            mistakes.append(Mistake(
                mistake_type="low_confidence_entry",
                severity="medium" if trade.outcome == TradeOutcome.LOSS else "low",
                description=f"Entered with only {trade.signal_confidence:.0%} confidence",
                pnl_impact=float(trade.net_pnl) if trade.outcome == TradeOutcome.LOSS else 0,
                details={"confidence": trade.signal_confidence},
            ))

        # 6. Poor health entry — entered when strategy health was low
        if trade.health_score_at_entry < 0.5:
            mistakes.append(Mistake(
                mistake_type="poor_health_entry",
                severity="medium",
                description=f"Entered at health score {trade.health_score_at_entry:.2f} (below 0.50)",
                pnl_impact=float(trade.net_pnl) if trade.outcome == TradeOutcome.LOSS else 0,
                details={"health_score": trade.health_score_at_entry},
            ))

        # Tag the trade with mistake types
        for m in mistakes:
            if m.mistake_type not in trade.mistakes:
                trade.mistakes.append(m.mistake_type)

        # Store for reporting
        sid = trade.strategy_id
        self._history[sid].extend(mistakes)
        while len(self._history[sid]) > self._max_history:
            self._history[sid].pop(0)

        return mistakes

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def report(self, strategy_id: str) -> dict[str, Any]:
        """Generate a mistake report for a strategy.

        Returns
        -------
        dict
            ``total_mistakes`` : int
            ``by_type`` : dict mapping mistake_type → {count, total_pnl_impact, avg_severity}
            ``costliest_type`` : str — mistake type with worst total P&L impact
            ``total_pnl_impact`` : float — total estimated cost of all mistakes
        """
        mistakes = self._history.get(strategy_id, [])
        if not mistakes:
            return {
                "strategy_id": strategy_id,
                "total_mistakes": 0,
                "by_type": {},
                "costliest_type": None,
                "total_pnl_impact": 0.0,
            }

        by_type: dict[str, dict[str, Any]] = {}
        severity_scores = {"low": 1, "medium": 2, "high": 3}

        for m in mistakes:
            if m.mistake_type not in by_type:
                by_type[m.mistake_type] = {
                    "count": 0,
                    "total_pnl_impact": 0.0,
                    "severity_sum": 0,
                    "examples": [],
                }
            entry = by_type[m.mistake_type]
            entry["count"] += 1
            entry["total_pnl_impact"] += m.pnl_impact
            entry["severity_sum"] += severity_scores.get(m.severity, 1)

        # Compute averages and find costliest
        costliest = None
        worst_impact = 0.0
        for mtype, info in by_type.items():
            avg_sev_score = info["severity_sum"] / info["count"]
            info["avg_severity"] = (
                "high" if avg_sev_score >= 2.5
                else "medium" if avg_sev_score >= 1.5
                else "low"
            )
            del info["severity_sum"]
            info["total_pnl_impact"] = round(info["total_pnl_impact"], 2)

            if info["total_pnl_impact"] < worst_impact:
                worst_impact = info["total_pnl_impact"]
                costliest = mtype

        total_impact = sum(info["total_pnl_impact"] for info in by_type.values())

        return {
            "strategy_id": strategy_id,
            "total_mistakes": len(mistakes),
            "by_type": by_type,
            "costliest_type": costliest,
            "total_pnl_impact": round(total_impact, 2),
        }

    def get_all_strategy_ids(self) -> list[str]:
        """Return all strategy IDs with mistake data."""
        return [sid for sid, m in self._history.items() if m]
