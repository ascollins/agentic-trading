"""Efficacy analysis data models.

Defines the structures used to diagnose loss drivers and segment
trade performance for the Optimizer Efficacy Agent.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class LossDriverCategory(str, enum.Enum):
    """Categories of loss drivers, ordered by diagnostic priority.

    The investigation order follows the user's mandate:
    costs/execution first, then exit geometry, regime mismatch,
    risk/sizing, and signal edge last.
    """

    EXECUTION_COST = "execution_cost"
    EXIT_GEOMETRY = "exit_geometry"
    REGIME_MISMATCH = "regime_mismatch"
    RISK_SIZING = "risk_sizing"
    SIGNAL_EDGE = "signal_edge"


# Priority order for investigation (costs first, signal last)
DRIVER_PRIORITY: list[LossDriverCategory] = [
    LossDriverCategory.EXECUTION_COST,
    LossDriverCategory.EXIT_GEOMETRY,
    LossDriverCategory.REGIME_MISMATCH,
    LossDriverCategory.RISK_SIZING,
    LossDriverCategory.SIGNAL_EDGE,
]


@dataclass
class LossDriverBreakdown:
    """Quantified breakdown of a single loss driver category."""

    category: LossDriverCategory
    trade_count: int = 0
    avg_loss_pct: float = 0.0
    total_pnl_impact: float = 0.0        # Sum of returns attributed to this driver
    share_of_total_losses: float = 0.0    # What % of total losses this explains
    examples: list[dict[str, Any]] = field(default_factory=list)  # Sample trades
    diagnosis: str = ""                   # Human-readable explanation
    severity: str = ""                    # "critical", "warning", "info"


@dataclass
class SegmentAnalysis:
    """Performance analysis for a single segment (symbol, direction, etc.)."""

    segment_name: str = ""
    segment_type: str = ""           # "symbol", "direction", "strategy", "exit_reason"
    trade_count: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    total_return: float = 0.0
    profit_factor: float = 0.0
    avg_hold_seconds: float = 0.0
    avg_mae_pct: float = 0.0        # Avg maximum adverse excursion
    avg_mfe_pct: float = 0.0        # Avg maximum favorable excursion
    management_efficiency: float = 0.0  # avg(actual_return / mfe) for winners
    fee_drag_pct: float = 0.0       # avg(fee / notional)
    stop_hit_rate: float = 0.0      # Fraction of trades closed by stop-loss


@dataclass
class DataIntegrityResult:
    """Result of Phase 0 data integrity checks."""

    passed: bool = True
    total_trades: int = 0
    issues: list[str] = field(default_factory=list)
    has_prices: bool = True
    has_fees: bool = True
    has_mae_mfe: bool = True
    has_timestamps: bool = True


@dataclass
class EfficacyReport:
    """Complete efficacy analysis report.

    Contains the full diagnostic output: data integrity result,
    per-segment breakdown, loss driver analysis, and prioritised
    recommendations.
    """

    # Metadata
    timestamp: str = ""
    strategy_id: str = ""
    data_window: str = ""

    # Phase 0: data integrity
    data_integrity: DataIntegrityResult = field(
        default_factory=DataIntegrityResult
    )

    # Aggregate stats
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_return: float = 0.0
    total_return: float = 0.0

    # Phase 1: segmentation
    segments: dict[str, SegmentAnalysis] = field(default_factory=dict)

    # Phase 1: loss driver diagnosis (sorted by impact)
    loss_drivers: list[LossDriverBreakdown] = field(default_factory=list)

    # Phase 2: prioritised recommendations
    recommendations: list[str] = field(default_factory=list)

    # Flags
    min_trades_met: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Export to a flat dictionary for JSON serialisation."""
        return {
            "timestamp": self.timestamp,
            "strategy_id": self.strategy_id,
            "data_window": self.data_window,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_return": self.avg_return,
            "total_return": self.total_return,
            "min_trades_met": self.min_trades_met,
            "data_integrity": {
                "passed": self.data_integrity.passed,
                "total_trades": self.data_integrity.total_trades,
                "issues": self.data_integrity.issues,
            },
            "loss_drivers": [
                {
                    "category": d.category.value,
                    "trade_count": d.trade_count,
                    "avg_loss_pct": round(d.avg_loss_pct, 6),
                    "total_pnl_impact": round(d.total_pnl_impact, 6),
                    "share_of_total_losses": round(d.share_of_total_losses, 4),
                    "severity": d.severity,
                    "diagnosis": d.diagnosis,
                }
                for d in self.loss_drivers
            ],
            "segments": {
                k: {
                    "segment_type": v.segment_type,
                    "trade_count": v.trade_count,
                    "win_rate": round(v.win_rate, 4),
                    "avg_return": round(v.avg_return, 6),
                    "profit_factor": round(v.profit_factor, 4),
                    "avg_hold_seconds": round(v.avg_hold_seconds, 1),
                    "avg_mae_pct": round(v.avg_mae_pct, 6),
                    "avg_mfe_pct": round(v.avg_mfe_pct, 6),
                    "management_efficiency": round(v.management_efficiency, 4),
                    "fee_drag_pct": round(v.fee_drag_pct, 6),
                    "stop_hit_rate": round(v.stop_hit_rate, 4),
                }
                for k, v in self.segments.items()
            },
            "recommendations": self.recommendations,
        }
