"""Structured trade plan model.

Captures a complete trade plan including entry zone, stop loss, targets,
R:R analysis, macro context, and conviction level.  Bridges to the
existing :class:`~agentic_trading.core.events.Signal` pipeline via
:meth:`TradePlan.to_signal`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.enums import (
    ConvictionLevel,
    MarketStructureBias,
    SetupGrade,
    SignalDirection,
    Timeframe,
)


class EntryZone(BaseModel):
    """Entry price zone with optional laddered levels."""

    primary_entry: float
    entry_low: float | None = None
    entry_high: float | None = None
    scaled_entries: list[tuple[float, float]] = Field(
        default_factory=list,
        description="List of (price, allocation_pct) for laddered entries",
    )


class TargetSpec(BaseModel):
    """Take-profit target specification."""

    price: float
    rr_ratio: float = 0.0
    scale_out_pct: float = 0.0
    rationale: str = ""


class TradePlan(BaseModel):
    """Complete structured trade plan.

    Captures all aspects of a trade plan following the technical analyst
    framework: multi-timeframe context, entry/exit levels, risk management,
    and conviction assessment.

    Usage::

        plan = TradePlan(
            symbol="BTC/USDT",
            direction=SignalDirection.LONG,
            entry=EntryZone(primary_entry=95000),
            stop_loss=92000,
            targets=[TargetSpec(price=98000, rr_ratio=1.0, scale_out_pct=0.4)],
            ...
        )
        signal_kwargs = plan.to_signal()
    """

    # Identity
    plan_id: str = ""
    strategy_id: str = ""
    symbol: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Direction & conviction
    direction: SignalDirection
    conviction: ConvictionLevel = ConvictionLevel.MODERATE
    setup_grade: SetupGrade = SetupGrade.C
    confidence: float = 0.0  # 0.0–1.0, maps to Signal.confidence

    # Multi-timeframe context
    htf_bias: MarketStructureBias = MarketStructureBias.UNCLEAR
    htf_timeframe: Timeframe = Timeframe.D1
    trade_timeframe: Timeframe = Timeframe.H1
    ltf_trigger_timeframe: Timeframe = Timeframe.M15
    structure_notes: str = ""

    # Entry
    entry: EntryZone

    # Risk management
    stop_loss: float
    risk_pct: float = 0.01  # Default 1% risk
    invalidation_notes: str = ""

    # Targets
    targets: list[TargetSpec] = Field(default_factory=list)

    # R:R summary
    blended_rr: float = 0.0
    expected_r: float = 0.0

    # Macro / contextual
    macro_context: str = ""
    key_levels: dict[str, float] = Field(
        default_factory=dict,
        description="Key S/R levels, e.g. {'weekly_support': 64000}",
    )
    catalysts: list[str] = Field(default_factory=list)

    # Technical indicators snapshot
    indicators_snapshot: dict[str, float] = Field(default_factory=dict)

    # Rationale
    rationale: str = ""
    edge_description: str = ""

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_signal_risk_constraints(self) -> dict[str, Any]:
        """Convert trade plan to ``risk_constraints`` dict for Signal events.

        This allows a :class:`TradePlan` to be consumed by the existing
        ``PortfolioManager`` sizing pipeline which dispatches on
        ``risk_constraints["sizing_method"]``.
        """
        constraints: dict[str, Any] = {
            "sizing_method": "stop_loss_based",
            "entry": self.entry.primary_entry,
            "stop_loss": self.stop_loss,
            "risk_pct": self.risk_pct,
            "blended_rr": self.blended_rr,
            "setup_grade": self.setup_grade.value,
            "conviction": self.conviction.value,
        }
        if self.targets:
            constraints["targets"] = [t.price for t in self.targets]
            constraints["scale_out_pcts"] = [
                t.scale_out_pct for t in self.targets
            ]
        if self.entry.scaled_entries:
            constraints["sizing_method"] = "scaled_entry"
            constraints["scaled_entries"] = self.entry.scaled_entries
        return constraints

    def to_signal(self) -> dict[str, Any]:
        """Return kwargs suitable for constructing a Signal event.

        Example::

            from agentic_trading.core.events import Signal
            signal = Signal(**plan.to_signal())
        """
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "timeframe": self.trade_timeframe,
            "risk_constraints": self.to_signal_risk_constraints(),
            "features_used": dict(self.indicators_snapshot),
        }
