"""Pydantic models for CMT (Chartered Market Technician) analysis.

Defines the structured data models for the 9-layer CMT analytical
framework, confluence scoring, trade plans, and API request/response
schemas.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Layer results
# ---------------------------------------------------------------------------


class CMTLayerResult(BaseModel):
    """Result from a single CMT analytical layer."""

    layer: int  # 1-9
    name: str  # e.g. "Trend Identification"
    direction: str = ""  # bullish / bearish / neutral
    confidence: str = ""  # high / medium / low
    score: float = 0.0  # Layer-specific score contribution
    key_findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Confluence scoring
# ---------------------------------------------------------------------------


class CMTConfluenceScore(BaseModel):
    """7-dimension CMT confluence scoring per spec Section 2.2.2."""

    trend_alignment: float = 0.0  # -2 to +2
    key_level_proximity: float = 0.0  # 0 to 2
    pattern_signal: float = 0.0  # -2 to +2
    indicator_consensus: float = 0.0  # -2 to +2
    sentiment_alignment: float = 0.0  # -1 to +1
    volatility_regime: float = 0.0  # -1 to +1
    macro_alignment: float = 0.0  # -1 to +1
    total: float = 0.0  # Sum: -10 to +11
    threshold_met: bool = False  # total >= min_confluence_score
    veto: bool = False  # Any single layer scoring -2

    def compute_total(self) -> float:
        """Recalculate total from dimensions."""
        self.total = (
            self.trend_alignment
            + self.key_level_proximity
            + self.pattern_signal
            + self.indicator_consensus
            + self.sentiment_alignment
            + self.volatility_regime
            + self.macro_alignment
        )
        return self.total

    def check_veto(self) -> bool:
        """Check if any dimension triggers a veto (-2)."""
        self.veto = any(
            v <= -2.0
            for v in [
                self.trend_alignment,
                self.pattern_signal,
                self.indicator_consensus,
            ]
        )
        return self.veto


# ---------------------------------------------------------------------------
# Trade plan
# ---------------------------------------------------------------------------


class CMTTarget(BaseModel):
    """Single price target with scale-out percentage."""

    price: float
    pct: float  # Scale-out percentage (e.g. 50.0 for 50%)
    source: str = ""  # e.g. "measured_move", "fib_extension", "sr_level"


class CMTTradePlan(BaseModel):
    """Trade plan produced by the CMT analysis framework."""

    direction: str  # LONG / SHORT
    entry_price: float
    entry_trigger: str = ""  # Description of what triggers entry
    stop_loss: float
    stop_reasoning: str = ""  # Why stop is at this level
    targets: list[CMTTarget] = Field(default_factory=list)
    rr_ratio: float = 0.0  # Risk:reward ratio (first target)
    blended_rr: float = 0.0  # Weighted R:R across all targets
    position_size_pct: float = 0.0  # Suggested risk % of equity
    invalidation: str = ""  # What kills the thesis
    thesis: str = ""  # Human-readable trade thesis


# ---------------------------------------------------------------------------
# Assessment request / response
# ---------------------------------------------------------------------------


class CMTAssessmentRequest(BaseModel):
    """Input data for a CMT analysis cycle."""

    symbol: str
    timeframes: list[str] = Field(default_factory=list)
    ohlcv_summary: dict[str, Any] = Field(default_factory=dict)
    indicator_values: dict[str, float] = Field(default_factory=dict)
    htf_assessment: dict[str, Any] = Field(default_factory=dict)
    smc_confluence: dict[str, Any] = Field(default_factory=dict)
    regime_state: dict[str, Any] = Field(default_factory=dict)
    portfolio_state: dict[str, Any] = Field(default_factory=dict)
    performance_metrics: dict[str, Any] = Field(default_factory=dict)


class CMTAssessmentResponse(BaseModel):
    """Structured output from the CMT analysis engine."""

    symbol: str
    timeframes_analyzed: list[str] = Field(default_factory=list)
    layers: list[CMTLayerResult] = Field(default_factory=list)
    confluence: CMTConfluenceScore = Field(default_factory=CMTConfluenceScore)
    trade_plan: CMTTradePlan | None = None
    thesis: str = ""
    system_health: str = "green"  # green / amber / red
    watchlist_action: str = ""  # "enter" / "monitor" / "no_action"
    no_trade_reason: str = ""

    def layer_dict(self) -> dict[str, Any]:
        """Serialize layers as a dict keyed by layer number."""
        return {
            f"layer_{lr.layer}_{lr.name.lower().replace(' ', '_')}": lr.model_dump()
            for lr in self.layers
        }
