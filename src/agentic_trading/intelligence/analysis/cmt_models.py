"""Pydantic models for CMT (Chartered Market Technician) analysis.

Defines the structured data models for the 9-layer CMT analytical
framework, confluence scoring, trade plans, and API request/response
schemas.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Target price must be positive, got {v}")
        return v

    @field_validator("pct")
    @classmethod
    def pct_must_be_valid(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError(f"Target pct must be 0-100, got {v}")
        return v


# Maximum stop distance as fraction of entry price.  An LLM-generated
# stop that is more than 50% away from entry is almost certainly a
# hallucination (e.g. stop=0 or stop=entry*10).
_MAX_STOP_DISTANCE_PCT = 0.50

# Maximum position size the LLM is allowed to suggest (% of equity).
# The governance/risk layer will further cap this, but we clamp early
# to prevent wildly over-leveraged suggestions from propagating.
_MAX_POSITION_SIZE_PCT = 5.0

# Maximum R:R the LLM can suggest.  Anything above 20:1 is unrealistic.
_MAX_RR_RATIO = 20.0


class CMTTradePlan(BaseModel):
    """Trade plan produced by the CMT analysis framework.

    All fields from LLM output are validated at parse time.  Invalid
    values raise ``ValidationError``, which the caller (CMTAnalysisEngine)
    catches and treats as a no-trade response.
    """

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

    @field_validator("direction")
    @classmethod
    def direction_must_be_valid(cls, v: str) -> str:
        if v.upper() not in ("LONG", "SHORT"):
            raise ValueError(f"direction must be LONG or SHORT, got '{v}'")
        return v.upper()

    @field_validator("entry_price")
    @classmethod
    def entry_price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"entry_price must be positive, got {v}")
        return v

    @field_validator("stop_loss")
    @classmethod
    def stop_loss_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"stop_loss must be positive, got {v}")
        return v

    @field_validator("rr_ratio")
    @classmethod
    def rr_ratio_must_be_reasonable(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"rr_ratio must be non-negative, got {v}")
        if v > _MAX_RR_RATIO:
            raise ValueError(
                f"rr_ratio {v} exceeds maximum {_MAX_RR_RATIO} — likely hallucinated"
            )
        return v

    @field_validator("position_size_pct")
    @classmethod
    def position_size_must_be_bounded(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"position_size_pct must be non-negative, got {v}")
        if v > _MAX_POSITION_SIZE_PCT:
            raise ValueError(
                f"position_size_pct {v}% exceeds max {_MAX_POSITION_SIZE_PCT}%"
            )
        return v

    @model_validator(mode="after")
    def stop_on_correct_side(self) -> CMTTradePlan:
        """Verify stop_loss is on the correct side of entry_price.

        For LONG trades the stop must be below entry; for SHORT trades
        the stop must be above entry.  Also reject stops that are
        unreasonably far from entry (> ``_MAX_STOP_DISTANCE_PCT``).
        """
        if self.direction == "LONG" and self.stop_loss >= self.entry_price:
            raise ValueError(
                f"LONG stop_loss ({self.stop_loss}) must be below "
                f"entry_price ({self.entry_price})"
            )
        if self.direction == "SHORT" and self.stop_loss <= self.entry_price:
            raise ValueError(
                f"SHORT stop_loss ({self.stop_loss}) must be above "
                f"entry_price ({self.entry_price})"
            )

        # Check stop distance is reasonable
        distance_pct = abs(self.stop_loss - self.entry_price) / self.entry_price
        if distance_pct > _MAX_STOP_DISTANCE_PCT:
            raise ValueError(
                f"Stop distance {distance_pct:.1%} exceeds maximum "
                f"{_MAX_STOP_DISTANCE_PCT:.0%} of entry price — "
                f"likely hallucinated (entry={self.entry_price}, "
                f"stop={self.stop_loss})"
            )

        return self


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
