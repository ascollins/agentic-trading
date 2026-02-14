"""Narration data schemas.

``DecisionExplanation`` is the single source of truth for all narration.
The script generator may only use fields from this schema — no invented reasons.

``NarrationItem`` is the stored output: script text + metadata for both channels.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Source of Truth: DecisionExplanation
# ---------------------------------------------------------------------------

class ConsideredSetup(BaseModel):
    """A setup the strategy considered (max 3 per explanation)."""
    name: str
    direction: str = ""  # "long" / "short" / "flat"
    confidence: float = 0.0
    status: str = ""  # "triggered" / "watching" / "invalidated"


class RiskSummary(BaseModel):
    """Risk context snapshot."""
    intended_size_pct: float = 0.0
    stop_invalidation: str = ""
    active_blocks: list[str] = Field(default_factory=list)
    governance_action: str = ""
    health_score: float = 1.0
    maturity_level: str = ""


class PositionSnapshot(BaseModel):
    """Current exposure snapshot."""
    open_positions: int = 0
    gross_exposure_usd: float = 0.0
    net_exposure_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    available_balance_usd: float = 0.0


class DecisionExplanation(BaseModel):
    """Single source of truth for all narration content.

    Every field here is populated from actual platform state.
    The narration service may ONLY use these fields to generate scripts.
    """
    timestamp: datetime = Field(default_factory=_now)
    symbol: str = ""
    timeframe: str = ""

    # Market context (plain English, no jargon)
    market_summary: str = ""

    # Strategy / regime
    active_strategy: str = ""
    active_regime: str = ""  # "trending" / "ranging" / "unknown"
    regime_confidence: float = 0.0

    # Setups considered
    considered_setups: list[ConsideredSetup] = Field(default_factory=list)

    # Decision
    action: str = ""  # ENTER / EXIT / HOLD / NO_TRADE / ADJUST
    reasons: list[str] = Field(default_factory=list)  # max 3
    reason_confidences: list[float] = Field(default_factory=list)

    # NO_TRADE specifics
    why_not: list[str] = Field(default_factory=list)  # max 3
    what_would_change: list[str] = Field(default_factory=list)  # max 3

    # Risk
    risk: RiskSummary = Field(default_factory=RiskSummary)

    # Position / exposure
    position: PositionSnapshot = Field(default_factory=PositionSnapshot)

    # Trace
    trace_id: str = ""
    signal_id: str = ""

    def content_hash(self) -> str:
        """Stable hash for deduplication (ignores timestamp)."""
        payload = {
            "symbol": self.symbol,
            "action": self.action,
            "active_strategy": self.active_strategy,
            "active_regime": self.active_regime,
            "reasons": self.reasons,
            "why_not": self.why_not,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Output: NarrationItem
# ---------------------------------------------------------------------------

class NarrationItem(BaseModel):
    """Stored narration output shared by both text and avatar channels."""

    script_id: str = ""  # Stable hash for caching / dedupe
    timestamp: datetime = Field(default_factory=_now)
    script_text: str = ""
    verbosity: str = "normal"  # quiet / normal / detailed

    # Provenance
    decision_ref: str = ""  # trace_id linking to DecisionExplanation
    sources: list[str] = Field(default_factory=list)  # fields used

    # Avatar
    playback_url: str = ""
    tavus_session_id: str = ""

    # Channel flags
    published_text: bool = False
    published_avatar: bool = False

    # Extra
    metadata: dict[str, Any] = Field(default_factory=dict)
