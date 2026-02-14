"""NarrationService — converts DecisionExplanation into plain-English scripts.

Responsibilities:
  - Generate desk-style narration from platform state
  - Enforce verbosity limits (QUIET <= 30w, NORMAL <= 70w, DETAILED <= 130w)
  - Ban technical jargon (RSI, MACD, ATR, Sharpe, etc.)
  - Deduplicate: only narrate meaningful changes
  - Throttle: configurable heartbeat interval for quiet periods
"""

from __future__ import annotations

import hashlib
import logging
import time
from enum import Enum
from typing import Any

from .schema import DecisionExplanation, NarrationItem

logger = logging.getLogger(__name__)


class Verbosity(str, Enum):
    QUIET = "quiet"        # <= 30 words
    NORMAL = "normal"      # <= 70 words
    DETAILED = "detailed"  # <= 130 words


# Jargon that must never appear in narration output
BANNED_JARGON = frozenset({
    "rsi", "macd", "atr", "sharpe", "z-score", "zscore",
    "orderbook imbalance", "order book imbalance", "bollinger",
    "stochastic", "ichimoku", "fibonacci", "ema crossover",
    "adx", "obv", "vwap", "keltner", "donchian",
})

# Word limits per verbosity level
WORD_LIMITS = {
    Verbosity.QUIET: 30,
    Verbosity.NORMAL: 70,
    Verbosity.DETAILED: 130,
}

# Actions that trigger narration (not just heartbeats)
MEANINGFUL_ACTIONS = {"ENTER", "EXIT", "ADJUST"}


class NarrationService:
    """Generates plain-English narration scripts from platform decisions.

    Parameters
    ----------
    verbosity:
        Default verbosity level for generated scripts.
    heartbeat_seconds:
        Interval between heartbeat narrations when nothing meaningful changes.
    dedupe_window_seconds:
        Minimum time between identical narrations (by content hash).
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        heartbeat_seconds: float = 60.0,
        dedupe_window_seconds: float = 30.0,
    ) -> None:
        self._verbosity = verbosity
        self._heartbeat_seconds = heartbeat_seconds
        self._dedupe_window_seconds = dedupe_window_seconds

        # Deduplication state
        self._last_hash: str = ""
        self._last_hash_time: float = 0.0
        self._last_narration_time: float = 0.0
        self._narration_count: int = 0

    @property
    def narration_count(self) -> int:
        return self._narration_count

    def generate(
        self,
        explanation: DecisionExplanation,
        verbosity: Verbosity | None = None,
        force: bool = False,
    ) -> NarrationItem | None:
        """Generate a narration script from a DecisionExplanation.

        Returns ``None`` if the narration is deduplicated or throttled.
        Set ``force=True`` to bypass deduplication.
        """
        v = verbosity or self._verbosity
        now = time.monotonic()

        # Dedupe: skip if same content hash within window
        content_hash = explanation.content_hash()
        if not force:
            if (
                content_hash == self._last_hash
                and (now - self._last_hash_time) < self._dedupe_window_seconds
            ):
                return None

        # Throttle: for non-meaningful actions, enforce heartbeat interval
        is_meaningful = (
            explanation.action.upper() in MEANINGFUL_ACTIONS
            or force
        )
        if not is_meaningful and not force:
            if (now - self._last_narration_time) < self._heartbeat_seconds:
                return None

        # Build the script
        script, sources = self._build_script(explanation, v)

        # Enforce word limit
        script = self._enforce_word_limit(script, WORD_LIMITS[v])

        # Enforce jargon ban
        script = self._scrub_jargon(script)

        # Build stable script_id
        script_id = hashlib.sha256(
            f"{content_hash}:{explanation.timestamp.isoformat()}".encode()
        ).hexdigest()[:16]

        item = NarrationItem(
            script_id=script_id,
            timestamp=explanation.timestamp,
            script_text=script,
            verbosity=v.value,
            decision_ref=explanation.trace_id,
            sources=sources,
            published_text=True,
        )

        # Update state
        self._last_hash = content_hash
        self._last_hash_time = now
        self._last_narration_time = now
        self._narration_count += 1

        logger.debug(
            "Narration generated: id=%s action=%s words=%d",
            script_id, explanation.action, len(script.split()),
        )
        return item

    # ------------------------------------------------------------------
    # Script construction
    # ------------------------------------------------------------------

    def _build_script(
        self, exp: DecisionExplanation, verbosity: Verbosity
    ) -> tuple[str, list[str]]:
        """Build a narration script from the explanation fields.

        Returns (script_text, list_of_source_fields_used).
        """
        parts: list[str] = []
        sources: list[str] = []

        # 1. What's happening? (Market context)
        if exp.market_summary:
            parts.append(exp.market_summary)
            sources.append("market_summary")
        elif exp.symbol:
            parts.append(f"Watching {exp.symbol}.")
            sources.append("symbol")

        # 2. What are we watching? (Strategy + regime)
        if exp.active_strategy or exp.active_regime:
            regime_desc = self._regime_plain(exp.active_regime)
            if exp.active_strategy and regime_desc:
                parts.append(
                    f"The {exp.active_strategy} strategy is active in a {regime_desc} market."
                )
            elif exp.active_strategy:
                parts.append(f"The {exp.active_strategy} strategy is active.")
            elif regime_desc:
                parts.append(f"Market is {regime_desc}.")
            sources.extend(["active_strategy", "active_regime"])

        # 3. Considered setups (DETAILED only)
        if verbosity == Verbosity.DETAILED and exp.considered_setups:
            setup_names = [s.name for s in exp.considered_setups[:3]]
            parts.append(f"Setups considered: {', '.join(setup_names)}.")
            sources.append("considered_setups")

        # 4. What did we decide?
        action_text = self._action_plain(exp)
        if action_text:
            parts.append(action_text)
            sources.append("action")

        # 5. Why?
        if exp.reasons:
            reason_text = self._reasons_plain(exp.reasons[:3])
            parts.append(reason_text)
            sources.append("reasons")

        # 6. If NO_TRADE — why not + what would change
        if exp.action.upper() == "NO_TRADE":
            if exp.why_not:
                blocker = exp.why_not[0]
                parts.append(f"Top blocker: {blocker}.")
                sources.append("why_not")
            if exp.what_would_change and verbosity != Verbosity.QUIET:
                changes = exp.what_would_change[:2]
                parts.append(
                    f"Would reconsider if: {'; '.join(changes)}."
                )
                sources.append("what_would_change")

        # 7. Risk controls (NORMAL + DETAILED)
        if verbosity != Verbosity.QUIET:
            risk_text = self._risk_plain(exp)
            if risk_text:
                parts.append(risk_text)
                sources.append("risk")

        # 8. Position snapshot (DETAILED only)
        if verbosity == Verbosity.DETAILED and exp.position.open_positions > 0:
            pos = exp.position
            parts.append(
                f"Currently holding {pos.open_positions} "
                f"position{'s' if pos.open_positions != 1 else ''}, "
                f"unrealised P&L ${pos.unrealized_pnl_usd:+,.0f}."
            )
            sources.append("position")

        script = " ".join(parts)
        return script, sources

    # ------------------------------------------------------------------
    # Plain-English translators
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_plain(regime: str) -> str:
        """Translate regime to plain English."""
        mapping = {
            "trend": "trending",
            "trending": "trending",
            "range": "sideways",
            "ranging": "sideways",
            "unknown": "uncertain",
            "": "",
        }
        return mapping.get(regime.lower(), regime.lower())

    @staticmethod
    def _action_plain(exp: DecisionExplanation) -> str:
        """Translate action to plain English."""
        action = exp.action.upper()
        sym = exp.symbol or "the market"
        mapping = {
            "ENTER": f"Decided to enter a position on {sym}.",
            "EXIT": f"Decided to close the position on {sym}.",
            "HOLD": f"Holding the current position on {sym}.",
            "NO_TRADE": f"No trade on {sym} right now.",
            "ADJUST": f"Adjusting the position on {sym}.",
        }
        return mapping.get(action, f"Action: {action} on {sym}.")

    @staticmethod
    def _reasons_plain(reasons: list[str]) -> str:
        if len(reasons) == 1:
            return f"Reason: {reasons[0]}."
        return "Reasons: " + "; ".join(reasons) + "."

    @staticmethod
    def _risk_plain(exp: DecisionExplanation) -> str:
        parts = []
        if exp.risk.active_blocks:
            parts.append(f"Risk blocks active: {', '.join(exp.risk.active_blocks)}")
        if exp.risk.governance_action and exp.risk.governance_action not in ("allow", ""):
            parts.append(f"Governance: {exp.risk.governance_action}")
        if exp.risk.health_score < 0.8:
            parts.append(f"Strategy health at {exp.risk.health_score:.0%}")
        if not parts:
            return ""
        return " ".join(parts) + "."

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_word_limit(text: str, max_words: int) -> str:
        """Truncate to word limit, preserving sentence endings where possible."""
        words = text.split()
        if len(words) <= max_words:
            return text
        truncated = " ".join(words[:max_words])
        # Try to end at a sentence boundary
        last_period = truncated.rfind(".")
        if last_period > len(truncated) * 0.5:
            return truncated[: last_period + 1]
        return truncated + "..."

    @staticmethod
    def _scrub_jargon(text: str) -> str:
        """Remove banned jargon terms from the narration."""
        lower = text.lower()
        for term in BANNED_JARGON:
            if term in lower:
                # Replace the jargon (case-insensitive) with a generic substitute
                import re
                text = re.sub(
                    re.escape(term),
                    "technical signal",
                    text,
                    flags=re.IGNORECASE,
                )
        return text

    @staticmethod
    def check_jargon(text: str) -> list[str]:
        """Return list of banned jargon found in text (for testing)."""
        lower = text.lower()
        return [term for term in BANNED_JARGON if term in lower]
