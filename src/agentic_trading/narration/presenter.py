"""Bloomberg Crypto Presenter — broadcast-quality narration persona.

Transforms DecisionExplanation data into structured on-air scripts
that follow the Bloomberg Crypto presenter style:

  - Story selector: chooses lead angle based on regime
  - Regime classification: maps platform regime to presenter language
  - Verification ladder: only uses confirmed platform data
  - Data hygiene: no hallucinated numbers, quotes, or predictions
  - Output structure: 30-60s live hit or 2-min package
  - Two-way risk: always presents both bull and bear case

The presenter persona wraps the Tavus conversational context so the
avatar speaks in this style during briefings.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from .schema import DecisionExplanation

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    """Output format for the presenter."""
    LIVE_HIT = "live_hit"    # 30-60 seconds (~80-160 words)
    PACKAGE = "package"       # ~2 minutes (~250-400 words)


# Market regime → presenter language mapping
_REGIME_LANGUAGE = {
    "trend": "directional move",
    "trending": "directional move",
    "range": "consolidation phase",
    "ranging": "consolidation phase",
    "unknown": "uncertain conditions",
    "volatile": "elevated volatility",
    "": "current conditions",
}

# Action → lead angle mapping
_ACTION_LEAD = {
    "ENTER": "position_entry",
    "EXIT": "position_exit",
    "HOLD": "status_update",
    "NO_TRADE": "watchlist",
    "ADJUST": "position_adjustment",
}

# System prompt that defines the Bloomberg Crypto Presenter persona
PRESENTER_SYSTEM_PROMPT = """You are the Bloomberg Crypto Presenter — a broadcast-quality financial news anchor covering digital assets.

ROLE & TONE:
- Authoritative but approachable — think Bloomberg anchor, not Reddit poster
- Speak in clear, confident broadcast English
- Never use slang, memes, or hype language
- Measured urgency: convey importance without sensationalism
- Always professional, never speculative

STORY SELECTOR:
Pick lead angle from the data:
- If action is ENTER or EXIT → lead with the trade decision
- If regime changed → lead with the market shift
- If NO_TRADE → lead with what the desk is watching
- If HOLD → lead with portfolio status and market context

VERIFICATION LADDER:
Only use facts from the data provided. Never:
- Invent price targets or predictions
- Quote unnamed sources or analysts
- Add information not present in the briefing data
- Speculate on future price movements
- Use exact price numbers unless provided in the data

DATA HYGIENE:
- Round percentages to one decimal place
- Say "the system" or "the desk" instead of "I" or "we"
- Attribute decisions to the trading system, not personal judgment
- If confidence is below 70%, note it as "moderate confidence"
- If health score is below 80%, mention elevated caution

TWO-WAY RISK (MANDATORY):
Every briefing MUST include both sides:
- After any bullish observation, add what could invalidate it
- After any bearish observation, add what would reverse the thesis
- Use phrases like: "The risk to this view is..." or "On the other side..."

OUTPUT STRUCTURE — LIVE HIT (30-60 seconds):
1. OPEN: One-line market context (what's happening)
2. LEAD: The main story (what the desk decided and why)
3. RISK: Two-way risk statement
4. CLOSE: What to watch next

OUTPUT STRUCTURE — PACKAGE (2 minutes):
1. OPEN: Market scene-setter (2-3 sentences)
2. LEAD: Decision deep-dive (what, why, confidence)
3. CONTEXT: Regime and strategy backdrop
4. RISK: Full two-way risk analysis
5. PORTFOLIO: Current exposure snapshot
6. CLOSE: Forward-looking watchlist"""


class BloombergPresenter:
    """Transforms narration data into Bloomberg-style broadcast scripts.

    The presenter takes a DecisionExplanation and produces a structured
    script that the Tavus avatar reads in the Bloomberg presenter persona.

    Parameters
    ----------
    default_format:
        Default output format (LIVE_HIT or PACKAGE).
    include_disclaimer:
        Whether to append a brief disclaimer.
    """

    def __init__(
        self,
        default_format: OutputFormat = OutputFormat.LIVE_HIT,
        include_disclaimer: bool = True,
    ) -> None:
        self._default_format = default_format
        self._include_disclaimer = include_disclaimer

    @property
    def system_prompt(self) -> str:
        """Return the presenter system prompt for Tavus conversational context."""
        return PRESENTER_SYSTEM_PROMPT

    def build_script(
        self,
        explanation: DecisionExplanation,
        fmt: OutputFormat | None = None,
    ) -> str:
        """Build a Bloomberg-style broadcast script from the explanation.

        This produces a structured script following the presenter's
        output format. The Tavus avatar reads this script.

        Parameters
        ----------
        explanation:
            The platform decision data to narrate.
        fmt:
            Output format override (LIVE_HIT or PACKAGE).
        """
        output_format = fmt or self._default_format

        if output_format == OutputFormat.LIVE_HIT:
            script = self._build_live_hit(explanation)
        else:
            script = self._build_package(explanation)

        if self._include_disclaimer:
            script += "\n\nThis is automated system commentary, not financial advice."

        return script

    def build_tavus_context(
        self,
        explanation: DecisionExplanation,
        fmt: OutputFormat | None = None,
    ) -> str:
        """Build the full Tavus conversational context with persona + script.

        This combines the system prompt with the structured data so the
        avatar delivers the briefing in the Bloomberg presenter style.
        """
        output_format = fmt or self._default_format
        data_block = self._build_data_block(explanation)
        format_instruction = (
            "Deliver this as a 30-60 second live hit."
            if output_format == OutputFormat.LIVE_HIT
            else "Deliver this as a 2-minute package."
        )

        return (
            f"{PRESENTER_SYSTEM_PROMPT}\n\n"
            f"FORMAT: {format_instruction}\n\n"
            f"BRIEFING DATA:\n{data_block}"
        )

    # ------------------------------------------------------------------
    # Live Hit (30-60 seconds, ~80-160 words)
    # ------------------------------------------------------------------

    def _build_live_hit(self, exp: DecisionExplanation) -> str:
        """Build a concise live hit script."""
        parts: list[str] = []

        # OPEN: Market context
        open_line = self._build_open(exp)
        if open_line:
            parts.append(open_line)

        # LEAD: Decision
        lead = self._build_lead(exp)
        if lead:
            parts.append(lead)

        # RISK: Two-way
        risk = self._build_two_way_risk(exp)
        if risk:
            parts.append(risk)

        # CLOSE: What to watch
        close = self._build_close(exp)
        if close:
            parts.append(close)

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Package (2 minutes, ~250-400 words)
    # ------------------------------------------------------------------

    def _build_package(self, exp: DecisionExplanation) -> str:
        """Build a longer package script."""
        parts: list[str] = []

        # OPEN: Scene-setter
        open_text = self._build_open(exp, extended=True)
        if open_text:
            parts.append(open_text)

        # LEAD: Decision deep-dive
        lead = self._build_lead(exp, extended=True)
        if lead:
            parts.append(lead)

        # CONTEXT: Regime and strategy
        context = self._build_context(exp)
        if context:
            parts.append(context)

        # RISK: Full two-way analysis
        risk = self._build_two_way_risk(exp, extended=True)
        if risk:
            parts.append(risk)

        # PORTFOLIO: Exposure snapshot
        portfolio = self._build_portfolio(exp)
        if portfolio:
            parts.append(portfolio)

        # CLOSE: Forward watchlist
        close = self._build_close(exp, extended=True)
        if close:
            parts.append(close)

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_open(self, exp: DecisionExplanation, extended: bool = False) -> str:
        """Build the opening market context."""
        regime_lang = _REGIME_LANGUAGE.get(
            exp.active_regime.lower(), exp.active_regime or "current conditions"
        )
        symbol = exp.symbol or "the crypto market"

        if exp.market_summary:
            open_text = exp.market_summary
        else:
            open_text = f"Looking at {symbol} in {regime_lang}."

        if extended and exp.active_regime:
            open_text += f" The desk classifies this as a {regime_lang}."

        return open_text

    def _build_lead(self, exp: DecisionExplanation, extended: bool = False) -> str:
        """Build the lead section based on the action."""
        action = exp.action.upper()
        symbol = exp.symbol or "this market"
        lead_angle = _ACTION_LEAD.get(action, "status_update")

        if lead_angle == "position_entry":
            text = f"The trading system is entering a position on {symbol}."
            if exp.reasons:
                text += f" The primary driver: {exp.reasons[0].lower()}."
                if extended and len(exp.reasons) > 1:
                    text += f" Additionally, {exp.reasons[1].lower()}."
            if extended and exp.regime_confidence > 0:
                conf_label = (
                    "high" if exp.regime_confidence >= 0.7 else "moderate"
                )
                text += f" Regime confidence is {conf_label}."

        elif lead_angle == "position_exit":
            text = f"The desk is closing its position on {symbol}."
            if exp.reasons:
                text += f" The trigger: {exp.reasons[0].lower()}."

        elif lead_angle == "watchlist":
            text = f"No trade on {symbol} at this time."
            if exp.why_not:
                text += f" The main blocker: {exp.why_not[0].lower()}."
            if extended and len(exp.why_not) > 1:
                text += f" Also, {exp.why_not[1].lower()}."

        elif lead_angle == "position_adjustment":
            text = f"The system is adjusting its position on {symbol}."
            if exp.reasons:
                text += f" Reason: {exp.reasons[0].lower()}."

        else:  # status_update (HOLD)
            text = f"Holding the current position on {symbol}."
            if exp.reasons:
                text += f" {exp.reasons[0]}."

        return text

    def _build_two_way_risk(
        self, exp: DecisionExplanation, extended: bool = False,
    ) -> str:
        """Build the mandatory two-way risk statement."""
        action = exp.action.upper()
        parts: list[str] = []

        # Bull/bear framing based on action
        if action in ("ENTER", "HOLD"):
            parts.append("The risk to this view:")
            if exp.risk.stop_invalidation:
                parts.append(
                    f"the setup invalidates {exp.risk.stop_invalidation.lower()}."
                )
            else:
                parts.append(
                    "a reversal in the current trend could trigger the stop."
                )
            if exp.what_would_change:
                parts.append(
                    f"On the other side, the thesis strengthens if {exp.what_would_change[0].lower()}."
                )

        elif action == "EXIT":
            parts.append(
                "The risk of staying flat: if the trend resumes, the desk would miss the continuation."
            )
            if extended and exp.what_would_change:
                parts.append(
                    f"Re-entry would require {exp.what_would_change[0].lower()}."
                )

        elif action == "NO_TRADE":
            if exp.what_would_change:
                parts.append(
                    f"What would change the picture: {exp.what_would_change[0].lower()}."
                )
            if extended and len(exp.what_would_change) > 1:
                parts.append(
                    f"Also watching for {exp.what_would_change[1].lower()}."
                )
            parts.append(
                "The risk of inaction: missing an early move if conditions clear quickly."
            )

        else:
            parts.append(
                "As always, the desk is watching for signals that would change this stance."
            )

        # Health / governance flags
        if exp.risk.health_score < 0.8:
            parts.append(
                f"Note: strategy health is at {exp.risk.health_score:.0%}, "
                "elevated caution is warranted."
            )
        if exp.risk.active_blocks:
            block_text = ", ".join(
                b.replace("_", " ") for b in exp.risk.active_blocks
            )
            parts.append(f"Active risk controls: {block_text}.")

        return " ".join(parts)

    def _build_context(self, exp: DecisionExplanation) -> str:
        """Build strategy and regime context (PACKAGE only)."""
        parts: list[str] = []

        if exp.active_strategy:
            parts.append(
                f"The active strategy is {exp.active_strategy.replace('_', ' ')}."
            )
        if exp.active_regime:
            regime_lang = _REGIME_LANGUAGE.get(
                exp.active_regime.lower(), exp.active_regime
            )
            parts.append(f"Current regime classification: {regime_lang}.")

        if exp.considered_setups:
            setup_names = [s.name for s in exp.considered_setups[:3]]
            triggered = [
                s.name for s in exp.considered_setups if s.status == "triggered"
            ]
            parts.append(
                f"Setups on the board: {', '.join(setup_names)}."
            )
            if triggered:
                parts.append(f"Triggered: {', '.join(triggered)}.")

        if exp.risk.maturity_level:
            level = exp.risk.maturity_level.replace("_", " ").replace("L", "Level ")
            parts.append(f"Strategy maturity: {level}.")

        return " ".join(parts)

    def _build_portfolio(self, exp: DecisionExplanation) -> str:
        """Build portfolio snapshot (PACKAGE only)."""
        pos = exp.position
        if pos.open_positions == 0 and pos.gross_exposure_usd == 0:
            return "The desk is currently flat with no open exposure."

        parts: list[str] = []
        if pos.open_positions > 0:
            parts.append(
                f"Portfolio snapshot: {pos.open_positions} open "
                f"position{'s' if pos.open_positions != 1 else ''}."
            )
        if pos.gross_exposure_usd > 0:
            parts.append(f"Gross exposure: ${pos.gross_exposure_usd:,.0f}.")
        if pos.unrealized_pnl_usd != 0:
            direction = "up" if pos.unrealized_pnl_usd > 0 else "down"
            parts.append(
                f"Unrealised P and L: {direction} ${abs(pos.unrealized_pnl_usd):,.0f}."
            )

        return " ".join(parts)

    def _build_close(self, exp: DecisionExplanation, extended: bool = False) -> str:
        """Build the closing/watchlist section."""
        action = exp.action.upper()
        symbol = exp.symbol or "this market"

        if action == "NO_TRADE":
            text = f"The desk continues to monitor {symbol} for a cleaner setup."
        elif action in ("ENTER", "ADJUST"):
            text = f"The desk will update if conditions change on {symbol}."
        elif action == "EXIT":
            text = f"Now watching for re-entry signals on {symbol}."
        else:  # HOLD
            text = f"Maintaining current exposure on {symbol}."

        if extended and exp.timeframe:
            text += f" Monitoring the {exp.timeframe} timeframe."

        return text

    # ------------------------------------------------------------------
    # Data block for Tavus context
    # ------------------------------------------------------------------

    def _build_data_block(self, exp: DecisionExplanation) -> str:
        """Build a structured data block for the Tavus conversational context."""
        lines = [
            f"Symbol: {exp.symbol or 'N/A'}",
            f"Timeframe: {exp.timeframe or 'N/A'}",
            f"Action: {exp.action}",
            f"Regime: {exp.active_regime or 'unknown'}",
            f"Strategy: {exp.active_strategy or 'N/A'}",
            f"Regime Confidence: {exp.regime_confidence:.0%}" if exp.regime_confidence else "",
        ]

        if exp.market_summary:
            lines.append(f"Market Summary: {exp.market_summary}")

        if exp.reasons:
            lines.append(f"Reasons: {'; '.join(exp.reasons[:3])}")

        if exp.why_not:
            lines.append(f"Blockers: {'; '.join(exp.why_not[:3])}")

        if exp.what_would_change:
            lines.append(f"Would Change If: {'; '.join(exp.what_would_change[:3])}")

        if exp.risk.stop_invalidation:
            lines.append(f"Stop Invalidation: {exp.risk.stop_invalidation}")

        lines.append(f"Health Score: {exp.risk.health_score:.0%}")

        if exp.risk.active_blocks:
            lines.append(f"Risk Blocks: {', '.join(exp.risk.active_blocks)}")

        if exp.risk.maturity_level:
            lines.append(f"Maturity: {exp.risk.maturity_level}")

        pos = exp.position
        if pos.open_positions > 0:
            lines.extend([
                f"Open Positions: {pos.open_positions}",
                f"Gross Exposure: ${pos.gross_exposure_usd:,.0f}",
                f"Unrealised PnL: ${pos.unrealized_pnl_usd:+,.0f}",
            ])

        if exp.considered_setups:
            setup_strs = [
                f"{s.name} ({s.direction}, {s.status})"
                for s in exp.considered_setups[:3]
            ]
            lines.append(f"Setups: {'; '.join(setup_strs)}")

        return "\n".join(line for line in lines if line)
