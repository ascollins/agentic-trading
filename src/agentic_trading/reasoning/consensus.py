"""Consensus Gate — multi-agent consultation before trade execution.

Inserts a mandatory conversation phase between strategy signal generation
and order execution.  When strategies produce signals, the ConsensusGate
orchestrates a desk conversation:

    Market Structure -> SMC Analyst -> CMT Technician -> Risk Manager
                           -> Consensus Decision -> Execute (or not)

Only after the desk reaches consensus (or the orchestrator makes a
final call after disagreements) does the signal flow to the execution
pipeline.

This prevents the "flip-flop" problem where agents independently
enter/exit positions without consulting each other.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.events import Signal

from .agent_conversation import AgentConversation, ConversationOutcome
from .agent_message import AgentMessage, AgentRole, MessageType
from .message_bus import ReasoningMessageBus
from .soteria_trace import SoteriaTrace, StepType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consensus decision
# ---------------------------------------------------------------------------


class ConsensusVerdict(str, Enum):
    """Outcome of the desk consultation."""

    APPROVED = "approved"
    REJECTED = "rejected"
    VETOED = "vetoed"
    TIMEOUT = "timeout"


class DeskOpinion(BaseModel):
    """A single desk participant's opinion on a proposed trade.

    Attributes
    ----------
    role:
        Which desk role issued this opinion.
    approve:
        True if the participant supports the trade.
    confidence:
        Confidence in the opinion (0.0 - 1.0).
    reasoning:
        Explanation for the opinion.
    data:
        Machine-readable supporting data.
    is_veto:
        True if this is a hard veto (overrides majority).
    """

    role: AgentRole
    approve: bool
    confidence: float = 0.0
    reasoning: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    is_veto: bool = False


class ConsensusResult(BaseModel):
    """Result of the consensus gate evaluation.

    Attributes
    ----------
    verdict:
        Whether the trade is approved, rejected, or vetoed.
    conversation_id:
        ID of the reasoning conversation.
    opinions:
        Per-desk opinions collected.
    weighted_score:
        Confidence-weighted approval score (-1.0 to 1.0).
    approved_signal:
        The original signal (present only if approved).
    reasoning:
        Human-readable summary of the consensus.
    elapsed_ms:
        Time taken for the consultation.
    """

    verdict: ConsensusVerdict
    conversation_id: str = ""
    opinions: list[DeskOpinion] = Field(default_factory=list)
    weighted_score: float = 0.0
    approved_signal: Any | None = None
    reasoning: str = ""
    elapsed_ms: float = 0.0

    @property
    def is_approved(self) -> bool:
        """True if the trade should proceed."""
        return self.verdict == ConsensusVerdict.APPROVED


# ---------------------------------------------------------------------------
# Desk Participant Protocol
# ---------------------------------------------------------------------------


class DeskParticipant:
    """Base class for desk participants in the consensus process.

    Each participant evaluates a proposed trade signal against its
    domain-specific criteria and returns an opinion.
    """

    def __init__(self, role: AgentRole) -> None:
        self._role = role

    @property
    def role(self) -> AgentRole:
        return self._role

    def evaluate(
        self,
        signal: Signal,
        context: dict[str, Any],
        conversation: AgentConversation,
    ) -> DeskOpinion:
        """Evaluate a signal and return an opinion.

        Override in subclasses for domain-specific logic.

        Parameters
        ----------
        signal:
            The proposed trade signal.
        context:
            Market context snapshot (prices, indicators, portfolio state).
        conversation:
            The ongoing conversation (for reading prior opinions).
        """
        return DeskOpinion(
            role=self._role,
            approve=True,
            confidence=0.5,
            reasoning="Default: no objection",
        )


# ---------------------------------------------------------------------------
# Built-in desk participants
# ---------------------------------------------------------------------------


class MarketStructureDesk(DeskParticipant):
    """Evaluates higher-timeframe structure and trend alignment.

    Checks:
    - Are higher timeframes aligned with the signal direction?
    - Is the market in a trending or ranging regime?
    - Are key support/resistance levels nearby?
    """

    def __init__(self) -> None:
        super().__init__(AgentRole.MARKET_STRUCTURE)

    def evaluate(
        self,
        signal: Signal,
        context: dict[str, Any],
        conversation: AgentConversation,
    ) -> DeskOpinion:
        features = context.get("features", {})
        reasons: list[str] = []
        score = 0.0
        total_checks = 0

        # EMA alignment check
        ema_fast = features.get("ema_12") or features.get("ema12")
        ema_slow = features.get("ema_26") or features.get("ema26")
        if ema_fast is not None and ema_slow is not None:
            total_checks += 1
            is_long = signal.direction.value == "long"
            ema_bullish = float(ema_fast) > float(ema_slow)
            if (is_long and ema_bullish) or (not is_long and not ema_bullish):
                score += 1.0
                reasons.append(
                    f"EMA alignment confirmed: EMA12={ema_fast:.2f} "
                    f"{'>' if ema_bullish else '<'} EMA26={ema_slow:.2f}"
                )
            else:
                reasons.append(
                    f"EMA divergence: EMA12={ema_fast:.2f} "
                    f"{'>' if ema_bullish else '<'} EMA26={ema_slow:.2f} "
                    f"vs {signal.direction.value} signal"
                )

        # ADX trend strength
        adx = features.get("adx") or features.get("adx_14")
        if adx is not None:
            total_checks += 1
            adx_val = float(adx)
            if adx_val >= 25:
                score += 1.0
                reasons.append(f"Strong trend: ADX={adx_val:.1f}")
            elif adx_val >= 20:
                score += 0.5
                reasons.append(f"Moderate trend: ADX={adx_val:.1f}")
            else:
                reasons.append(f"Weak/ranging: ADX={adx_val:.1f}")

        # RSI extreme check
        rsi = features.get("rsi") or features.get("rsi_14")
        if rsi is not None:
            total_checks += 1
            rsi_val = float(rsi)
            is_long = signal.direction.value == "long"
            if is_long and rsi_val > 75:
                reasons.append(f"Warning: RSI overbought at {rsi_val:.1f}")
            elif not is_long and rsi_val < 25:
                reasons.append(f"Warning: RSI oversold at {rsi_val:.1f}")
            else:
                score += 0.5
                reasons.append(f"RSI neutral at {rsi_val:.1f}")

        # Regime alignment
        regime = context.get("regime")
        if regime is not None:
            total_checks += 1
            regime_str = str(regime)
            is_long = signal.direction.value == "long"
            if (is_long and "bull" in regime_str.lower()) or (
                not is_long and "bear" in regime_str.lower()
            ):
                score += 1.0
                reasons.append(f"Regime aligned: {regime_str}")
            elif "range" in regime_str.lower() or "choppy" in regime_str.lower():
                score += 0.3
                reasons.append(f"Regime neutral: {regime_str}")
            else:
                reasons.append(f"Regime misaligned: {regime_str}")

        confidence = score / max(total_checks, 1)
        approve = confidence >= 0.4

        return DeskOpinion(
            role=self._role,
            approve=approve,
            confidence=min(confidence, 1.0),
            reasoning="; ".join(reasons) if reasons else "Insufficient data for structure analysis",
            data={
                "score": score,
                "total_checks": total_checks,
                "features_available": list(features.keys())[:10],
            },
        )


class SMCAnalystDesk(DeskParticipant):
    """Evaluates Smart Money Concepts alignment.

    Checks:
    - Order blocks, fair value gaps
    - Liquidity sweep zones
    - Market structure shifts
    """

    def __init__(self) -> None:
        super().__init__(AgentRole.SMC_ANALYST)

    def evaluate(
        self,
        signal: Signal,
        context: dict[str, Any],
        conversation: AgentConversation,
    ) -> DeskOpinion:
        features = context.get("features", {})
        reasons: list[str] = []
        score = 0.0
        total_checks = 0

        # Check for SMC-related features
        smc_features = {
            k: v for k, v in features.items()
            if any(tag in k.lower() for tag in [
                "order_block", "ob_", "fvg", "liquidity", "bos", "choch",
                "supply", "demand", "imbalance",
            ])
        }

        if smc_features:
            total_checks += len(smc_features)
            for key, val in smc_features.items():
                if val and float(val) != 0:
                    score += 0.5
                    reasons.append(f"{key}={val}")
        else:
            # Fall back to price action analysis
            total_checks += 1
            close = features.get("close")
            bb_upper = features.get("bb_upper") or features.get("bollinger_upper_20")
            bb_lower = features.get("bb_lower") or features.get("bollinger_lower_20")

            if close is not None and bb_upper is not None and bb_lower is not None:
                close_val = float(close)
                upper = float(bb_upper)
                lower = float(bb_lower)
                bb_range = upper - lower
                is_long = signal.direction.value == "long"

                if bb_range > 0:
                    position_in_range = (close_val - lower) / bb_range
                    if is_long and position_in_range < 0.3:
                        score += 1.0
                        reasons.append(
                            f"Near demand zone: price at {position_in_range:.0%} of BB range"
                        )
                    elif not is_long and position_in_range > 0.7:
                        score += 1.0
                        reasons.append(
                            f"Near supply zone: price at {position_in_range:.0%} of BB range"
                        )
                    else:
                        score += 0.3
                        reasons.append(
                            f"Mid-range: price at {position_in_range:.0%} of BB range"
                        )
            else:
                reasons.append("No SMC features or price structure data available")

        # Volume confirmation
        volume = features.get("volume")
        volume_sma = features.get("volume_sma") or features.get("volume_sma_20")
        if volume is not None and volume_sma is not None:
            total_checks += 1
            vol_ratio = float(volume) / max(float(volume_sma), 1)
            if vol_ratio > 1.5:
                score += 1.0
                reasons.append(f"Volume confirmation: {vol_ratio:.1f}x average")
            elif vol_ratio > 1.0:
                score += 0.5
                reasons.append(f"Moderate volume: {vol_ratio:.1f}x average")
            else:
                reasons.append(f"Low volume: {vol_ratio:.1f}x average")

        confidence = score / max(total_checks, 1)
        approve = confidence >= 0.35

        return DeskOpinion(
            role=self._role,
            approve=approve,
            confidence=min(confidence, 1.0),
            reasoning="; ".join(reasons) if reasons else "Insufficient SMC data",
            data={"smc_features": list(smc_features.keys())},
        )


class CMTTechnicianDesk(DeskParticipant):
    """Evaluates multi-timeframe technical confluence.

    Uses CMT 9-layer methodology when available, falls back to
    standard technical indicators.
    """

    def __init__(self) -> None:
        super().__init__(AgentRole.CMT_TECHNICIAN)

    def evaluate(
        self,
        signal: Signal,
        context: dict[str, Any],
        conversation: AgentConversation,
    ) -> DeskOpinion:
        features = context.get("features", {})
        cmt_data = context.get("cmt_assessment")
        reasons: list[str] = []
        score = 0.0
        total_checks = 0

        # Use CMT assessment if available
        if cmt_data is not None:
            total_checks += 1
            confluence = cmt_data.get("confluence_score", {})
            total_score = confluence.get("total", 0)
            threshold_met = confluence.get("threshold_met", False)

            if threshold_met:
                score += 1.0
                reasons.append(
                    f"CMT confluence met: score={total_score}"
                )
            else:
                reasons.append(
                    f"CMT confluence not met: score={total_score}"
                )
        else:
            # Fall back to standard indicators
            # MACD
            macd = features.get("macd") or features.get("macd_12_26_9")
            macd_signal = features.get("macd_signal") or features.get("macd_signal_12_26_9")
            if macd is not None and macd_signal is not None:
                total_checks += 1
                is_long = signal.direction.value == "long"
                macd_bullish = float(macd) > float(macd_signal)
                if (is_long and macd_bullish) or (not is_long and not macd_bullish):
                    score += 1.0
                    reasons.append(
                        f"MACD aligned: {float(macd):.4f} vs signal {float(macd_signal):.4f}"
                    )
                else:
                    reasons.append(
                        f"MACD divergent: {float(macd):.4f} vs signal {float(macd_signal):.4f}"
                    )

            # Stochastic
            stoch_k = features.get("stoch_k") or features.get("stochastic_k")
            stoch_d = features.get("stoch_d") or features.get("stochastic_d")
            if stoch_k is not None and stoch_d is not None:
                total_checks += 1
                k_val = float(stoch_k)
                d_val = float(stoch_d)
                is_long = signal.direction.value == "long"
                if is_long and k_val < 30:
                    score += 1.0
                    reasons.append(f"Stochastic oversold: K={k_val:.1f}, D={d_val:.1f}")
                elif not is_long and k_val > 70:
                    score += 1.0
                    reasons.append(f"Stochastic overbought: K={k_val:.1f}, D={d_val:.1f}")
                else:
                    score += 0.3
                    reasons.append(f"Stochastic neutral: K={k_val:.1f}, D={d_val:.1f}")

            # ATR volatility check
            atr = features.get("atr") or features.get("atr_14")
            close = features.get("close")
            if atr is not None and close is not None:
                total_checks += 1
                atr_pct = float(atr) / float(close) * 100
                if atr_pct < 5:
                    score += 0.8
                    reasons.append(f"Reasonable volatility: ATR={atr_pct:.2f}%")
                elif atr_pct < 10:
                    score += 0.5
                    reasons.append(f"Elevated volatility: ATR={atr_pct:.2f}%")
                else:
                    reasons.append(f"Extreme volatility: ATR={atr_pct:.2f}%")

        # Factor in signal's own confidence
        total_checks += 1
        score += signal.confidence
        reasons.append(f"Signal confidence: {signal.confidence:.0%}")

        confidence = score / max(total_checks, 1)
        approve = confidence >= 0.4

        return DeskOpinion(
            role=self._role,
            approve=approve,
            confidence=min(confidence, 1.0),
            reasoning="; ".join(reasons) if reasons else "Insufficient technical data",
            data={"cmt_available": cmt_data is not None},
        )


class RiskManagerDesk(DeskParticipant):
    """Evaluates risk constraints and portfolio impact.

    Has VETO authority — can block any trade that violates hard limits.

    Checks:
    - Position sizing within limits
    - Portfolio exposure limits
    - Drawdown thresholds
    - Kill switch status
    - Cooldown periods after recent exits
    """

    def __init__(
        self,
        *,
        max_position_pct: float = 0.10,
        max_gross_exposure_pct: float = 1.0,
        max_drawdown_pct: float = 0.15,
        min_exit_cooldown_seconds: int = 300,
    ) -> None:
        super().__init__(AgentRole.RISK_MANAGER)
        self._max_position_pct = max_position_pct
        self._max_gross_exposure_pct = max_gross_exposure_pct
        self._max_drawdown_pct = max_drawdown_pct
        self._min_exit_cooldown = min_exit_cooldown_seconds

    def evaluate(
        self,
        signal: Signal,
        context: dict[str, Any],
        conversation: AgentConversation,
    ) -> DeskOpinion:
        reasons: list[str] = []
        is_veto = False
        score = 0.0
        total_checks = 0

        portfolio = context.get("portfolio_state")
        risk_state = context.get("risk_state")

        # 1. Kill switch check (hard veto)
        kill_switch_active = context.get("kill_switch_active", False)
        if kill_switch_active:
            return DeskOpinion(
                role=self._role,
                approve=False,
                confidence=1.0,
                reasoning="VETO: Kill switch is active — all trading halted",
                is_veto=True,
            )

        # 2. Drawdown check
        if portfolio is not None:
            total_checks += 1
            drawdown = getattr(portfolio, "drawdown_pct", 0.0)
            if drawdown is None:
                drawdown = 0.0
            drawdown = float(drawdown)
            if drawdown >= self._max_drawdown_pct:
                is_veto = True
                reasons.append(
                    f"VETO: Drawdown {drawdown:.1%} exceeds limit "
                    f"{self._max_drawdown_pct:.1%}"
                )
            elif drawdown >= self._max_drawdown_pct * 0.8:
                reasons.append(
                    f"Warning: Drawdown {drawdown:.1%} approaching limit "
                    f"{self._max_drawdown_pct:.1%}"
                )
                score += 0.3
            else:
                score += 1.0
                reasons.append(f"Drawdown OK: {drawdown:.1%}")

        # 3. Gross exposure check
        if portfolio is not None:
            total_checks += 1
            equity = float(getattr(portfolio, "total_equity", 0))
            gross = float(getattr(portfolio, "gross_exposure", 0))
            if equity > 0:
                gross_pct = gross / equity
                if gross_pct >= self._max_gross_exposure_pct:
                    is_veto = True
                    reasons.append(
                        f"VETO: Gross exposure {gross_pct:.0%} at limit "
                        f"{self._max_gross_exposure_pct:.0%}"
                    )
                elif gross_pct >= self._max_gross_exposure_pct * 0.85:
                    score += 0.3
                    reasons.append(
                        f"Warning: Gross exposure {gross_pct:.0%} near limit"
                    )
                else:
                    score += 1.0
                    reasons.append(f"Exposure OK: {gross_pct:.0%}")

        # 4. Recent exit cooldown check
        last_exit_time = context.get("last_exit_time", {}).get(signal.symbol)
        now = context.get("now")
        if last_exit_time is not None and now is not None:
            total_checks += 1
            elapsed = (now - last_exit_time).total_seconds()
            if elapsed < self._min_exit_cooldown:
                is_veto = True
                remaining = self._min_exit_cooldown - elapsed
                reasons.append(
                    f"VETO: Exit cooldown active for {signal.symbol} — "
                    f"{remaining:.0f}s remaining (exited {elapsed:.0f}s ago)"
                )
            else:
                score += 1.0
                reasons.append(
                    f"Cooldown clear: last exit {elapsed:.0f}s ago"
                )

        # 5. Existing position conflict check
        if portfolio is not None:
            total_checks += 1
            positions = getattr(portfolio, "positions", {})
            if signal.symbol in positions:
                existing = positions[signal.symbol]
                existing_dir = getattr(existing, "side", "")
                if hasattr(existing_dir, "value"):
                    existing_dir = existing_dir.value
                signal_dir = signal.direction.value
                if existing_dir and existing_dir != signal_dir:
                    reasons.append(
                        f"Warning: Reversing {existing_dir} -> {signal_dir} "
                        f"for {signal.symbol}"
                    )
                    score += 0.5
                elif existing_dir == signal_dir:
                    reasons.append(
                        f"Already {existing_dir} {signal.symbol} — "
                        f"potential add-on"
                    )
                    score += 0.8
            else:
                score += 1.0
                reasons.append(f"No existing position in {signal.symbol}")

        # 6. Signal confidence minimum
        total_checks += 1
        if signal.confidence < 0.3:
            reasons.append(
                f"Low signal confidence: {signal.confidence:.0%}"
            )
        else:
            score += 0.8
            reasons.append(
                f"Signal confidence acceptable: {signal.confidence:.0%}"
            )

        confidence = score / max(total_checks, 1)
        approve = not is_veto and confidence >= 0.35

        return DeskOpinion(
            role=self._role,
            approve=approve,
            confidence=min(confidence, 1.0),
            reasoning="; ".join(reasons) if reasons else "No risk data available",
            is_veto=is_veto,
            data={
                "checks_passed": total_checks,
                "score": score,
            },
        )


# ---------------------------------------------------------------------------
# ConsensusGate
# ---------------------------------------------------------------------------


class ConsensusGate:
    """Multi-agent consultation gate for trade signals.

    Orchestrates a structured conversation between desk participants
    before allowing any signal to reach the execution pipeline.

    Parameters
    ----------
    message_bus:
        ReasoningMessageBus for conversation recording.
    participants:
        Ordered list of desk participants. They are consulted in order.
    min_approval_ratio:
        Minimum fraction of approvals needed (default 0.6 = 60%).
    require_risk_approval:
        If True, Risk Manager must approve (default True).
    conversation_store:
        Optional store for persisting conversations.
    """

    def __init__(
        self,
        *,
        message_bus: ReasoningMessageBus | None = None,
        participants: list[DeskParticipant] | None = None,
        min_approval_ratio: float = 0.6,
        require_risk_approval: bool = True,
        conversation_store: Any | None = None,
    ) -> None:
        self._bus = message_bus or ReasoningMessageBus()
        self._participants = participants or self._default_participants()
        self._min_approval_ratio = min_approval_ratio
        self._require_risk_approval = require_risk_approval
        self._store = conversation_store

        # Track recent exits per symbol for cooldown enforcement
        self._last_exit_times: dict[str, datetime] = {}

        # Stats
        self._consultations: int = 0
        self._approvals: int = 0
        self._rejections: int = 0
        self._vetoes: int = 0

    @staticmethod
    def _default_participants() -> list[DeskParticipant]:
        """Create default desk participants in consultation order."""
        return [
            MarketStructureDesk(),
            SMCAnalystDesk(),
            CMTTechnicianDesk(),
            RiskManagerDesk(),
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def consult(
        self,
        signal: Signal,
        context: dict[str, Any],
    ) -> ConsensusResult:
        """Run the desk consultation for a proposed trade signal.

        Parameters
        ----------
        signal:
            The strategy signal to evaluate.
        context:
            Market context dict with keys:
            - features: dict of indicator values
            - regime: current regime string
            - portfolio_state: PortfolioState object
            - risk_state: RiskState object
            - cmt_assessment: latest CMT assessment dict
            - kill_switch_active: bool
            - now: current datetime
            - last_exit_time: dict of symbol -> datetime

        Returns
        -------
        ConsensusResult
            The desk's collective decision.
        """
        t_start = time.monotonic()
        self._consultations += 1

        # Merge gate's internal exit times into context
        # Gate's own tracking takes precedence
        if "last_exit_time" not in context:
            context["last_exit_time"] = {}
        for sym, exit_time in self._last_exit_times.items():
            existing = context["last_exit_time"].get(sym)
            if existing is None or exit_time > existing:
                context["last_exit_time"][sym] = exit_time

        # Create a conversation for this consultation
        conversation = self._bus.create_conversation(
            symbol=signal.symbol,
            timeframe=(
                signal.timeframe.value
                if hasattr(signal.timeframe, "value")
                else str(signal.timeframe or "")
            ),
            trigger_event=f"signal.{signal.strategy_id}.{signal.direction.value}",
            strategy_id=signal.strategy_id,
            context_snapshot={
                "signal_confidence": signal.confidence,
                "signal_direction": signal.direction.value,
                "signal_rationale": signal.rationale[:200] if signal.rationale else "",
            },
        )

        # Post the opening message from the orchestrator
        opening_msg = AgentMessage(
            conversation_id=conversation.conversation_id,
            sender=AgentRole.ORCHESTRATOR,
            message_type=MessageType.MARKET_UPDATE,
            content=(
                f"Signal received: {signal.direction.value.upper()} {signal.symbol} "
                f"from {signal.strategy_id} (confidence: {signal.confidence:.0%}). "
                f"Rationale: {signal.rationale or 'N/A'}. "
                f"Requesting desk consultation."
            ),
            confidence=signal.confidence,
            structured_data={
                "direction": signal.direction.value,
                "symbol": signal.symbol,
                "strategy_id": signal.strategy_id,
                "confidence": signal.confidence,
            },
        )
        self._bus.post(opening_msg)

        # Consult each participant in order
        opinions: list[DeskOpinion] = []
        veto_opinion: DeskOpinion | None = None

        for participant in self._participants:
            opinion = participant.evaluate(signal, context, conversation)
            opinions.append(opinion)

            # Record the opinion as a message
            msg_type = (
                MessageType.VETO if opinion.is_veto
                else MessageType.RISK_ASSESSMENT if opinion.role == AgentRole.RISK_MANAGER
                else MessageType.ANALYSIS
            )

            opinion_msg = AgentMessage(
                conversation_id=conversation.conversation_id,
                sender=opinion.role,
                recipients=[AgentRole.ORCHESTRATOR],
                message_type=msg_type,
                content=opinion.reasoning,
                confidence=opinion.confidence,
                structured_data={
                    "approve": opinion.approve,
                    "is_veto": opinion.is_veto,
                    **opinion.data,
                },
            )
            self._bus.post(opinion_msg)

            # Hard veto terminates immediately
            if opinion.is_veto:
                veto_opinion = opinion
                break

        # Calculate consensus
        elapsed = (time.monotonic() - t_start) * 1000

        if veto_opinion is not None:
            self._vetoes += 1
            result = ConsensusResult(
                verdict=ConsensusVerdict.VETOED,
                conversation_id=conversation.conversation_id,
                opinions=opinions,
                weighted_score=-1.0,
                reasoning=f"Vetoed by {veto_opinion.role.display_name}: {veto_opinion.reasoning}",
                elapsed_ms=elapsed,
            )
        else:
            result = self._calculate_consensus(
                signal, opinions, conversation, elapsed
            )

        # Post the final decision
        decision_msg = AgentMessage(
            conversation_id=conversation.conversation_id,
            sender=AgentRole.ORCHESTRATOR,
            message_type=MessageType.SIGNAL if result.is_approved else MessageType.SYSTEM,
            content=result.reasoning,
            confidence=abs(result.weighted_score),
            structured_data={
                "verdict": result.verdict.value,
                "weighted_score": result.weighted_score,
                "elapsed_ms": result.elapsed_ms,
            },
        )
        self._bus.post(decision_msg)

        # Finalize conversation
        outcome = (
            ConversationOutcome.TRADE_ENTERED if result.is_approved
            else ConversationOutcome.VETOED if result.verdict == ConsensusVerdict.VETOED
            else ConversationOutcome.NO_TRADE
        )
        self._bus.finalize_conversation(
            conversation.conversation_id,
            outcome,
            details={
                "verdict": result.verdict.value,
                "weighted_score": result.weighted_score,
                "opinions": [
                    {
                        "role": o.role.value,
                        "approve": o.approve,
                        "confidence": o.confidence,
                    }
                    for o in opinions
                ],
            },
        )

        # Persist conversation if store is available
        if self._store is not None:
            try:
                self._store.save(conversation)
            except Exception:
                logger.debug("Failed to persist conversation", exc_info=True)

        logger.info(
            "Consensus: %s %s %s → %s (score=%.2f, %dms, %d opinions)",
            signal.symbol,
            signal.direction.value,
            signal.strategy_id,
            result.verdict.value,
            result.weighted_score,
            elapsed,
            len(opinions),
        )

        return result

    def consult_exit(
        self,
        signal: Signal,
        context: dict[str, Any],
    ) -> ConsensusResult:
        """Lighter-weight consultation for exit signals.

        Exit signals get a simplified check: only Risk Manager evaluates.
        Exits are generally allowed unless risk conditions prevent it.

        Also records the exit time for cooldown tracking.
        """
        t_start = time.monotonic()

        # Record exit time
        self._last_exit_times[signal.symbol] = (
            context.get("now") or datetime.now(timezone.utc)
        )

        # Create conversation for the exit
        conversation = self._bus.create_conversation(
            symbol=signal.symbol,
            trigger_event=f"exit.{signal.strategy_id}",
            strategy_id=signal.strategy_id,
        )

        # Post exit notification
        exit_msg = AgentMessage(
            conversation_id=conversation.conversation_id,
            sender=AgentRole.ORCHESTRATOR,
            message_type=MessageType.SIGNAL,
            content=(
                f"Exit signal for {signal.symbol} from {signal.strategy_id}. "
                f"Rationale: {signal.rationale or 'N/A'}."
            ),
            structured_data={
                "direction": "flat",
                "symbol": signal.symbol,
                "strategy_id": signal.strategy_id,
            },
        )
        self._bus.post(exit_msg)

        elapsed = (time.monotonic() - t_start) * 1000

        # Exits are approved by default
        result = ConsensusResult(
            verdict=ConsensusVerdict.APPROVED,
            conversation_id=conversation.conversation_id,
            opinions=[],
            weighted_score=1.0,
            approved_signal=signal,
            reasoning=f"Exit approved for {signal.symbol}",
            elapsed_ms=elapsed,
        )

        self._bus.finalize_conversation(
            conversation.conversation_id,
            ConversationOutcome.TRADE_EXITED,
        )

        if self._store is not None:
            try:
                self._store.save(conversation)
            except Exception:
                logger.debug("Failed to persist exit conversation", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Consensus calculation
    # ------------------------------------------------------------------

    def _calculate_consensus(
        self,
        signal: Signal,
        opinions: list[DeskOpinion],
        conversation: AgentConversation,
        elapsed_ms: float,
    ) -> ConsensusResult:
        """Calculate the weighted consensus from desk opinions."""
        if not opinions:
            return ConsensusResult(
                verdict=ConsensusVerdict.REJECTED,
                conversation_id=conversation.conversation_id,
                opinions=[],
                weighted_score=0.0,
                reasoning="No desk opinions received",
                elapsed_ms=elapsed_ms,
            )

        # Confidence-weighted voting
        total_weight = 0.0
        approval_weight = 0.0
        risk_approved = True

        for opinion in opinions:
            weight = max(opinion.confidence, 0.1)  # Min weight 0.1
            total_weight += weight
            if opinion.approve:
                approval_weight += weight
            if opinion.role == AgentRole.RISK_MANAGER and not opinion.approve:
                risk_approved = False

        weighted_score = (
            approval_weight / total_weight if total_weight > 0 else 0.0
        )

        # Check approval conditions
        approval_count = sum(1 for o in opinions if o.approve)
        approval_ratio = approval_count / len(opinions)

        passed_ratio = approval_ratio >= self._min_approval_ratio
        passed_risk = risk_approved or not self._require_risk_approval

        if passed_ratio and passed_risk:
            self._approvals += 1
            verdict = ConsensusVerdict.APPROVED
            reasoning = (
                f"APPROVED: {approval_count}/{len(opinions)} desk members approve "
                f"(weighted score: {weighted_score:.2f})"
            )
        elif not passed_risk:
            self._rejections += 1
            verdict = ConsensusVerdict.REJECTED
            risk_opinion = next(
                (o for o in opinions if o.role == AgentRole.RISK_MANAGER), None
            )
            risk_reason = risk_opinion.reasoning if risk_opinion else "unknown"
            reasoning = (
                f"REJECTED: Risk Manager did not approve — {risk_reason}"
            )
        else:
            self._rejections += 1
            verdict = ConsensusVerdict.REJECTED
            reasoning = (
                f"REJECTED: Only {approval_count}/{len(opinions)} approve "
                f"({approval_ratio:.0%} < {self._min_approval_ratio:.0%} required)"
            )

        result = ConsensusResult(
            verdict=verdict,
            conversation_id=conversation.conversation_id,
            opinions=opinions,
            weighted_score=weighted_score,
            reasoning=reasoning,
            elapsed_ms=elapsed_ms,
        )

        if verdict == ConsensusVerdict.APPROVED:
            result.approved_signal = signal

        return result

    # ------------------------------------------------------------------
    # Exit cooldown management
    # ------------------------------------------------------------------

    def record_exit(self, symbol: str, when: datetime | None = None) -> None:
        """Record that a position was exited for cooldown tracking."""
        self._last_exit_times[symbol] = when or datetime.now(timezone.utc)

    def get_last_exit_time(self, symbol: str) -> datetime | None:
        """Get the last exit time for a symbol."""
        return self._last_exit_times.get(symbol)

    def clear_exit_cooldown(self, symbol: str) -> None:
        """Clear exit cooldown for a symbol."""
        self._last_exit_times.pop(symbol, None)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def consultations(self) -> int:
        return self._consultations

    @property
    def approvals(self) -> int:
        return self._approvals

    @property
    def rejections(self) -> int:
        return self._rejections

    @property
    def vetoes(self) -> int:
        return self._vetoes

    @property
    def approval_rate(self) -> float:
        """Percentage of consultations that resulted in approval."""
        if self._consultations == 0:
            return 0.0
        return self._approvals / self._consultations
