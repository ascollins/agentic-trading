"""Tests for the ConsensusGate — multi-agent consultation before trade execution.

Tests cover:
- Individual desk participants (Market Structure, SMC, CMT, Risk Manager)
- ConsensusGate overall flow (approval, rejection, veto)
- Exit cooldown tracking
- Conversation recording via ReasoningMessageBus
- Edge cases (empty features, no data, extreme values)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from agentic_trading.core.enums import SignalDirection, Timeframe
from agentic_trading.core.events import Signal
from agentic_trading.reasoning.agent_message import AgentRole, MessageType
from agentic_trading.reasoning.consensus import (
    CMTTechnicianDesk,
    ConsensusGate,
    ConsensusResult,
    ConsensusVerdict,
    DeskOpinion,
    DeskParticipant,
    MarketStructureDesk,
    RiskManagerDesk,
    SMCAnalystDesk,
)
from agentic_trading.reasoning.message_bus import ReasoningMessageBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    *,
    symbol: str = "BTC/USDT",
    direction: str = "long",
    confidence: float = 0.7,
    strategy_id: str = "test_strategy",
    rationale: str = "Test signal rationale",
) -> Signal:
    """Create a test Signal."""
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=SignalDirection(direction),
        confidence=confidence,
        rationale=rationale,
        timeframe=Timeframe.M15,
    )


def _make_context(
    *,
    features: dict[str, Any] | None = None,
    regime: str | None = None,
    kill_switch: bool = False,
    equity: float = 100_000,
    gross_exposure: float = 10_000,
    drawdown: float = 0.02,
    positions: dict | None = None,
    last_exit_time: dict | None = None,
) -> dict[str, Any]:
    """Create a test context dict."""

    class _MockPortfolio:
        def __init__(self):
            self.total_equity = Decimal(str(equity))
            self.gross_exposure = Decimal(str(gross_exposure))
            self.net_exposure = Decimal(str(gross_exposure * 0.8))
            self.drawdown_pct = drawdown
            self.positions = positions or {}

    return {
        "features": features or {
            "close": 43000.0,
            "ema_12": 43100.0,
            "ema_26": 42900.0,
            "adx_14": 28.0,
            "rsi_14": 55.0,
            "atr_14": 500.0,
            "macd": 0.002,
            "macd_signal": 0.001,
            "bb_upper": 44000.0,
            "bb_lower": 42000.0,
            "volume": 1500,
            "volume_sma": 1000,
        },
        "regime": regime,
        "portfolio_state": _MockPortfolio(),
        "risk_state": None,
        "cmt_assessment": None,
        "kill_switch_active": kill_switch,
        "now": datetime.now(timezone.utc),
        "last_exit_time": last_exit_time or {},
    }


# ---------------------------------------------------------------------------
# MarketStructureDesk tests
# ---------------------------------------------------------------------------


class TestMarketStructureDesk:
    def test_approves_aligned_ema_trend(self):
        desk = MarketStructureDesk()
        signal = _make_signal(direction="long")
        context = _make_context(features={
            "ema_12": 43100, "ema_26": 42900,
            "adx_14": 30, "rsi_14": 55,
        })
        opinion = desk.evaluate(signal, context, None)
        assert opinion.approve is True
        assert opinion.confidence > 0.5
        assert "EMA alignment" in opinion.reasoning

    def test_rejects_misaligned_ema(self):
        desk = MarketStructureDesk()
        signal = _make_signal(direction="long")
        # EMA12 < EMA26 => bearish, conflicting with long signal
        context = _make_context(features={
            "ema_12": 42800, "ema_26": 43100,
            "adx_14": 15, "rsi_14": 55,
        })
        opinion = desk.evaluate(signal, context, None)
        assert opinion.confidence < 0.5
        assert "EMA divergence" in opinion.reasoning

    def test_regime_aligned_bullish(self):
        desk = MarketStructureDesk()
        signal = _make_signal(direction="long")
        context = _make_context(
            features={"ema_12": 100, "ema_26": 99, "adx_14": 30},
            regime="bullish_trending",
        )
        opinion = desk.evaluate(signal, context, None)
        assert opinion.approve is True
        assert "Regime aligned" in opinion.reasoning

    def test_regime_misaligned(self):
        desk = MarketStructureDesk()
        signal = _make_signal(direction="long")
        context = _make_context(
            features={"ema_12": 100, "ema_26": 99},
            regime="bearish_trending",
        )
        opinion = desk.evaluate(signal, context, None)
        assert "Regime misaligned" in opinion.reasoning

    def test_empty_features_returns_opinion(self):
        desk = MarketStructureDesk()
        signal = _make_signal()
        context = _make_context(features={})
        opinion = desk.evaluate(signal, context, None)
        assert isinstance(opinion, DeskOpinion)
        assert opinion.role == AgentRole.MARKET_STRUCTURE

    def test_weak_adx_reduces_confidence(self):
        desk = MarketStructureDesk()
        signal = _make_signal(direction="long")
        context = _make_context(features={
            "ema_12": 100, "ema_26": 99, "adx_14": 12,
        })
        opinion = desk.evaluate(signal, context, None)
        assert "Weak/ranging" in opinion.reasoning


# ---------------------------------------------------------------------------
# SMCAnalystDesk tests
# ---------------------------------------------------------------------------


class TestSMCAnalystDesk:
    def test_volume_confirmation(self):
        desk = SMCAnalystDesk()
        signal = _make_signal(direction="long")
        context = _make_context(features={
            "close": 43000, "bb_upper": 44000, "bb_lower": 42000,
            "volume": 2000, "volume_sma": 1000,
        })
        opinion = desk.evaluate(signal, context, None)
        assert "Volume confirmation" in opinion.reasoning

    def test_demand_zone_for_long(self):
        desk = SMCAnalystDesk()
        signal = _make_signal(direction="long")
        # Price near lower BB => demand zone
        context = _make_context(features={
            "close": 42100, "bb_upper": 44000, "bb_lower": 42000,
            "volume": 800, "volume_sma": 1000,
        })
        opinion = desk.evaluate(signal, context, None)
        assert "demand zone" in opinion.reasoning.lower() or "Near demand" in opinion.reasoning

    def test_supply_zone_for_short(self):
        desk = SMCAnalystDesk()
        signal = _make_signal(direction="short")
        # Price near upper BB => supply zone
        context = _make_context(features={
            "close": 43900, "bb_upper": 44000, "bb_lower": 42000,
            "volume": 800, "volume_sma": 1000,
        })
        opinion = desk.evaluate(signal, context, None)
        assert "supply zone" in opinion.reasoning.lower() or "Near supply" in opinion.reasoning

    def test_smc_features_detected(self):
        desk = SMCAnalystDesk()
        signal = _make_signal()
        context = _make_context(features={
            "order_block_bullish": 1,
            "fvg_15m": 0.5,
            "liquidity_sweep": 0,
        })
        opinion = desk.evaluate(signal, context, None)
        assert opinion.data.get("smc_features")

    def test_no_data(self):
        desk = SMCAnalystDesk()
        signal = _make_signal()
        context = _make_context(features={})
        opinion = desk.evaluate(signal, context, None)
        assert isinstance(opinion, DeskOpinion)


# ---------------------------------------------------------------------------
# CMTTechnicianDesk tests
# ---------------------------------------------------------------------------


class TestCMTTechnicianDesk:
    def test_macd_aligned(self):
        desk = CMTTechnicianDesk()
        signal = _make_signal(direction="long")
        context = _make_context(features={
            "macd": 0.005, "macd_signal": 0.002,
            "atr": 500, "close": 43000,
        })
        opinion = desk.evaluate(signal, context, None)
        assert "MACD aligned" in opinion.reasoning

    def test_macd_divergent(self):
        desk = CMTTechnicianDesk()
        signal = _make_signal(direction="long")
        context = _make_context(features={
            "macd": -0.003, "macd_signal": 0.002,
        })
        opinion = desk.evaluate(signal, context, None)
        assert "MACD divergent" in opinion.reasoning

    def test_stochastic_oversold_for_long(self):
        desk = CMTTechnicianDesk()
        signal = _make_signal(direction="long")
        context = _make_context(features={
            "stoch_k": 18, "stoch_d": 15,
        })
        opinion = desk.evaluate(signal, context, None)
        assert "oversold" in opinion.reasoning.lower()

    def test_cmt_assessment_used_when_available(self):
        desk = CMTTechnicianDesk()
        signal = _make_signal(direction="long")
        context = _make_context()
        context["cmt_assessment"] = {
            "confluence_score": {"total": 7, "threshold_met": True},
        }
        opinion = desk.evaluate(signal, context, None)
        assert "CMT confluence met" in opinion.reasoning
        assert opinion.data["cmt_available"] is True

    def test_cmt_confluence_not_met(self):
        desk = CMTTechnicianDesk()
        signal = _make_signal()
        context = _make_context()
        context["cmt_assessment"] = {
            "confluence_score": {"total": 2, "threshold_met": False},
        }
        opinion = desk.evaluate(signal, context, None)
        assert "not met" in opinion.reasoning

    def test_high_volatility_warning(self):
        desk = CMTTechnicianDesk()
        signal = _make_signal()
        context = _make_context(features={
            "atr": 6000, "close": 43000,
        })
        opinion = desk.evaluate(signal, context, None)
        assert "volatility" in opinion.reasoning.lower()


# ---------------------------------------------------------------------------
# RiskManagerDesk tests
# ---------------------------------------------------------------------------


class TestRiskManagerDesk:
    def test_kill_switch_vetoes(self):
        desk = RiskManagerDesk()
        signal = _make_signal()
        context = _make_context(kill_switch=True)
        opinion = desk.evaluate(signal, context, None)
        assert opinion.approve is False
        assert opinion.is_veto is True
        assert "Kill switch" in opinion.reasoning

    def test_drawdown_veto(self):
        desk = RiskManagerDesk(max_drawdown_pct=0.10)
        signal = _make_signal()
        context = _make_context(drawdown=0.12)
        opinion = desk.evaluate(signal, context, None)
        assert opinion.is_veto is True
        assert "Drawdown" in opinion.reasoning

    def test_drawdown_warning(self):
        desk = RiskManagerDesk(max_drawdown_pct=0.15)
        signal = _make_signal()
        context = _make_context(drawdown=0.13)
        opinion = desk.evaluate(signal, context, None)
        assert "approaching limit" in opinion.reasoning

    def test_gross_exposure_veto(self):
        desk = RiskManagerDesk(max_gross_exposure_pct=0.5)
        signal = _make_signal()
        context = _make_context(equity=100000, gross_exposure=60000)
        opinion = desk.evaluate(signal, context, None)
        assert opinion.is_veto is True
        assert "Gross exposure" in opinion.reasoning

    def test_exit_cooldown_veto(self):
        desk = RiskManagerDesk(min_exit_cooldown_seconds=300)
        signal = _make_signal()
        now = datetime.now(timezone.utc)
        context = _make_context()
        context["now"] = now
        context["last_exit_time"] = {
            "BTC/USDT": now - timedelta(seconds=60),
        }
        opinion = desk.evaluate(signal, context, None)
        assert opinion.is_veto is True
        assert "cooldown" in opinion.reasoning.lower()

    def test_cooldown_clear_after_period(self):
        desk = RiskManagerDesk(min_exit_cooldown_seconds=300)
        signal = _make_signal()
        now = datetime.now(timezone.utc)
        context = _make_context()
        context["now"] = now
        context["last_exit_time"] = {
            "BTC/USDT": now - timedelta(seconds=600),
        }
        opinion = desk.evaluate(signal, context, None)
        assert opinion.is_veto is False
        assert "Cooldown clear" in opinion.reasoning

    def test_existing_position_conflict(self):
        desk = RiskManagerDesk()
        signal = _make_signal(direction="long")

        class _MockPos:
            side = "short"

        context = _make_context(positions={"BTC/USDT": _MockPos()})
        opinion = desk.evaluate(signal, context, None)
        assert "Reversing" in opinion.reasoning

    def test_no_portfolio_returns_opinion(self):
        desk = RiskManagerDesk()
        signal = _make_signal()
        context = {"kill_switch_active": False}
        opinion = desk.evaluate(signal, context, None)
        assert isinstance(opinion, DeskOpinion)

    def test_approves_healthy_conditions(self):
        desk = RiskManagerDesk()
        signal = _make_signal(confidence=0.8)
        context = _make_context(
            drawdown=0.03,
            equity=100000,
            gross_exposure=20000,
        )
        opinion = desk.evaluate(signal, context, None)
        assert opinion.approve is True
        assert opinion.is_veto is False

    def test_low_confidence_noted(self):
        desk = RiskManagerDesk()
        signal = _make_signal(confidence=0.1)
        context = _make_context()
        opinion = desk.evaluate(signal, context, None)
        assert "Low signal confidence" in opinion.reasoning


# ---------------------------------------------------------------------------
# ConsensusGate tests
# ---------------------------------------------------------------------------


class TestConsensusGate:
    def test_approves_strong_signal(self):
        """Strong signal with aligned features should be approved."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus)
        signal = _make_signal(direction="long", confidence=0.8)
        context = _make_context(features={
            "close": 43000,
            "ema_12": 43200, "ema_26": 42800,
            "adx_14": 30, "rsi_14": 55,
            "macd": 0.005, "macd_signal": 0.002,
            "atr_14": 500,
            "bb_upper": 44000, "bb_lower": 42000,
            "volume": 1500, "volume_sma": 1000,
        })

        result = gate.consult(signal, context)
        assert isinstance(result, ConsensusResult)
        assert result.verdict == ConsensusVerdict.APPROVED
        assert result.weighted_score > 0.5
        assert len(result.opinions) == 4
        assert result.elapsed_ms > 0
        assert gate.consultations == 1
        assert gate.approvals == 1

    def test_rejects_weak_signal(self):
        """Very weak signal with misaligned everything should be rejected."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus, min_approval_ratio=0.6)
        signal = _make_signal(direction="long", confidence=0.1)
        # All indicators against the signal
        context = _make_context(features={
            "close": 43000,
            "ema_12": 42500, "ema_26": 43500,  # bearish EMA
            "adx_14": 10,  # weak trend
            "rsi_14": 80,  # overbought for long
            "macd": -0.005, "macd_signal": 0.001,  # MACD divergent
            "bb_upper": 44000, "bb_lower": 42000,
            "volume": 300, "volume_sma": 1000,  # low volume
        })

        result = gate.consult(signal, context)
        assert result.verdict in (ConsensusVerdict.REJECTED, ConsensusVerdict.VETOED)
        assert gate.rejections + gate.vetoes == 1

    def test_veto_stops_consultation(self):
        """Kill switch veto from Risk Manager stops all further checks."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus)
        signal = _make_signal()
        context = _make_context(kill_switch=True)

        result = gate.consult(signal, context)
        assert result.verdict == ConsensusVerdict.VETOED
        assert "Kill switch" in result.reasoning
        assert gate.vetoes == 1

    def test_exit_cooldown_prevents_reentry(self):
        """After an exit, re-entry within cooldown period is vetoed."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(
            message_bus=bus,
            participants=[
                MarketStructureDesk(),
                SMCAnalystDesk(),
                CMTTechnicianDesk(),
                RiskManagerDesk(min_exit_cooldown_seconds=300),
            ],
        )
        now = datetime.now(timezone.utc)

        # First: record an exit
        exit_signal = _make_signal(direction="flat")
        gate.record_exit("BTC/USDT", when=now)

        # Then: try re-entry 60 seconds later
        entry_signal = _make_signal(direction="long")
        context = _make_context()
        context["now"] = now + timedelta(seconds=60)

        result = gate.consult(entry_signal, context)
        assert result.verdict == ConsensusVerdict.VETOED
        assert "cooldown" in result.reasoning.lower()

    def test_exit_cooldown_clears_after_period(self):
        """After cooldown expires, re-entry is allowed."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(
            message_bus=bus,
            participants=[
                MarketStructureDesk(),
                SMCAnalystDesk(),
                CMTTechnicianDesk(),
                RiskManagerDesk(min_exit_cooldown_seconds=300),
            ],
        )
        now = datetime.now(timezone.utc)

        gate.record_exit("BTC/USDT", when=now)

        # Try re-entry 600 seconds later (past cooldown)
        entry_signal = _make_signal(direction="long", confidence=0.8)
        context = _make_context(features={
            "close": 43000,
            "ema_12": 43200, "ema_26": 42800,
            "adx_14": 30, "rsi_14": 55,
            "macd": 0.005, "macd_signal": 0.002,
            "atr_14": 500,
            "bb_upper": 44000, "bb_lower": 42000,
            "volume": 1500, "volume_sma": 1000,
        })
        context["now"] = now + timedelta(seconds=600)

        result = gate.consult(entry_signal, context)
        # Should not be vetoed by cooldown
        assert "cooldown" not in result.reasoning.lower()

    def test_conversation_recorded(self):
        """Consultation creates a recorded conversation."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus)
        signal = _make_signal()
        context = _make_context()

        result = gate.consult(signal, context)

        # Check conversation was recorded
        conv = bus.get_conversation(result.conversation_id)
        assert conv is not None
        # At least: opening msg + 4 opinions + decision
        assert len(conv.messages) >= 3
        assert conv.completed_at is not None
        assert conv.symbol == "BTC/USDT"

    def test_conversation_has_desk_opinions(self):
        """Each desk participant records an opinion message."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus)
        signal = _make_signal()
        context = _make_context()

        result = gate.consult(signal, context)
        conv = bus.get_conversation(result.conversation_id)

        # Find opinion messages (not from orchestrator)
        opinion_msgs = [
            m for m in conv.messages
            if m.sender != AgentRole.ORCHESTRATOR
        ]

        # Should have opinions from the 4 desk participants
        senders = {m.sender for m in opinion_msgs}
        assert AgentRole.MARKET_STRUCTURE in senders
        assert AgentRole.SMC_ANALYST in senders
        assert AgentRole.CMT_TECHNICIAN in senders
        assert AgentRole.RISK_MANAGER in senders

    def test_consult_exit_records_time(self):
        """Exit consultation records the exit time."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus)
        signal = _make_signal(direction="flat")
        now = datetime.now(timezone.utc)

        result = gate.consult_exit(signal, {"now": now})
        assert result.is_approved
        assert gate.get_last_exit_time("BTC/USDT") == now

    def test_risk_rejection_overrides_majority(self):
        """Risk Manager rejection blocks even if others approve."""
        bus = ReasoningMessageBus()

        class AlwaysApprove(DeskParticipant):
            def evaluate(self, sig, ctx, conv):
                return DeskOpinion(
                    role=self._role, approve=True, confidence=0.9,
                    reasoning="Always approve",
                )

        gate = ConsensusGate(
            message_bus=bus,
            participants=[
                AlwaysApprove(AgentRole.MARKET_STRUCTURE),
                AlwaysApprove(AgentRole.SMC_ANALYST),
                AlwaysApprove(AgentRole.CMT_TECHNICIAN),
                RiskManagerDesk(max_drawdown_pct=0.05),
            ],
            require_risk_approval=True,
        )
        signal = _make_signal()
        context = _make_context(drawdown=0.06)

        result = gate.consult(signal, context)
        # Risk Manager vetoes due to drawdown
        assert result.verdict in (ConsensusVerdict.VETOED, ConsensusVerdict.REJECTED)

    def test_stats_tracking(self):
        """Gate tracks consultation, approval, rejection stats."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus)

        # Approved signal
        sig1 = _make_signal(confidence=0.8)
        ctx1 = _make_context(features={
            "close": 43000,
            "ema_12": 43200, "ema_26": 42800,
            "adx_14": 30, "rsi_14": 55,
            "macd": 0.005, "macd_signal": 0.002,
            "atr_14": 500,
            "bb_upper": 44000, "bb_lower": 42000,
            "volume": 1500, "volume_sma": 1000,
        })
        gate.consult(sig1, ctx1)

        # Vetoed signal
        sig2 = _make_signal()
        ctx2 = _make_context(kill_switch=True)
        gate.consult(sig2, ctx2)

        assert gate.consultations == 2
        assert gate.vetoes >= 1

    def test_custom_participants(self):
        """Gate works with custom desk participants."""
        bus = ReasoningMessageBus()

        class CustomDesk(DeskParticipant):
            def evaluate(self, sig, ctx, conv):
                return DeskOpinion(
                    role=self._role,
                    approve=True,
                    confidence=1.0,
                    reasoning="Custom always approve",
                )

        gate = ConsensusGate(
            message_bus=bus,
            participants=[CustomDesk(AgentRole.ORCHESTRATOR)],
            require_risk_approval=False,
        )
        signal = _make_signal()
        context = _make_context()

        result = gate.consult(signal, context)
        assert result.is_approved
        assert len(result.opinions) == 1

    def test_approval_rate_property(self):
        """Approval rate calculated correctly."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus)
        assert gate.approval_rate == 0.0

    def test_consult_exit_approved_by_default(self):
        """Exit signals are approved by default."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus)
        signal = _make_signal(direction="flat")

        result = gate.consult_exit(signal, {"now": datetime.now(timezone.utc)})
        assert result.is_approved
        assert result.verdict == ConsensusVerdict.APPROVED


# ---------------------------------------------------------------------------
# Conversation persistence tests
# ---------------------------------------------------------------------------


class TestConsensusGatePersistence:
    def test_store_called_on_consult(self):
        """Conversation is saved to store when provided."""
        bus = ReasoningMessageBus()
        saved = []

        class _MockStore:
            def save(self, conv):
                saved.append(conv)

        gate = ConsensusGate(
            message_bus=bus,
            conversation_store=_MockStore(),
        )
        signal = _make_signal()
        context = _make_context()

        gate.consult(signal, context)
        assert len(saved) == 1
        assert saved[0].symbol == "BTC/USDT"

    def test_store_called_on_exit(self):
        """Exit conversation is saved to store."""
        bus = ReasoningMessageBus()
        saved = []

        class _MockStore:
            def save(self, conv):
                saved.append(conv)

        gate = ConsensusGate(
            message_bus=bus,
            conversation_store=_MockStore(),
        )
        signal = _make_signal(direction="flat")
        gate.consult_exit(signal, {"now": datetime.now(timezone.utc)})
        assert len(saved) == 1

    def test_store_failure_doesnt_crash(self):
        """Store failure is caught and logged, doesn't crash the gate."""
        bus = ReasoningMessageBus()

        class _FailingStore:
            def save(self, conv):
                raise RuntimeError("Store unavailable")

        gate = ConsensusGate(
            message_bus=bus,
            conversation_store=_FailingStore(),
        )
        signal = _make_signal()
        context = _make_context()

        # Should not raise
        result = gate.consult(signal, context)
        assert isinstance(result, ConsensusResult)


# ---------------------------------------------------------------------------
# Integration: full signal flow simulation
# ---------------------------------------------------------------------------


class TestConsensusGateIntegration:
    def test_full_cycle_entry_exit_re_entry(self):
        """Simulate: approve entry → exit → cooldown blocks re-entry → cooldown clears."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(
            message_bus=bus,
            participants=[
                MarketStructureDesk(),
                SMCAnalystDesk(),
                CMTTechnicianDesk(),
                RiskManagerDesk(min_exit_cooldown_seconds=120),
            ],
        )
        now = datetime.now(timezone.utc)

        # Step 1: Strong entry signal
        entry_signal = _make_signal(direction="long", confidence=0.85)
        good_features = {
            "close": 43000,
            "ema_12": 43200, "ema_26": 42800,
            "adx_14": 32, "rsi_14": 52,
            "macd": 0.005, "macd_signal": 0.002,
            "atr_14": 500,
            "bb_upper": 44000, "bb_lower": 42000,
            "volume": 1500, "volume_sma": 1000,
        }
        context1 = _make_context(features=good_features)
        context1["now"] = now

        result1 = gate.consult(entry_signal, context1)
        assert result1.is_approved, f"Expected approved, got {result1.verdict}: {result1.reasoning}"

        # Step 2: Exit
        exit_signal = _make_signal(direction="flat")
        exit_result = gate.consult_exit(
            exit_signal, {"now": now + timedelta(minutes=10)}
        )
        assert exit_result.is_approved

        # Step 3: Immediate re-entry blocked by cooldown
        reentry_signal = _make_signal(direction="long", confidence=0.85)
        context3 = _make_context(features=good_features)
        context3["now"] = now + timedelta(minutes=10, seconds=30)

        result3 = gate.consult(reentry_signal, context3)
        assert result3.verdict == ConsensusVerdict.VETOED
        assert "cooldown" in result3.reasoning.lower()

        # Step 4: Re-entry succeeds after cooldown expires
        context4 = _make_context(features=good_features)
        context4["now"] = now + timedelta(minutes=15)  # > 120 seconds after exit

        result4 = gate.consult(reentry_signal, context4)
        assert "cooldown" not in result4.reasoning.lower()

        # Verify stats
        assert gate.consultations == 3
        assert gate.vetoes >= 1

    def test_multiple_symbols_independent_cooldowns(self):
        """Cooldowns are tracked per-symbol."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(
            message_bus=bus,
            participants=[RiskManagerDesk(min_exit_cooldown_seconds=300)],
            require_risk_approval=True,
        )
        now = datetime.now(timezone.utc)

        # Exit BTC
        gate.record_exit("BTC/USDT", when=now)

        # ETH entry should not be affected
        eth_signal = _make_signal(symbol="ETH/USDT", direction="long", confidence=0.8)
        ctx = _make_context()
        ctx["now"] = now + timedelta(seconds=30)

        result = gate.consult(eth_signal, ctx)
        # ETH should not be vetoed by BTC cooldown
        assert "cooldown" not in result.reasoning.lower() or "ETH" not in result.reasoning

    def test_short_signal_flow(self):
        """Short signals are evaluated correctly."""
        bus = ReasoningMessageBus()
        gate = ConsensusGate(message_bus=bus)
        signal = _make_signal(direction="short", confidence=0.7)
        # Features aligned for short: EMA12 < EMA26
        context = _make_context(features={
            "close": 43000,
            "ema_12": 42700, "ema_26": 43200,
            "adx_14": 28, "rsi_14": 72,
            "macd": -0.003, "macd_signal": 0.001,
            "atr_14": 500,
            "bb_upper": 44000, "bb_lower": 42000,
            "volume": 1400, "volume_sma": 1000,
            "stoch_k": 82, "stoch_d": 78,
        })

        result = gate.consult(signal, context)
        assert isinstance(result, ConsensusResult)
        # Short with aligned bearish features should be considered
        assert len(result.opinions) == 4
