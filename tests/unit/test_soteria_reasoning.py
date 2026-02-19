"""Tests for Soteria Agent Reasoning & Inter-Agent Messaging Layer.

Three fixture conversations:
1. Disagreement + Veto — SMC and CMT disagree, Risk vetoes
2. Risk Veto — clean signal, risk blocks on exposure
3. Post-Trade Debrief — successful trade with fill report and review
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from agentic_trading.reasoning.agent_message import (
    AgentMessage,
    AgentRole,
    MessageType,
    role_to_agent_type,
)
from agentic_trading.reasoning.soteria_trace import (
    SoteriaStep,
    SoteriaTrace,
    StepType,
)
from agentic_trading.reasoning.agent_conversation import (
    AgentConversation,
    ConversationOutcome,
)
from agentic_trading.reasoning.message_bus import ReasoningMessageBus
from agentic_trading.reasoning.reasoning_mixin import ReasoningMixin
from agentic_trading.reasoning.conversation_store import (
    InMemoryConversationStore,
    JsonFileConversationStore,
)
from agentic_trading.reasoning.system_prompts import (
    get_system_prompt,
    get_extended_thinking_config,
    REASONING_XML_SCHEMA,
)
from agentic_trading.core.enums import AgentType


# ===========================================================================
# Fixture builders
# ===========================================================================


def _ts(offset_secs: int = 0) -> datetime:
    """Helper: UTC timestamp with optional offset."""
    return datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset_secs)


def _make_disagreement_veto_conversation() -> AgentConversation:
    """Fixture 1: Disagreement + Veto.

    Market Structure → bullish, SMC → long signal at 42800,
    CMT → disagrees (RSI overbought), Risk → vetoes (exposure limit).
    """
    conv = AgentConversation(
        conversation_id="conv-disagree-veto-001",
        symbol="BTC/USDT",
        timeframe="15m",
        trigger_event="candle.BTC/USDT.15m",
        strategy_id="smc_ob_strategy",
        started_at=_ts(0),
    )

    # Message 1: Market Structure broadcast
    msg1 = AgentMessage(
        message_id="msg-001",
        conversation_id=conv.conversation_id,
        sender=AgentRole.MARKET_STRUCTURE,
        recipients=[],
        message_type=MessageType.ANALYSIS,
        content="Higher timeframe shows bullish structure. BOS confirmed on H4. "
                "Key resistance at 43,200, support at 42,500.",
        confidence=0.82,
        timestamp=_ts(1),
        structured_data={"bias": "bullish", "resistance": 43200, "support": 42500},
    )
    conv.add_message(msg1)

    # Message 2: SMC signal
    msg2 = AgentMessage(
        message_id="msg-002",
        conversation_id=conv.conversation_id,
        sender=AgentRole.SMC_ANALYST,
        recipients=[],
        message_type=MessageType.SIGNAL,
        content="Order block identified at 42,800. Entry: 42,800, SL: 42,400, "
                "TP: 43,600. R:R = 2.0. Bullish OB with FVG fill.",
        confidence=0.85,
        timestamp=_ts(2),
        structured_data={
            "direction": "long",
            "entry_price": 42800,
            "stop_loss": 42400,
            "take_profit": 43600,
            "risk_reward": 2.0,
        },
    )
    conv.add_message(msg2)

    # Message 3: CMT challenges
    msg3 = AgentMessage(
        message_id="msg-003",
        conversation_id=conv.conversation_id,
        sender=AgentRole.CMT_TECHNICIAN,
        recipients=[],
        message_type=MessageType.CHALLENGE,
        content="RSI at 78 — overbought territory. MACD histogram declining. "
                "Bollinger upper band touch. Technical divergence with SMC signal.",
        confidence=0.70,
        timestamp=_ts(3),
        references=["msg-002"],
        structured_data={"rsi": 78, "macd_histogram": "declining", "bb_touch": "upper"},
    )
    conv.add_message(msg3)

    # Message 4: SMC responds to challenge
    msg4 = AgentMessage(
        message_id="msg-004",
        conversation_id=conv.conversation_id,
        sender=AgentRole.SMC_ANALYST,
        recipients=[AgentRole.CMT_TECHNICIAN],
        message_type=MessageType.RESPONSE,
        content="Acknowledged overbought RSI. However, OB structure is strong and "
                "volume confirms institutional interest. Reducing confidence to 0.72.",
        confidence=0.72,
        timestamp=_ts(4),
        references=["msg-003"],
    )
    conv.add_message(msg4)

    # Message 5: Risk vetoes
    msg5 = AgentMessage(
        message_id="msg-005",
        conversation_id=conv.conversation_id,
        sender=AgentRole.RISK_MANAGER,
        recipients=[AgentRole.ORCHESTRATOR],
        message_type=MessageType.VETO,
        content="VETO: Current gross exposure at 2.8x (limit 3.0x). Adding BTC long "
                "would push to 3.4x. Also, daily drawdown at -4.2% (limit -5%). "
                "Insufficient margin of safety.",
        confidence=1.0,
        timestamp=_ts(5),
        structured_data={
            "gross_exposure": 2.8,
            "limit": 3.0,
            "projected_exposure": 3.4,
            "daily_drawdown_pct": -4.2,
            "drawdown_limit_pct": -5.0,
        },
    )
    conv.add_message(msg5)

    # Add traces
    trace1 = SoteriaTrace(
        trace_id="trace-mkt-001",
        conversation_id=conv.conversation_id,
        agent_role=AgentRole.MARKET_STRUCTURE,
        agent_id="mkt-struct-agent-001",
        symbol="BTC/USDT",
        trigger="candle.BTC/USDT.15m",
        started_at=_ts(0),
        completed_at=_ts(1),
        outcome="analysis_complete",
    )
    trace1.add_step(StepType.PERCEPTION, "H4 candle closed bullish, BOS confirmed", confidence=0.82)
    trace1.add_step(StepType.DECISION, "Bullish bias, key levels marked", confidence=0.82)
    conv.add_trace(trace1)

    trace2 = SoteriaTrace(
        trace_id="trace-smc-001",
        conversation_id=conv.conversation_id,
        agent_role=AgentRole.SMC_ANALYST,
        agent_id="smc-agent-001",
        symbol="BTC/USDT",
        trigger="market_structure.analysis",
        started_at=_ts(1),
        completed_at=_ts(2),
        outcome="signal_emitted",
    )
    trace2.add_step(StepType.PERCEPTION, "Scanned for OB + FVG on 15m", confidence=0.85)
    trace2.add_step(StepType.HYPOTHESIS, "Bullish OB at 42800 with FVG fill", confidence=0.85)
    trace2.add_step(StepType.DECISION, "Long signal: 42800 entry, 2.0 R:R", confidence=0.85)
    conv.add_trace(trace2)

    trace3 = SoteriaTrace(
        trace_id="trace-cmt-001",
        conversation_id=conv.conversation_id,
        agent_role=AgentRole.CMT_TECHNICIAN,
        agent_id="cmt-agent-001",
        symbol="BTC/USDT",
        trigger="smc.signal",
        started_at=_ts(2),
        completed_at=_ts(3),
        thinking_raw="Looking at RSI on 15m... 78 is quite high. MACD histogram "
                     "has been declining for 3 bars. This suggests momentum exhaustion.",
        outcome="challenge_issued",
    )
    trace3.add_step(StepType.PERCEPTION, "RSI 78, MACD declining, BB upper touch", confidence=0.70)
    trace3.add_step(StepType.EVALUATION, "Technical divergence with price action", confidence=0.70)
    trace3.add_step(StepType.VETO, "Challenging SMC signal on overbought conditions", confidence=0.70)
    conv.add_trace(trace3)

    trace4 = SoteriaTrace(
        trace_id="trace-risk-001",
        conversation_id=conv.conversation_id,
        agent_role=AgentRole.RISK_MANAGER,
        agent_id="risk-agent-001",
        symbol="BTC/USDT",
        trigger="smc.signal",
        started_at=_ts(4),
        completed_at=_ts(5),
        outcome="vetoed",
    )
    trace4.add_step(StepType.CONTEXT_LOAD, "Portfolio: 2.8x gross, -4.2% daily", confidence=1.0)
    trace4.add_step(StepType.EVALUATION, "Adding BTC long → 3.4x (exceeds 3.0x limit)", confidence=1.0)
    trace4.add_step(StepType.VETO, "VETO: exposure + drawdown limits insufficient margin", confidence=1.0)
    conv.add_trace(trace4)

    conv.finalize(ConversationOutcome.VETOED, {
        "reason": "exposure_limit",
        "gross_exposure": 2.8,
        "projected": 3.4,
        "limit": 3.0,
    })

    return conv


def _make_risk_veto_conversation() -> AgentConversation:
    """Fixture 2: Risk Veto — clean signal, risk blocks on exposure."""
    conv = AgentConversation(
        conversation_id="conv-risk-veto-002",
        symbol="ETH/USDT",
        timeframe="15m",
        trigger_event="candle.ETH/USDT.15m",
        strategy_id="smc_ob_strategy",
        started_at=_ts(0),
    )

    msg1 = AgentMessage(
        message_id="msg-r01",
        conversation_id=conv.conversation_id,
        sender=AgentRole.MARKET_STRUCTURE,
        recipients=[],
        message_type=MessageType.ANALYSIS,
        content="ETH showing bearish structure. CHoCH on H1. Support at 2,200.",
        confidence=0.78,
        timestamp=_ts(1),
    )
    conv.add_message(msg1)

    msg2 = AgentMessage(
        message_id="msg-r02",
        conversation_id=conv.conversation_id,
        sender=AgentRole.SMC_ANALYST,
        recipients=[],
        message_type=MessageType.SIGNAL,
        content="Short signal: bearish OB at 2,280. SL: 2,310, TP: 2,200. R:R 2.67.",
        confidence=0.80,
        timestamp=_ts(2),
        structured_data={"direction": "short", "entry_price": 2280, "stop_loss": 2310, "take_profit": 2200},
    )
    conv.add_message(msg2)

    msg3 = AgentMessage(
        message_id="msg-r03",
        conversation_id=conv.conversation_id,
        sender=AgentRole.CMT_TECHNICIAN,
        recipients=[],
        message_type=MessageType.ANALYSIS,
        content="Confirmed: RSI 32, MACD bearish crossover. Signal validated.",
        confidence=0.82,
        timestamp=_ts(3),
    )
    conv.add_message(msg3)

    msg4 = AgentMessage(
        message_id="msg-r04",
        conversation_id=conv.conversation_id,
        sender=AgentRole.RISK_MANAGER,
        recipients=[AgentRole.ORCHESTRATOR],
        message_type=MessageType.VETO,
        content="VETO: Kill switch active due to exchange API degradation.",
        confidence=1.0,
        timestamp=_ts(4),
        structured_data={"kill_switch_active": True, "reason": "api_degradation"},
    )
    conv.add_message(msg4)

    conv.finalize(ConversationOutcome.VETOED, {"reason": "kill_switch"})
    return conv


def _make_debrief_conversation() -> AgentConversation:
    """Fixture 3: Post-Trade Debrief — successful trade + review."""
    conv = AgentConversation(
        conversation_id="conv-debrief-003",
        symbol="BTC/USDT",
        timeframe="15m",
        trigger_event="candle.BTC/USDT.15m",
        strategy_id="smc_ob_strategy",
        started_at=_ts(0),
    )

    # Pre-trade messages
    msg1 = AgentMessage(
        message_id="msg-d01",
        conversation_id=conv.conversation_id,
        sender=AgentRole.MARKET_STRUCTURE,
        recipients=[],
        message_type=MessageType.ANALYSIS,
        content="Bullish BOS on H4. Strong trend continuation setup.",
        confidence=0.88,
        timestamp=_ts(1),
    )
    conv.add_message(msg1)

    msg2 = AgentMessage(
        message_id="msg-d02",
        conversation_id=conv.conversation_id,
        sender=AgentRole.SMC_ANALYST,
        recipients=[],
        message_type=MessageType.SIGNAL,
        content="Long at 42,500. SL: 42,100. TP: 43,300. R:R 2.0.",
        confidence=0.90,
        timestamp=_ts(2),
        structured_data={"direction": "long", "entry_price": 42500, "stop_loss": 42100, "take_profit": 43300},
    )
    conv.add_message(msg2)

    msg3 = AgentMessage(
        message_id="msg-d03",
        conversation_id=conv.conversation_id,
        sender=AgentRole.CMT_TECHNICIAN,
        recipients=[],
        message_type=MessageType.ANALYSIS,
        content="Confirmed: RSI 55, MACD bullish, volume increasing.",
        confidence=0.87,
        timestamp=_ts(3),
    )
    conv.add_message(msg3)

    msg4 = AgentMessage(
        message_id="msg-d04",
        conversation_id=conv.conversation_id,
        sender=AgentRole.RISK_MANAGER,
        recipients=[AgentRole.ORCHESTRATOR],
        message_type=MessageType.RISK_ASSESSMENT,
        content="APPROVED: Position within limits. Sized at 0.5 BTC.",
        confidence=0.95,
        timestamp=_ts(4),
        structured_data={"approved": True, "size": 0.5, "symbol": "BTC/USDT"},
    )
    conv.add_message(msg4)

    msg5 = AgentMessage(
        message_id="msg-d05",
        conversation_id=conv.conversation_id,
        sender=AgentRole.EXECUTION,
        recipients=[],
        message_type=MessageType.EXECUTION_PLAN,
        content="Market order for 0.5 BTC. Expected fill ~42,505.",
        confidence=0.90,
        timestamp=_ts(5),
    )
    conv.add_message(msg5)

    msg6 = AgentMessage(
        message_id="msg-d06",
        conversation_id=conv.conversation_id,
        sender=AgentRole.EXECUTION,
        recipients=[],
        message_type=MessageType.FILL_REPORT,
        content="Filled 0.5 BTC at 42,508. Slippage: 0.8bps. Fee: $1.27.",
        confidence=1.0,
        timestamp=_ts(6),
        structured_data={"fill_price": 42508, "slippage_bps": 0.8, "fee": 1.27},
    )
    conv.add_message(msg6)

    # Debrief message
    msg7 = AgentMessage(
        message_id="msg-d07",
        conversation_id=conv.conversation_id,
        sender=AgentRole.EXECUTION,
        recipients=[],
        message_type=MessageType.DEBRIEF,
        content="Trade executed successfully. Slippage within expectations. "
                "All agents aligned on bullish thesis.",
        confidence=1.0,
        timestamp=_ts(10),
    )
    conv.add_message(msg7)

    # Add traces
    trace1 = SoteriaTrace(
        trace_id="trace-d-mkt",
        conversation_id=conv.conversation_id,
        agent_role=AgentRole.MARKET_STRUCTURE,
        agent_id="mkt-agent-001",
        symbol="BTC/USDT",
        trigger="candle.BTC/USDT.15m",
        started_at=_ts(0),
        completed_at=_ts(1),
        outcome="analysis_complete",
    )
    trace1.add_step(StepType.PERCEPTION, "H4 BOS confirmed", confidence=0.88)
    trace1.add_step(StepType.DECISION, "Bullish context set", confidence=0.88)
    conv.add_trace(trace1)

    trace2 = SoteriaTrace(
        trace_id="trace-d-exec",
        conversation_id=conv.conversation_id,
        agent_role=AgentRole.EXECUTION,
        agent_id="exec-agent-001",
        symbol="BTC/USDT",
        trigger="risk.approved",
        started_at=_ts(5),
        completed_at=_ts(6),
        outcome="filled",
    )
    trace2.add_step(StepType.ACTION, "Market order submitted", confidence=0.90)
    trace2.add_step(StepType.OUTCOME, "Filled at 42,508, 0.8bps slippage", confidence=1.0)
    conv.add_trace(trace2)

    conv.finalize(ConversationOutcome.TRADE_ENTERED, {
        "fill_price": 42508,
        "size": 0.5,
        "slippage_bps": 0.8,
    })
    return conv


# ===========================================================================
# AgentMessage tests
# ===========================================================================


class TestAgentMessage:
    def test_create_message(self):
        msg = AgentMessage(
            sender=AgentRole.SMC_ANALYST,
            message_type=MessageType.SIGNAL,
            content="Long signal at 42,800",
            confidence=0.85,
        )
        assert msg.sender == AgentRole.SMC_ANALYST
        assert msg.message_type == MessageType.SIGNAL
        assert msg.confidence == 0.85
        assert msg.message_id  # auto-generated

    def test_broadcast_detection(self):
        msg = AgentMessage(
            sender=AgentRole.MARKET_STRUCTURE,
            recipients=[],
            message_type=MessageType.ANALYSIS,
        )
        assert msg.is_broadcast is True

        msg2 = AgentMessage(
            sender=AgentRole.RISK_MANAGER,
            recipients=[AgentRole.ORCHESTRATOR],
            message_type=MessageType.VETO,
        )
        assert msg2.is_broadcast is False

    def test_veto_detection(self):
        msg = AgentMessage(
            sender=AgentRole.RISK_MANAGER,
            message_type=MessageType.VETO,
            content="VETO: exposure limit",
        )
        assert msg.is_veto is True

        msg2 = AgentMessage(
            sender=AgentRole.SMC_ANALYST,
            message_type=MessageType.SIGNAL,
        )
        assert msg2.is_veto is False

    def test_challenge_detection(self):
        msg = AgentMessage(
            sender=AgentRole.CMT_TECHNICIAN,
            message_type=MessageType.CHALLENGE,
        )
        assert msg.is_challenge is True

        msg2 = AgentMessage(
            sender=AgentRole.CMT_TECHNICIAN,
            message_type=MessageType.DISAGREEMENT,
        )
        assert msg2.is_challenge is True

    def test_short_summary(self):
        msg = AgentMessage(
            sender=AgentRole.SMC_ANALYST,
            recipients=[],
            message_type=MessageType.SIGNAL,
            content="Long signal at 42,800 with bullish order block",
            confidence=0.85,
        )
        summary = msg.short_summary()
        assert "SMC Analyst" in summary
        assert "ALL" in summary
        assert "85%" in summary
        assert "signal" in summary

    def test_role_to_agent_type_mapping(self):
        assert role_to_agent_type(AgentRole.CMT_TECHNICIAN) == AgentType.CMT_ANALYST
        assert role_to_agent_type(AgentRole.RISK_MANAGER) == AgentType.RISK_GATE
        assert role_to_agent_type(AgentRole.EXECUTION) == AgentType.EXECUTION
        assert role_to_agent_type(AgentRole.MARKET_STRUCTURE) == AgentType.MARKET_INTELLIGENCE

    def test_agent_role_display_names(self):
        assert AgentRole.CMT_TECHNICIAN.display_name == "CMT Technician"
        assert AgentRole.RISK_MANAGER.display_name == "Risk Manager"
        assert AgentRole.ORCHESTRATOR.display_name == "Orchestrator"

    def test_message_type_values(self):
        assert MessageType.MORNING_BRIEF.value == "morning_brief"
        assert MessageType.VETO.value == "veto"
        assert MessageType.DEBRIEF.value == "debrief"
        assert MessageType.DISAGREEMENT.value == "disagreement"


# ===========================================================================
# SoteriaTrace tests
# ===========================================================================


class TestSoteriaTrace:
    def test_create_trace(self):
        trace = SoteriaTrace(
            agent_role=AgentRole.SMC_ANALYST,
            agent_id="smc-001",
            symbol="BTC/USDT",
            trigger="candle.BTC/USDT.15m",
        )
        assert trace.agent_role == AgentRole.SMC_ANALYST
        assert trace.symbol == "BTC/USDT"
        assert trace.trace_id  # auto-generated
        assert trace.steps == []

    def test_add_step(self):
        trace = SoteriaTrace(agent_role=AgentRole.CMT_TECHNICIAN)
        step = trace.add_step(
            StepType.PERCEPTION,
            "RSI at 78, overbought",
            confidence=0.70,
            evidence={"rsi": 78},
        )
        assert step.step_type == StepType.PERCEPTION
        assert step.confidence == 0.70
        assert len(trace.steps) == 1

    def test_complete_trace(self):
        trace = SoteriaTrace(agent_role=AgentRole.RISK_MANAGER)
        trace.add_step(StepType.EVALUATION, "Checking limits")
        trace.complete("vetoed", {"reason": "exposure"})
        assert trace.outcome == "vetoed"
        assert trace.completed_at is not None
        assert trace.final_output == {"reason": "exposure"}

    def test_duration_ms(self):
        trace = SoteriaTrace(
            agent_role=AgentRole.EXECUTION,
            started_at=_ts(0),
        )
        trace.completed_at = _ts(2)  # 2 seconds later
        assert trace.duration_ms == 2000.0

    def test_format_trace(self):
        trace = SoteriaTrace(
            agent_role=AgentRole.SMC_ANALYST,
            agent_id="smc-analyst-12345678",
            symbol="BTC/USDT",
            trigger="candle.BTC/USDT.15m",
            started_at=_ts(0),
        )
        trace.add_step(StepType.PERCEPTION, "OB scan complete", confidence=0.85)
        trace.add_step(StepType.DECISION, "Long signal at 42800", confidence=0.85)
        trace.complete("signal_emitted")

        text = trace.format_trace()
        assert "SMC Analyst" in text
        assert "BTC/USDT" in text
        assert "signal_emitted" in text
        assert "Perception" in text or "perception" in text.lower()

    def test_step_types_all_present(self):
        """Verify all 9 StepType values exist."""
        assert len(StepType) == 9
        expected = {
            "perception", "context_load", "hypothesis", "evaluation",
            "decision", "action", "handoff", "veto", "outcome",
        }
        actual = {s.value for s in StepType}
        assert actual == expected

    def test_step_with_context_and_messages(self):
        trace = SoteriaTrace(agent_role=AgentRole.RISK_MANAGER)
        step = trace.add_step(
            StepType.CONTEXT_LOAD,
            "Loaded portfolio state",
            context_used={"gross_exposure": 2.8},
            messages_sent=["msg-001", "msg-002"],
        )
        assert step.context_used == {"gross_exposure": 2.8}
        assert step.messages_sent == ["msg-001", "msg-002"]

    def test_extended_thinking_capture(self):
        trace = SoteriaTrace(
            agent_role=AgentRole.CMT_TECHNICIAN,
            thinking_raw="Deep analysis: RSI divergence with price...",
        )
        assert "RSI divergence" in trace.thinking_raw
        text = trace.format_trace()
        assert "extended_thinking" in text


# ===========================================================================
# AgentConversation tests
# ===========================================================================


class TestAgentConversation:
    def test_create_conversation(self):
        conv = AgentConversation(
            symbol="BTC/USDT",
            timeframe="15m",
            trigger_event="candle.BTC/USDT.15m",
            strategy_id="smc_ob_strategy",
        )
        assert conv.symbol == "BTC/USDT"
        assert conv.conversation_id  # auto-generated
        assert conv.messages == []
        assert conv.traces == []

    def test_add_message_sets_conversation_id(self):
        conv = AgentConversation(conversation_id="conv-test")
        msg = AgentMessage(
            sender=AgentRole.SMC_ANALYST,
            message_type=MessageType.SIGNAL,
        )
        conv.add_message(msg)
        assert msg.conversation_id == "conv-test"
        assert len(conv.messages) == 1

    def test_has_veto(self):
        conv = _make_disagreement_veto_conversation()
        assert conv.has_veto is True

    def test_has_disagreement(self):
        conv = _make_disagreement_veto_conversation()
        assert conv.has_disagreement is True

    def test_no_disagreement(self):
        conv = _make_risk_veto_conversation()
        assert conv.has_disagreement is False
        assert conv.has_veto is True

    def test_participating_roles(self):
        conv = _make_disagreement_veto_conversation()
        roles = conv.participating_roles
        assert AgentRole.MARKET_STRUCTURE in roles
        assert AgentRole.SMC_ANALYST in roles
        assert AgentRole.CMT_TECHNICIAN in roles
        assert AgentRole.RISK_MANAGER in roles

    def test_finalize(self):
        conv = AgentConversation(started_at=_ts(0))
        conv.finalize(ConversationOutcome.TRADE_ENTERED, {"fill_price": 42508})
        assert conv.outcome == ConversationOutcome.TRADE_ENTERED
        assert conv.completed_at is not None
        assert conv.outcome_details["fill_price"] == 42508

    def test_duration_ms(self):
        conv = AgentConversation(started_at=_ts(0))
        conv.completed_at = _ts(5)
        assert conv.duration_ms == 5000.0


class TestDeskConversationRendering:
    def test_desk_rendering_disagreement_veto(self):
        conv = _make_disagreement_veto_conversation()
        text = conv.print_desk_conversation()

        assert "DESK CONVERSATION" in text
        assert "BTC/USDT" in text
        assert "Market Structure" in text
        assert "SMC Analyst" in text
        assert "CMT Technician" in text
        assert "Risk Manager" in text
        assert "VETO" in text
        assert "VETOED" in text
        assert "Messages: 5" in text

    def test_desk_rendering_debrief(self):
        conv = _make_debrief_conversation()
        text = conv.print_desk_conversation()

        assert "TRADE_ENTERED" in text
        assert "FILL_REPORT" in text
        assert "DEBRIEF" in text
        assert "Messages: 7" in text

    def test_desk_rendering_shows_structured_data(self):
        conv = _make_disagreement_veto_conversation()
        text = conv.print_desk_conversation()
        # Structured data highlights should appear
        assert "📊" in text


class TestChainOfThoughtRendering:
    def test_chain_rendering(self):
        conv = _make_disagreement_veto_conversation()
        text = conv.print_chain_of_thought()

        assert "CHAIN OF THOUGHT" in text
        assert "BTC/USDT" in text
        assert "Market Structure" in text
        assert "SMC Analyst" in text
        assert "Perception" in text or "perception" in text.lower()

    def test_chain_shows_extended_thinking(self):
        conv = _make_disagreement_veto_conversation()
        text = conv.print_chain_of_thought()
        # CMT trace has thinking_raw
        assert "extended_thinking" in text


class TestExplainRendering:
    def test_explain_vetoed(self):
        conv = _make_disagreement_veto_conversation()
        text = conv.explain()

        assert "BTC/USDT" in text
        assert "15m" in text
        assert "Vetoed" in text
        assert "Risk Manager" in text
        assert "disagreed" in text

    def test_explain_trade_entered(self):
        conv = _make_debrief_conversation()
        text = conv.explain()

        assert "BTC/USDT" in text
        assert "Trade Entered" in text

    def test_explain_risk_veto(self):
        conv = _make_risk_veto_conversation()
        text = conv.explain()

        assert "ETH/USDT" in text
        assert "Vetoed" in text


# ===========================================================================
# ReasoningMessageBus tests
# ===========================================================================


class TestReasoningMessageBus:
    def test_subscribe_and_post(self):
        bus = ReasoningMessageBus()
        received: list[AgentMessage] = []

        bus.subscribe(AgentRole.ORCHESTRATOR, lambda m: received.append(m))

        msg = AgentMessage(
            conversation_id="conv-test",
            sender=AgentRole.RISK_MANAGER,
            recipients=[AgentRole.ORCHESTRATOR],
            message_type=MessageType.VETO,
            content="VETO: exposure",
        )
        delivered = bus.post(msg)

        assert delivered == 1
        assert len(received) == 1
        assert received[0].content == "VETO: exposure"

    def test_broadcast_delivers_to_all(self):
        bus = ReasoningMessageBus()
        received_smc: list[AgentMessage] = []
        received_cmt: list[AgentMessage] = []
        received_risk: list[AgentMessage] = []

        bus.subscribe(AgentRole.SMC_ANALYST, lambda m: received_smc.append(m))
        bus.subscribe(AgentRole.CMT_TECHNICIAN, lambda m: received_cmt.append(m))
        bus.subscribe(AgentRole.RISK_MANAGER, lambda m: received_risk.append(m))

        msg = AgentMessage(
            conversation_id="conv-test",
            sender=AgentRole.MARKET_STRUCTURE,
            recipients=[],  # broadcast
            message_type=MessageType.ANALYSIS,
            content="Bullish structure",
        )
        delivered = bus.post(msg)

        assert delivered == 3
        assert len(received_smc) == 1
        assert len(received_cmt) == 1
        assert len(received_risk) == 1

    def test_broadcast_skips_sender(self):
        bus = ReasoningMessageBus()
        received: list[AgentMessage] = []

        bus.subscribe(AgentRole.MARKET_STRUCTURE, lambda m: received.append(m))

        msg = AgentMessage(
            conversation_id="conv-test",
            sender=AgentRole.MARKET_STRUCTURE,
            recipients=[],
            message_type=MessageType.ANALYSIS,
        )
        bus.post(msg)

        # Should not deliver to sender
        assert len(received) == 0

    def test_create_conversation(self):
        bus = ReasoningMessageBus()
        conv = bus.create_conversation(
            symbol="BTC/USDT",
            timeframe="15m",
            trigger_event="candle.BTC/USDT.15m",
        )
        assert conv.symbol == "BTC/USDT"
        assert bus.conversation_count == 1

        # Retrieve
        loaded = bus.get_conversation(conv.conversation_id)
        assert loaded is not None
        assert loaded.symbol == "BTC/USDT"

    def test_message_recorded_in_conversation(self):
        bus = ReasoningMessageBus()
        conv = bus.create_conversation(symbol="BTC/USDT")

        msg = AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.SMC_ANALYST,
            message_type=MessageType.SIGNAL,
            content="Long at 42,800",
        )
        bus.post(msg)

        loaded = bus.get_conversation(conv.conversation_id)
        assert loaded is not None
        assert len(loaded.messages) == 1

    def test_get_thread_by_role(self):
        bus = ReasoningMessageBus()
        conv = bus.create_conversation(symbol="BTC/USDT")

        # Post messages from different senders
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.SMC_ANALYST,
            recipients=[],
            message_type=MessageType.SIGNAL,
        ))
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.RISK_MANAGER,
            recipients=[AgentRole.ORCHESTRATOR],
            message_type=MessageType.VETO,
        ))

        # Thread for ORCHESTRATOR should include targeted message + broadcast
        thread = bus.get_thread(conv.conversation_id, AgentRole.ORCHESTRATOR)
        assert len(thread) == 2  # broadcast + directed

        # Thread for RISK_MANAGER should only include broadcast (sent by them)
        thread_risk = bus.get_thread(conv.conversation_id, AgentRole.RISK_MANAGER)
        # Includes: broadcast from SMC (risk is not sender) + veto from risk (risk is sender)
        assert len(thread_risk) == 2

    def test_finalize_conversation(self):
        bus = ReasoningMessageBus()
        conv = bus.create_conversation(symbol="ETH/USDT")

        result = bus.finalize_conversation(
            conv.conversation_id,
            ConversationOutcome.VETOED,
            {"reason": "kill_switch"},
        )
        assert result is not None
        assert result.outcome == ConversationOutcome.VETOED
        assert result.completed_at is not None

    def test_list_conversations_with_filters(self):
        bus = ReasoningMessageBus()
        conv1 = bus.create_conversation(symbol="BTC/USDT")
        bus.finalize_conversation(conv1.conversation_id, ConversationOutcome.TRADE_ENTERED)

        conv2 = bus.create_conversation(symbol="ETH/USDT")
        bus.finalize_conversation(conv2.conversation_id, ConversationOutcome.VETOED)

        conv3 = bus.create_conversation(symbol="BTC/USDT")
        bus.finalize_conversation(conv3.conversation_id, ConversationOutcome.VETOED)

        btc = bus.list_conversations(symbol="BTC/USDT")
        assert len(btc) == 2

        vetoed = bus.list_conversations(outcome=ConversationOutcome.VETOED)
        assert len(vetoed) == 2

    def test_conversation_eviction(self):
        bus = ReasoningMessageBus(max_conversations=3)
        for i in range(5):
            bus.create_conversation(symbol=f"SYM{i}")

        assert bus.conversation_count == 3

    def test_stats(self):
        bus = ReasoningMessageBus()
        bus.subscribe(AgentRole.ORCHESTRATOR, lambda m: None)

        conv = bus.create_conversation(symbol="BTC/USDT")
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.RISK_MANAGER,
            recipients=[AgentRole.ORCHESTRATOR],
            message_type=MessageType.VETO,
        ))

        assert bus.messages_posted == 1
        assert bus.messages_delivered == 1
        assert bus.subscriber_count == 1

    def test_unsubscribe(self):
        bus = ReasoningMessageBus()
        received: list[AgentMessage] = []
        handler = lambda m: received.append(m)

        bus.subscribe(AgentRole.ORCHESTRATOR, handler)
        bus.unsubscribe(AgentRole.ORCHESTRATOR, handler)

        bus.post(AgentMessage(
            conversation_id="c1",
            sender=AgentRole.RISK_MANAGER,
            recipients=[AgentRole.ORCHESTRATOR],
            message_type=MessageType.VETO,
        ))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_post_async(self):
        bus = ReasoningMessageBus()
        received: list[AgentMessage] = []

        async def async_handler(msg: AgentMessage) -> None:
            received.append(msg)

        bus.subscribe_async(AgentRole.ORCHESTRATOR, async_handler)

        conv = bus.create_conversation(symbol="BTC/USDT")
        msg = AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.RISK_MANAGER,
            recipients=[AgentRole.ORCHESTRATOR],
            message_type=MessageType.VETO,
        )
        await bus.post_async(msg)

        assert len(received) == 1

    def test_handler_error_does_not_crash(self):
        bus = ReasoningMessageBus()

        def bad_handler(msg: AgentMessage) -> None:
            raise ValueError("handler crash")

        bus.subscribe(AgentRole.ORCHESTRATOR, bad_handler)

        msg = AgentMessage(
            conversation_id="c1",
            sender=AgentRole.RISK_MANAGER,
            recipients=[AgentRole.ORCHESTRATOR],
            message_type=MessageType.VETO,
        )
        # Should not raise
        delivered = bus.post(msg)
        assert delivered == 0  # Handler errored, not counted as delivered

    def test_clear(self):
        bus = ReasoningMessageBus()
        bus.subscribe(AgentRole.ORCHESTRATOR, lambda m: None)
        bus.create_conversation(symbol="BTC/USDT")
        bus.clear()

        assert bus.conversation_count == 0
        assert bus.subscriber_count == 0
        assert bus.messages_posted == 0


# ===========================================================================
# ReasoningMixin tests
# ===========================================================================


class _MockAgent(ReasoningMixin):
    """Mock agent that uses the ReasoningMixin."""

    def __init__(self) -> None:
        ReasoningMixin.__init__(self, role=AgentRole.SMC_ANALYST)
        self._agent_id = "mock-agent-001"


class TestReasoningMixin:
    def test_mixin_properties(self):
        agent = _MockAgent()
        assert agent.reasoning_role == AgentRole.SMC_ANALYST
        assert agent.reasoning_bus is None
        assert agent.has_reasoning is False

    def test_set_reasoning_bus(self):
        agent = _MockAgent()
        bus = ReasoningMessageBus()
        agent.set_reasoning_bus(bus)
        assert agent.has_reasoning is True

    def test_begin_and_end_reasoning(self):
        agent = _MockAgent()
        bus = ReasoningMessageBus()
        agent.set_reasoning_bus(bus)
        conv = bus.create_conversation(symbol="BTC/USDT")

        trace = agent.begin_reasoning(
            conversation_id=conv.conversation_id,
            symbol="BTC/USDT",
            trigger="candle.BTC/USDT.15m",
        )
        assert trace.agent_role == AgentRole.SMC_ANALYST
        assert trace.symbol == "BTC/USDT"

        agent.add_reasoning_step(StepType.PERCEPTION, "OB scan complete", confidence=0.85)
        agent.add_reasoning_step(StepType.DECISION, "Long signal", confidence=0.85)

        result = agent.end_reasoning("signal_emitted")
        assert result is not None
        assert result.outcome == "signal_emitted"
        assert len(result.steps) == 2

    def test_post_message_without_bus_returns_none(self):
        agent = _MockAgent()
        result = agent.post_message(MessageType.SIGNAL, "test")
        assert result is None

    def test_post_message_with_bus(self):
        agent = _MockAgent()
        bus = ReasoningMessageBus()
        agent.set_reasoning_bus(bus)
        conv = bus.create_conversation(symbol="BTC/USDT")
        agent.begin_reasoning(conversation_id=conv.conversation_id, symbol="BTC/USDT")

        msg = agent.post_analysis("OB identified", confidence=0.85)
        assert msg is not None
        assert msg.sender == AgentRole.SMC_ANALYST
        assert msg.message_type == MessageType.ANALYSIS

        # Message should be in conversation
        loaded = bus.get_conversation(conv.conversation_id)
        assert loaded is not None
        assert len(loaded.messages) == 1

    def test_post_veto(self):
        agent = _MockAgent()
        agent._reasoning_role = AgentRole.RISK_MANAGER
        bus = ReasoningMessageBus()
        agent.set_reasoning_bus(bus)
        conv = bus.create_conversation(symbol="BTC/USDT")
        agent.begin_reasoning(conversation_id=conv.conversation_id)

        msg = agent.post_veto("Exposure limit exceeded")
        assert msg is not None
        assert msg.is_veto
        assert msg.confidence == 1.0
        assert AgentRole.ORCHESTRATOR in msg.recipients

    def test_post_challenge(self):
        agent = _MockAgent()
        agent._reasoning_role = AgentRole.CMT_TECHNICIAN
        bus = ReasoningMessageBus()
        agent.set_reasoning_bus(bus)
        conv = bus.create_conversation(symbol="BTC/USDT")
        agent.begin_reasoning(conversation_id=conv.conversation_id)

        msg = agent.post_challenge("RSI overbought", confidence=0.70)
        assert msg is not None
        assert msg.message_type == MessageType.CHALLENGE
        assert msg.confidence == 0.70

    def test_message_tracked_in_trace(self):
        agent = _MockAgent()
        bus = ReasoningMessageBus()
        agent.set_reasoning_bus(bus)
        conv = bus.create_conversation(symbol="BTC/USDT")
        trace = agent.begin_reasoning(conversation_id=conv.conversation_id)

        msg = agent.post_signal("Long at 42,800", confidence=0.85)
        assert msg is not None
        assert msg.message_id in trace.messages_sent


# ===========================================================================
# ConversationStore tests
# ===========================================================================


class TestInMemoryConversationStore:
    def test_save_and_load(self):
        store = InMemoryConversationStore()
        conv = _make_disagreement_veto_conversation()
        store.save(conv)

        loaded = store.load(conv.conversation_id)
        assert loaded is not None
        assert loaded.symbol == "BTC/USDT"

    def test_explain(self):
        store = InMemoryConversationStore()
        conv = _make_disagreement_veto_conversation()
        store.save(conv)

        text = store.explain(conv.conversation_id)
        assert "BTC/USDT" in text
        assert "Vetoed" in text

    def test_explain_not_found(self):
        store = InMemoryConversationStore()
        text = store.explain("nonexistent")
        assert "not found" in text

    def test_find_by_symbol(self):
        store = InMemoryConversationStore()
        store.save(_make_disagreement_veto_conversation())
        store.save(_make_risk_veto_conversation())
        store.save(_make_debrief_conversation())

        btc = store.find_by_symbol("BTC/USDT")
        assert len(btc) == 2

        eth = store.find_by_symbol("ETH/USDT")
        assert len(eth) == 1

    def test_find_vetoed(self):
        store = InMemoryConversationStore()
        store.save(_make_disagreement_veto_conversation())
        store.save(_make_risk_veto_conversation())
        store.save(_make_debrief_conversation())

        vetoed = store.find_vetoed()
        assert len(vetoed) == 2

        vetoed_btc = store.find_vetoed("BTC/USDT")
        assert len(vetoed_btc) == 1

    def test_find_disagreements(self):
        store = InMemoryConversationStore()
        store.save(_make_disagreement_veto_conversation())
        store.save(_make_risk_veto_conversation())
        store.save(_make_debrief_conversation())

        disagreements = store.find_disagreements()
        assert len(disagreements) == 1
        assert disagreements[0].conversation_id == "conv-disagree-veto-001"

    def test_find_by_outcome(self):
        store = InMemoryConversationStore()
        store.save(_make_disagreement_veto_conversation())
        store.save(_make_debrief_conversation())

        entered = store.find_by_outcome(ConversationOutcome.TRADE_ENTERED)
        assert len(entered) == 1

        vetoed = store.find_by_outcome(ConversationOutcome.VETOED)
        assert len(vetoed) == 1

    def test_replay(self):
        store = InMemoryConversationStore()
        conv = _make_debrief_conversation()
        store.save(conv)

        replayed = store.replay(conv.conversation_id)
        assert replayed is not None
        assert len(replayed.messages) == 7
        assert replayed.outcome == ConversationOutcome.TRADE_ENTERED

    def test_eviction(self):
        store = InMemoryConversationStore(max_entries=2)
        store.save(_make_disagreement_veto_conversation())
        store.save(_make_risk_veto_conversation())
        store.save(_make_debrief_conversation())

        assert store.count == 2
        # First one should be evicted
        assert store.load("conv-disagree-veto-001") is None

    def test_query_multi_filter(self):
        store = InMemoryConversationStore()
        store.save(_make_disagreement_veto_conversation())
        store.save(_make_risk_veto_conversation())
        store.save(_make_debrief_conversation())

        results = store.query(
            symbol="BTC/USDT",
            outcome=ConversationOutcome.VETOED,
        )
        assert len(results) == 1
        assert results[0].conversation_id == "conv-disagree-veto-001"

    def test_clear(self):
        store = InMemoryConversationStore()
        store.save(_make_disagreement_veto_conversation())
        store.clear()
        assert store.count == 0


class TestJsonFileConversationStore:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "conversations.jsonl")
        store = JsonFileConversationStore(path)

        conv = _make_disagreement_veto_conversation()
        store.save(conv)

        loaded = store.load(conv.conversation_id)
        assert loaded is not None
        assert loaded.symbol == "BTC/USDT"

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "conversations.jsonl")

        store1 = JsonFileConversationStore(path)
        store1.save(_make_disagreement_veto_conversation())

        # New instance loads from file
        store2 = JsonFileConversationStore(path)
        loaded = store2.load("conv-disagree-veto-001")
        assert loaded is not None
        assert loaded.symbol == "BTC/USDT"

    def test_query_after_reload(self, tmp_path):
        path = str(tmp_path / "conversations.jsonl")

        store1 = JsonFileConversationStore(path)
        store1.save(_make_disagreement_veto_conversation())
        store1.save(_make_debrief_conversation())

        store2 = JsonFileConversationStore(path)
        vetoed = store2.find_vetoed()
        assert len(vetoed) == 1


# ===========================================================================
# SystemPrompt tests
# ===========================================================================


class TestSystemPrompts:
    def test_get_prompt_for_each_role(self):
        for role in AgentRole:
            prompt = get_system_prompt(role)
            assert len(prompt) > 100
            assert "<reasoning>" in prompt
            assert "<perception" in prompt
            assert "<decision" in prompt

    def test_prompt_includes_xml_schema(self):
        prompt = get_system_prompt(AgentRole.CMT_TECHNICIAN)
        assert "<reasoning>" in prompt
        assert "<handoff" in prompt
        assert "<evaluation>" in prompt

    def test_prompt_with_symbol_context(self):
        prompt = get_system_prompt(
            AgentRole.SMC_ANALYST,
            symbol="BTC/USDT",
            timeframe="15m",
        )
        assert "BTC/USDT" in prompt
        assert "15m" in prompt

    def test_cmt_prompt_mentions_extended_thinking(self):
        prompt = get_system_prompt(AgentRole.CMT_TECHNICIAN)
        assert "extended thinking" in prompt

    def test_risk_prompt_mentions_veto(self):
        prompt = get_system_prompt(AgentRole.RISK_MANAGER)
        assert "VETO" in prompt
        assert "veto" in prompt.lower()

    def test_extended_thinking_config(self):
        config = get_extended_thinking_config()
        assert config["type"] == "enabled"
        assert config["budget_tokens"] == 8000

    def test_market_structure_prompt_mentions_first(self):
        prompt = get_system_prompt(AgentRole.MARKET_STRUCTURE)
        assert "FIRST" in prompt


# ===========================================================================
# Integration: Full conversation lifecycle through bus
# ===========================================================================


class TestConversationLifecycle:
    """End-to-end test: create conversation, post messages, finalize, store, query."""

    def test_full_disagreement_veto_lifecycle(self):
        bus = ReasoningMessageBus()
        store = InMemoryConversationStore()

        # Agents subscribe
        risk_received: list[AgentMessage] = []
        bus.subscribe(AgentRole.RISK_MANAGER, lambda m: risk_received.append(m))

        orch_received: list[AgentMessage] = []
        bus.subscribe(AgentRole.ORCHESTRATOR, lambda m: orch_received.append(m))

        # Create conversation
        conv = bus.create_conversation(
            symbol="BTC/USDT",
            timeframe="15m",
            trigger_event="candle.BTC/USDT.15m",
            strategy_id="smc_ob_strategy",
        )

        # Market Structure broadcasts
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.MARKET_STRUCTURE,
            recipients=[],
            message_type=MessageType.ANALYSIS,
            content="Bullish structure on H4",
            confidence=0.82,
        ))

        # SMC broadcasts signal
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.SMC_ANALYST,
            recipients=[],
            message_type=MessageType.SIGNAL,
            content="Long at 42,800",
            confidence=0.85,
        ))

        # CMT challenges
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.CMT_TECHNICIAN,
            recipients=[],
            message_type=MessageType.CHALLENGE,
            content="RSI overbought at 78",
            confidence=0.70,
        ))

        # Risk vetoes
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.RISK_MANAGER,
            recipients=[AgentRole.ORCHESTRATOR],
            message_type=MessageType.VETO,
            content="Exposure limit exceeded",
            confidence=1.0,
        ))

        # Finalize
        result = bus.finalize_conversation(
            conv.conversation_id,
            ConversationOutcome.VETOED,
            {"reason": "exposure_limit"},
        )

        assert result is not None
        assert result.outcome == ConversationOutcome.VETOED
        assert result.has_veto is True
        assert result.has_disagreement is True
        assert len(result.messages) == 4

        # Store and query
        store.save(result)
        vetoed = store.find_vetoed("BTC/USDT")
        assert len(vetoed) == 1

        disagreements = store.find_disagreements()
        assert len(disagreements) == 1

        # Risk received broadcasts (not veto — that was targeted)
        assert len(risk_received) == 3  # analysis + signal + challenge

        # Orchestrator received veto + broadcasts
        assert len(orch_received) == 4  # analysis + signal + challenge + veto

    def test_full_debrief_lifecycle(self):
        bus = ReasoningMessageBus()
        store = InMemoryConversationStore()

        conv = bus.create_conversation(
            symbol="BTC/USDT",
            timeframe="15m",
            strategy_id="smc_ob_strategy",
        )

        # Analysis → Signal → Confirmation → Risk OK → Execute → Fill → Debrief
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.MARKET_STRUCTURE,
            recipients=[],
            message_type=MessageType.ANALYSIS,
            content="Bullish",
            confidence=0.88,
        ))
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.SMC_ANALYST,
            recipients=[],
            message_type=MessageType.SIGNAL,
            content="Long at 42,500",
            confidence=0.90,
        ))
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.RISK_MANAGER,
            recipients=[AgentRole.ORCHESTRATOR],
            message_type=MessageType.RISK_ASSESSMENT,
            content="Approved, 0.5 BTC",
            confidence=0.95,
        ))
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.EXECUTION,
            recipients=[],
            message_type=MessageType.FILL_REPORT,
            content="Filled at 42,508",
            confidence=1.0,
            structured_data={"fill_price": 42508},
        ))
        bus.post(AgentMessage(
            conversation_id=conv.conversation_id,
            sender=AgentRole.EXECUTION,
            recipients=[],
            message_type=MessageType.DEBRIEF,
            content="Trade executed successfully",
            confidence=1.0,
        ))

        result = bus.finalize_conversation(
            conv.conversation_id,
            ConversationOutcome.TRADE_ENTERED,
            {"fill_price": 42508},
        )

        store.save(result)

        # Queries
        entered = store.find_by_outcome(ConversationOutcome.TRADE_ENTERED)
        assert len(entered) == 1

        assert not result.has_veto
        assert not result.has_disagreement
        assert len(result.messages) == 5

        # Explain
        explanation = store.explain(result.conversation_id)
        assert "BTC/USDT" in explanation

    def test_serialization_roundtrip(self):
        """Verify AgentConversation survives JSON serialization."""
        conv = _make_disagreement_veto_conversation()
        json_str = conv.to_json()
        assert len(json_str) > 500

        # Roundtrip
        loaded = AgentConversation.model_validate_json(json_str)
        assert loaded.conversation_id == conv.conversation_id
        assert loaded.outcome == conv.outcome
        assert len(loaded.messages) == len(conv.messages)
        assert len(loaded.traces) == len(conv.traces)

        # Verify nested objects survived
        assert loaded.messages[0].sender == AgentRole.MARKET_STRUCTURE
        assert loaded.traces[0].agent_role == AgentRole.MARKET_STRUCTURE

    def test_rendering_all_fixtures(self):
        """Verify all three fixtures render without errors."""
        for conv in [
            _make_disagreement_veto_conversation(),
            _make_risk_veto_conversation(),
            _make_debrief_conversation(),
        ]:
            # All rendering methods should succeed
            desk = conv.print_desk_conversation()
            assert len(desk) > 100

            chain = conv.print_chain_of_thought()
            assert len(chain) > 50

            explanation = conv.explain()
            assert len(explanation) > 50
