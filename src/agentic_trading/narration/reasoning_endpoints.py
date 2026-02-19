"""Reasoning & Chain-of-Thought HTTP endpoints for Grafana.

Endpoints:
  GET /reasoning/pipelines?limit=20&symbol=...    — Recent pipeline results (table)
  GET /reasoning/pipeline/{pipeline_id}            — Single pipeline chain of thought
  GET /reasoning/traces?limit=30&symbol=...        — Flattened reasoning traces (table)
  GET /reasoning/steps?limit=50&symbol=...         — Flattened reasoning steps (table)
  GET /reasoning/context                           — Current fact table snapshot
  GET /reasoning/memories?limit=20&symbol=...      — Recent memory entries
  GET /reasoning/stats                             — Aggregate stats for stat panels

All endpoints return JSON arrays or objects consumable by the
Grafana Infinity datasource plugin.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)


def register_reasoning_routes(app: web.Application) -> None:
    """Register all reasoning endpoints on an existing aiohttp app."""
    app.router.add_get("/reasoning/pipelines", handle_pipelines)
    app.router.add_get(
        "/reasoning/pipeline/{pipeline_id}", handle_pipeline_detail,
    )
    app.router.add_get("/reasoning/traces", handle_traces)
    app.router.add_get("/reasoning/steps", handle_steps)
    app.router.add_get("/reasoning/context", handle_context)
    app.router.add_get("/reasoning/memories", handle_memories)
    app.router.add_get("/reasoning/stats", handle_stats)
    app.router.add_get("/reasoning/consensus", handle_consensus_stats)
    app.router.add_get("/reasoning/conversations", handle_conversations)


# ---------------------------------------------------------------------------
# Pipeline list — table of recent pipeline runs
# ---------------------------------------------------------------------------

async def handle_pipelines(request: web.Request) -> web.Response:
    """GET /reasoning/pipelines — recent pipeline results as flat table rows.

    Query params:
      limit (int): Max rows (default 20)
      symbol (str): Filter by trigger symbol
    """
    pipeline_log = request.app.get("pipeline_log")
    if pipeline_log is None:
        return web.json_response([])

    limit = int(request.query.get("limit", "20"))
    symbol = request.query.get("symbol") or None

    results = pipeline_log.query(symbol=symbol, limit=limit)

    rows: list[dict[str, Any]] = []
    for r in results:
        # Count traces and signals for summary
        trace_count = len(r.reasoning_traces)
        signal_count = len(r.signals)
        agent_types = ", ".join(
            t.get("agent_type", "?") for t in r.reasoning_traces
        )

        rows.append({
            "pipeline_id": r.pipeline_id,
            "timestamp": r.started_at.isoformat(),
            "symbol": r.trigger_symbol,
            "timeframe": r.trigger_timeframe,
            "outcome": r.outcome.value,
            "duration_ms": round(r.duration_ms, 1),
            "agents": agent_types,
            "trace_count": trace_count,
            "signal_count": signal_count,
        })

    return web.json_response(rows)


# ---------------------------------------------------------------------------
# Pipeline detail — full chain of thought for one pipeline
# ---------------------------------------------------------------------------

async def handle_pipeline_detail(request: web.Request) -> web.Response:
    """GET /reasoning/pipeline/{pipeline_id} — full chain of thought.

    Returns a single object with the rendered chain of thought text
    and all structured trace data.
    """
    pipeline_log = request.app.get("pipeline_log")
    if pipeline_log is None:
        return web.json_response(
            {"error": "Pipeline log not available"}, status=404,
        )

    pipeline_id = request.match_info["pipeline_id"]
    result = pipeline_log.load(pipeline_id)
    if result is None:
        return web.json_response(
            {"error": f"Pipeline {pipeline_id} not found"}, status=404,
        )

    return web.json_response({
        "pipeline_id": result.pipeline_id,
        "chain_of_thought": result.print_chain_of_thought(),
        "timestamp": result.started_at.isoformat(),
        "symbol": result.trigger_symbol,
        "outcome": result.outcome.value,
        "duration_ms": round(result.duration_ms, 1),
        "traces": result.reasoning_traces,
        "signals": result.signals,
        "context_at_start": result.context_at_start,
        "context_at_end": result.context_at_end,
    })


# ---------------------------------------------------------------------------
# Traces — flattened trace table (one row per agent per pipeline)
# ---------------------------------------------------------------------------

async def handle_traces(request: web.Request) -> web.Response:
    """GET /reasoning/traces — flattened reasoning traces as table rows.

    Query params:
      limit (int): Max pipeline results to scan (default 30)
      symbol (str): Filter by trigger symbol
    """
    pipeline_log = request.app.get("pipeline_log")
    if pipeline_log is None:
        return web.json_response([])

    limit = int(request.query.get("limit", "30"))
    symbol = request.query.get("symbol") or None

    results = pipeline_log.query(symbol=symbol, limit=limit)

    rows: list[dict[str, Any]] = []
    for r in results:
        for trace in r.reasoning_traces:
            steps = trace.get("steps", [])
            # Extract the decision step content (if any)
            decision_text = ""
            max_confidence = 0.0
            for step in steps:
                conf = step.get("confidence", 0.0)
                if conf > max_confidence:
                    max_confidence = conf
                if step.get("phase") == "decision":
                    decision_text = step.get("content", "")

            # Build a short narrative from all phases
            phase_summary = " -> ".join(
                f"{s.get('phase', '?')}: {s.get('content', '')[:60]}"
                for s in steps
            )

            has_thinking = bool(trace.get("raw_thinking", ""))
            trace_started = trace.get("started_at", "")
            trace_completed = trace.get("completed_at", "")

            # Calculate trace duration
            duration = 0.0
            if trace_started and trace_completed:
                try:
                    t0 = datetime.fromisoformat(trace_started)
                    t1 = datetime.fromisoformat(trace_completed)
                    duration = (t1 - t0).total_seconds() * 1000
                except (ValueError, TypeError):
                    pass

            rows.append({
                "pipeline_id": r.pipeline_id,
                "pipeline_time": r.started_at.isoformat(),
                "agent_id": str(trace.get("agent_id", ""))[:8],
                "agent_type": trace.get("agent_type", "unknown"),
                "symbol": trace.get("symbol", r.trigger_symbol),
                "outcome": trace.get("outcome", ""),
                "decision": decision_text[:120],
                "confidence": round(max_confidence, 2),
                "step_count": len(steps),
                "duration_ms": round(duration, 1),
                "has_extended_thinking": has_thinking,
                "phase_summary": phase_summary[:300],
            })

    return web.json_response(rows)


# ---------------------------------------------------------------------------
# Steps — every reasoning step flattened (one row per step)
# ---------------------------------------------------------------------------

async def handle_steps(request: web.Request) -> web.Response:
    """GET /reasoning/steps — every reasoning step as a flat table row.

    Query params:
      limit (int): Max pipeline results to scan (default 20)
      symbol (str): Filter by trigger symbol
      phase (str): Filter by reasoning phase
    """
    pipeline_log = request.app.get("pipeline_log")
    if pipeline_log is None:
        return web.json_response([])

    limit = int(request.query.get("limit", "20"))
    symbol = request.query.get("symbol") or None
    phase_filter = request.query.get("phase") or None

    results = pipeline_log.query(symbol=symbol, limit=limit)

    rows: list[dict[str, Any]] = []
    for r in results:
        for trace in r.reasoning_traces:
            agent_type = trace.get("agent_type", "unknown")
            agent_id = str(trace.get("agent_id", ""))[:8]
            trace_symbol = trace.get("symbol", r.trigger_symbol)

            for step in trace.get("steps", []):
                phase = step.get("phase", "")
                if phase_filter and phase != phase_filter:
                    continue

                rows.append({
                    "pipeline_id": r.pipeline_id,
                    "pipeline_time": r.started_at.isoformat(),
                    "agent_type": agent_type,
                    "agent_id": agent_id,
                    "symbol": trace_symbol,
                    "phase": phase,
                    "content": step.get("content", ""),
                    "confidence": step.get("confidence", 0.0),
                    "timestamp": step.get("timestamp", ""),
                    "evidence": str(step.get("evidence", {}))[:200],
                })

    return web.json_response(rows)


# ---------------------------------------------------------------------------
# Context — current fact table snapshot
# ---------------------------------------------------------------------------

async def handle_context(request: web.Request) -> web.Response:
    """GET /reasoning/context — current fact table as JSON.

    Returns portfolio, risk, prices, and regime state.
    """
    context_manager = request.app.get("context_manager")
    if context_manager is None:
        return web.json_response({"error": "Context not available"})

    facts = context_manager.facts
    snapshot = facts.snapshot()

    # Build a flat structure Grafana can consume
    # Regimes is a dict[symbol, dict] — summarise into a single string
    regime_summary = "unknown"
    if snapshot.regimes:
        # Use the first symbol's regime or join all
        regime_labels = []
        for sym, rdata in snapshot.regimes.items():
            label = rdata.get("regime", rdata.get("label", str(rdata)))
            regime_labels.append(f"{sym}: {label}" if len(snapshot.regimes) > 1 else str(label))
        regime_summary = ", ".join(regime_labels) if regime_labels else "unknown"
    data: dict[str, Any] = {
        "regime": regime_summary,
    }

    # Portfolio
    p = snapshot.portfolio
    data["equity"] = float(p.total_equity)
    data["gross_exposure"] = float(p.gross_exposure)
    data["net_exposure"] = float(p.net_exposure)
    data["daily_pnl"] = float(p.daily_pnl)
    data["open_positions"] = p.open_position_count

    # Risk
    r = snapshot.risk
    data["kill_switch"] = r.kill_switch_active
    data["max_leverage"] = float(r.max_portfolio_leverage)
    data["drawdown_pct"] = float(r.current_drawdown_pct)
    data["degraded_mode"] = r.degraded_mode
    data["circuit_breakers"] = ", ".join(r.circuit_breakers_tripped) or "none"

    # Prices
    price_rows: list[dict[str, Any]] = []
    for sym, levels in snapshot.prices.items():
        price_rows.append({
            "symbol": sym,
            "bid": float(levels.bid),
            "ask": float(levels.ask),
            "last": float(levels.last),
            "funding_rate": float(levels.funding_rate),
            "updated_at": levels.updated_at.isoformat() if levels.updated_at else "",
        })
    data["prices"] = price_rows

    # Positions
    position_rows: list[dict[str, Any]] = []
    for sym, pos in p.positions.items():
        position_rows.append({
            "symbol": sym,
            "qty": float(pos.get("qty", 0)),
            "entry_price": float(pos.get("entry_price", 0)),
        })
    data["positions"] = position_rows

    return web.json_response(data)


# ---------------------------------------------------------------------------
# Memories — recent memory store entries
# ---------------------------------------------------------------------------

async def handle_memories(request: web.Request) -> web.Response:
    """GET /reasoning/memories — recent memory entries as table rows.

    Query params:
      limit (int): Max rows (default 20)
      symbol (str): Filter by symbol
      type (str): Filter by entry type
    """
    context_manager = request.app.get("context_manager")
    if context_manager is None:
        return web.json_response([])

    limit = int(request.query.get("limit", "20"))
    symbol = request.query.get("symbol") or None
    entry_type = request.query.get("type") or None

    # Import MemoryEntryType for filtering
    entry_type_enum = None
    if entry_type:
        try:
            from agentic_trading.core.enums import MemoryEntryType
            entry_type_enum = MemoryEntryType(entry_type)
        except (ValueError, KeyError):
            pass

    memories = context_manager.memory.query(
        symbol=symbol,
        entry_type=entry_type_enum,
        limit=limit,
    )

    rows: list[dict[str, Any]] = []
    for m in memories:
        rows.append({
            "entry_id": m.entry_id,
            "timestamp": m.timestamp.isoformat(),
            "entry_type": m.entry_type.value if hasattr(m.entry_type, "value") else str(m.entry_type),
            "symbol": m.symbol,
            "timeframe": m.timeframe,
            "strategy_id": m.strategy_id,
            "summary": m.summary,
            "relevance": round(m.relevance_score, 3),
            "tags": ", ".join(m.tags),
            "ttl_hours": m.ttl_hours,
        })

    return web.json_response(rows)


# ---------------------------------------------------------------------------
# Stats — aggregate stats for Grafana stat/gauge panels
# ---------------------------------------------------------------------------

async def handle_stats(request: web.Request) -> web.Response:
    """GET /reasoning/stats — aggregate reasoning stats.

    Returns counts and distributions for stat panels.
    """
    pipeline_log = request.app.get("pipeline_log")
    context_manager = request.app.get("context_manager")

    stats: dict[str, Any] = {
        "total_pipelines": 0,
        "total_memories": 0,
        "signals_emitted": 0,
        "no_signals": 0,
        "errors": 0,
        "avg_confidence": 0.0,
        "avg_duration_ms": 0.0,
        "agents_active": 0,
    }

    if pipeline_log is not None:
        stats["total_pipelines"] = pipeline_log.count

        # Scan recent pipelines for stats
        recent = pipeline_log.query(limit=100)
        outcome_counts: dict[str, int] = {}
        confidence_sum = 0.0
        confidence_count = 0
        duration_sum = 0.0
        duration_count = 0
        agent_types_seen: set[str] = set()

        for r in recent:
            oc = r.outcome.value
            outcome_counts[oc] = outcome_counts.get(oc, 0) + 1

            if r.duration_ms > 0:
                duration_sum += r.duration_ms
                duration_count += 1

            for trace in r.reasoning_traces:
                agent_types_seen.add(trace.get("agent_type", "unknown"))
                for step in trace.get("steps", []):
                    c = step.get("confidence", 0.0)
                    if c > 0:
                        confidence_sum += c
                        confidence_count += 1

        stats["signals_emitted"] = outcome_counts.get("signal_emitted", 0)
        stats["no_signals"] = outcome_counts.get("no_signal", 0)
        stats["errors"] = outcome_counts.get("error", 0)
        stats["agents_active"] = len(agent_types_seen)

        if confidence_count > 0:
            stats["avg_confidence"] = round(
                confidence_sum / confidence_count, 3,
            )
        if duration_count > 0:
            stats["avg_duration_ms"] = round(
                duration_sum / duration_count, 1,
            )

    if context_manager is not None:
        try:
            stats["total_memories"] = context_manager.memory.entry_count
        except AttributeError:
            pass

    return web.json_response(stats)


# ---------------------------------------------------------------------------
# Consensus — consensus gate stats and recent conversations
# ---------------------------------------------------------------------------

async def handle_consensus_stats(request: web.Request) -> web.Response:
    """GET /reasoning/consensus — consensus gate statistics.

    Returns approval/rejection/veto counts and approval rate.
    """
    consensus_gate = request.app.get("consensus_gate")
    reasoning_bus = request.app.get("reasoning_bus")

    stats: dict[str, Any] = {
        "total_consultations": 0,
        "approvals": 0,
        "rejections": 0,
        "vetoes": 0,
        "approval_rate": 0.0,
        "total_conversations": 0,
        "messages_posted": 0,
        "messages_delivered": 0,
    }

    if consensus_gate is not None:
        stats["total_consultations"] = consensus_gate.consultations
        stats["approvals"] = consensus_gate.approvals
        stats["rejections"] = consensus_gate.rejections
        stats["vetoes"] = consensus_gate.vetoes
        stats["approval_rate"] = round(consensus_gate.approval_rate, 3)

    if reasoning_bus is not None:
        stats["total_conversations"] = reasoning_bus.conversation_count
        stats["messages_posted"] = reasoning_bus.messages_posted
        stats["messages_delivered"] = reasoning_bus.messages_delivered

    return web.json_response(stats)


async def handle_conversations(request: web.Request) -> web.Response:
    """GET /reasoning/conversations — recent desk conversations.

    Query params:
      limit (int): Max rows (default 20)
      symbol (str): Filter by symbol
      outcome (str): Filter by outcome
    """
    reasoning_bus = request.app.get("reasoning_bus")
    if reasoning_bus is None:
        return web.json_response([])

    limit = int(request.query.get("limit", "20"))
    symbol = request.query.get("symbol") or None

    outcome_filter = None
    outcome_str = request.query.get("outcome")
    if outcome_str:
        try:
            from agentic_trading.reasoning.agent_conversation import (
                ConversationOutcome,
            )
            outcome_filter = ConversationOutcome(outcome_str)
        except (ValueError, KeyError):
            pass

    conversations = reasoning_bus.list_conversations(
        symbol=symbol,
        outcome=outcome_filter,
        limit=limit,
    )

    rows: list[dict[str, Any]] = []
    for conv in conversations:
        roles = [r.value for r in conv.participating_roles]
        rows.append({
            "conversation_id": conv.conversation_id[:12],
            "timestamp": conv.started_at.isoformat(),
            "symbol": conv.symbol,
            "timeframe": conv.timeframe,
            "strategy_id": conv.strategy_id,
            "trigger": conv.trigger_event,
            "outcome": conv.outcome.value,
            "message_count": len(conv.messages),
            "has_veto": conv.has_veto,
            "has_disagreement": conv.has_disagreement,
            "duration_ms": round(conv.duration_ms, 1),
            "participants": ", ".join(roles),
        })

    return web.json_response(rows)
