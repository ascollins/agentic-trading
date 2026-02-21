"""CMT Analyst Agent — autonomous technical analysis via Claude API.

Consumes platform-computed features (indicators, SMC patterns, HTF
assessments) and calls the Claude API with the CMT skill to produce
structured 9-layer CMT assessments. When confluence thresholds are met,
emits Signal events that flow through the existing governance -> sizing ->
execution pipeline.

Write ownership
~~~~~~~~~~~~~~~
This agent is the sole producer of:

*  ``CMTAssessment`` events on the ``intelligence.cmt`` topic
*  ``Signal`` events with ``strategy_id="cmt_analyst"``

Usage::

    agent = CMTAnalystAgent(
        intelligence_manager=intelligence_manager,
        event_bus=event_bus,
        symbols=["BTC/USDT", "ETH/USDT"],
        engine=cmt_engine,
        config=cmt_config,
    )
    await agent.start()
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.config import CMTConfig
from agentic_trading.core.enums import (
    AgentType,
    MemoryEntryType,
    ReasoningPhase,
    SignalDirection,
    Timeframe,
)
from agentic_trading.core.events import (
    AgentCapabilities,
    CMTAssessment,
    Signal,
)
from agentic_trading.core.interfaces import IEventBus
from agentic_trading.intelligence.analysis.cmt_engine import CMTAnalysisEngine
from agentic_trading.intelligence.analysis.cmt_models import (
    CMTAssessmentRequest,
    CMTAssessmentResponse,
)
from agentic_trading.reasoning.models import ReasoningTrace

logger = logging.getLogger(__name__)

_STRATEGY_ID = "cmt_analyst"


class CMTAnalystAgent(BaseAgent):
    """Periodic agent that runs CMT analysis for each symbol in the universe.

    Parameters
    ----------
    intelligence_manager:
        Platform ``IntelligenceManager`` for feature access.
    event_bus:
        Event bus for publishing CMTAssessment and Signal events.
    symbols:
        Trading universe symbols to analyze.
    engine:
        ``CMTAnalysisEngine`` instance (Claude API integration).
    config:
        ``CMTConfig`` with thresholds and scheduling.
    """

    def __init__(
        self,
        *,
        intelligence_manager: Any,  # IntelligenceManager (avoid circular import)
        event_bus: IEventBus,
        symbols: list[str],
        engine: CMTAnalysisEngine,
        config: CMTConfig,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            interval=config.analysis_interval_seconds,
        )
        self._im = intelligence_manager
        self._bus = event_bus
        self._symbols = symbols
        self._engine = engine
        self._config = config
        self._enable_thinking = True

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CMT_ANALYST

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=[],
            publishes_to=["intelligence.cmt", "strategy.signal"],
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        logger.info(
            "CMTAnalystAgent starting: symbols=%s, interval=%ds, "
            "model=%s, min_confluence=%d",
            self._symbols,
            self._config.analysis_interval_seconds,
            self._config.model,
            self._config.min_confluence_score,
        )

    # ------------------------------------------------------------------
    # Periodic work
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Run CMT analysis for each symbol in the universe."""
        for symbol in self._symbols:
            try:
                await self._analyze_symbol(symbol)
            except Exception:
                logger.exception("CMT analysis failed for %s", symbol)

    async def _analyze_symbol(self, symbol: str) -> ReasoningTrace | None:
        """Run the full CMT analysis pipeline for a single symbol.

        Returns the reasoning trace for the analysis (or None if
        reasoning capture is not available).
        """
        trace = self._start_reasoning(symbol=symbol)

        # 1. PERCEPTION: Gather platform data
        ctx = self._read_context(symbol=symbol)
        memory_count = (
            len(ctx.relevant_memories) if ctx is not None else 0
        )
        trace.add_step(
            ReasoningPhase.PERCEPTION,
            f"Gathering market data for {symbol} across "
            f"{len(self._config.timeframes)} timeframes, "
            f"{memory_count} relevant memories available",
        )

        request = self._build_request(symbol)
        if request is None:
            trace.add_step(
                ReasoningPhase.DECISION,
                "Insufficient data for CMT analysis — skipping",
                confidence=0.0,
            )
            trace.complete("skipped")
            logger.debug("Insufficient data for CMT analysis of %s", symbol)
            return trace

        trace.add_step(
            ReasoningPhase.PERCEPTION,
            f"Built request: {len(request.ohlcv_summary)} timeframes, "
            f"{len(request.indicator_values)} indicators",
            confidence=1.0,
            evidence={
                "timeframes": list(request.ohlcv_summary.keys()),
                "indicator_count": len(request.indicator_values),
            },
        )

        # 2. Call Claude API via engine (with or without extended thinking)
        response: CMTAssessmentResponse | None = None
        raw_thinking = ""

        if self._enable_thinking:
            response, raw_thinking = await self._engine.assess_with_thinking(
                request
            )
        else:
            response = await self._engine.assess(request)

        if response is None:
            trace.add_step(
                ReasoningPhase.EVALUATION,
                "API call failed or budget exhausted",
                confidence=0.0,
            )
            trace.complete("error")
            return trace

        # 3. HYPOTHESIS: Record what Claude thinks
        confidence = max(0.0, min(1.0, (response.confluence.total - 3) / 8))
        trace.add_step(
            ReasoningPhase.HYPOTHESIS,
            response.thesis[:300],
            confidence=confidence,
            evidence={
                "confluence_total": response.confluence.total,
                "system_health": response.system_health,
                "layer_count": len(response.layers),
            },
        )

        if raw_thinking:
            trace.raw_thinking = raw_thinking

        # 4. EVALUATION: Confluence check
        trace.add_step(
            ReasoningPhase.EVALUATION,
            f"Confluence score: {response.confluence.total:.1f}, "
            f"threshold_met={response.confluence.threshold_met}, "
            f"veto={response.confluence.veto}",
            confidence=1.0 if response.confluence.threshold_met else 0.3,
        )

        # 5. Publish CMTAssessment event
        assessment_event = CMTAssessment(
            symbol=symbol,
            timeframes_analyzed=response.timeframes_analyzed,
            layers=response.layer_dict(),
            confluence_score=response.confluence.model_dump(),
            trade_plan=(
                response.trade_plan.model_dump()
                if response.trade_plan
                else None
            ),
            thesis=response.thesis,
            system_health=response.system_health,
            raw_llm_response="",  # Omit raw response for storage efficiency
        )
        await self._bus.publish("intelligence.cmt", assessment_event)

        # 6. DECISION + ACTION
        if (
            response.confluence.threshold_met
            and response.trade_plan is not None
        ):
            plan = response.trade_plan
            trace.add_step(
                ReasoningPhase.DECISION,
                f"Emit {plan.direction} signal for {symbol}, "
                f"rr={plan.rr_ratio:.1f}",
                confidence=confidence,
            )
            await self._emit_signal(symbol, response)
            trace.add_step(
                ReasoningPhase.ACTION,
                "Signal published to strategy.signal",
            )
            trace.complete("signal_emitted")
        else:
            reason = (
                response.no_trade_reason
                or "Confluence below threshold"
            )
            trace.add_step(
                ReasoningPhase.DECISION,
                f"No trade: {reason}",
                confidence=1.0 - confidence,
            )
            trace.complete("no_signal")

        # 7. Write analysis to memory store
        self._write_analysis(
            entry_type=MemoryEntryType.CMT_ASSESSMENT,
            content={
                "thesis": response.thesis,
                "confluence_total": response.confluence.total,
                "threshold_met": response.confluence.threshold_met,
                "system_health": response.system_health,
                "layer_count": len(response.layers),
                "has_trade_plan": response.trade_plan is not None,
            },
            symbol=symbol,
            summary=response.thesis[:100],
            tags=["cmt", response.system_health],
        )

        return trace

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------

    def _build_request(self, symbol: str) -> CMTAssessmentRequest | None:
        """Gather platform data into a CMTAssessmentRequest."""
        timeframes = self._config.timeframes

        # Collect OHLCV summaries and indicator values across timeframes
        ohlcv_summary: dict[str, Any] = {}
        indicator_values: dict[str, float] = {}

        for tf_str in timeframes:
            try:
                tf = Timeframe(tf_str) if isinstance(tf_str, str) else tf_str
            except ValueError:
                continue

            buffer = self._im.get_buffer(symbol, tf)
            if not buffer:
                continue

            # OHLCV summary: last 5 candles for context
            recent = buffer[-5:] if len(buffer) >= 5 else buffer
            ohlcv_summary[tf_str] = [
                {
                    "t": str(getattr(c, "timestamp", "")),
                    "o": float(getattr(c, "open", 0)),
                    "h": float(getattr(c, "high", 0)),
                    "l": float(getattr(c, "low", 0)),
                    "c": float(getattr(c, "close", 0)),
                    "v": float(getattr(c, "volume", 0)),
                }
                for c in recent
            ]

        if not ohlcv_summary:
            return None  # No data available yet

        # Gather aligned features for HTF/SMC analysis
        htf_assessment: dict[str, Any] = {}
        smc_confluence: dict[str, Any] = {}

        try:
            # Use the feature engine to get the latest features
            fe = self._im.feature_engine
            latest_features: dict[str, float] = {}
            for tf_str in timeframes:
                try:
                    tf = (
                        Timeframe(tf_str)
                        if isinstance(tf_str, str)
                        else tf_str
                    )
                except ValueError:
                    continue
                buf = fe.get_buffer(symbol, tf)
                if buf:
                    fv = fe.compute_features(symbol, tf, buf)
                    if fv and hasattr(fv, "features"):
                        for k, v in fv.features.items():
                            indicator_values[f"{tf_str}_{k}"] = v
                            latest_features[f"{tf_str}_{k}"] = v
        except Exception:
            logger.debug(
                "Could not gather features for %s", symbol, exc_info=True
            )

        # HTF assessment
        try:
            if latest_features and hasattr(self._im, "analyze_htf"):
                htf = self._im.analyze_htf(symbol, latest_features)
                if htf is not None:
                    htf_assessment = (
                        htf.model_dump()
                        if hasattr(htf, "model_dump")
                        else {}
                    )
        except Exception:
            logger.debug(
                "HTF analysis unavailable for %s", symbol, exc_info=True
            )

        # SMC confluence
        try:
            if latest_features and hasattr(
                self._im, "score_smc_confluence"
            ):
                smc = self._im.score_smc_confluence(
                    symbol, latest_features
                )
                if smc is not None:
                    smc_confluence = (
                        smc.model_dump()
                        if hasattr(smc, "model_dump")
                        else {}
                    )
        except Exception:
            logger.debug(
                "SMC scoring unavailable for %s", symbol, exc_info=True
            )

        return CMTAssessmentRequest(
            symbol=symbol,
            timeframes=timeframes,
            ohlcv_summary=ohlcv_summary,
            indicator_values=indicator_values,
            htf_assessment=htf_assessment,
            smc_confluence=smc_confluence,
        )

    # ------------------------------------------------------------------
    # Signal emission
    # ------------------------------------------------------------------

    async def _emit_signal(
        self,
        symbol: str,
        response: CMTAssessmentResponse,
    ) -> None:
        """Emit a Signal event from a CMT trade plan.

        Performs pre-emission sanity checks on LLM-generated values
        before publishing.  If any check fails the signal is dropped
        with a warning — no trade is emitted.
        """
        plan = response.trade_plan
        if plan is None:
            return

        direction_map = {
            "LONG": SignalDirection.LONG,
            "SHORT": SignalDirection.SHORT,
        }
        direction = direction_map.get(plan.direction.upper())
        if direction is None:
            logger.warning(
                "Unknown CMT trade direction: %s", plan.direction
            )
            return

        # ── Pre-emission sanity gate (defense-in-depth) ──────────
        # The Pydantic validators on CMTTradePlan catch most issues at
        # parse time.  These checks guard against edge cases where a
        # valid-looking plan is still dangerous (e.g. stop == entry
        # after rounding, or zero-price target).
        stop_dec = Decimal(str(plan.stop_loss)) if plan.stop_loss else None
        tp_dec = (
            Decimal(str(plan.targets[0].price))
            if plan.targets
            else None
        )

        if stop_dec is not None and stop_dec <= 0:
            logger.warning(
                "Dropping CMT signal for %s: stop_loss=%s is non-positive",
                symbol, stop_dec,
            )
            return

        if tp_dec is not None and tp_dec <= 0:
            logger.warning(
                "Dropping CMT signal for %s: take_profit=%s is non-positive",
                symbol, tp_dec,
            )
            return

        if stop_dec is not None and tp_dec is not None:
            entry_dec = Decimal(str(plan.entry_price))
            if direction == SignalDirection.LONG:
                if stop_dec >= entry_dec or tp_dec <= entry_dec:
                    logger.warning(
                        "Dropping CMT LONG signal for %s: "
                        "stop=%s entry=%s tp=%s — invalid geometry",
                        symbol, stop_dec, entry_dec, tp_dec,
                    )
                    return
            elif direction == SignalDirection.SHORT:
                if stop_dec <= entry_dec or tp_dec >= entry_dec:
                    logger.warning(
                        "Dropping CMT SHORT signal for %s: "
                        "stop=%s entry=%s tp=%s — invalid geometry",
                        symbol, stop_dec, entry_dec, tp_dec,
                    )
                    return

        # Map confluence total to 0-1 confidence
        # Confluence range is -10 to +11, threshold is 5
        raw_conf = response.confluence.total
        confidence = max(0.0, min(1.0, (raw_conf - 3) / 8))

        signal = Signal(
            strategy_id=_STRATEGY_ID,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            rationale=response.thesis[:500],
            timeframe=Timeframe.H4,
            stop_loss=stop_dec,
            take_profit=tp_dec,
            risk_constraints={
                "cmt_confluence": response.confluence.total,
                "cmt_rr_ratio": plan.rr_ratio,
                "cmt_system_health": response.system_health,
            },
        )

        await self._bus.publish("strategy.signal", signal)
        logger.info(
            "CMT signal emitted: %s %s %s, confidence=%.2f, rr=%.1f",
            symbol,
            direction.value,
            plan.entry_trigger,
            confidence,
            plan.rr_ratio,
        )
