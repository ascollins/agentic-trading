"""Top-level Orchestrator — single bootstrap entry point.

Composes the six layer managers (:class:`IntelligenceManager`,
:class:`SignalManager`, :class:`BusManager`, :class:`ExecutionGateway`,
:class:`PolicyGate`, :class:`ReconciliationManager`), the clock, the
:class:`TradingContext`, the :class:`ContextManager`, and the
:class:`PipelineLog` into a single facade that replaces the
procedural wiring in ``main.py``.

Usage::

    from agentic_trading.orchestrator import Orchestrator

    orch = Orchestrator.from_config(settings)
    await orch.start()
    # ... trading runs ...
    await orch.stop()

    # Replay any historical pipeline decision
    print(orch.explain("pipeline-id-here"))

Backtest shortcut (no event bus, no feeds)::

    orch = Orchestrator.from_config(settings)
    # access sub-managers directly
    candles = orch.intelligence.load_candles(...)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_trading.core.config import Settings
from agentic_trading.core.enums import Mode
from agentic_trading.core.interfaces import PortfolioState, TradingContext

logger = logging.getLogger(__name__)


class Orchestrator:
    """Top-level facade that wires all layer managers together.

    Owns six layer managers, the clock, the ``TradingContext``,
    the ``ContextManager`` (fact table + memory store), and the
    ``PipelineLog`` for reasoning chain persistence.

    Provides a ``from_config()`` factory for construction, a clean
    lifecycle (start / stop), accessor properties for each layer,
    and ``explain()`` / ``get_pipeline_history()`` for decision replay.

    Parameters
    ----------
    settings:
        Application settings.
    ctx:
        The ``TradingContext`` shared across layers.
    bus:
        BusManager for topic-routed and domain-event messaging.
    intelligence:
        IntelligenceManager for data ingestion and feature computation.
    signal:
        SignalManager for strategy dispatch and portfolio sizing.
    execution:
        ExecutionGateway for order submission and risk management.
        May be ``None`` in backtest mode (backtest engine owns execution).
    policy:
        PolicyGate for governance and policy evaluation.
        May be ``None`` when governance is disabled.
    reconciliation:
        ReconciliationManager for trade journaling and exchange reconciliation.
    context_manager:
        ContextManager for fact table and memory store.
    pipeline_log:
        PipelineLog for persisting pipeline results.
    fact_sync:
        FactTableEventSync for auto-updating the fact table from
        the event bus (paper/live only).
    """

    def __init__(
        self,
        settings: Settings,
        ctx: TradingContext,
        bus: Any,
        intelligence: Any,
        signal: Any,
        execution: Any | None = None,
        policy: Any | None = None,
        reconciliation: Any | None = None,
        cmt_agent: Any | None = None,
        context_manager: Any | None = None,
        pipeline_log: Any | None = None,
        fact_sync: Any | None = None,
    ) -> None:
        self._settings = settings
        self._ctx = ctx
        self._bus = bus
        self._intelligence = intelligence
        self._signal = signal
        self._execution = execution
        self._policy = policy
        self._reconciliation = reconciliation
        self._cmt_agent = cmt_agent
        self._context_manager = context_manager
        self._pipeline_log = pipeline_log
        self._fact_sync = fact_sync

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        settings: Settings,
        *,
        adapter: Any | None = None,
        tool_gateway: Any | None = None,
    ) -> Orchestrator:
        """Build a fully wired Orchestrator from application settings.

        Creates the clock, event bus, TradingContext, ContextManager,
        PipelineLog, and all six layer managers, wiring them together
        according to the trading mode.

        Parameters
        ----------
        settings:
            Application settings (``Settings``).
        adapter:
            Optional pre-built exchange adapter.  When ``None``, an
            adapter is created only for paper/live modes.
        tool_gateway:
            Optional ``ToolGateway`` for institutional control-plane
            side effects.

        Returns
        -------
        Orchestrator
        """
        mode = settings.mode

        # --- Clock ---
        from agentic_trading.core.clock import SimClock, WallClock

        if mode == Mode.BACKTEST:
            clock = SimClock()
        else:
            clock = WallClock()

        # --- Bus layer ---
        from agentic_trading.bus.manager import BusManager

        bus = BusManager.from_config(
            mode=mode,
            redis_url=settings.redis_url,
        )

        # --- TradingContext ---
        ctx = TradingContext(
            clock=clock,
            event_bus=bus.legacy_bus,
            instruments={},
            portfolio_state=PortfolioState(),
            risk_limits=settings.risk.model_dump(),
        )

        # --- Context Manager ---
        from agentic_trading.context.manager import ContextManager

        context_mgr = ContextManager.from_config(
            mode=mode,
            data_dir=settings.data_dir,
            memory_ttl_hours=settings.context.memory_ttl_hours,
            max_memory_entries=settings.context.max_memory_entries,
            memory_store_path=(
                settings.context.memory_store_path
                if mode != Mode.BACKTEST
                else None
            ),
        )
        context_mgr.sync_from_trading_context(ctx)

        # --- Pipeline Log ---
        from agentic_trading.reasoning.pipeline_log import (
            InMemoryPipelineLog,
            PipelineLog,
        )

        pipeline_log: InMemoryPipelineLog | PipelineLog
        if mode == Mode.BACKTEST:
            pipeline_log = InMemoryPipelineLog()
        else:
            log_dir = Path(settings.context.pipeline_log_dir)
            pipeline_log = PipelineLog(log_dir / "pipelines.jsonl")

        # --- Fact Table Event Sync (paper/live only) ---
        fact_sync: Any | None = None
        if mode != Mode.BACKTEST:
            from agentic_trading.context.event_sync import FactTableEventSync

            fact_sync = FactTableEventSync(
                context_mgr.facts, bus.legacy_bus
            )

        # --- Intelligence layer ---
        from agentic_trading.intelligence.manager import IntelligenceManager

        intel_kwargs: dict[str, Any] = {
            "data_dir": settings.backtest.data_dir,
            "indicator_config": None,
        }
        if mode != Mode.BACKTEST:
            intel_kwargs["event_bus"] = bus.legacy_bus
            intel_kwargs["exchange_configs"] = (
                settings.exchanges if settings.exchanges else None
            )
            intel_kwargs["symbols"] = settings.symbols.symbols or None

        intelligence = IntelligenceManager.from_config(**intel_kwargs)

        # --- Signal layer ---
        from agentic_trading.signal.manager import SignalManager

        # Resolve governance sizing function (if governance enabled)
        governance_sizing_fn = None

        sizing_multiplier = 1.0
        if settings.safe_mode.enabled:
            sizing_multiplier = settings.safe_mode.position_size_multiplier

        signal_kwargs: dict[str, Any] = {
            "strategy_ids": [
                s.strategy_id
                for s in settings.strategies
                if s.enabled
            ] or None,
            "max_position_pct": settings.risk.max_single_position_pct,
            "sizing_multiplier": sizing_multiplier,
            "governance_sizing_fn": governance_sizing_fn,
        }
        if mode != Mode.BACKTEST:
            signal_kwargs["feature_engine"] = intelligence.feature_engine
            signal_kwargs["event_bus"] = bus.legacy_bus

        signal_mgr = SignalManager.from_config(**signal_kwargs)

        # --- Policy layer (optional) ---
        policy: Any | None = None
        if settings.governance.enabled:
            from agentic_trading.policy.gate import PolicyGate

            policy = PolicyGate.from_config(
                governance_config=settings.governance,
                risk_config=settings.risk,
                event_bus=bus.legacy_bus,
            )

            # Wire governance sizing into signal manager
            try:
                signal_mgr.set_sizing_multiplier(sizing_multiplier)
            except Exception:
                pass

        # --- Execution layer (paper/live only) ---
        execution: Any | None = None
        if mode != Mode.BACKTEST:
            from agentic_trading.execution.gateway import ExecutionGateway

            exec_kwargs: dict[str, Any] = {
                "adapter": adapter,
                "event_bus": bus.legacy_bus,
                "risk_config": settings.risk,
                "instruments": ctx.instruments,
                "portfolio_state_provider": lambda: ctx.portfolio_state,
                "tool_gateway": tool_gateway,
            }
            if policy is not None:
                exec_kwargs["governance_gate"] = policy.governance_gate

            # Only create if an adapter is available
            if adapter is not None:
                execution = ExecutionGateway.from_config(**exec_kwargs)

        # --- Reconciliation layer ---
        from agentic_trading.reconciliation.manager import (
            ReconciliationManager,
        )

        recon_kwargs: dict[str, Any] = {
            "max_closed_trades": 10_000,
            "rolling_window": 100,
        }
        if policy is not None:
            recon_kwargs["health_tracker"] = policy.health
            recon_kwargs["drift_detector"] = policy.drift

        if (
            mode != Mode.BACKTEST
            and adapter is not None
            and execution is not None
        ):
            recon_kwargs["adapter"] = adapter
            recon_kwargs["event_bus"] = bus.legacy_bus
            recon_kwargs["order_manager"] = execution.order_manager
            recon_kwargs["recon_interval"] = float(
                settings.risk.reconciliation_interval_seconds
            )

        reconciliation = ReconciliationManager.from_config(**recon_kwargs)

        # --- CMT Analyst Agent (optional) ---
        cmt_agent: Any | None = None
        if settings.cmt.enabled and mode != Mode.BACKTEST:
            try:
                from agentic_trading.agents.cmt_analyst import (
                    CMTAnalystAgent,
                )
                from agentic_trading.intelligence.analysis.cmt_engine import (
                    CMTAnalysisEngine,
                )

                cmt_engine = CMTAnalysisEngine(
                    skill_path=settings.cmt.skill_path,
                    api_key_env=settings.cmt.api_key_env,
                    model=settings.cmt.model,
                    max_daily_calls=settings.cmt.max_daily_api_calls,
                    min_confluence=settings.cmt.min_confluence_score,
                )
                cmt_agent = CMTAnalystAgent(
                    intelligence_manager=intelligence,
                    event_bus=bus.legacy_bus,
                    symbols=settings.symbols.symbols or [],
                    engine=cmt_engine,
                    config=settings.cmt,
                )
                # Inject context manager into CMT agent
                cmt_agent.set_context_manager(context_mgr)
                logger.info("CMT Analyst Agent created")
            except Exception:
                logger.exception("Failed to create CMT Analyst Agent")

        return cls(
            settings=settings,
            ctx=ctx,
            bus=bus,
            intelligence=intelligence,
            signal=signal_mgr,
            execution=execution,
            policy=policy,
            reconciliation=reconciliation,
            cmt_agent=cmt_agent,
            context_manager=context_mgr,
            pipeline_log=pipeline_log,
            fact_sync=fact_sync,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all layers in dependency order.

        Order: bus -> fact_sync -> intelligence -> execution ->
        reconciliation -> signal (which subscribes to feature.vector
        and begins dispatching to strategies).
        """
        await self._bus.start()

        if self._fact_sync is not None:
            await self._fact_sync.start()

        await self._intelligence.start()

        if self._execution is not None:
            await self._execution.start()

        if self._reconciliation is not None:
            await self._reconciliation.start()

        if self._cmt_agent is not None:
            await self._cmt_agent.start()

        if self._signal.runner is not None:
            await self._signal.start(self._ctx)

        logger.info(
            "Orchestrator started (mode=%s, layers=%s)",
            self._settings.mode.value,
            ", ".join(self._active_layer_names()),
        )

    async def stop(self) -> None:
        """Stop all layers in reverse dependency order."""
        if self._signal.runner is not None:
            # Signal runner has no explicit stop, but log for completeness
            pass

        if self._cmt_agent is not None:
            await self._cmt_agent.stop()

        if self._reconciliation is not None:
            await self._reconciliation.stop()

        if self._execution is not None:
            await self._execution.stop()

        await self._intelligence.stop()

        await self._bus.stop()

        logger.info("Orchestrator stopped")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        """Application settings."""
        return self._settings

    @property
    def ctx(self) -> TradingContext:
        """The TradingContext shared across layers."""
        return self._ctx

    @property
    def bus(self):
        """BusManager — event bus layer."""
        return self._bus

    @property
    def intelligence(self):
        """IntelligenceManager — data ingestion and feature computation."""
        return self._intelligence

    @property
    def signal(self):
        """SignalManager — strategy dispatch and portfolio sizing."""
        return self._signal

    @property
    def execution(self):
        """ExecutionGateway — order execution and risk (``None`` in backtest)."""
        return self._execution

    @property
    def policy(self):
        """PolicyGate — governance and policy (``None`` when disabled)."""
        return self._policy

    @property
    def reconciliation(self):
        """ReconciliationManager — trade journaling and reconciliation."""
        return self._reconciliation

    @property
    def cmt_agent(self):
        """CMTAnalystAgent — autonomous CMT analysis (``None`` when disabled)."""
        return self._cmt_agent

    @property
    def context_manager(self):
        """ContextManager — fact table and memory store."""
        return self._context_manager

    @property
    def pipeline_log(self):
        """PipelineLog — persistent pipeline result storage."""
        return self._pipeline_log

    @property
    def mode(self) -> Mode:
        """Current trading mode."""
        return self._settings.mode

    @property
    def is_backtest(self) -> bool:
        """Whether running in backtest mode."""
        return self._settings.mode == Mode.BACKTEST

    # ------------------------------------------------------------------
    # Pipeline Result & Reasoning
    # ------------------------------------------------------------------

    def explain(self, pipeline_id: str) -> str:
        """Replay a historical pipeline decision.

        Loads the pipeline result from the log and renders the full
        chain of thought as human-readable text.

        Parameters
        ----------
        pipeline_id:
            The ``pipeline_id`` of the pipeline to explain.

        Returns
        -------
        str
            Human-readable chain of thought, or a "not found" message.
        """
        if self._pipeline_log is None:
            return "Pipeline log not available"

        result = self._pipeline_log.load(pipeline_id)
        if result is None:
            return f"Pipeline {pipeline_id} not found"

        return result.print_chain_of_thought()

    def get_pipeline_history(
        self,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
        since: datetime | None = None,
        limit: int = 20,
    ) -> list:
        """Query past pipeline results.

        Parameters
        ----------
        symbol:
            Filter by trigger symbol.
        strategy:
            Filter by strategy ID in signals.
        since:
            Only return pipelines after this time.
        limit:
            Maximum number of results.

        Returns
        -------
        list[PipelineResult]
        """
        if self._pipeline_log is None:
            return []

        return self._pipeline_log.query(
            symbol=symbol,
            strategy=strategy,
            since=since,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Collect metrics from all layers into a single dict."""
        metrics: dict[str, Any] = {
            "mode": self._settings.mode.value,
        }

        # Bus metrics
        try:
            metrics["bus"] = self._bus.get_metrics()
        except Exception:
            pass

        # Signal metrics
        try:
            metrics["signal_count"] = self._signal.signal_count
            metrics["strategy_count"] = len(self._signal.strategies)
        except Exception:
            pass

        # Reconciliation metrics
        if self._reconciliation is not None:
            try:
                metrics["open_trades"] = len(
                    self._reconciliation.get_open_trades()
                )
                metrics["closed_trades"] = len(
                    self._reconciliation.get_closed_trades()
                )
            except Exception:
                pass

        # Pipeline log metrics
        if self._pipeline_log is not None:
            try:
                metrics["pipeline_count"] = self._pipeline_log.count
            except Exception:
                pass

        # Context manager metrics
        if self._context_manager is not None:
            try:
                metrics["memory_entries"] = (
                    self._context_manager.memory.entry_count
                )
            except Exception:
                pass

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _active_layer_names(self) -> list[str]:
        """Return names of active (non-None) layers."""
        names = ["bus", "intelligence", "signal"]
        if self._execution is not None:
            names.append("execution")
        if self._policy is not None:
            names.append("policy")
        if self._reconciliation is not None:
            names.append("reconciliation")
        if self._cmt_agent is not None:
            names.append("cmt_agent")
        if self._context_manager is not None:
            names.append("context")
        if self._pipeline_log is not None:
            names.append("pipeline_log")
        return names
