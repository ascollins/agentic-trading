"""ReportingAgent — periodic report generation (design spec §7.2).

Runs on a configurable interval (default: daily) and compiles:
    - PnL summary from the trade journal
    - Risk summary (current exposure, drawdown)
    - Reconciliation status
    - Active incident / surveillance case counts
    - Overall system health

Publishes a :class:`DailyReportEvent` on the ``reporting`` topic.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentCapabilities, BaseEvent
from agentic_trading.core.ids import new_id, utc_now
from agentic_trading.core.interfaces import IEventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report event
# ---------------------------------------------------------------------------


class DailyReportEvent(BaseEvent):
    """Daily aggregated report published by the ReportingAgent."""

    source_module: str = "reporting"
    report_id: str = ""
    report_date: str = ""  # ISO date string (YYYY-MM-DD)

    pnl_summary: dict[str, Any] = {}
    risk_summary: dict[str, Any] = {}
    recon_summary: dict[str, Any] = {}
    surveillance_summary: dict[str, Any] = {}
    system_health: dict[str, Any] = {}

    strategies_active: int = 0
    open_positions: int = 0
    total_fills_today: int = 0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ReportingAgent(BaseAgent):
    """Periodic reporting agent that compiles system-wide summaries.

    Parameters
    ----------
    event_bus:
        Event bus for publishing DailyReportEvent.
    journal:
        Optional trade journal for PnL data.
    risk_manager:
        Optional risk manager for exposure data.
    case_manager:
        Optional CaseManager for surveillance case counts.
    agent_registry:
        Optional AgentRegistry for system health.
    interval:
        Seconds between report cycles (default: 86400 = daily).
    agent_id:
        Optional agent identifier.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        *,
        journal: Any = None,
        risk_manager: Any = None,
        case_manager: Any = None,
        agent_registry: Any = None,
        interval: float = 86400.0,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id or "reporting", interval=interval)
        self._event_bus = event_bus
        self._journal = journal
        self._risk_manager = risk_manager
        self._case_manager = case_manager
        self._agent_registry = agent_registry
        self._reports_generated: int = 0

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.REPORTING

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=[],
            publishes_to=["reporting"],
            description="Periodic system-wide report generation",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _on_start(self) -> None:
        logger.info("ReportingAgent started (interval=%ds)", self._interval)

    async def _on_stop(self) -> None:
        logger.info(
            "ReportingAgent stopped (reports_generated=%d)",
            self._reports_generated,
        )

    # ------------------------------------------------------------------
    # Work cycle
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Generate and publish a daily report."""
        report = self._compile_report()
        await self._event_bus.publish("reporting", report)
        self._reports_generated += 1
        logger.info(
            "Daily report published: report_id=%s date=%s",
            report.report_id, report.report_date,
        )

    def _compile_report(self) -> DailyReportEvent:
        """Compile data from all available sources."""
        now = utc_now()

        return DailyReportEvent(
            report_id=new_id(),
            report_date=now.strftime("%Y-%m-%d"),
            pnl_summary=self._collect_pnl(),
            risk_summary=self._collect_risk(),
            recon_summary=self._collect_recon(),
            surveillance_summary=self._collect_surveillance(),
            system_health=self._collect_health(),
            strategies_active=self._count_strategies(),
            open_positions=self._count_positions(),
            total_fills_today=self._count_fills(),
        )

    # ------------------------------------------------------------------
    # Data collectors (gracefully degrade if source unavailable)
    # ------------------------------------------------------------------

    def _collect_pnl(self) -> dict[str, Any]:
        if self._journal is None:
            return {"available": False}
        try:
            closed = getattr(self._journal, "closed_trades", [])
            if callable(closed):
                closed = closed()
            total_pnl = sum(
                float(getattr(t, "realized_pnl", 0)) for t in closed
            )
            return {
                "available": True,
                "total_closed_trades": len(closed),
                "total_realized_pnl": total_pnl,
            }
        except Exception:
            logger.debug("Failed to collect PnL", exc_info=True)
            return {"available": False, "error": "collection_failed"}

    def _collect_risk(self) -> dict[str, Any]:
        if self._risk_manager is None:
            return {"available": False}
        try:
            state = {}
            if hasattr(self._risk_manager, "get_risk_state"):
                raw = self._risk_manager.get_risk_state()
                state = raw if isinstance(raw, dict) else {"state": str(raw)}
            return {"available": True, **state}
        except Exception:
            logger.debug("Failed to collect risk", exc_info=True)
            return {"available": False, "error": "collection_failed"}

    def _collect_recon(self) -> dict[str, Any]:
        return {"available": False}

    def _collect_surveillance(self) -> dict[str, Any]:
        if self._case_manager is None:
            return {"available": False}
        try:
            return {
                "available": True,
                "total_cases": self._case_manager.total_cases,
                "open_cases": self._case_manager.open_count,
            }
        except Exception:
            logger.debug("Failed to collect surveillance", exc_info=True)
            return {"available": False, "error": "collection_failed"}

    def _collect_health(self) -> dict[str, Any]:
        if self._agent_registry is None:
            return {"available": False}
        try:
            health = self._agent_registry.health_check_all()
            unhealthy = [
                aid for aid, h in health.items()
                if not getattr(h, "healthy", True)
            ]
            return {
                "available": True,
                "total_agents": len(health),
                "unhealthy_agents": unhealthy,
            }
        except Exception:
            logger.debug("Failed to collect health", exc_info=True)
            return {"available": False, "error": "collection_failed"}

    def _count_strategies(self) -> int:
        if self._journal is None:
            return 0
        try:
            trades = getattr(self._journal, "open_trades", [])
            if callable(trades):
                trades = trades()
            return len({getattr(t, "strategy_id", "") for t in trades})
        except Exception:
            return 0

    def _count_positions(self) -> int:
        if self._journal is None:
            return 0
        try:
            trades = getattr(self._journal, "open_trades", [])
            if callable(trades):
                trades = trades()
            return len(trades)
        except Exception:
            return 0

    def _count_fills(self) -> int:
        if self._journal is None:
            return 0
        try:
            return getattr(self._journal, "fill_count_today", 0)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def reports_generated(self) -> int:
        return self._reports_generated
