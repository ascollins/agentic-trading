"""RiskManager facade.

Orchestrates all risk sub-systems (pre-trade checks, post-trade checks,
circuit breakers, kill switch, drawdown monitor, exposure tracker, alert
engine) behind a single unified interface.

The :class:`RiskManager` is the only risk-module class that the
execution engine and portfolio manager need to interact with.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agentic_trading.core.config import RiskConfig
from agentic_trading.core.enums import RiskAlertSeverity
from agentic_trading.core.events import (
    CircuitBreakerEvent,
    FillEvent,
    KillSwitchEvent,
    OrderIntent,
    RiskAlert,
    RiskCheckResult,
)
from agentic_trading.core.interfaces import IEventBus, PortfolioState
from agentic_trading.core.models import Fill, Instrument

from .alerts import AlertEngine
from .circuit_breakers import CircuitBreakerManager
from .drawdown import DrawdownMonitor
from .exposure import ExposureTracker
from .kill_switch import KillSwitch
from .post_trade import PostTradeChecker
from .pre_trade import PreTradeChecker
from .var_es import RiskMetrics

logger = logging.getLogger(__name__)


class RiskManager:
    """Top-level facade for the risk management subsystem.

    Coordinates:

    * **Pre-trade checks** -- position size, notional, leverage, exposure,
      instrument limits.
    * **Post-trade checks** -- position consistency, PnL sanity, leverage
      spike, fill price deviation.
    * **Circuit breakers** -- volatility, spread, liquidity, staleness,
      error rate.
    * **Kill switch** -- global halt, backed by Redis or in-memory.
    * **Drawdown monitor** -- peak drawdown and daily loss limits.
    * **Exposure tracker** -- gross/net/per-asset/per-exchange exposure.
    * **Alert engine** -- configurable rules with hit-rate tracking.
    * **VaR / ES** -- value-at-risk and expected shortfall computation.

    Args:
        config: A :class:`~agentic_trading.core.config.RiskConfig` instance.
        event_bus: Optional event bus for publishing risk events.
        instruments: Optional instrument metadata mapping.
        redis_url: Optional Redis URL for the kill switch.
        kill_switch: Optional pre-built kill switch instance.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        event_bus: IEventBus | None = None,
        instruments: dict[str, Instrument] | None = None,
        redis_url: str | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self._config = config or RiskConfig()
        self._event_bus = event_bus
        self._instruments = instruments or {}

        # Sub-systems
        self.pre_trade = PreTradeChecker(
            max_position_pct=self._config.max_single_position_pct,
            max_notional=500_000.0,  # Could be added to RiskConfig
            max_portfolio_leverage=self._config.max_portfolio_leverage,
            max_gross_exposure_pct=self._config.max_portfolio_leverage,
            instruments=self._instruments,
        )
        self.post_trade = PostTradeChecker(
            max_unexpected_loss_pct=self._config.max_daily_loss_pct,
            max_leverage_after_fill=self._config.max_portfolio_leverage + 1.0,
            max_fill_deviation_pct=0.05,
        )
        self.circuit_breakers = CircuitBreakerManager()
        self.kill_switch = kill_switch or KillSwitch(redis_url=redis_url)
        self.drawdown = DrawdownMonitor()
        self.exposure = ExposureTracker()
        self.alerts = AlertEngine()
        self.metrics = RiskMetrics()

    # ------------------------------------------------------------------
    # Pre-trade pipeline
    # ------------------------------------------------------------------

    async def pre_trade_check(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Run full pre-trade risk check pipeline.

        Steps:
        1. Kill switch check
        2. Circuit breaker check (for the intent's symbol)
        3. All pre-trade sub-checks

        Args:
            intent: The order intent to validate.
            portfolio: Current portfolio snapshot.

        Returns:
            An aggregate :class:`RiskCheckResult`.  ``passed=True`` only
            if *every* sub-check passes.
        """
        # 1. Kill switch
        if await self.kill_switch.is_active():
            result = RiskCheckResult(
                check_name="kill_switch",
                passed=False,
                reason="Kill switch is active -- all orders rejected",
                order_intent_id=intent.event_id,
            )
            await self._publish_risk_event(result)
            return result

        # 2. Circuit breakers
        if self.circuit_breakers.is_any_tripped(symbol=intent.symbol):
            tripped = self.circuit_breakers.get_tripped_breakers()
            names = ", ".join(b.breaker_type.value for b in tripped)
            result = RiskCheckResult(
                check_name="circuit_breaker",
                passed=False,
                reason=f"Circuit breaker(s) tripped: {names}",
                order_intent_id=intent.event_id,
                details={"tripped_breakers": names},
            )
            await self._publish_risk_event(result)
            return result

        # 3. Pre-trade sub-checks
        sub_results = self.pre_trade.check(intent, portfolio)

        # Aggregate: first failure determines the outcome
        for sub in sub_results:
            if not sub.passed:
                await self._publish_risk_event(sub)
                return sub

        # All passed
        aggregate = RiskCheckResult(
            check_name="pre_trade_aggregate",
            passed=True,
            order_intent_id=intent.event_id,
            details={
                "checks_run": len(sub_results),
                "all_passed": True,
            },
        )
        return aggregate

    # ------------------------------------------------------------------
    # Post-trade pipeline
    # ------------------------------------------------------------------

    async def post_trade_check(
        self,
        fill: Fill,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Run full post-trade risk check pipeline.

        Steps:
        1. All post-trade sub-checks
        2. Update exposure tracker
        3. Update drawdown monitor
        4. Evaluate alert rules

        Args:
            fill: The fill that was just received.
            portfolio: Current portfolio snapshot after the fill.

        Returns:
            An aggregate :class:`RiskCheckResult`.
        """
        # 1. Post-trade sub-checks
        sub_results = self.post_trade.check(fill, portfolio)

        # 2. Update exposure
        self.exposure.update(portfolio.positions)

        # 3. Update drawdown (using total equity from balances)
        total_equity = float(sum(
            (b.total for b in portfolio.balances.values()),
            Decimal("0"),
        ))
        if total_equity > 0:
            self.drawdown.update_equity(total_equity)

            # Check drawdown limit
            if self.drawdown.check_drawdown(
                total_equity, self._config.max_drawdown_pct
            ):
                logger.critical(
                    "Max drawdown breached after fill %s -- "
                    "consider activating kill switch",
                    fill.fill_id,
                )
                await self._publish_alert(
                    "drawdown_limit_breached",
                    RiskAlertSeverity.CRITICAL,
                    f"Max drawdown breached: "
                    f"{self.drawdown.current_drawdown_pct:.2%}",
                )

            # Check daily loss
            if self.drawdown.check_daily_loss(
                self.drawdown.daily_pnl,
                self._config.max_daily_loss_pct,
                total_equity,
            ):
                logger.critical(
                    "Daily loss limit breached after fill %s",
                    fill.fill_id,
                )
                await self._publish_alert(
                    "daily_loss_limit_breached",
                    RiskAlertSeverity.CRITICAL,
                    f"Daily loss limit breached: "
                    f"PnL={self.drawdown.daily_pnl:.2f}",
                )

        # 4. Evaluate alert rules
        context = self._build_alert_context(portfolio, total_equity)
        alerts = self.alerts.evaluate(context)
        for alert in alerts:
            await self._publish_event("risk", alert)

        # Aggregate
        failed = [r for r in sub_results if not r.passed]
        if failed:
            first_fail = failed[0]
            await self._publish_risk_event(first_fail)
            return first_fail

        return RiskCheckResult(
            check_name="post_trade_aggregate",
            passed=True,
            details={
                "checks_run": len(sub_results),
                "all_passed": True,
                "alerts_fired": len(alerts),
            },
        )

    # ------------------------------------------------------------------
    # Circuit breaker passthrough
    # ------------------------------------------------------------------

    async def evaluate_circuit_breakers(
        self,
        values: dict[str, float],
        symbol: str = "",
    ) -> list[CircuitBreakerEvent]:
        """Evaluate circuit breakers and publish transition events.

        Args:
            values: Mapping of breaker type name -> measured value.
            symbol: Scope symbol.

        Returns:
            List of state-transition events.
        """
        events = self.circuit_breakers.evaluate_all(values, symbol=symbol)
        for evt in events:
            await self._publish_event("risk", evt)
        return events

    # ------------------------------------------------------------------
    # Kill switch passthrough
    # ------------------------------------------------------------------

    async def activate_kill_switch(
        self,
        reason: str,
        triggered_by: str = "risk_engine",
    ) -> KillSwitchEvent:
        """Activate the global kill switch.

        Args:
            reason: Human-readable reason.
            triggered_by: Source identifier.

        Returns:
            A :class:`KillSwitchEvent`.
        """
        event = await self.kill_switch.activate(
            reason=reason, triggered_by=triggered_by,
        )
        await self._publish_event("system", event)
        return event

    async def deactivate_kill_switch(self) -> KillSwitchEvent:
        """Deactivate the global kill switch."""
        event = await self.kill_switch.deactivate()
        await self._publish_event("system", event)
        return event

    # ------------------------------------------------------------------
    # Instrument management
    # ------------------------------------------------------------------

    def update_instruments(self, instruments: dict[str, Instrument]) -> None:
        """Update instrument metadata (e.g. after a reconciliation cycle)."""
        self._instruments = instruments
        self.pre_trade.instruments = instruments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_alert_context(
        self,
        portfolio: PortfolioState,
        total_equity: float,
    ) -> dict[str, Any]:
        """Build the context dict passed to the alert engine."""
        snapshot = self.exposure.get_snapshot()
        return {
            "total_equity": total_equity,
            "gross_exposure": float(snapshot.gross_exposure),
            "net_exposure": float(snapshot.net_exposure),
            "drawdown_pct": self.drawdown.current_drawdown_pct,
            "daily_pnl": self.drawdown.daily_pnl,
            "peak_equity": self.drawdown.peak_equity,
            "leverage": (
                float(snapshot.gross_exposure) / total_equity
                if total_equity > 0 else 0.0
            ),
            "position_count": len(portfolio.positions),
            "open_order_count": len(portfolio.open_orders),
        }

    async def _publish_risk_event(self, result: RiskCheckResult) -> None:
        """Publish a risk check result on the event bus."""
        await self._publish_event("risk", result)

    async def _publish_alert(
        self,
        alert_type: str,
        severity: RiskAlertSeverity,
        message: str,
    ) -> None:
        """Convenience to create and publish a RiskAlert."""
        alert = RiskAlert(
            severity=severity,
            alert_type=alert_type,
            message=message,
        )
        await self._publish_event("risk", alert)

    async def _publish_event(self, topic: str, event: Any) -> None:
        """Publish an event on the bus, if one is configured."""
        if self._event_bus is not None:
            try:
                await self._event_bus.publish(topic, event)
            except Exception:
                logger.exception(
                    "Failed to publish event on topic=%s: %s",
                    topic,
                    type(event).__name__,
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Clean up resources (Redis connections, etc.)."""
        await self.kill_switch.close()
