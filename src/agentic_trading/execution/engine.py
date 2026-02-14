"""Execution engine: order lifecycle management.

Receives OrderIntent events, checks kill switch and risk, submits orders via
the exchange adapter, and manages the full lifecycle from intent through
acknowledgement to fill.  Publishes OrderAck, FillEvent, and PositionUpdate
events onto the event bus.
"""

from __future__ import annotations

import logging
from typing import Any

from decimal import Decimal

from agentic_trading.core.enums import GovernanceAction, OrderStatus
from agentic_trading.core.errors import (
    DuplicateOrderError,
    ExchangeError,
    KillSwitchActive,
    OrderRejectedError,
)
from agentic_trading.core.events import (
    BaseEvent,
    FillEvent,
    KillSwitchEvent,
    OrderAck,
    OrderIntent,
    OrderUpdate,
    PositionUpdate,
    RiskCheckResult,
)
from agentic_trading.core.interfaces import (
    IEventBus,
    IExchangeAdapter,
    IRiskChecker,
    PortfolioState,
)

from .order_manager import OrderManager

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Orchestrates the full order lifecycle.

    Responsibilities:
      - Listen for ``OrderIntent`` events on the ``execution`` topic.
      - Check the kill switch before every submission.
      - Run pre-trade risk checks via the ``IRiskChecker``.
      - Deduplicate intents via ``OrderManager``.
      - Submit to the exchange adapter and publish ``OrderAck``.
      - Forward ``FillEvent`` / ``OrderUpdate`` / ``PositionUpdate``.
      - Retry failed submissions through ``OrderManager``.

    Parameters
    ----------
    adapter:
        Exchange adapter implementing ``IExchangeAdapter``.
    event_bus:
        Event bus implementing ``IEventBus``.
    risk_manager:
        Pre/post-trade risk checker implementing ``IRiskChecker``.
    kill_switch:
        Callable returning ``True`` when the kill switch is active.
    portfolio_state_provider:
        Callable returning the latest ``PortfolioState`` snapshot.
    max_retries:
        Maximum submission retries before giving up (default 3).
    """

    def __init__(
        self,
        adapter: IExchangeAdapter,
        event_bus: IEventBus,
        risk_manager: IRiskChecker,
        kill_switch: Any = None,
        portfolio_state_provider: Any = None,
        max_retries: int = 3,
        governance_gate: Any = None,
    ) -> None:
        self._adapter = adapter
        self._event_bus = event_bus
        self._risk_manager = risk_manager
        self._kill_switch_active: bool = False
        self._kill_switch = kill_switch
        self._portfolio_state_provider = portfolio_state_provider
        self._order_manager = OrderManager(max_retries=max_retries)
        self._governance_gate = governance_gate
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to relevant event-bus topics and start processing."""
        logger.info("ExecutionEngine starting")
        await self._event_bus.subscribe(
            topic="execution",
            group="execution_engine",
            handler=self._on_order_intent,
        )
        await self._event_bus.subscribe(
            topic="system",
            group="execution_engine",
            handler=self._on_kill_switch,
        )
        self._running = True
        logger.info("ExecutionEngine started")

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("ExecutionEngine stopping")
        self._running = False
        logger.info("ExecutionEngine stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_kill_switch(self, event: BaseEvent) -> None:
        """React to kill-switch activation / deactivation."""
        if isinstance(event, KillSwitchEvent):
            self._kill_switch_active = event.activated
            if event.activated:
                logger.warning(
                    "Kill switch ACTIVATED: %s (triggered_by=%s)",
                    event.reason,
                    event.triggered_by,
                )
            else:
                logger.info("Kill switch deactivated")

    async def _on_order_intent(self, event: BaseEvent) -> None:
        """Handle an incoming ``OrderIntent``."""
        if not isinstance(event, OrderIntent):
            return
        await self.handle_intent(event)

    # ------------------------------------------------------------------
    # Core intent processing
    # ------------------------------------------------------------------

    async def handle_intent(self, intent: OrderIntent) -> OrderAck | None:
        """Process a single ``OrderIntent`` through the full lifecycle.

        Steps:
            1. Kill switch check
            2. Deduplicate via order manager
            3. Pre-trade risk check
            4. Submit to exchange adapter (with retry)
            5. Publish OrderAck
            6. Register the order in the order manager

        Returns the ``OrderAck`` on success, or ``None`` on rejection.
        """
        # 1. Kill-switch gate
        if self._is_kill_switch_active():
            logger.warning(
                "Order rejected (kill switch active): dedupe_key=%s symbol=%s",
                intent.dedupe_key,
                intent.symbol,
            )
            ack = OrderAck(
                order_id="",
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=intent.exchange,
                status=OrderStatus.REJECTED,
                message="Kill switch is active",
                trace_id=intent.trace_id,
            )
            await self._publish_ack(ack)
            return ack

        # 2. Deduplication
        if self._order_manager.dedupe_check(intent.dedupe_key):
            logger.info(
                "Duplicate order intent suppressed: dedupe_key=%s",
                intent.dedupe_key,
            )
            raise DuplicateOrderError(
                f"Duplicate dedupe_key: {intent.dedupe_key}"
            )

        # Register the intent as PENDING
        self._order_manager.register_intent(intent)

        # 2.5 Governance gate (if enabled)
        if self._governance_gate is not None:
            gov_decision = await self._governance_gate.evaluate(
                strategy_id=intent.strategy_id,
                symbol=intent.symbol,
                notional_usd=float(intent.qty * (intent.price or Decimal("0"))),
                portfolio_pct=0.0,
                is_reduce_only=intent.reduce_only,
                leverage=intent.leverage or 1,
                existing_positions=0,
                trace_id=intent.trace_id,
            )
            if gov_decision.action in (
                GovernanceAction.BLOCK,
                GovernanceAction.PAUSE,
                GovernanceAction.KILL,
            ):
                logger.warning(
                    "Order rejected by governance: %s (strategy=%s symbol=%s)",
                    gov_decision.reason,
                    intent.strategy_id,
                    intent.symbol,
                )
                self._order_manager.update_order(
                    OrderUpdate(
                        order_id="",
                        client_order_id=intent.dedupe_key,
                        symbol=intent.symbol,
                        exchange=intent.exchange,
                        status=OrderStatus.REJECTED,
                        trace_id=intent.trace_id,
                    )
                )
                ack = OrderAck(
                    order_id="",
                    client_order_id=intent.dedupe_key,
                    symbol=intent.symbol,
                    exchange=intent.exchange,
                    status=OrderStatus.REJECTED,
                    message=f"Governance: {gov_decision.reason}",
                    trace_id=intent.trace_id,
                )
                await self._publish_ack(ack)
                return ack
            elif gov_decision.action == GovernanceAction.REDUCE_SIZE:
                # Apply governance sizing reduction
                new_qty = Decimal(
                    str(float(intent.qty) * gov_decision.sizing_multiplier)
                )
                intent = intent.model_copy(update={"qty": new_qty})
                logger.info(
                    "Governance reduced sizing: %s → %s (mult=%.2f)",
                    intent.dedupe_key,
                    new_qty,
                    gov_decision.sizing_multiplier,
                )

        # 3. Pre-trade risk check
        portfolio_state = self._get_portfolio_state()
        risk_result: RiskCheckResult = self._risk_manager.pre_trade_check(
            intent, portfolio_state
        )
        if not risk_result.passed:
            logger.warning(
                "Order rejected by risk check '%s': %s (dedupe_key=%s)",
                risk_result.check_name,
                risk_result.reason,
                intent.dedupe_key,
            )
            self._order_manager.update_order(
                OrderUpdate(
                    order_id="",
                    client_order_id=intent.dedupe_key,
                    symbol=intent.symbol,
                    exchange=intent.exchange,
                    status=OrderStatus.REJECTED,
                    trace_id=intent.trace_id,
                )
            )
            ack = OrderAck(
                order_id="",
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=intent.exchange,
                status=OrderStatus.REJECTED,
                message=f"Risk check failed: {risk_result.reason}",
                trace_id=intent.trace_id,
            )
            await self._publish_ack(ack)
            await self._event_bus.publish("risk", risk_result)
            return ack

        # 4. Submit with retry
        ack = await self._submit_with_retry(intent)
        return ack

    # ------------------------------------------------------------------
    # Submission with retry
    # ------------------------------------------------------------------

    async def _submit_with_retry(self, intent: OrderIntent) -> OrderAck:
        """Submit the order via the adapter, retrying on transient failures.

        Uses ``OrderManager.should_retry`` and
        ``OrderManager.record_attempt`` to track attempts.
        """
        last_error: Exception | None = None

        while self._order_manager.should_retry(intent.dedupe_key):
            self._order_manager.record_attempt(intent.dedupe_key)
            try:
                ack: OrderAck = await self._adapter.submit_order(intent)
                logger.info(
                    "Order submitted: order_id=%s client_order_id=%s "
                    "symbol=%s status=%s",
                    ack.order_id,
                    ack.client_order_id,
                    ack.symbol,
                    ack.status.value,
                )
                # Transition to SUBMITTED in order manager
                self._order_manager.update_order(
                    OrderUpdate(
                        order_id=ack.order_id,
                        client_order_id=intent.dedupe_key,
                        symbol=intent.symbol,
                        exchange=intent.exchange,
                        status=OrderStatus.SUBMITTED,
                        remaining_qty=intent.qty,
                        trace_id=intent.trace_id,
                    )
                )
                await self._publish_ack(ack)
                return ack

            except ExchangeError as exc:
                last_error = exc
                attempt = self._order_manager.get_attempt_count(
                    intent.dedupe_key
                )
                logger.warning(
                    "Submission attempt %d failed for dedupe_key=%s: %s",
                    attempt,
                    intent.dedupe_key,
                    exc,
                )

        # All retries exhausted
        logger.error(
            "Order submission failed after max retries: dedupe_key=%s "
            "last_error=%s",
            intent.dedupe_key,
            last_error,
        )
        self._order_manager.update_order(
            OrderUpdate(
                order_id="",
                client_order_id=intent.dedupe_key,
                symbol=intent.symbol,
                exchange=intent.exchange,
                status=OrderStatus.REJECTED,
                trace_id=intent.trace_id,
            )
        )
        ack = OrderAck(
            order_id="",
            client_order_id=intent.dedupe_key,
            symbol=intent.symbol,
            exchange=intent.exchange,
            status=OrderStatus.REJECTED,
            message=f"Max retries exhausted: {last_error}",
            trace_id=intent.trace_id,
        )
        await self._publish_ack(ack)
        return ack

    # ------------------------------------------------------------------
    # Fill handling (called externally, e.g. by websocket feed or recon)
    # ------------------------------------------------------------------

    async def handle_fill(self, fill: FillEvent) -> None:
        """Process an incoming fill event.

        Updates local order state, publishes FillEvent and PositionUpdate
        onto the event bus, and runs a post-trade risk check.
        """
        logger.info(
            "Fill received: fill_id=%s order_id=%s symbol=%s "
            "side=%s price=%s qty=%s",
            fill.fill_id,
            fill.order_id,
            fill.symbol,
            fill.side.value,
            fill.price,
            fill.qty,
        )
        # Update order state in order manager
        order_state = self._order_manager.get_order(fill.client_order_id)
        if order_state is not None:
            filled_qty = order_state.filled_qty + fill.qty
            remaining_qty = order_state.total_qty - filled_qty
            if remaining_qty <= 0:
                new_status = OrderStatus.FILLED
            else:
                new_status = OrderStatus.PARTIALLY_FILLED

            self._order_manager.update_order(
                OrderUpdate(
                    order_id=fill.order_id,
                    client_order_id=fill.client_order_id,
                    symbol=fill.symbol,
                    exchange=fill.exchange,
                    status=new_status,
                    filled_qty=filled_qty,
                    remaining_qty=max(remaining_qty, fill.qty.__class__("0")),
                    avg_fill_price=fill.price,
                    trace_id=fill.trace_id,
                )
            )

        # Publish fill
        await self._event_bus.publish("execution", fill)

        # Publish position update (consumers will reconcile real positions)
        position_update = PositionUpdate(
            symbol=fill.symbol,
            exchange=fill.exchange,
            qty=fill.qty,
            entry_price=fill.price,
            mark_price=fill.price,
            trace_id=fill.trace_id,
        )
        await self._event_bus.publish("state", position_update)

        # Post-trade risk check
        portfolio_state = self._get_portfolio_state()
        from agentic_trading.core.models import Fill as FillModel

        fill_model = FillModel(
            fill_id=fill.fill_id,
            order_id=fill.order_id,
            client_order_id=fill.client_order_id,
            symbol=fill.symbol,
            exchange=fill.exchange,
            side=fill.side,
            price=fill.price,
            qty=fill.qty,
            fee=fill.fee,
            fee_currency=fill.fee_currency,
            is_maker=fill.is_maker,
            timestamp=fill.timestamp,
            trace_id=fill.trace_id,
        )
        post_risk = self._risk_manager.post_trade_check(
            fill_model, portfolio_state
        )
        if not post_risk.passed:
            logger.warning(
                "Post-trade risk alert: check=%s reason=%s",
                post_risk.check_name,
                post_risk.reason,
            )
            await self._event_bus.publish("risk", post_risk)

    # ------------------------------------------------------------------
    # Order update handling
    # ------------------------------------------------------------------

    async def handle_order_update(self, update: OrderUpdate) -> None:
        """Process an external order-status update (e.g. from websocket).

        Transitions the order state and publishes the update event.
        """
        logger.info(
            "Order update: order_id=%s status=%s filled=%s remaining=%s",
            update.order_id,
            update.status.value,
            update.filled_qty,
            update.remaining_qty,
        )
        self._order_manager.update_order(update)
        await self._event_bus.publish("execution", update)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_kill_switch_active(self) -> bool:
        """Check kill switch from both cached flag and callable."""
        if self._kill_switch_active:
            return True
        if self._kill_switch is not None and callable(self._kill_switch):
            return self._kill_switch()
        return False

    def _get_portfolio_state(self) -> PortfolioState:
        """Return the current portfolio snapshot."""
        if (
            self._portfolio_state_provider is not None
            and callable(self._portfolio_state_provider)
        ):
            return self._portfolio_state_provider()
        return PortfolioState()

    async def _publish_ack(self, ack: OrderAck) -> None:
        """Publish an ``OrderAck`` to the execution topic."""
        await self._event_bus.publish("execution", ack)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def order_manager(self) -> OrderManager:
        """Expose the order manager for reconciliation and monitoring."""
        return self._order_manager

    @property
    def is_running(self) -> bool:
        return self._running
