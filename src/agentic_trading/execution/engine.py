"""Execution engine: order lifecycle management.

Receives OrderIntent events, checks kill switch and risk, submits orders via
the exchange adapter (or ToolGateway when available), and manages the full
lifecycle from intent through acknowledgement to fill.  Publishes OrderAck,
FillEvent, and PositionUpdate events onto the event bus.

When a ToolGateway is provided, the engine uses the institutional control
plane for all exchange side effects.  Policy evaluation, approval, audit
logging, and kill-switch checks are handled by the control plane.  An
OrderLifecycleManager tracks each order through a strict FSM.

Without a ToolGateway (legacy mode), the engine falls back to direct adapter
calls with inline governance/risk checks (preserved for backward compatibility
during migration).
"""

from __future__ import annotations

import inspect
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

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

    When ``tool_gateway`` is provided, the engine uses the institutional
    control plane for all exchange mutations.  The ToolGateway handles
    policy evaluation, approval, audit logging, kill-switch, and rate
    limiting.  An ``OrderLifecycleManager`` tracks each order's FSM.

    Parameters
    ----------
    adapter:
        Exchange adapter implementing ``IExchangeAdapter``.
        Used directly only in legacy mode (no tool_gateway).
    event_bus:
        Event bus implementing ``IEventBus``.
    risk_manager:
        Pre/post-trade risk checker implementing ``IRiskChecker``.
    kill_switch:
        Callable returning ``True`` when the kill switch is active.
        In CP mode, ToolGateway handles kill switch.
    portfolio_state_provider:
        Callable returning the latest ``PortfolioState`` snapshot.
    max_retries:
        Maximum submission retries before giving up (default 3).
    governance_gate:
        Legacy governance gate.  Ignored when tool_gateway is provided.
    tool_gateway:
        Institutional control plane gateway.  When provided, all exchange
        mutations go through the control plane.
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
        tool_gateway: Any = None,
    ) -> None:
        self._adapter = adapter
        self._event_bus = event_bus
        self._risk_manager = risk_manager
        self._kill_switch_active: bool = False
        self._kill_switch = kill_switch
        self._portfolio_state_provider = portfolio_state_provider
        self._order_manager = OrderManager(max_retries=max_retries)
        self._governance_gate = governance_gate
        self._tool_gateway = tool_gateway
        self._running: bool = False

        # OrderLifecycleManager for FSM tracking (CP mode)
        self._lifecycle_manager: Any = None
        if self._tool_gateway is not None:
            from agentic_trading.control_plane.state_machine import (
                OrderLifecycleManager,
            )
            self._lifecycle_manager = OrderLifecycleManager()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to relevant event-bus topics and start processing."""
        logger.info("ExecutionEngine starting (cp_mode=%s)", self._tool_gateway is not None)
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
        try:
            await self.handle_intent(event)
        except DuplicateOrderError:
            logger.debug("Duplicate order suppressed: %s", event.dedupe_key)
        except Exception:
            logger.warning(
                "Failed to process order intent %s", event.dedupe_key,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Core intent processing
    # ------------------------------------------------------------------

    async def handle_intent(self, intent: OrderIntent) -> OrderAck | None:
        """Process a single ``OrderIntent`` through the full lifecycle.

        When tool_gateway is set, uses the control plane path.
        Otherwise, falls back to the legacy direct-adapter path.
        """
        if self._tool_gateway is not None:
            return await self._handle_intent_cp(intent)
        return await self._handle_intent_legacy(intent)

    # ------------------------------------------------------------------
    # Control Plane path (ToolGateway)
    # ------------------------------------------------------------------

    async def _handle_intent_cp(self, intent: OrderIntent) -> OrderAck | None:
        """Process an order intent through the institutional control plane.

        Steps:
            1. Deduplicate via order manager
            2. Pre-trade risk check (still engine-local for now)
            3. Create OrderLifecycle FSM
            4. Build ProposedAction
            5. Submit via ToolGateway.call() (policy, approval, audit,
               kill-switch, rate-limit all handled by the gateway)
            6. Interpret ToolCallResult → OrderAck
            7. Handle immediate fills
        """
        from agentic_trading.control_plane.action_types import (
            ActionScope,
            ProposedAction,
            ToolName,
        )
        from agentic_trading.control_plane.state_machine import OrderState

        # 1. Deduplication
        if self._order_manager.dedupe_check(intent.dedupe_key):
            raise DuplicateOrderError(
                f"Duplicate dedupe_key: {intent.dedupe_key}"
            )

        # Register the intent as PENDING
        self._order_manager.register_intent(intent)

        # 2. Create OrderLifecycle FSM
        lifecycle = self._lifecycle_manager.create(
            action_id=intent.dedupe_key,
            correlation_id=intent.trace_id,
        )
        # INTENT_RECEIVED -> PREFLIGHT_POLICY
        lifecycle.transition(OrderState.PREFLIGHT_POLICY)

        # 3. Pre-trade risk check
        portfolio_state = self._get_portfolio_state()
        _risk_result = self._risk_manager.pre_trade_check(intent, portfolio_state)
        if inspect.isawaitable(_risk_result):
            _risk_result = await _risk_result
        risk_result: RiskCheckResult = _risk_result
        if not risk_result.passed:
            logger.warning(
                "Order rejected by risk check '%s': %s (dedupe_key=%s)",
                risk_result.check_name,
                risk_result.reason,
                intent.dedupe_key,
            )
            # PREFLIGHT_POLICY -> BLOCKED
            lifecycle.transition(OrderState.BLOCKED)
            lifecycle.error = f"risk_check_failed: {risk_result.reason}"
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

        # 4. Build ProposedAction for ToolGateway
        # Stay in PREFLIGHT_POLICY while ToolGateway evaluates policy,
        # approval, and submits.  Transition based on outcome.
        proposed = ProposedAction(
            tool_name=ToolName.SUBMIT_ORDER,
            scope=ActionScope(
                strategy_id=intent.strategy_id,
                symbol=intent.symbol,
                exchange=intent.exchange,
                actor="execution_engine",
            ),
            request_params={"intent": intent.model_dump(mode="json")},
            idempotency_key=intent.dedupe_key,
            causation_id=intent.event_id,
        )

        # 5. Submit via ToolGateway (policy + approval + audit + adapter)
        result = await self._tool_gateway.call(proposed)
        lifecycle.tool_result = result

        if not result.success:
            error_str = result.error or "unknown"

            # Pending approval: policy passed but human approval needed
            if error_str.startswith("pending_approval:"):
                # PREFLIGHT_POLICY -> AWAITING_APPROVAL
                lifecycle.transition(OrderState.AWAITING_APPROVAL)
                ack = OrderAck(
                    order_id="",
                    client_order_id=intent.dedupe_key,
                    symbol=intent.symbol,
                    exchange=intent.exchange,
                    status=OrderStatus.PENDING,
                    message=f"Awaiting approval: {error_str}",
                    trace_id=intent.trace_id,
                )
                await self._publish_ack(ack)
                return ack

            # Policy block: order rejected at policy evaluation
            if error_str.startswith("policy_blocked:"):
                # PREFLIGHT_POLICY -> BLOCKED
                lifecycle.transition(OrderState.BLOCKED)
                lifecycle.error = error_str
            else:
                # Submission failure: policy passed but adapter failed
                # PREFLIGHT_POLICY -> SUBMITTING -> SUBMIT_FAILED
                lifecycle.transition(OrderState.SUBMITTING)
                lifecycle.transition(OrderState.SUBMIT_FAILED)
                lifecycle.error = error_str

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
                message=error_str,
                trace_id=intent.trace_id,
            )
            await self._publish_ack(ack)
            return ack

        # 6. Success — interpret the ToolCallResult response as OrderAck
        # PREFLIGHT_POLICY -> SUBMITTING -> SUBMITTED
        lifecycle.transition(OrderState.SUBMITTING)
        lifecycle.transition(OrderState.SUBMITTED)

        resp = result.response
        ack_status_str = resp.get("status", "submitted")
        # Map string status back to OrderStatus enum
        try:
            ack_status = OrderStatus(ack_status_str)
        except ValueError:
            ack_status = OrderStatus.SUBMITTED

        ack = OrderAck(
            order_id=resp.get("order_id", ""),
            client_order_id=resp.get("client_order_id", intent.dedupe_key),
            symbol=resp.get("symbol", intent.symbol),
            exchange=resp.get("exchange", intent.exchange),
            status=ack_status,
            message=resp.get("message", ""),
            trace_id=intent.trace_id,
        )

        logger.info(
            "Order submitted via CP: order_id=%s symbol=%s status=%s",
            ack.order_id,
            ack.symbol,
            ack.status.value,
        )

        # Emit order metric
        try:
            from agentic_trading.observability.metrics import record_order
            side_val = intent.side.value if hasattr(intent.side, "value") else str(intent.side)
            otype = intent.order_type.value if hasattr(intent.order_type, "value") else str(intent.order_type)
            record_order(intent.symbol, side_val, otype, ack.status.value)
        except Exception:
            pass

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

        # SUBMITTED -> MONITORING
        lifecycle.transition(OrderState.MONITORING)

        # 7. Handle immediate fill (PaperAdapter or instant market fill)
        if ack.status == OrderStatus.FILLED:
            # MONITORING -> COMPLETE
            lifecycle.transition(OrderState.COMPLETE)

            synthetic_fill = self._synthesize_fill(intent, ack, resp)
            lifecycle.fills.append({"fill_id": synthetic_fill.fill_id})
            await self.handle_fill(synthetic_fill)

            # COMPLETE -> POST_TRADE -> TERMINAL
            lifecycle.transition(OrderState.POST_TRADE)
            lifecycle.transition(OrderState.TERMINAL)

        return ack

    # ------------------------------------------------------------------
    # Legacy path (direct adapter)
    # ------------------------------------------------------------------

    async def _handle_intent_legacy(self, intent: OrderIntent) -> OrderAck | None:
        """Process an order via direct adapter calls (pre-control-plane)."""
        # 1. Kill-switch gate
        if await self._is_kill_switch_active():
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
        _risk_result = self._risk_manager.pre_trade_check(intent, portfolio_state)
        if inspect.isawaitable(_risk_result):
            _risk_result = await _risk_result
        risk_result: RiskCheckResult = _risk_result
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

        # 5. If the adapter returned an immediate fill (e.g. PaperAdapter or
        #    a live market order that filled instantly), synthesize a FillEvent
        #    so the journal and narration see it.
        if ack is not None and ack.status == OrderStatus.FILLED:
            fill_price = self._resolve_fill_price(intent, ack, response=None)
            synthetic_fill = FillEvent(
                fill_id=str(uuid.uuid4()),
                order_id=ack.order_id,
                client_order_id=ack.client_order_id or intent.dedupe_key,
                symbol=intent.symbol,
                exchange=intent.exchange,
                side=intent.side,
                price=fill_price,
                qty=intent.qty,
                fee=Decimal("0"),
                fee_currency="USDT",
                is_maker=False,
                trace_id=intent.trace_id,
                timestamp=datetime.now(timezone.utc),
            )
            await self.handle_fill(synthetic_fill)

        return ack

    # ------------------------------------------------------------------
    # Submission with retry (legacy path)
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
                # Emit order metric
                try:
                    from agentic_trading.observability.metrics import record_order
                    side_val = intent.side.value if hasattr(intent.side, "value") else str(intent.side)
                    otype = intent.order_type.value if hasattr(intent.order_type, "value") else str(intent.order_type)
                    record_order(intent.symbol, side_val, otype, ack.status.value)
                except Exception:
                    pass
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

        # Refresh portfolio state for post-trade check.
        # In CP mode, use ToolGateway for reads; otherwise use adapter.
        portfolio_state = await self._refresh_portfolio_for_post_trade()

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
        _post_risk = self._risk_manager.post_trade_check(
            fill_model, portfolio_state
        )
        if inspect.isawaitable(_post_risk):
            _post_risk = await _post_risk
        post_risk = _post_risk
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

    def _synthesize_fill(
        self,
        intent: OrderIntent,
        ack: OrderAck,
        response: dict[str, Any],
    ) -> FillEvent:
        """Build a synthetic FillEvent from a successful submission.

        Used when the adapter reports an immediate fill (PaperAdapter,
        or a live market order that filled instantly).
        """
        fill_price = self._resolve_fill_price(intent, ack, response)

        return FillEvent(
            fill_id=str(uuid.uuid4()),
            order_id=ack.order_id,
            client_order_id=ack.client_order_id or intent.dedupe_key,
            symbol=intent.symbol,
            exchange=intent.exchange,
            side=intent.side,
            price=fill_price,
            qty=intent.qty,
            fee=Decimal("0"),
            fee_currency="USDT",
            is_maker=False,
            trace_id=intent.trace_id,
            timestamp=datetime.now(timezone.utc),
        )

    def _resolve_fill_price(
        self,
        intent: OrderIntent,
        ack: OrderAck,
        response: dict[str, Any] | None,
    ) -> Decimal:
        """Resolve the actual fill price from all available sources.

        Checks (in priority order):
        1. Response ``avg_fill_price`` (from ToolGateway/CP path).
        2. PaperAdapter order record (``_orders[order_id].avg_fill_price``).
        3. CCXTAdapter last fill price (``_last_fill_price``).
        4. Intent ``price`` (limit orders only).
        5. Adapter ``get_market_price`` (latest candle price).

        Logs a warning if none of the above produce a valid price.
        """
        # 1. Response avg_fill_price (CP path)
        if response and response.get("avg_fill_price"):
            try:
                price = Decimal(str(response["avg_fill_price"]))
                if price > Decimal("0"):
                    return price
            except Exception:
                pass

        # 2. PaperAdapter order record
        try:
            if hasattr(self._adapter, "_orders"):
                paper_order = self._adapter._orders.get(ack.order_id)
                if paper_order is not None:
                    fp = paper_order.avg_fill_price
                    if isinstance(fp, Decimal) and fp > Decimal("0"):
                        return fp
        except Exception:
            pass

        # 3. CCXTAdapter last fill price
        try:
            if hasattr(self._adapter, "_last_fill_price"):
                last_price = self._adapter._last_fill_price
                if isinstance(last_price, Decimal) and last_price > Decimal("0"):
                    return last_price
        except Exception:
            pass

        # 4. Intent price (limit orders)
        if intent.price is not None and intent.price > Decimal("0"):
            return intent.price

        # 5. Adapter market price (latest candle)
        try:
            if hasattr(self._adapter, "get_market_price"):
                market_price = self._adapter.get_market_price(intent.symbol)
                if isinstance(market_price, Decimal) and market_price > Decimal("0"):
                    logger.warning(
                        "Fill price resolved from market price for %s: %s "
                        "(order_id=%s, trace_id=%s)",
                        intent.symbol,
                        market_price,
                        ack.order_id,
                        intent.trace_id[:8] if intent.trace_id else "?",
                    )
                    return market_price
        except Exception:
            pass

        logger.warning(
            "Could not resolve fill price for %s — defaulting to 0 "
            "(order_id=%s, trace_id=%s). TP/SL will use fallback.",
            intent.symbol,
            ack.order_id,
            intent.trace_id[:8] if intent.trace_id else "?",
        )
        return Decimal("0")

    async def _refresh_portfolio_for_post_trade(self) -> PortfolioState:
        """Refresh portfolio state from exchange for post-trade risk check.

        In CP mode, uses ToolGateway for reads; otherwise uses adapter.
        """
        try:
            if self._tool_gateway is not None:
                from agentic_trading.control_plane.action_types import ToolName
                pos_resp = await self._tool_gateway.read(
                    ToolName.GET_POSITIONS, actor="execution_engine",
                )
                bal_resp = await self._tool_gateway.read(
                    ToolName.GET_BALANCES, actor="execution_engine",
                )
                # ToolGateway returns dicts, need to reconstruct
                # For now, fall back to cached state since reconstructing
                # Pydantic models from dicts requires the model classes.
                # The read responses are audited; we use cached state for
                # the actual risk check.
                return self._get_portfolio_state()
            else:
                positions = await self._adapter.get_positions()
                balances = await self._adapter.get_balances()
                return PortfolioState(
                    positions={p.symbol: p for p in positions},
                    balances={b.currency: b for b in balances},
                )
        except Exception:
            logger.debug(
                "Could not refresh portfolio for post-trade check, using cached state",
                exc_info=True,
            )
            return self._get_portfolio_state()

    async def _is_kill_switch_active(self) -> bool:
        """Check kill switch from both cached flag and callable."""
        if self._kill_switch_active:
            return True
        if self._kill_switch is not None and callable(self._kill_switch):
            result = self._kill_switch()
            if inspect.isawaitable(result):
                return await result
            return bool(result)
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
    def lifecycle_manager(self) -> Any:
        """Expose the OrderLifecycleManager (CP mode only, None in legacy)."""
        return self._lifecycle_manager

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uses_control_plane(self) -> bool:
        """Whether this engine is using the institutional control plane."""
        return self._tool_gateway is not None
