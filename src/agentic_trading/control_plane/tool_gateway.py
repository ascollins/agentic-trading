"""ToolGateway — the ONLY path for all exchange side effects.

No agent may call the exchange adapter directly. Every mutation goes:
    ProposedAction -> PolicyEvaluator -> ApprovalService -> AuditLog -> ToolGateway -> Adapter

Enforces:
    1. Allowlisted tools only (ToolName enum)
    2. Policy evaluation BEFORE execution (mutating tools)
    3. Approval BEFORE execution (if policy requires it)
    4. Audit log append BEFORE execution (mandatory, fail-closed)
    5. Idempotency keys prevent double-submit
    6. Kill switch check (final gate before adapter call)
    7. Rate limiting per tool
    8. Signed ToolCallRecorded events (request_hash + response_hash)
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Protocol, runtime_checkable

from agentic_trading.core.errors import ControlPlaneUnavailable
from agentic_trading.core.events import OrderIntent, SystemHealth
from agentic_trading.core.ids import content_hash

from .action_types import (
    MUTATING_TOOLS,
    ActionScope,
    ApprovalDecision,
    ApprovalTier,
    AuditEntry,
    CPPolicyDecision,
    ProposedAction,
    ToolCallResult,
    ToolName,
)
from .audit_log import AuditLog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for pluggable dependencies (Day 3 provides real impls)
# ---------------------------------------------------------------------------


@runtime_checkable
class IPolicyEvaluator(Protocol):
    """Minimal interface for policy evaluation (Day 3 implementation)."""

    def evaluate(self, proposed: ProposedAction) -> CPPolicyDecision: ...


@runtime_checkable
class IApprovalService(Protocol):
    """Minimal interface for approval service (Day 3 implementation)."""

    async def request(
        self, policy_decision: CPPolicyDecision, proposed: ProposedAction,
    ) -> ApprovalDecision: ...


# ---------------------------------------------------------------------------
# Stub implementations for Day 2 (replaced in Day 3)
# ---------------------------------------------------------------------------


class AllowAllPolicy:
    """Stub: allows everything at T0. Replaced by real PolicyEvaluator on Day 3."""

    def evaluate(self, proposed: ProposedAction) -> CPPolicyDecision:
        return CPPolicyDecision(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            allowed=True,
            tier=ApprovalTier.T0_AUTONOMOUS,
            sizing_multiplier=1.0,
            reasons=["stub_allow_all"],
            policy_set_version="stub_v0",
        )


class AutoApproveService:
    """Stub: auto-approves everything. Replaced by real ApprovalService on Day 3."""

    async def request(
        self, policy_decision: CPPolicyDecision, proposed: ProposedAction,
    ) -> ApprovalDecision:
        return ApprovalDecision(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            approved=True,
            tier=policy_decision.tier,
            decided_by=["system:stub_auto"],
            reason="stub_auto_approved",
        )


# ---------------------------------------------------------------------------
# ToolGateway
# ---------------------------------------------------------------------------


class ToolGateway:
    """The ONLY path for exchange side effects.

    Construction requires:
        - adapter: IExchangeAdapter (sole reference in the entire system)
        - audit_log: AuditLog (fail-closed dependency)
        - event_bus: IEventBus (for incident/tool-call events)

    Optional (stubs used until Day 3):
        - policy_evaluator: IPolicyEvaluator
        - approval_service: IApprovalService
        - kill_switch_fn: callable returning bool
        - rate_limits: dict[str, int] (tool_name -> max calls per minute)
    """

    def __init__(
        self,
        adapter: Any,  # IExchangeAdapter
        audit_log: AuditLog,
        event_bus: Any,  # IEventBus
        policy_evaluator: Any | None = None,
        approval_service: Any | None = None,
        kill_switch_fn: Any = None,
        rate_limits: dict[str, int] | None = None,
    ) -> None:
        self._adapter = adapter
        self._audit = audit_log
        self._event_bus = event_bus
        self._policy = policy_evaluator or AllowAllPolicy()
        self._approval = approval_service or AutoApproveService()
        self._kill_switch_fn = kill_switch_fn
        self._rate_limits = rate_limits or {}
        self._idempotency_cache: dict[str, ToolCallResult] = {}
        self._call_counts: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def call(self, proposed: ProposedAction) -> ToolCallResult:
        """Execute a proposed action through the full control plane.

        Flow:
            1. Validate tool is allowlisted
            2. Check idempotency cache
            3. For mutating tools: evaluate policy
            4. For mutating tools: check approval
            5. Append to audit log (MANDATORY — fail if unavailable)
            6. Check kill switch (mutating only, FINAL gate)
            7. Check rate limit
            8. Execute via adapter
            9. Record result in audit log
            10. Cache for idempotency

        Returns:
            ToolCallResult with success/failure + response/error.

        Never raises to callers. Errors are encoded in ToolCallResult.
        """
        t0 = time.monotonic()

        # 1. Validate tool name
        if proposed.tool_name not in ToolName:
            return self._reject(proposed, "unknown_tool", t0)

        # 2. Idempotency check
        if proposed.idempotency_key:
            cached = self._idempotency_cache.get(proposed.idempotency_key)
            if cached is not None:
                return cached.model_copy(update={"was_idempotent_replay": True})

        is_mutating = proposed.tool_name in MUTATING_TOOLS

        # 2.5. Information barrier: role check (spec §7.2)
        if proposed.required_role and proposed.scope.actor_role:
            if proposed.scope.actor_role != proposed.required_role:
                logger.warning(
                    "Information barrier: actor_role=%s lacks required_role=%s "
                    "for action %s",
                    proposed.scope.actor_role,
                    proposed.required_role,
                    proposed.action_id,
                )
                await self._audit_event(
                    proposed, "information_barrier_blocked",
                    {
                        "actor_role": proposed.scope.actor_role,
                        "required_role": proposed.required_role,
                    },
                )
                return self._reject(
                    proposed,
                    f"information_barrier: role '{proposed.scope.actor_role}' "
                    f"cannot perform action requiring '{proposed.required_role}'",
                    t0,
                )

        # 3. Policy evaluation (mutating only)
        policy_decision: CPPolicyDecision | None = None
        if is_mutating:
            try:
                policy_decision = self._policy.evaluate(proposed)
            except Exception as exc:
                # FAIL CLOSED
                logger.error(
                    "Policy evaluator failed, blocking: %s", exc, exc_info=True,
                )
                await self._emit_incident(
                    "policy_evaluator_unavailable", proposed, str(exc),
                )
                return self._reject(proposed, f"policy_evaluator_error: {exc}", t0)

            if not policy_decision.allowed:
                await self._audit_event(
                    proposed, "policy_blocked",
                    {"reasons": policy_decision.reasons},
                )
                return self._reject(
                    proposed,
                    f"policy_blocked: {'; '.join(policy_decision.reasons)}",
                    t0,
                )

        # 4. Approval check (mutating, non-T0 only)
        approval_decision: ApprovalDecision | None = None
        if (
            is_mutating
            and policy_decision
            and policy_decision.tier != ApprovalTier.T0_AUTONOMOUS
        ):
            try:
                approval_decision = await self._approval.request(
                    policy_decision, proposed,
                )
            except Exception as exc:
                logger.error(
                    "Approval service failed, blocking: %s", exc, exc_info=True,
                )
                await self._emit_incident(
                    "approval_service_unavailable", proposed, str(exc),
                )
                return self._reject(
                    proposed, f"approval_service_error: {exc}", t0,
                )

            if not approval_decision.approved:
                if approval_decision.pending_request_id:
                    return self._pending(proposed, approval_decision, t0)
                return self._reject(
                    proposed,
                    f"approval_denied: {approval_decision.reason}",
                    t0,
                )

        # 5. Audit log (MANDATORY — fail closed)
        try:
            await self._audit.append(AuditEntry(
                correlation_id=proposed.correlation_id,
                causation_id=proposed.action_id,
                actor=proposed.scope.actor,
                scope=proposed.scope,
                event_type="tool_call_pre_execution",
                payload={
                    "action_id": proposed.action_id,
                    "tool_name": proposed.tool_name.value,
                    "request_hash": proposed.request_hash,
                    "policy_decision_id": (
                        policy_decision.decision_id if policy_decision else None
                    ),
                    "approval_id": (
                        approval_decision.approval_id if approval_decision else None
                    ),
                },
            ))
        except Exception as exc:
            logger.error(
                "Audit log unavailable, blocking all side effects: %s", exc,
                exc_info=True,
            )
            await self._emit_incident(
                "audit_log_unavailable", proposed, str(exc),
            )
            return self._reject(proposed, f"audit_log_unavailable: {exc}", t0)

        # 6. Kill switch (mutating only, FINAL gate)
        if is_mutating and await self._is_kill_switch_active():
            return self._reject(proposed, "kill_switch_active", t0)

        # 7. Rate limit
        if not self._check_rate_limit(proposed.tool_name):
            return self._reject(proposed, "rate_limit_exceeded", t0)

        # 8. Execute
        try:
            response = await self._dispatch(proposed)
            elapsed_ms = (time.monotonic() - t0) * 1000
            response_hash = content_hash(str(response))

            result = ToolCallResult(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                tool_name=proposed.tool_name,
                success=True,
                response=response,
                response_hash=response_hash,
                latency_ms=round(elapsed_ms, 2),
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "Tool dispatch failed: tool=%s error=%s",
                proposed.tool_name.value, exc,
                exc_info=True,
            )
            result = ToolCallResult(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                tool_name=proposed.tool_name,
                success=False,
                error=str(exc),
                latency_ms=round(elapsed_ms, 2),
            )

        # 9. Record result in audit log (best-effort post-execution)
        try:
            await self._audit.append(AuditEntry(
                correlation_id=proposed.correlation_id,
                causation_id=proposed.action_id,
                actor=proposed.scope.actor,
                scope=proposed.scope,
                event_type="tool_call_recorded",
                payload={
                    "action_id": proposed.action_id,
                    "tool_name": proposed.tool_name.value,
                    "success": result.success,
                    "request_hash": proposed.request_hash,
                    "response_hash": result.response_hash,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                },
            ))
        except Exception:
            logger.error(
                "Failed to record tool call result in audit", exc_info=True,
            )

        # 10. Cache for idempotency
        if proposed.idempotency_key:
            self._idempotency_cache[proposed.idempotency_key] = result

        return result

    # ------------------------------------------------------------------
    # Convenience: read-only calls (no policy/approval needed)
    # ------------------------------------------------------------------

    async def read(
        self,
        tool_name: ToolName,
        params: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        """Execute a read-only tool call.

        Raises ValueError if called with a mutating tool name.
        Returns the response dict directly (unwrapped from ToolCallResult).
        """
        if tool_name in MUTATING_TOOLS:
            raise ValueError(f"Use call() for mutating tool: {tool_name}")

        proposed = ProposedAction(
            tool_name=tool_name,
            scope=ActionScope(
                strategy_id="",
                symbol=(params or {}).get("symbol", ""),
                actor=actor,
            ),
            request_params=params or {},
        )
        result = await self.call(proposed)
        if not result.success:
            raise RuntimeError(f"Read failed: {result.error}")
        return result.response

    # ------------------------------------------------------------------
    # Dispatch to adapter
    # ------------------------------------------------------------------

    async def _dispatch(self, proposed: ProposedAction) -> dict[str, Any]:
        """Route the proposed action to the correct adapter method.

        Returns a serializable dict of the response.
        """
        p = proposed.request_params
        match proposed.tool_name:
            case ToolName.SUBMIT_ORDER:
                intent = OrderIntent(**p["intent"])
                ack = await self._adapter.submit_order(intent)
                return ack.model_dump()

            case ToolName.CANCEL_ORDER:
                ack = await self._adapter.cancel_order(
                    p["order_id"], p["symbol"],
                )
                return ack.model_dump()

            case ToolName.CANCEL_ALL_ORDERS:
                acks = await self._adapter.cancel_all_orders(
                    p.get("symbol"),
                )
                return {"acks": [a.model_dump() for a in acks]}

            case ToolName.AMEND_ORDER:
                from decimal import Decimal
                kwargs: dict[str, Any] = {
                    "order_id": p["order_id"],
                    "symbol": p["symbol"],
                }
                if "qty" in p:
                    kwargs["qty"] = Decimal(str(p["qty"]))
                if "price" in p:
                    kwargs["price"] = Decimal(str(p["price"]))
                if "stop_price" in p:
                    kwargs["stop_price"] = Decimal(str(p["stop_price"]))
                ack = await self._adapter.amend_order(**kwargs)
                return ack.model_dump()

            case ToolName.BATCH_SUBMIT_ORDERS:
                intents = [OrderIntent(**i) for i in p["intents"]]
                acks = await self._adapter.batch_submit_orders(intents)
                return {"acks": [a.model_dump() for a in acks]}

            case ToolName.SET_TRADING_STOP:
                from decimal import Decimal
                kwargs_ts: dict[str, Any] = {"symbol": p["symbol"]}
                if "take_profit" in p and p["take_profit"] is not None:
                    kwargs_ts["take_profit"] = Decimal(str(p["take_profit"]))
                if "stop_loss" in p and p["stop_loss"] is not None:
                    kwargs_ts["stop_loss"] = Decimal(str(p["stop_loss"]))
                if "trailing_stop" in p and p["trailing_stop"] is not None:
                    kwargs_ts["trailing_stop"] = Decimal(str(p["trailing_stop"]))
                if "active_price" in p and p["active_price"] is not None:
                    kwargs_ts["active_price"] = Decimal(str(p["active_price"]))
                result = await self._adapter.set_trading_stop(**kwargs_ts)
                return result if isinstance(result, dict) else {"result": result}

            case ToolName.SET_LEVERAGE:
                result = await self._adapter.set_leverage(
                    p["symbol"], p["leverage"],
                )
                return result if isinstance(result, dict) else {"result": result}

            case ToolName.SET_POSITION_MODE:
                result = await self._adapter.set_position_mode(
                    p["symbol"], p["mode"],
                )
                return result if isinstance(result, dict) else {"result": result}

            case ToolName.GET_POSITIONS:
                positions = await self._adapter.get_positions(p.get("symbol"))
                return {"positions": [pos.model_dump() for pos in positions]}

            case ToolName.GET_BALANCES:
                balances = await self._adapter.get_balances()
                return {"balances": [b.model_dump() for b in balances]}

            case ToolName.GET_OPEN_ORDERS:
                orders = await self._adapter.get_open_orders(p.get("symbol"))
                return {"orders": [o.model_dump() for o in orders]}

            case ToolName.GET_INSTRUMENT:
                inst = await self._adapter.get_instrument(p["symbol"])
                return inst.model_dump()

            case ToolName.GET_FUNDING_RATE:
                rate = await self._adapter.get_funding_rate(p["symbol"])
                return {"rate": str(rate)}

            case ToolName.GET_CLOSED_PNL:
                pnl = await self._adapter.get_closed_pnl(
                    p["symbol"], limit=p.get("limit", 50),
                )
                return {"entries": pnl}

            case _:
                raise ValueError(f"Unhandled tool: {proposed.tool_name}")

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    @staticmethod
    def _reject(
        proposed: ProposedAction, reason: str, t0: float,
    ) -> ToolCallResult:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return ToolCallResult(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            tool_name=proposed.tool_name,
            success=False,
            error=reason,
            latency_ms=round(elapsed_ms, 2),
        )

    @staticmethod
    def _pending(
        proposed: ProposedAction,
        approval: ApprovalDecision,
        t0: float,
    ) -> ToolCallResult:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return ToolCallResult(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            tool_name=proposed.tool_name,
            success=False,
            error=f"pending_approval:{approval.pending_request_id}",
            response={"pending_request_id": approval.pending_request_id},
            latency_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    async def _is_kill_switch_active(self) -> bool:
        if self._kill_switch_fn is None:
            return False
        result = self._kill_switch_fn()
        if inspect.isawaitable(result):
            return await result
        return bool(result)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self, tool_name: ToolName) -> bool:
        limit = self._rate_limits.get(tool_name.value)
        if limit is None:
            return True
        now = time.monotonic()
        key = tool_name.value
        timestamps = self._call_counts.setdefault(key, [])
        # Prune entries older than 60 seconds
        timestamps[:] = [t for t in timestamps if now - t < 60]
        if len(timestamps) >= limit:
            return False
        timestamps.append(now)
        return True

    # ------------------------------------------------------------------
    # Audit helper
    # ------------------------------------------------------------------

    async def _audit_event(
        self,
        proposed: ProposedAction,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Best-effort audit append (non-critical path)."""
        try:
            await self._audit.append(AuditEntry(
                correlation_id=proposed.correlation_id,
                causation_id=proposed.action_id,
                actor=proposed.scope.actor,
                scope=proposed.scope,
                event_type=event_type,
                payload=payload,
            ))
        except Exception:
            logger.warning("Non-critical audit append failed", exc_info=True)

    # ------------------------------------------------------------------
    # Incident emission
    # ------------------------------------------------------------------

    async def _emit_incident(
        self, incident_type: str, proposed: ProposedAction, detail: str,
    ) -> None:
        """Emit a SystemHealth incident event."""
        try:
            event = SystemHealth(
                component=f"control_plane.{incident_type}",
                healthy=False,
                message=f"{incident_type}: {detail}",
                details={
                    "action_id": proposed.action_id,
                    "tool_name": proposed.tool_name.value,
                },
            )
            await self._event_bus.publish("system", event)
        except Exception:
            logger.critical(
                "Cannot emit incident event: %s", incident_type, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def idempotency_cache_size(self) -> int:
        return len(self._idempotency_cache)

    def clear_idempotency_cache(self) -> None:
        """Clear the idempotency cache (for testing or periodic cleanup)."""
        self._idempotency_cache.clear()
