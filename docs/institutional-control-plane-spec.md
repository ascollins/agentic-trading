# Institutional Control Plane Refactor Spec

**Date**: 2026-02-16
**Status**: DRAFT - Pending Review
**Goal**: Provable control, deterministic governance, full auditability. Minimum delta.

---

## 1) BYPASS PATH AUDIT (RUTHLESS)

### CRITICAL (Must Fix Before Live)

| # | Bypass | Location | Mechanism | Fix | Proof It's Closed |
|---|--------|----------|-----------|-----|-------------------|
| B1 | **GovernanceGate is optional** | `execution/engine.py:84,203` | `governance_gate` defaults to `None`; `if self._governance_gate is not None:` skips all checks | Make `governance_gate` required in `__init__`. Remove the `None` default. Constructor raises `ValueError` if `None` in live/paper mode. | Type checker + runtime assertion. No code path can construct engine without gate. |
| B2 | **`governance.enabled` defaults to `False`** | `core/config.py:192`, `governance/gate.py:106-114` | If config omits `governance.enabled`, gate returns `ALLOW` unconditionally | Remove the `enabled` flag entirely. Governance is always on. If you want to skip it, run backtest mode. For live/paper: gate MUST evaluate. | Config validation at startup: `if mode in (PAPER, LIVE) and not governance_gate: sys.exit(1)` |
| B3 | **Direct `adapter.set_trading_stop()` in main.py** | `main.py:1225,1490` | After fill and at startup, TP/SL set directly on adapter bypassing governance | Route ALL adapter mutations through `ToolGateway`. TP/SL becomes a `SetTradingStopRequest` evaluated by policy. | Grep for `adapter.set_trading_stop` must return zero hits outside `ToolGateway`. |
| B4 | **Direct `adapter._ccxt.fetch_positions()` in main.py** | `main.py:1458` | Reaches through adapter abstraction to raw CCXT | Use `adapter.get_positions()` (the protocol method). Delete all `adapter._ccxt` references outside the adapter itself. | Grep for `adapter._ccxt` must return zero hits outside `ccxt_adapter.py`. |
| B5 | **Policy engine exception = fail-open** | `governance/gate.py:328-331` | Bare `except Exception:` lets order proceed when policy engine crashes | Change to **fail-closed**: on exception, return `GovernanceAction.BLOCK` with reason `"policy_engine_error"`. Emit incident event. | Unit test: mock policy engine to raise; assert BLOCK returned. |
| B6 | **Approval manager exception = fail-open** | `governance/gate.py:214-217` | Same pattern as B5 for approval manager | Same fix: fail-closed. On exception, BLOCK + incident. | Unit test: mock approval manager to raise; assert BLOCK returned. |
| B7 | **Orchestrator wires `governance_gate=None`** | `agents/orchestrator.py:130-147` | Comment says "will be available after risk gate starts" but never wired | Wire governance gate explicitly during orchestrator construction. Or: remove orchestrator's separate engine creation; use a single engine instance. | Integration test: orchestrator-created engine rejects ungoverned orders. |
| B8 | **`ExecutionAgent.adapter` public property** | `agents/execution.py:132-135` | Anyone with agent reference can call `agent.adapter.submit_order()` directly | Remove `.adapter` property. Make `_adapter` truly private. All mutations through ToolGateway. | `hasattr(ExecutionAgent, 'adapter')` returns `False`. |
| B9 | **`AgentOrchestrator.adapter` public property** | `agents/orchestrator.py:209-212` | Same as B8 for orchestrator | Remove property. | Same check. |

### HIGH (Fix in Week 1)

| # | Bypass | Location | Mechanism | Fix | Proof |
|---|--------|----------|-----------|-----|-------|
| B10 | **Missing context fields = rule passes** | `governance/policy_engine.py:225-240` | `field_value is None` -> `passed=True` | Fail-closed: missing field = `passed=False` with reason `"required_field_missing"` | Property test: random field omission always fails. |
| B11 | **Type mismatch = rule passes** | `governance/policy_engine.py:310-318` | `except (TypeError, ValueError): return True` | Fail-closed: type error = `return False` | Unit test with wrong-type field values. |
| B12 | **Shadow mode passes all violations** | `governance/policy_engine.py:131,144-148` | Shadow mode never adds to `failed` list | Add startup check: `if mode == LIVE and any policy in SHADOW: raise ConfigError`. Shadow mode only allowed in backtest/paper. | Config validation test. |
| B13 | **Escalation allows downgrade** | `governance/approval_manager.py:288-317` | `escalate()` accepts any level, even lower | Add validation: `if new_level.rank <= request.escalation_level.rank: raise ValueError` | Unit test: downgrade attempt raises. |
| B14 | **No auth on approve/reject/escalate** | `governance/approval_manager.py` | `decided_by` is unchecked string | Add `required_authority: EscalationLevel` mapping. Caller must provide credential matching the required level. | Unit test: L1 credential cannot approve L3 request. |
| B15 | **Kill switch not at adapter level** | Adapter protocol | Kill switch checked only in ExecutionEngine; direct adapter callers bypass it | ToolGateway checks kill switch BEFORE dispatching any mutating tool call. Adapter itself is never called directly. | All adapter mutation paths go through ToolGateway (enforced by removing direct references). |
| B16 | **`amend_order` / `batch_submit_orders` ungoverned** | `ccxt_adapter.py:561-707` | Engine only wraps `submit_order`; these methods callable directly | Route through ToolGateway as distinct tool types with their own policy evaluation. | ToolGateway has registered handlers for `amend_order` and `batch_submit`. |
| B17 | **Canary created without `kill_switch_fn`** | `main.py:319-325` | Kill action silently skipped when `_kill_switch_fn is None` | Pass `risk_manager.kill_switch.activate` as `kill_switch_fn`. | Integration test: canary health fail -> kill switch activates. |
| B18 | **Recon auto-repair without governance** | `execution/reconciliation.py:275-336` | Auto-repair modifies local state, no policy check | Recon produces `ProposedAction` events for state repairs. ToolGateway evaluates policy before applying. Read-only recon is fine; write operations need governance. | Recon tests: auto-repair generates ProposedAction, not direct state mutation. |

### MEDIUM (Fix in Week 2)

| # | Bypass | Location | Fix |
|---|--------|----------|-----|
| B19 | No event bus publish authorization | `redis_streams.py`, `memory_bus.py` | ToolGateway is the only publisher of mutating events. Event bus gets topic-level write ACLs (allowlist of source_module per topic). |
| B20 | Kill switch deactivation via event bus | `execution/engine.py:126-137` | Only ToolGateway (with ADMIN scope) can publish `KillSwitchEvent(activated=False)`. Engine ignores deactivation events from other sources. |
| B21 | Journal force-close without governance | `main.py:967-1016` | Journal force-close is a read operation on local state, acceptable. But it should emit an audit event. |
| B22 | Shutdown does not cancel pending orders | `main.py:1517-1533` | Graceful shutdown calls `ToolGateway.cancel_all_open_orders()` with governance override flag. |
| B23 | Dual kill switch state | `execution/engine.py:89-90` | Remove cached boolean. Always check via callable (single source of truth). |

---

## 2) TARGET ARCHITECTURE (MINIMUM DELTA)

### Current -> Target

```
CURRENT:
  Agent -> ExecutionEngine -> [optional GovernanceGate] -> adapter.submit_order()
  main.py -> adapter.set_trading_stop()  [UNGOVERNED]
  main.py -> adapter._ccxt.fetch_positions()  [ABSTRACTION LEAK]
  GovernanceGate -> [optional PolicyEngine] -> [optional ApprovalManager]
  All components hold adapter references

TARGET:
  Agent -> ProposedAction -> PolicyEvaluator -> ApprovalService -> ToolGateway -> adapter
  ToolGateway is the ONLY holder of adapter reference
  PolicyEvaluator is MANDATORY, deterministic, fail-closed
  ApprovalService enforces tiers BEFORE ToolGateway
  AuditLog.append() is MANDATORY before ToolGateway.call()
  If PolicyEvaluator OR AuditLog unavailable -> BLOCK (fail-closed)
```

### New Modules (5 total, all in `src/agentic_trading/control_plane/`)

```
control_plane/
  __init__.py
  tool_gateway.py        # ToolGateway: sole side-effect executor
  policy_evaluator.py    # PolicyEvaluator: deterministic rule engine
  approval_service.py    # ApprovalService: tiered approval workflow
  audit_log.py           # AuditLog: append-only event journal
  action_types.py        # ProposedAction, ToolCallResult, etc.
  state_machine.py       # OrderStateMachine: execution lifecycle FSM
```

### Module Boundaries (Who May Call Whom)

```
                    ┌─────────────┐
                    │   Agents    │  (MarketIntel, Execution, Risk, etc.)
                    └──────┬──────┘
                           │ emit ProposedAction
                           ▼
                    ┌─────────────┐
                    │   Policy    │  deterministic, no side effects
                    │  Evaluator  │  reads: PolicyRegistry, PortfolioState
                    └──────┬──────┘
                           │ PolicyDecision
                           ▼
                    ┌─────────────┐
                    │  Approval   │  checks tiers, may block/hold/auto-approve
                    │  Service    │  reads: ApprovalRules, PolicyDecision
                    └──────┬──────┘
                           │ ApprovalDecision
                           ▼
                    ┌─────────────┐
                    │  Audit Log  │  append-only, MUST succeed before tool call
                    │             │  writes: ToolCallRecorded (pre-execution)
                    └──────┬──────┘
                           │ audit_receipt
                           ▼
                    ┌─────────────┐
                    │    Tool     │  ONLY module with adapter reference
                    │   Gateway   │  enforces: allowlist, rate limits, idempotency
                    └──────┬──────┘  kill switch check HERE (last gate)
                           │
                           ▼
                    ┌─────────────┐
                    │  Exchange   │  PaperAdapter / CCXTAdapter
                    │  Adapter    │  NO direct callers except ToolGateway
                    └─────────────┘
```

**Rules:**
- Agents MAY read from adapter (positions, balances) via ToolGateway read methods (no policy needed for reads).
- Agents MUST NOT hold adapter references. Period.
- PolicyEvaluator has NO side effects. It is a pure function: `(action, snapshot) -> decision`.
- ApprovalService MAY block and emit events. It MUST NOT call ToolGateway.
- AuditLog MUST be available. If unavailable, ToolGateway refuses all calls.
- ToolGateway checks kill switch as final gate (after policy + approval + audit).

### What Gets Refactored (Not Replaced)

| Existing Module | Change |
|----------------|--------|
| `GovernanceGate` | Becomes a thin wrapper calling `PolicyEvaluator` + `ApprovalService`. Eventually deprecated in favor of direct control plane calls. |
| `PolicyEngine` | Renamed/upgraded to `PolicyEvaluator`. Made fail-closed. Missing fields fail. Type errors fail. |
| `PolicyStore` | Renamed to `PolicyRegistry`. Integrity checks on load. Production lock for shadow mode. |
| `ApprovalManager` | Upgraded to `ApprovalService`. Escalation validation. Authority checks. |
| `ExecutionEngine` | Stops holding adapter. Constructs `ProposedAction` and submits to control plane. Becomes orchestrator of the state machine. |
| `main.py` | All direct adapter calls removed. TP/SL, startup recon, position reads all go through ToolGateway. |

---

## 3) CODE-LEVEL SPEC (INTERFACES + TYPES)

### 3.1 Core Types (`control_plane/action_types.py`)

```python
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ToolName(str, Enum):
    """Allowlisted tool names. No other tools exist."""
    SUBMIT_ORDER = "submit_order"
    CANCEL_ORDER = "cancel_order"
    CANCEL_ALL_ORDERS = "cancel_all_orders"
    AMEND_ORDER = "amend_order"
    BATCH_SUBMIT_ORDERS = "batch_submit_orders"
    SET_TRADING_STOP = "set_trading_stop"
    SET_LEVERAGE = "set_leverage"
    SET_POSITION_MODE = "set_position_mode"
    # Read-only (no policy required, but audited):
    GET_POSITIONS = "get_positions"
    GET_BALANCES = "get_balances"
    GET_OPEN_ORDERS = "get_open_orders"
    GET_INSTRUMENT = "get_instrument"
    GET_FUNDING_RATE = "get_funding_rate"


MUTATING_TOOLS: frozenset[ToolName] = frozenset({
    ToolName.SUBMIT_ORDER,
    ToolName.CANCEL_ORDER,
    ToolName.CANCEL_ALL_ORDERS,
    ToolName.AMEND_ORDER,
    ToolName.BATCH_SUBMIT_ORDERS,
    ToolName.SET_TRADING_STOP,
    ToolName.SET_LEVERAGE,
    ToolName.SET_POSITION_MODE,
})


class ApprovalTier(str, Enum):
    T0_AUTONOMOUS = "T0_autonomous"     # No human needed
    T1_NOTIFY = "T1_notify"             # Execute + notify
    T2_APPROVE = "T2_approve"           # Hold until 1 approval
    T3_DUAL_APPROVE = "T3_dual_approve" # Hold until 2 approvals


class ActionScope(BaseModel):
    """Scope of a proposed action. Used for policy scoping and audit."""
    strategy_id: str
    symbol: str
    exchange: str = "bybit"
    actor: str = ""  # agent_id that proposed this


class ProposedAction(BaseModel):
    """An agent's request to perform a side effect. Immutable after creation."""
    action_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    correlation_id: str = Field(default_factory=_uuid)
    causation_id: str = ""  # event_id that caused this action

    tool_name: ToolName
    scope: ActionScope
    request_params: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""

    # Computed at creation
    request_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.request_hash:
            payload = f"{self.tool_name.value}:{self.idempotency_key}:{self.scope.model_dump_json()}"
            self.request_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]


class PolicyDecision(BaseModel):
    """Result of deterministic policy evaluation. Immutable."""
    decision_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    action_id: str  # references ProposedAction.action_id
    correlation_id: str = ""

    allowed: bool
    tier: ApprovalTier = ApprovalTier.T0_AUTONOMOUS
    sizing_multiplier: float = 1.0
    reasons: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    shadow_violations: list[str] = Field(default_factory=list)

    policy_set_version: str = ""
    snapshot_hash: str = ""  # hash of the portfolio/context snapshot used

    # For replay: exact inputs used
    context_snapshot: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    """Result of the approval service check."""
    approval_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    action_id: str
    correlation_id: str = ""

    approved: bool
    tier: ApprovalTier
    decided_by: list[str] = Field(default_factory=list)  # who approved
    reason: str = ""

    # If pending, this is the request ID to poll
    pending_request_id: str | None = None
    expires_at: datetime | None = None


class ToolCallResult(BaseModel):
    """Result of a ToolGateway.call() invocation."""
    result_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    action_id: str
    correlation_id: str = ""

    tool_name: ToolName
    success: bool
    response: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    response_hash: str = ""

    # Timing
    latency_ms: float = 0.0

    # Idempotency: was this a replay of a previous call?
    was_idempotent_replay: bool = False


class AuditEntry(BaseModel):
    """Single entry in the append-only audit log."""
    entry_id: str = Field(default_factory=_uuid)
    timestamp: datetime = Field(default_factory=_now)
    correlation_id: str
    causation_id: str = ""
    actor: str = ""
    scope: ActionScope | None = None

    event_type: str  # "proposed_action", "policy_evaluated", "approval_decision", etc.
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.payload_hash:
            import json
            raw = json.dumps(self.payload, sort_keys=True, default=str)
            self.payload_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
```

### 3.2 ToolGateway (`control_plane/tool_gateway.py`)

```python
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from .action_types import (
    ApprovalDecision,
    ApprovalTier,
    AuditEntry,
    MUTATING_TOOLS,
    PolicyDecision,
    ProposedAction,
    ToolCallResult,
    ToolName,
)

logger = logging.getLogger(__name__)


class ToolGateway:
    """The ONLY path for side effects. No agent may call the adapter directly.

    Enforces:
        1. Allowlisted tools only (ToolName enum)
        2. Policy evaluation BEFORE execution (for mutating tools)
        3. Approval BEFORE execution (if policy decision requires it)
        4. Audit log append BEFORE execution (mandatory, fail-closed)
        5. Idempotency keys prevent double-submit
        6. Kill switch check (final gate)
        7. Rate limiting per scope
        8. Signed ToolCallRecorded events (request_hash + response_hash)
    """

    def __init__(
        self,
        adapter: "IExchangeAdapter",
        policy_evaluator: "PolicyEvaluator",
        approval_service: "ApprovalService",
        audit_log: "AuditLog",
        event_bus: "IEventBus",
        kill_switch_fn: Any = None,
        rate_limits: dict[str, int] | None = None,  # tool_name -> max per minute
    ) -> None:
        self._adapter = adapter  # ONLY reference to adapter in entire system
        self._policy = policy_evaluator
        self._approval = approval_service
        self._audit = audit_log
        self._event_bus = event_bus
        self._kill_switch_fn = kill_switch_fn
        self._rate_limits = rate_limits or {}
        self._idempotency_cache: dict[str, ToolCallResult] = {}
        self._call_counts: dict[str, list[float]] = {}  # tool -> [timestamps]

    async def call(
        self,
        proposed: ProposedAction,
    ) -> ToolCallResult:
        """Execute a proposed action through the full control plane.

        Flow:
            1. Validate tool is allowlisted
            2. Check idempotency cache
            3. For mutating tools: evaluate policy
            4. For mutating tools: check approval
            5. Append to audit log (MANDATORY - fail if unavailable)
            6. Check kill switch (for mutating tools)
            7. Check rate limit
            8. Execute via adapter
            9. Record result in audit log
            10. Publish ToolCallRecorded event

        Raises:
            ControlPlaneUnavailable: if audit log or policy evaluator is down
        """
        t0 = time.monotonic()

        # 1. Validate tool name (already typed, but defense in depth)
        if proposed.tool_name not in ToolName:
            return self._reject(proposed, "unknown_tool")

        # 2. Idempotency check
        if proposed.idempotency_key and proposed.idempotency_key in self._idempotency_cache:
            cached = self._idempotency_cache[proposed.idempotency_key]
            return cached.model_copy(update={"was_idempotent_replay": True})

        is_mutating = proposed.tool_name in MUTATING_TOOLS

        # 3. Policy evaluation (mutating only)
        policy_decision: PolicyDecision | None = None
        if is_mutating:
            try:
                policy_decision = self._policy.evaluate(proposed)
            except Exception as exc:
                # FAIL CLOSED: policy evaluator error = BLOCK
                logger.error("Policy evaluator failed, blocking action: %s", exc)
                await self._emit_incident("policy_evaluator_unavailable", proposed, str(exc))
                return self._reject(proposed, f"policy_evaluator_error: {exc}")

            if not policy_decision.allowed:
                await self._audit_decision(proposed, policy_decision, None)
                return self._reject(proposed, f"policy_blocked: {'; '.join(policy_decision.reasons)}")

        # 4. Approval check (mutating only, if policy requires non-T0 tier)
        approval_decision: ApprovalDecision | None = None
        if is_mutating and policy_decision and policy_decision.tier != ApprovalTier.T0_AUTONOMOUS:
            try:
                approval_decision = await self._approval.request(
                    policy_decision, proposed
                )
            except Exception as exc:
                logger.error("Approval service failed, blocking action: %s", exc)
                await self._emit_incident("approval_service_unavailable", proposed, str(exc))
                return self._reject(proposed, f"approval_service_error: {exc}")

            if not approval_decision.approved:
                if approval_decision.pending_request_id:
                    return self._pending(proposed, approval_decision)
                return self._reject(proposed, f"approval_denied: {approval_decision.reason}")

        # 5. Audit log (MANDATORY - fail closed)
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
                    "policy_decision_id": policy_decision.decision_id if policy_decision else None,
                    "approval_id": approval_decision.approval_id if approval_decision else None,
                },
            ))
        except Exception as exc:
            logger.error("Audit log unavailable, blocking all side effects: %s", exc)
            await self._emit_incident("audit_log_unavailable", proposed, str(exc))
            return self._reject(proposed, f"audit_log_unavailable: {exc}")

        # 6. Kill switch (mutating only, FINAL gate)
        if is_mutating and await self._is_kill_switch_active():
            return self._reject(proposed, "kill_switch_active")

        # 7. Rate limit
        if not self._check_rate_limit(proposed.tool_name):
            return self._reject(proposed, "rate_limit_exceeded")

        # 8. Execute
        try:
            response = await self._dispatch(proposed)
            elapsed = (time.monotonic() - t0) * 1000
            response_hash = hashlib.sha256(
                str(response).encode()
            ).hexdigest()[:16]

            result = ToolCallResult(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                tool_name=proposed.tool_name,
                success=True,
                response=response,
                response_hash=response_hash,
                latency_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            result = ToolCallResult(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                tool_name=proposed.tool_name,
                success=False,
                error=str(exc),
                latency_ms=elapsed,
            )

        # 9. Record result in audit log
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
            logger.error("Failed to record tool call result in audit", exc_info=True)

        # 10. Publish ToolCallRecorded event
        # (event schema defined in section 4)

        # Cache for idempotency
        if proposed.idempotency_key:
            self._idempotency_cache[proposed.idempotency_key] = result

        return result

    async def _dispatch(self, proposed: ProposedAction) -> dict[str, Any]:
        """Route the proposed action to the correct adapter method."""
        params = proposed.request_params
        match proposed.tool_name:
            case ToolName.SUBMIT_ORDER:
                from agentic_trading.core.events import OrderIntent
                intent = OrderIntent(**params["intent"])
                ack = await self._adapter.submit_order(intent)
                return ack.model_dump()
            case ToolName.CANCEL_ORDER:
                ack = await self._adapter.cancel_order(
                    params["order_id"], params["symbol"]
                )
                return ack.model_dump()
            case ToolName.CANCEL_ALL_ORDERS:
                acks = await self._adapter.cancel_all_orders(params.get("symbol"))
                return {"acks": [a.model_dump() for a in acks]}
            case ToolName.AMEND_ORDER:
                ack = await self._adapter.amend_order(**params)
                return ack.model_dump()
            case ToolName.SET_TRADING_STOP:
                result = await self._adapter.set_trading_stop(**params)
                return result
            case ToolName.SET_LEVERAGE:
                result = await self._adapter.set_leverage(
                    params["symbol"], params["leverage"]
                )
                return result
            case ToolName.GET_POSITIONS:
                positions = await self._adapter.get_positions(params.get("symbol"))
                return {"positions": [p.model_dump() for p in positions]}
            case ToolName.GET_BALANCES:
                balances = await self._adapter.get_balances()
                return {"balances": [b.model_dump() for b in balances]}
            case ToolName.GET_OPEN_ORDERS:
                orders = await self._adapter.get_open_orders(params.get("symbol"))
                return {"orders": [o.model_dump() for o in orders]}
            case ToolName.GET_INSTRUMENT:
                inst = await self._adapter.get_instrument(params["symbol"])
                return inst.model_dump()
            case ToolName.GET_FUNDING_RATE:
                rate = await self._adapter.get_funding_rate(params["symbol"])
                return {"rate": str(rate)}
            case _:
                raise ValueError(f"Unknown tool: {proposed.tool_name}")

    def _reject(self, proposed: ProposedAction, reason: str) -> ToolCallResult:
        return ToolCallResult(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            tool_name=proposed.tool_name,
            success=False,
            error=reason,
        )

    def _pending(self, proposed: ProposedAction, approval: ApprovalDecision) -> ToolCallResult:
        return ToolCallResult(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            tool_name=proposed.tool_name,
            success=False,
            error=f"pending_approval:{approval.pending_request_id}",
            response={"pending_request_id": approval.pending_request_id},
        )

    async def _is_kill_switch_active(self) -> bool:
        if self._kill_switch_fn is None:
            return False
        import inspect
        result = self._kill_switch_fn()
        if inspect.isawaitable(result):
            return await result
        return bool(result)

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

    async def _emit_incident(
        self, incident_type: str, proposed: ProposedAction, detail: str
    ) -> None:
        """Emit an incident event on the system topic."""
        try:
            from agentic_trading.core.events import SystemHealth
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
                "Cannot emit incident event: %s", incident_type, exc_info=True
            )

    async def read(self, tool_name: ToolName, params: dict[str, Any]) -> dict[str, Any]:
        """Convenience for read-only tool calls (no policy/approval needed)."""
        if tool_name in MUTATING_TOOLS:
            raise ValueError(f"Use call() for mutating tool: {tool_name}")
        proposed = ProposedAction(
            tool_name=tool_name,
            scope=ActionScope(strategy_id="", symbol=params.get("symbol", ""), actor="system"),
            request_params=params,
        )
        result = await self.call(proposed)
        return result.response
```

### 3.3 PolicyEvaluator (`control_plane/policy_evaluator.py`)

```python
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from .action_types import ApprovalTier, PolicyDecision, ProposedAction, ToolName

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """Deterministic, replayable policy evaluation. No side effects.

    Rules:
        - FAIL CLOSED: any error -> allowed=False
        - Missing context field -> allowed=False (not True)
        - Type mismatch -> allowed=False
        - Shadow mode only in paper/backtest
        - Deterministic: same input -> same output (no time-based logic)
    """

    def __init__(
        self,
        policy_registry: "PolicyRegistry",
        portfolio_state_provider: Any = None,
        mode: str = "live",  # "live" | "paper" | "backtest"
    ) -> None:
        self._registry = policy_registry
        self._portfolio_provider = portfolio_state_provider
        self._mode = mode

    def evaluate(self, proposed: ProposedAction) -> PolicyDecision:
        """Evaluate a proposed action against all active policy sets.

        Returns PolicyDecision with:
            - allowed: bool
            - tier: ApprovalTier (what level of approval is needed)
            - sizing_multiplier: 0.0 to 1.0
            - reasons: why each rule passed/failed
            - policy_set_version: version of the policy set used
            - snapshot_hash: hash of the context used (for replay)

        NEVER raises. Returns allowed=False on any internal error.
        """
        try:
            return self._evaluate_inner(proposed)
        except Exception as exc:
            logger.error(
                "PolicyEvaluator internal error, returning BLOCKED: %s", exc,
                exc_info=True,
            )
            return PolicyDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                allowed=False,
                reasons=[f"policy_evaluator_internal_error: {exc}"],
            )

    def _evaluate_inner(self, proposed: ProposedAction) -> PolicyDecision:
        """Core evaluation logic."""
        context = self._build_context(proposed)
        snapshot_hash = hashlib.sha256(
            json.dumps(context, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        all_reasons: list[str] = []
        all_failed: list[str] = []
        all_shadow: list[str] = []
        min_mult = 1.0
        max_tier = ApprovalTier.T0_AUTONOMOUS

        active_sets = self._registry.get_active_sets()
        policy_version = self._registry.version

        for policy_set in active_sets:
            # Shadow mode: only in paper/backtest
            if policy_set.mode == "shadow" and self._mode == "live":
                raise ValueError(
                    f"Shadow-mode policy set '{policy_set.set_id}' "
                    f"cannot be active in live mode"
                )

            for rule in policy_set.rules:
                if not self._rule_in_scope(rule, context):
                    continue

                result = self._evaluate_rule(rule, context)
                if not result["passed"]:
                    if policy_set.mode == "shadow":
                        all_shadow.append(f"{rule.rule_id}: {result['reason']}")
                    else:
                        all_failed.append(f"{rule.rule_id}: {result['reason']}")
                        all_reasons.append(result["reason"])
                        min_mult = min(min_mult, result.get("sizing_mult", 0.0))

                # Check if rule specifies approval tier
                if hasattr(rule, "approval_tier") and rule.approval_tier:
                    tier = ApprovalTier(rule.approval_tier)
                    if tier.value > max_tier.value:
                        max_tier = tier

        allowed = len(all_failed) == 0
        if not allowed:
            min_mult = 0.0  # BLOCK means zero sizing

        return PolicyDecision(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            allowed=allowed,
            tier=max_tier,
            sizing_multiplier=min_mult,
            reasons=all_reasons,
            failed_rules=all_failed,
            shadow_violations=all_shadow,
            policy_set_version=policy_version,
            snapshot_hash=snapshot_hash,
            context_snapshot=context,
        )

    def _build_context(self, proposed: ProposedAction) -> dict[str, Any]:
        """Build the evaluation context from the proposed action and portfolio state."""
        ctx: dict[str, Any] = {
            "tool_name": proposed.tool_name.value,
            "strategy_id": proposed.scope.strategy_id,
            "symbol": proposed.scope.symbol,
            "exchange": proposed.scope.exchange,
            "actor": proposed.scope.actor,
        }
        ctx.update(proposed.request_params)

        # Add portfolio state if available
        if self._portfolio_provider and callable(self._portfolio_provider):
            state = self._portfolio_provider()
            ctx["portfolio_gross_exposure"] = float(state.gross_exposure)
            ctx["portfolio_net_exposure"] = float(state.net_exposure)

        return ctx

    def _rule_in_scope(self, rule: Any, context: dict[str, Any]) -> bool:
        """Check if a rule applies to this context."""
        if hasattr(rule, "strategy_ids") and rule.strategy_ids is not None:
            sid = context.get("strategy_id", "")
            if sid and sid not in rule.strategy_ids:
                return False
        if hasattr(rule, "symbols") and rule.symbols is not None:
            sym = context.get("symbol", "")
            if sym and sym not in rule.symbols:
                return False
        if hasattr(rule, "tool_names") and rule.tool_names is not None:
            tool = context.get("tool_name", "")
            if tool and tool not in rule.tool_names:
                return False
        return True

    def _evaluate_rule(self, rule: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single rule. Returns dict with passed, reason, sizing_mult."""
        field_value = self._resolve_field(rule.field_path, context)

        # FAIL CLOSED: missing field
        if field_value is None:
            return {
                "passed": False,
                "reason": f"required_field_missing: {rule.field_path}",
                "sizing_mult": 0.0,
            }

        try:
            passed = self._check_condition(rule.operator, field_value, rule.threshold)
        except (TypeError, ValueError) as exc:
            # FAIL CLOSED: type mismatch
            return {
                "passed": False,
                "reason": f"type_mismatch: {rule.field_path} ({exc})",
                "sizing_mult": 0.0,
            }

        if passed:
            return {"passed": True, "reason": "ok", "sizing_mult": 1.0}
        return {
            "passed": False,
            "reason": f"{rule.field_path} {rule.operator} {rule.threshold} violated (actual={field_value})",
            "sizing_mult": getattr(rule, "sizing_multiplier", 0.0),
        }

    @staticmethod
    def _resolve_field(path: str, context: dict[str, Any]) -> Any:
        """Resolve a dot-path field from the context dict."""
        parts = path.split(".")
        current: Any = context
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            if current is None:
                return None
        return current

    @staticmethod
    def _check_condition(operator: str, value: Any, threshold: Any) -> bool:
        """Pure comparison. Raises on type mismatch (caller handles)."""
        match operator:
            case "LT": return float(value) < float(threshold)
            case "LE": return float(value) <= float(threshold)
            case "GT": return float(value) > float(threshold)
            case "GE": return float(value) >= float(threshold)
            case "EQ": return value == threshold
            case "NE": return value != threshold
            case "IN": return value in threshold
            case "NOT_IN": return value not in threshold
            case "BETWEEN":
                v = float(value)
                return float(threshold[0]) <= v <= float(threshold[1])
            case _:
                raise ValueError(f"Unknown operator: {operator}")
```

### 3.4 ApprovalService (`control_plane/approval_service.py`)

```python
from __future__ import annotations

import logging
from typing import Any

from .action_types import ApprovalDecision, ApprovalTier, PolicyDecision, ProposedAction

logger = logging.getLogger(__name__)


class ApprovalService:
    """Tiered approval workflow.

    Tiers:
        T0_AUTONOMOUS: Auto-approved. No human.
        T1_NOTIFY: Execute immediately, send notification.
        T2_APPROVE: Hold until 1 authorized approver signs off.
        T3_DUAL_APPROVE: Hold until 2 authorized approvers sign off.
    """

    def __init__(
        self,
        event_bus: "IEventBus",
        auto_approve_t1: bool = True,
        approval_timeout_seconds: int = 300,
    ) -> None:
        self._event_bus = event_bus
        self._auto_approve_t1 = auto_approve_t1
        self._timeout = approval_timeout_seconds
        self._pending: dict[str, _PendingApproval] = {}

    async def request(
        self, policy_decision: PolicyDecision, proposed: ProposedAction
    ) -> ApprovalDecision:
        """Evaluate approval requirement and return decision.

        For T0: auto-approve immediately.
        For T1: auto-approve + emit notification.
        For T2/T3: create pending request and return pending status.
        """
        tier = policy_decision.tier

        if tier == ApprovalTier.T0_AUTONOMOUS:
            return ApprovalDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                approved=True,
                tier=tier,
                decided_by=["system:autonomous"],
                reason="T0_autonomous",
            )

        if tier == ApprovalTier.T1_NOTIFY and self._auto_approve_t1:
            await self._notify(proposed, policy_decision)
            return ApprovalDecision(
                action_id=proposed.action_id,
                correlation_id=proposed.correlation_id,
                approved=True,
                tier=tier,
                decided_by=["system:t1_auto"],
                reason="T1_notify_auto_approved",
            )

        # T2 and T3: create pending approval
        from datetime import datetime, timezone, timedelta
        request_id = proposed.action_id  # reuse for correlation
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._timeout)

        self._pending[request_id] = _PendingApproval(
            request_id=request_id,
            proposed=proposed,
            policy_decision=policy_decision,
            tier=tier,
            required_approvals=1 if tier == ApprovalTier.T2_APPROVE else 2,
            approvals=[],
            expires_at=expires_at,
        )

        # Emit approval request event
        await self._emit_approval_requested(proposed, policy_decision, request_id, tier)

        return ApprovalDecision(
            action_id=proposed.action_id,
            correlation_id=proposed.correlation_id,
            approved=False,
            tier=tier,
            reason="awaiting_approval",
            pending_request_id=request_id,
            expires_at=expires_at,
        )

    async def approve(
        self,
        request_id: str,
        decided_by: str,
        authority_level: ApprovalTier,
    ) -> ApprovalDecision:
        """Approve a pending request. Validates authority level.

        The approver's authority_level must be >= the request's tier.
        For T3_DUAL_APPROVE, needs 2 distinct approvers.
        """
        pending = self._pending.get(request_id)
        if pending is None:
            raise ValueError(f"No pending approval: {request_id}")

        # Authority check: approver must have sufficient authority
        tier_rank = {
            ApprovalTier.T0_AUTONOMOUS: 0,
            ApprovalTier.T1_NOTIFY: 1,
            ApprovalTier.T2_APPROVE: 2,
            ApprovalTier.T3_DUAL_APPROVE: 3,
        }
        if tier_rank[authority_level] < tier_rank[pending.tier]:
            raise PermissionError(
                f"Approver authority {authority_level.value} insufficient "
                f"for {pending.tier.value} request"
            )

        # No self-approval: actor cannot approve their own action
        if decided_by == pending.proposed.scope.actor:
            raise PermissionError("Self-approval is prohibited")

        # No duplicate approvers
        if decided_by in pending.approvals:
            raise ValueError(f"Already approved by: {decided_by}")

        pending.approvals.append(decided_by)

        if len(pending.approvals) >= pending.required_approvals:
            del self._pending[request_id]
            return ApprovalDecision(
                action_id=pending.proposed.action_id,
                correlation_id=pending.proposed.correlation_id,
                approved=True,
                tier=pending.tier,
                decided_by=pending.approvals,
                reason="approved",
            )

        return ApprovalDecision(
            action_id=pending.proposed.action_id,
            correlation_id=pending.proposed.correlation_id,
            approved=False,
            tier=pending.tier,
            reason=f"need {pending.required_approvals - len(pending.approvals)} more approvals",
            pending_request_id=request_id,
        )

    async def reject(self, request_id: str, decided_by: str, reason: str) -> ApprovalDecision:
        """Reject a pending request."""
        pending = self._pending.pop(request_id, None)
        if pending is None:
            raise ValueError(f"No pending approval: {request_id}")
        return ApprovalDecision(
            action_id=pending.proposed.action_id,
            correlation_id=pending.proposed.correlation_id,
            approved=False,
            tier=pending.tier,
            decided_by=[decided_by],
            reason=f"rejected: {reason}",
        )

    async def expire_stale(self) -> list[str]:
        """Expire and clean up timed-out requests. Returns expired request IDs."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        expired = []
        for rid, pending in list(self._pending.items()):
            if pending.expires_at and now > pending.expires_at:
                del self._pending[rid]
                expired.append(rid)
        return expired

    async def _notify(self, proposed: ProposedAction, decision: PolicyDecision) -> None:
        """Emit T1 notification event."""
        try:
            from agentic_trading.core.events import SystemHealth
            await self._event_bus.publish("system", SystemHealth(
                component="approval_service",
                healthy=True,
                message=f"T1 notification: {proposed.tool_name.value} for {proposed.scope.symbol}",
                details={"action_id": proposed.action_id},
            ))
        except Exception:
            logger.warning("Failed to emit T1 notification", exc_info=True)

    async def _emit_approval_requested(
        self, proposed: ProposedAction, decision: PolicyDecision,
        request_id: str, tier: ApprovalTier,
    ) -> None:
        """Emit ApprovalRequested event."""
        try:
            from agentic_trading.core.events import ApprovalRequested
            await self._event_bus.publish("governance.approval", ApprovalRequested(
                request_id=request_id,
                strategy_id=proposed.scope.strategy_id,
                symbol=proposed.scope.symbol,
                action_type=proposed.tool_name.value,
                trigger="policy_tier",
                escalation_level=tier.value,
                reason=f"Policy requires {tier.value} approval",
            ))
        except Exception:
            logger.warning("Failed to emit approval request event", exc_info=True)


class _PendingApproval:
    """Internal state for a pending approval."""
    __slots__ = (
        "request_id", "proposed", "policy_decision", "tier",
        "required_approvals", "approvals", "expires_at",
    )

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
```

### 3.5 AuditLog (`control_plane/audit_log.py`)

```python
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .action_types import AuditEntry

logger = logging.getLogger(__name__)


class AuditLog:
    """Append-only audit event journal.

    Phase 1: In-memory with file persistence.
    Phase 2: Postgres-backed with WAL.

    Contract:
        - append() MUST succeed or raise (no silent drops)
        - read() returns all entries for a correlation_id
        - Entries are immutable after append
        - No delete/update operations
    """

    def __init__(self, persist_path: str | None = None, max_memory_entries: int = 100_000) -> None:
        self._entries: list[AuditEntry] = []
        self._by_correlation: dict[str, list[AuditEntry]] = defaultdict(list)
        self._by_action: dict[str, list[AuditEntry]] = defaultdict(list)
        self._persist_path = persist_path
        self._max = max_memory_entries
        self._available = True

    async def append(self, entry: AuditEntry) -> None:
        """Append an audit entry. Raises if the log is unavailable.

        This is the fail-closed contract: if we can't record it, the
        caller (ToolGateway) must not proceed with the side effect.
        """
        if not self._available:
            raise RuntimeError("AuditLog is unavailable")

        self._entries.append(entry)
        self._by_correlation[entry.correlation_id].append(entry)
        if entry.causation_id:
            self._by_action[entry.causation_id].append(entry)

        # Persist if configured
        if self._persist_path:
            try:
                import aiofiles
                async with aiofiles.open(self._persist_path, "a") as f:
                    await f.write(entry.model_dump_json() + "\n")
            except Exception as exc:
                # Persistence failure makes the log unavailable
                self._available = False
                raise RuntimeError(f"AuditLog persistence failed: {exc}") from exc

        # Memory cap
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]

    def read(self, correlation_id: str) -> list[AuditEntry]:
        """Read all entries for a correlation_id (the full action trace)."""
        return list(self._by_correlation.get(correlation_id, []))

    def read_by_action(self, action_id: str) -> list[AuditEntry]:
        """Read all entries causally linked to an action_id."""
        return list(self._by_action.get(action_id, []))

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        """For testing: toggle availability."""
        self._available = available
```

---

## 4) EVENT SCHEMAS (CANONICAL MINIMUM)

All events extend `BaseEvent` (already in `core/events.py`). New events added to same file.

### 4.1 Standard Envelope Fields (Already Present in BaseEvent)

```python
# Already in BaseEvent:
event_id: str       # UUID
timestamp: datetime  # UTC
trace_id: str       # correlation across the lifecycle
source_module: str

# NEW fields to add to BaseEvent:
correlation_id: str = ""   # links all events for one proposed action
causation_id: str = ""     # event_id of the event that caused this one
actor: str = ""            # agent_id or "system"
payload_hash: str = ""     # SHA-256 of the payload (for integrity)
scope: str = ""            # "strategy_id:symbol" for scoping
```

### 4.2 New Event Models

```python
# --- control_plane topic ---

class AgentProposedAction(BaseEvent):
    """Published when an agent proposes a side effect."""
    source_module: str = "control_plane"
    action_id: str
    tool_name: str
    scope_strategy: str = ""
    scope_symbol: str = ""
    scope_actor: str = ""
    request_hash: str = ""
    idempotency_key: str = ""


class PolicyEvaluated(BaseEvent):
    """Published after deterministic policy evaluation."""
    source_module: str = "control_plane.policy"
    action_id: str
    decision_id: str
    allowed: bool
    tier: str = "T0_autonomous"
    sizing_multiplier: float = 1.0
    reasons: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    shadow_violations: list[str] = Field(default_factory=list)
    policy_set_version: str = ""
    snapshot_hash: str = ""


class ApprovalRequestedCP(BaseEvent):
    """Published when a T2+ action is held for approval."""
    source_module: str = "control_plane.approval"
    action_id: str
    request_id: str
    tier: str
    scope_strategy: str = ""
    scope_symbol: str = ""
    required_approvals: int = 1
    expires_at: datetime | None = None
    reason: str = ""


class ApprovalGranted(BaseEvent):
    """Published when an approval request is fully approved."""
    source_module: str = "control_plane.approval"
    action_id: str
    request_id: str
    tier: str
    decided_by: list[str] = Field(default_factory=list)


class ApprovalDenied(BaseEvent):
    """Published when an approval request is rejected or expired."""
    source_module: str = "control_plane.approval"
    action_id: str
    request_id: str
    tier: str
    decided_by: str = ""
    reason: str = ""


class ToolCallRecorded(BaseEvent):
    """Published after every ToolGateway call (success or failure)."""
    source_module: str = "control_plane.tool_gateway"
    action_id: str
    tool_name: str
    success: bool
    request_hash: str = ""
    response_hash: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    was_idempotent_replay: bool = False


# --- execution topic (existing, RENAMED for clarity) ---
# OrderIntent -> already exists
# OrderAck -> already exists (now published BY ToolGateway)
# FillEvent -> already exists

class OrderSubmitted(BaseEvent):
    """Published when an order is submitted to the venue (after all gates)."""
    source_module: str = "control_plane.tool_gateway"
    action_id: str
    order_id: str
    client_order_id: str
    symbol: str
    exchange: str = "bybit"
    side: str
    order_type: str
    qty: str  # Decimal as string
    price: str | None = None


class VenueAck(BaseEvent):
    """Published when the venue acknowledges an order."""
    source_module: str = "control_plane.tool_gateway"
    action_id: str
    order_id: str
    venue_status: str
    venue_timestamp: datetime | None = None
    message: str = ""


class FillReceived(BaseEvent):
    """Published when a fill is received from the venue."""
    source_module: str = "control_plane.tool_gateway"
    action_id: str
    fill_id: str
    order_id: str
    symbol: str
    side: str
    price: str
    qty: str
    fee: str
    is_maker: bool = False


# --- system topic ---

class DegradedModeEnabled(BaseEvent):
    """Published when the system enters degraded mode."""
    source_module: str = "control_plane"
    mode: str  # "RISK_OFF_ONLY", "READ_ONLY", "FULL_STOP"
    reason: str = ""
    triggered_by: str = ""
    blocked_tools: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)


class IncidentCreated(BaseEvent):
    """Published when an incident is detected."""
    source_module: str = "control_plane"
    incident_id: str = Field(default_factory=_uuid)
    severity: str  # "warning", "critical", "emergency"
    component: str
    description: str
    auto_action: str = ""  # "degraded_mode", "kill_switch", "none"
```

### 4.3 Event Bus Topic Registry Update

```python
# Add to event_bus/schemas.py:
TOPIC_SCHEMAS = {
    # ... existing ...
    "control_plane": [
        AgentProposedAction, PolicyEvaluated,
        ApprovalRequestedCP, ApprovalGranted, ApprovalDenied,
        ToolCallRecorded,
        OrderSubmitted, VenueAck, FillReceived,
    ],
}

# Topic write ACLs (enforced by event bus):
TOPIC_WRITE_ACL = {
    "execution": ["control_plane.tool_gateway", "execution"],
    "control_plane": ["control_plane"],
    "system": ["control_plane", "governance.canary", "risk"],
    "governance": ["governance", "control_plane.policy"],
    "governance.approval": ["control_plane.approval", "governance.approval"],
}
```

---

## 5) EXECUTION STATE MACHINE (IMPLEMENTATION READY)

### 5.1 States

```
INTENT_RECEIVED        # OrderIntent received by ExecutionEngine
  |
  v
PREFLIGHT_POLICY       # PolicyEvaluator running (deterministic, <10ms)
  |
  +-- BLOCKED          # Policy rejected -> terminal
  |
  v
AWAITING_APPROVAL      # T2/T3 approval required (timeout: 300s default)
  |
  +-- APPROVAL_DENIED  # Rejected -> terminal
  +-- APPROVAL_EXPIRED # Timeout -> terminal
  |
  v
SUBMITTING             # ToolGateway dispatching to adapter (timeout: 30s)
  |
  +-- SUBMIT_FAILED    # Exchange error after retries -> terminal
  |
  v
SUBMITTED              # VenueAck received, order is live
  |
  v
MONITORING             # Watching for fills, status changes (timeout: configurable)
  |
  +-- PARTIALLY_FILLED # Partial fill received
  |     |
  |     v
  |   MONITORING       # Continue watching
  |
  v
COMPLETE               # Fully filled -> terminal (happy path)
  |
  v
ABORT                  # Cancel requested or deviation threshold exceeded
  |
  v
POST_TRADE             # Post-trade risk check, journal recording
  |
  v
TERMINAL               # Done. Immutable state. Audit sealed.
```

### 5.2 State Machine Implementation

```python
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    INTENT_RECEIVED = "intent_received"
    PREFLIGHT_POLICY = "preflight_policy"
    AWAITING_APPROVAL = "awaiting_approval"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    MONITORING = "monitoring"
    PARTIALLY_FILLED = "partially_filled"
    COMPLETE = "complete"
    ABORT = "abort"
    POST_TRADE = "post_trade"
    TERMINAL = "terminal"
    # Error terminals
    BLOCKED = "blocked"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_EXPIRED = "approval_expired"
    SUBMIT_FAILED = "submit_failed"


TERMINAL_STATES = frozenset({
    OrderState.TERMINAL,
    OrderState.BLOCKED,
    OrderState.APPROVAL_DENIED,
    OrderState.APPROVAL_EXPIRED,
    OrderState.SUBMIT_FAILED,
})

# Valid transitions (from -> to)
TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.INTENT_RECEIVED: frozenset({OrderState.PREFLIGHT_POLICY}),
    OrderState.PREFLIGHT_POLICY: frozenset({
        OrderState.AWAITING_APPROVAL, OrderState.SUBMITTING, OrderState.BLOCKED,
    }),
    OrderState.AWAITING_APPROVAL: frozenset({
        OrderState.SUBMITTING, OrderState.APPROVAL_DENIED, OrderState.APPROVAL_EXPIRED,
    }),
    OrderState.SUBMITTING: frozenset({
        OrderState.SUBMITTED, OrderState.SUBMIT_FAILED,
    }),
    OrderState.SUBMITTED: frozenset({OrderState.MONITORING}),
    OrderState.MONITORING: frozenset({
        OrderState.PARTIALLY_FILLED, OrderState.COMPLETE, OrderState.ABORT,
    }),
    OrderState.PARTIALLY_FILLED: frozenset({
        OrderState.MONITORING, OrderState.COMPLETE, OrderState.ABORT,
    }),
    OrderState.COMPLETE: frozenset({OrderState.POST_TRADE}),
    OrderState.ABORT: frozenset({OrderState.POST_TRADE}),
    OrderState.POST_TRADE: frozenset({OrderState.TERMINAL}),
}


class OrderLifecycle:
    """State machine for a single order's lifecycle.

    Enforces valid transitions, timeouts, and records all transitions
    for audit.
    """

    def __init__(
        self,
        action_id: str,
        correlation_id: str,
        timeouts: dict[OrderState, float] | None = None,
    ) -> None:
        self.action_id = action_id
        self.correlation_id = correlation_id
        self.state = OrderState.INTENT_RECEIVED
        self.history: list[tuple[OrderState, OrderState, float]] = []  # (from, to, timestamp)
        self.created_at = time.monotonic()
        self._timeouts = timeouts or {
            OrderState.PREFLIGHT_POLICY: 5.0,      # 5s for policy eval
            OrderState.AWAITING_APPROVAL: 300.0,    # 5min for human approval
            OrderState.SUBMITTING: 30.0,            # 30s for exchange submission
            OrderState.MONITORING: 3600.0,          # 1h for order to fill
        }
        self._state_entered_at: float = time.monotonic()

        # Context: accumulated during lifecycle
        self.policy_decision: Any = None
        self.approval_decision: Any = None
        self.tool_result: Any = None
        self.fills: list[Any] = []
        self.error: str | None = None

    def transition(self, new_state: OrderState) -> None:
        """Transition to a new state. Raises ValueError on invalid transition."""
        if self.state in TERMINAL_STATES:
            raise ValueError(
                f"Cannot transition from terminal state {self.state.value}"
            )
        valid = TRANSITIONS.get(self.state, frozenset())
        if new_state not in valid:
            raise ValueError(
                f"Invalid transition: {self.state.value} -> {new_state.value}. "
                f"Valid: {[s.value for s in valid]}"
            )
        now = time.monotonic()
        self.history.append((self.state, new_state, now))
        self.state = new_state
        self._state_entered_at = now

    def is_timed_out(self) -> bool:
        """Check if the current state has exceeded its timeout."""
        timeout = self._timeouts.get(self.state)
        if timeout is None:
            return False
        return (time.monotonic() - self._state_entered_at) > timeout

    def time_in_state(self) -> float:
        """Seconds spent in current state."""
        return time.monotonic() - self._state_entered_at

    def total_time(self) -> float:
        """Total seconds since creation."""
        return time.monotonic() - self.created_at

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_audit_dict(self) -> dict[str, Any]:
        """Serialize for audit logging."""
        return {
            "action_id": self.action_id,
            "correlation_id": self.correlation_id,
            "state": self.state.value,
            "history": [
                {"from": f.value, "to": t.value, "at": ts}
                for f, t, ts in self.history
            ],
            "total_time_s": self.total_time(),
            "fill_count": len(self.fills),
            "error": self.error,
        }
```

### 5.3 Timeout / Deviation / Rollback Behaviors

| State | Timeout | On Timeout |
|-------|---------|------------|
| `PREFLIGHT_POLICY` | 5s | -> BLOCKED (policy_evaluator_timeout) |
| `AWAITING_APPROVAL` | 300s | -> APPROVAL_EXPIRED |
| `SUBMITTING` | 30s | -> SUBMIT_FAILED (exchange_timeout) |
| `MONITORING` | 3600s | -> ABORT (fill_timeout) + cancel on exchange |

**Deviation thresholds leading to escalation:**
- Fill price deviation > 0.5% from expected -> emit `IncidentCreated(severity="warning")`
- Fill price deviation > 2.0% from expected -> emit `IncidentCreated(severity="critical")` + transition to ABORT
- Partial fill > 80% but no further fills for 5 minutes -> escalate to T2 approval for remaining

**Rollback actions (ABORT state):**
1. Cancel remaining open orders via `ToolGateway.call(CANCEL_ORDER)`
2. If partially filled, do NOT auto-close (that would require new governance evaluation)
3. Emit `IncidentCreated` with details of what was filled vs intended
4. Transition to POST_TRADE for risk check

**Stop conditions (enforced in SUBMITTING and MONITORING):**
- Policy engine down -> refuse new submissions; existing orders continue to monitor
- Audit log down -> refuse ALL tool calls; existing orders marked for manual review
- Degraded mode active -> check which tools are allowed

---

## 6) RECON + INCIDENT INTEGRATION (FAILURE BEHAVIOR)

### 6.1 Reconciliation Triggering Incidents

```
ReconciliationLoop runs every N seconds (configurable, default 60s)
  |
  v
Compare: local state vs exchange state
  |
  +-- Orders: local SUBMITTED not on exchange -> IncidentCreated(severity=warning)
  +-- Positions: size mismatch > threshold -> IncidentCreated(severity=critical)
  +-- Balances: deviation > threshold -> IncidentCreated(severity=warning)
  |
  v
Drift above threshold?
  |
  +-- Position drift > 5% -> DegradedModeEnabled(RISK_OFF_ONLY) for that symbol
  +-- Position drift > 15% -> KillSwitchActivated for that symbol
  +-- Balance drift > 10% -> DegradedModeEnabled(READ_ONLY) globally
  |
  v
Auto-repair ONLY via ToolGateway:
  - Recon proposes repairs as ProposedAction events
  - Each repair goes through PolicyEvaluator (with "recon_repair" scope)
  - No direct state mutation
```

### 6.2 State Uncertainty Handling

| Scenario | Detection | Response |
|----------|-----------|----------|
| Order submitted, no venue ack within 30s | SUBMITTING timeout | Cancel attempt via ToolGateway. If cancel also times out -> MANUAL_REVIEW incident |
| Fill received for unknown order | FillEvent with no matching local order | IncidentCreated(severity=critical). Do NOT auto-journal. Alert operator. |
| Position exists on exchange but not locally | Recon position mismatch | IncidentCreated(severity=critical). Enter RISK_OFF_ONLY for that symbol. |
| Network partition (adapter connectivity lost) | Health check failure | DegradedModeEnabled(FULL_STOP). All pending MONITORING orders escalated to manual review. |
| Redis event bus down | Bus health check failure | All new submissions blocked. Existing orders continue via adapter polling (degraded). |

### 6.3 Stop Conditions as Explicit Policies

These are declarative policy rules (loaded into PolicyEvaluator at startup):

```python
# In default policy set (always active, cannot be shadow-mode):
SYSTEM_POLICIES = [
    PolicyRule(
        rule_id="sys_policy_engine_required",
        field_path="system.policy_evaluator_healthy",
        operator="EQ",
        threshold=True,
        action_on_fail=GovernanceAction.BLOCK,
        description="Block all mutating actions if policy evaluator is unhealthy",
    ),
    PolicyRule(
        rule_id="sys_audit_log_required",
        field_path="system.audit_log_healthy",
        operator="EQ",
        threshold=True,
        action_on_fail=GovernanceAction.BLOCK,
        description="Block all mutating actions if audit log is unavailable",
    ),
    PolicyRule(
        rule_id="sys_degraded_risk_off",
        field_path="system.degraded_mode",
        operator="NE",
        threshold="FULL_STOP",
        action_on_fail=GovernanceAction.KILL,
        description="Kill switch if system is in FULL_STOP degraded mode",
    ),
    PolicyRule(
        rule_id="sys_degraded_reduce_only",
        field_path="system.degraded_mode",
        operator="NE",
        threshold="RISK_OFF_ONLY",
        action_on_fail=GovernanceAction.REDUCE_SIZE,
        description="Only reduce-only orders allowed in RISK_OFF_ONLY mode",
        # This rule only fails if degraded_mode == "RISK_OFF_ONLY"
        # The order must also be reduce_only=True to pass
    ),
]
```

### 6.4 Degraded Mode Definitions

| Mode | Allowed Tools | Blocked Tools | Trigger |
|------|--------------|---------------|---------|
| `NORMAL` | All | None | Default |
| `RISK_OFF_ONLY` | `cancel_order`, `cancel_all_orders`, read-only tools, `set_trading_stop` (reduce-only TP/SL) | `submit_order`, `batch_submit_orders`, `amend_order`, `set_leverage` | Position drift > 5%, recon critical |
| `READ_ONLY` | Read-only tools only | All mutating tools | Balance drift > 10%, adapter partial failure |
| `FULL_STOP` | None (not even reads) | All | Kill switch, audit log down, multiple critical incidents |

---

## 7) TEST PLAN (PROVE IT'S INSTITUTIONAL)

### 7.1 Acceptance Tests (GIVEN/WHEN/THEN)

```python
# ============================================================
# A1: No side effect without PolicyEvaluated + Approval + ToolCallRecorded
# ============================================================

class TestNoBypassPath:
    """GIVEN a running system with ToolGateway as sole adapter accessor
    WHEN any agent attempts to submit an order
    THEN the audit log contains: AgentProposedAction -> PolicyEvaluated ->
         (ApprovalGranted if tier > T0) -> ToolCallRecorded
    AND the ToolCallRecorded event references all prior events via correlation_id
    """

    async def test_submit_order_produces_full_audit_trail(self):
        # GIVEN
        gateway, audit, policy, adapter = _make_control_plane()

        # WHEN
        proposed = _make_proposed_action(ToolName.SUBMIT_ORDER)
        result = await gateway.call(proposed)

        # THEN
        entries = audit.read(proposed.correlation_id)
        event_types = [e.event_type for e in entries]
        assert "tool_call_pre_execution" in event_types
        assert "tool_call_recorded" in event_types
        assert result.success

    async def test_no_adapter_call_without_audit_entry(self):
        # GIVEN adapter that tracks calls
        tracker = AdapterCallTracker()
        gateway, audit, _, _ = _make_control_plane(adapter=tracker)

        # WHEN: audit log available
        proposed = _make_proposed_action(ToolName.SUBMIT_ORDER)
        await gateway.call(proposed)

        # THEN: adapter was called AND audit has entries
        assert tracker.submit_count == 1
        assert audit.entry_count >= 2  # pre + post


# ============================================================
# A2: Idempotency prevents double-submit
# ============================================================

class TestIdempotency:
    """GIVEN an order already submitted with idempotency_key="abc"
    WHEN the same idempotency_key is submitted again
    THEN the adapter is NOT called a second time
    AND the cached result is returned with was_idempotent_replay=True
    """

    async def test_idempotent_replay(self):
        tracker = AdapterCallTracker()
        gateway, _, _, _ = _make_control_plane(adapter=tracker)

        proposed = _make_proposed_action(
            ToolName.SUBMIT_ORDER, idempotency_key="abc-123"
        )
        result1 = await gateway.call(proposed)
        result2 = await gateway.call(proposed)

        assert tracker.submit_count == 1  # Only ONE adapter call
        assert not result1.was_idempotent_replay
        assert result2.was_idempotent_replay
        assert result1.response == result2.response


# ============================================================
# A3: Policy engine down -> trading blocked
# ============================================================

class TestPolicyEngineDown:
    """GIVEN the policy evaluator raises an exception
    WHEN a mutating action is proposed
    THEN the action is BLOCKED (not allowed through)
    AND an IncidentCreated event is emitted on the system topic
    AND the adapter is NOT called
    """

    async def test_policy_down_blocks_trading(self):
        tracker = AdapterCallTracker()
        broken_policy = BrokenPolicyEvaluator()  # raises on evaluate()
        gateway, _, _, bus = _make_control_plane(
            adapter=tracker, policy=broken_policy
        )

        proposed = _make_proposed_action(ToolName.SUBMIT_ORDER)
        result = await gateway.call(proposed)

        assert not result.success
        assert "policy_evaluator_error" in result.error
        assert tracker.submit_count == 0

    async def test_read_operations_still_work_when_policy_down(self):
        broken_policy = BrokenPolicyEvaluator()
        gateway, _, _, _ = _make_control_plane(policy=broken_policy)

        # Reads don't need policy evaluation
        result = await gateway.read(
            ToolName.GET_POSITIONS, {"symbol": "BTC/USDT"}
        )
        assert "positions" in result


# ============================================================
# A4: Audit store down -> side effects blocked
# ============================================================

class TestAuditDown:
    """GIVEN the audit log is unavailable (raises on append)
    WHEN any mutating action is proposed
    THEN the action is BLOCKED
    AND the adapter is NOT called
    AND an incident event is emitted
    """

    async def test_audit_down_blocks_all_mutations(self):
        tracker = AdapterCallTracker()
        gateway, audit, _, _ = _make_control_plane(adapter=tracker)
        audit.set_available(False)

        proposed = _make_proposed_action(ToolName.SUBMIT_ORDER)
        result = await gateway.call(proposed)

        assert not result.success
        assert "audit_log_unavailable" in result.error
        assert tracker.submit_count == 0

    async def test_audit_down_blocks_cancel_too(self):
        tracker = AdapterCallTracker()
        gateway, audit, _, _ = _make_control_plane(adapter=tracker)
        audit.set_available(False)

        proposed = _make_proposed_action(ToolName.CANCEL_ORDER)
        result = await gateway.call(proposed)

        assert not result.success
        assert "audit_log_unavailable" in result.error


# ============================================================
# A5: Degraded mode RISK_OFF_ONLY
# ============================================================

class TestDegradedMode:
    """GIVEN the system is in RISK_OFF_ONLY degraded mode
    WHEN a new SUBMIT_ORDER action is proposed
    THEN it is BLOCKED

    WHEN a CANCEL_ORDER (reduce-only) action is proposed
    THEN it is ALLOWED
    """

    async def test_risk_off_blocks_new_orders(self):
        gateway, _, policy, _ = _make_control_plane()
        # Simulate degraded mode in policy context
        policy.set_system_state("degraded_mode", "RISK_OFF_ONLY")

        proposed = _make_proposed_action(ToolName.SUBMIT_ORDER)
        result = await gateway.call(proposed)

        assert not result.success

    async def test_risk_off_allows_cancels(self):
        tracker = AdapterCallTracker()
        gateway, _, policy, _ = _make_control_plane(adapter=tracker)
        policy.set_system_state("degraded_mode", "RISK_OFF_ONLY")

        proposed = _make_proposed_action(ToolName.CANCEL_ORDER)
        result = await gateway.call(proposed)

        assert result.success


# ============================================================
# A6: Recon drift triggers incident and stops trading
# ============================================================

class TestReconDrift:
    """GIVEN reconciliation detects position drift > 5%
    WHEN the drift is reported
    THEN an IncidentCreated event is emitted
    AND DegradedModeEnabled(RISK_OFF_ONLY) is published
    AND subsequent SUBMIT_ORDER for that symbol is BLOCKED
    """

    async def test_position_drift_triggers_degraded_mode(self):
        gateway, _, policy, bus = _make_control_plane()
        events_captured = []
        await bus.subscribe("system", "test", lambda e: events_captured.append(e))

        # Simulate recon detecting drift
        await recon_report_drift(gateway, symbol="BTC/USDT", drift_pct=7.0)

        # Check events
        incident_events = [
            e for e in events_captured if hasattr(e, "incident_id")
        ]
        assert len(incident_events) >= 1

        # Check trading blocked
        proposed = _make_proposed_action(
            ToolName.SUBMIT_ORDER,
            symbol="BTC/USDT",
        )
        result = await gateway.call(proposed)
        assert not result.success


# ============================================================
# A7: Approval tiers enforced
# ============================================================

class TestApprovalTiers:
    """GIVEN a policy that requires T2 approval for orders > $50k
    WHEN a $100k order is proposed
    THEN it is held pending approval
    AND the ToolGateway does NOT call the adapter
    UNTIL an authorized approver approves it
    """

    async def test_t2_approval_holds_order(self):
        tracker = AdapterCallTracker()
        gateway, _, policy, _ = _make_control_plane(adapter=tracker)
        # Configure policy to require T2 for large orders
        policy.add_tier_rule("order_notional_usd", "GT", 50000, ApprovalTier.T2_APPROVE)

        proposed = _make_proposed_action(
            ToolName.SUBMIT_ORDER,
            request_params={"intent": _make_intent(notional=100_000).model_dump()},
        )
        result = await gateway.call(proposed)

        assert not result.success
        assert "pending_approval" in result.error
        assert tracker.submit_count == 0

    async def test_self_approval_rejected(self):
        gateway, _, _, _ = _make_control_plane()
        approval_service = gateway._approval

        pending_id = "test-pending"
        # ... create pending approval with actor="agent_1"

        with pytest.raises(PermissionError, match="Self-approval"):
            await approval_service.approve(
                pending_id, decided_by="agent_1", authority_level=ApprovalTier.T2_APPROVE
            )
```

### 7.2 Integration Test Harness

```python
class FakeExchangeAdapter:
    """Deterministic adapter for integration tests.

    Tracks all calls, returns configurable responses.
    Simulates latency, errors, partial fills.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.submit_count = 0
        self.cancel_count = 0
        self._positions: list[Position] = []
        self._balances: list[Balance] = []
        self._should_fail: bool = False
        self._fill_immediately: bool = True

    async def submit_order(self, intent: OrderIntent) -> OrderAck:
        self.calls.append({"method": "submit_order", "intent": intent})
        self.submit_count += 1
        if self._should_fail:
            raise ExchangeError("Simulated failure")
        status = OrderStatus.FILLED if self._fill_immediately else OrderStatus.SUBMITTED
        return OrderAck(
            order_id=f"fake-{self.submit_count}",
            client_order_id=intent.dedupe_key,
            symbol=intent.symbol,
            exchange=intent.exchange,
            status=status,
        )

    # ... implement all IExchangeAdapter methods similarly


def _make_control_plane(
    adapter=None, policy=None, approval=None, audit=None, bus=None
):
    """Factory for fully wired control plane for tests."""
    from agentic_trading.event_bus.memory_bus import MemoryEventBus

    adapter = adapter or FakeExchangeAdapter()
    bus = bus or MemoryEventBus()
    audit = audit or AuditLog()
    policy = policy or PolicyEvaluator(DefaultPolicyRegistry(), mode="paper")
    approval = approval or ApprovalService(bus)

    gateway = ToolGateway(
        adapter=adapter,
        policy_evaluator=policy,
        approval_service=approval,
        audit_log=audit,
        event_bus=bus,
    )
    return gateway, audit, policy, bus
```

---

## 8) IMPLEMENTATION PLAN (7 DAYS)

### Day 1: Foundation + Kill the Bypass Paths

**Tasks:**
1. Create `src/agentic_trading/control_plane/` directory
2. Implement `action_types.py` (all type definitions from section 3.1)
3. Implement `audit_log.py` (section 3.5)
4. Fix B5/B6: Change governance/gate.py exception handlers to fail-closed
5. Fix B10/B11: Change policy_engine.py missing field and type mismatch to fail-closed
6. Fix B17: Wire `kill_switch_fn` to GovernanceCanary in main.py

**Output:** Types compile. Audit log passes unit tests. Fail-closed fixes pass tests.
**Tests:** 15 unit tests for types + audit log + fail-closed behavior.

### Day 2: ToolGateway Core

**Tasks:**
1. Implement `tool_gateway.py` (section 3.2)
2. Implement the `_dispatch()` method for all tool types
3. Implement idempotency cache
4. Implement rate limiting
5. Write unit tests with FakeExchangeAdapter

**Output:** ToolGateway routes all IExchangeAdapter methods. Idempotency works. Rate limits work.
**Tests:** 20 unit tests covering each tool name, idempotency, rate limits.

### Day 3: PolicyEvaluator + ApprovalService

**Tasks:**
1. Implement `policy_evaluator.py` (section 3.3)
2. Implement `approval_service.py` (section 3.4)
3. Migrate existing PolicyEngine rules to new evaluator format
4. Write acceptance tests A1-A4 from section 7

**Output:** Policy evaluation is deterministic, fail-closed. Approval tiers work. Tests A1-A4 pass.
**Tests:** 25 tests covering policy evaluation, approval tiers, fail-closed behavior.

### Day 4: State Machine + ExecutionEngine Rewire

**Tasks:**
1. Implement `state_machine.py` (section 5.2)
2. Rewire `ExecutionEngine` to:
   - Construct `ProposedAction` instead of calling adapter directly
   - Use `ToolGateway.call()` for submission
   - Manage `OrderLifecycle` state machine per order
3. Remove `self._adapter` from ExecutionEngine (it goes through ToolGateway)
4. Remove `governance_gate` from ExecutionEngine (PolicyEvaluator handles it)
5. Fix B1: Make ToolGateway required (not optional) in ExecutionEngine

**Output:** ExecutionEngine uses control plane. No direct adapter references. State machine tracks lifecycle.
**Tests:** 15 tests for state machine transitions + rewired engine.

### Day 5: Eliminate Direct Adapter Access

**Tasks:**
1. Fix B3: Replace `adapter.set_trading_stop()` calls in main.py with ToolGateway calls
2. Fix B4: Replace `adapter._ccxt.fetch_positions()` with `ToolGateway.read(GET_POSITIONS)`
3. Fix B8/B9: Remove `.adapter` properties from ExecutionAgent and AgentOrchestrator
4. Fix B7: Wire governance gate in orchestrator
5. Ensure ALL adapter references in main.py go through ToolGateway
6. Run grep to verify: `adapter.submit_order`, `adapter.cancel`, `adapter.set_trading_stop`, `adapter._ccxt` appear ONLY in tool_gateway.py and ccxt_adapter.py

**Output:** Zero direct adapter calls outside ToolGateway. Grep proof.
**Tests:** Integration test that the full main.py startup path uses ToolGateway for all mutations.

### Day 6: Event Schemas + Recon Integration

**Tasks:**
1. Add new event types to `core/events.py` (section 4.2)
2. Update `event_bus/schemas.py` with new topic registry
3. Rewire ReconciliationLoop (B18): emit ProposedAction for repairs instead of direct state mutation
4. Implement degraded mode tracking in PolicyEvaluator
5. Write acceptance tests A5-A6 (degraded mode, recon drift)

**Output:** Full event schema. Recon uses control plane. Degraded mode works.
**Tests:** 15 tests for events, recon integration, degraded mode.

### Day 7: Integration Tests + Hardening + Documentation

**Tasks:**
1. Write full integration test harness (section 7.2)
2. Write acceptance test A7 (approval tiers)
3. Run ALL existing tests (expect ~1144 to still pass, minus those touching refactored code that need updates)
4. Fix any broken existing tests
5. Run the bypass path grep audit:
   ```bash
   grep -rn "adapter\.submit_order\|adapter\.cancel_order\|adapter\.set_trading_stop\|adapter\._ccxt\|adapter\.amend_order\|adapter\.batch_submit" src/ --include="*.py" | grep -v "tool_gateway\|ccxt_adapter\|paper\|backtest\|test_\|conftest"
   ```
6. Verify: zero hits = bypass paths are closed

**Output:** All tests pass. Grep audit clean. System is provably governed.
**Tests:** Full test suite green. Grep proof committed.

### What to DEFER (Not in 7 Days)

- **Postgres-backed AuditLog**: Phase 1 uses file persistence. Upgrade to Postgres WAL in week 2.
- **Topic-level write ACLs on event bus (B19/B20)**: Requires event bus refactor. Medium priority.
- **Backtest governance simulation (B9)**: Not needed for live safety. Deferred.
- **Approval UI/API**: T2/T3 approvals need a human interface. For now: CLI-based or webhook.
- **Policy file integrity checks (B17)**: Use checksums in week 2. No filesystem attacks in scope now.
- **Event bus signed events**: Nice to have but not blocking for control plane.

### Definition of Done

1. `grep` for direct adapter calls outside ToolGateway returns **zero results**
2. All acceptance tests A1-A7 **pass**
3. Existing test suite: **zero new failures** (fixing broken tests counts)
4. `make ci` (lint + test + typecheck) **passes**
5. AuditLog unavailable -> **zero mutating tool calls execute** (tested)
6. PolicyEvaluator unavailable -> **zero mutating tool calls execute** (tested)
7. Kill switch active -> **zero mutating tool calls execute** (tested)
8. Every `ToolCallRecorded` event has a matching `PolicyEvaluated` event for the same `correlation_id` (integration test asserts this)

---

## 9) AGENT TAXONOMY MAPPING (Current vs Target)

Your 13-agent institutional architecture defines clear boundaries. Here's the delta:

### Status Map

| # | Target Agent | Current State | Gap | Priority |
|---|-------------|---------------|-----|----------|
| 1 | **PolicyGateAgent** | Partial: `GovernanceGate` + `PolicyEngine` + `ApprovalManager` exist but wired poorly (optional, fail-open) | **CRITICAL**: Must become mandatory, fail-closed, deterministic. The control plane spec (sections 3.2-3.4) delivers this. Rename/refactor to `PolicyGateAgent`. | Day 1-3 |
| 2 | **MarketIntelligenceAgent** | **EXISTS**: `agents/market_intelligence.py` wraps FeedManager + FeatureEngine. Publishes FeatureVector events. | **LOW GAP**: Add lineage (feature version + input source hashes) to FeatureVector. Add data quality fields. | Week 2 |
| 3 | **DataQualityAgent** | **MISSING** entirely. No feed integrity verification. | **HIGH**: New agent. Compares WS vs REST, detects staleness, triggers RISK_OFF_ONLY. Feeds into PolicyGateAgent context (`system.data_quality_healthy`). | Day 6 (stub) + Week 2 |
| 4 | **SignalAgent** | Partial: Strategies implement `IStrategy.on_candle()` but are called procedurally in main.py, not as autonomous agents. | **MEDIUM**: Wrap strategies in SignalAgent that subscribes to FeatureVector topic and publishes Signal events. Must attach model version + feature hashes. Already partially done by the signal handling in main.py. | Week 2 |
| 5 | **PortfolioAgent** | Partial: `PortfolioManager` exists in `portfolio/manager.py` but called inline, not as an event-driven agent. | **MEDIUM**: Wrap in PortfolioAgent. Convert from sync call to event-driven: consumes Signal, produces TradeIntentProposed. Must do constraint satisfaction + netting. | Week 2 |
| 6 | **ExecutionPlannerAgent** | **MISSING**. Currently, OrderIntent goes directly from signal -> execution. No plan decomposition, no slicing, no venue routing. | **DEFER**: Not needed for single-venue (Bybit) single-strategy execution. Add when multi-venue or algorithmic execution is required. For now, the ExecutionAgent handles simple plans. |
| 7 | **ExecutionAgent** | **EXISTS**: `agents/execution.py` wraps `ExecutionEngine`. But it holds a raw adapter reference (B8) and governance is optional (B7). | **CRITICAL**: Rewire to use ToolGateway exclusively. Remove adapter property. State machine (section 5) tracks lifecycle. | Day 4-5 |
| 8 | **RFQNegotiationAgent** | **MISSING**. Not applicable for Bybit perps. | **DEFER**: Add when OTC/block liquidity is needed. |
| 9 | **SurveillanceAgent** | **MISSING**. No pattern detection, no case management. | **DEFER to Week 3**: Critical for institutional, but not blocking for the control plane. Stub the agent interface and wire it to the audit event stream. |
| 10 | **ReconciliationAgent** | **EXISTS**: `ReconciliationLoop` in `execution/reconciliation.py`. But auto-repair bypasses governance (B18). | **HIGH**: Rewire repairs to go through ToolGateway. Add materiality thresholds. Freeze trading scope on repeated breaks. | Day 6 |
| 11 | **ReportingAgent** | **MISSING** as an agent. Grafana dashboards exist but no automated report generation. | **DEFER to Week 3**: Stub interface. Block publishing if recon not green. |
| 12 | **IncidentResponseAgent** | **MISSING** as a distinct agent. Kill switch exists but incident management is ad-hoc (logging + kill switch). | **HIGH**: New agent. Consumes IncidentCreated events, applies degraded modes, executes runbook steps. Critical for the fail-closed architecture. | Day 6 (stub) + Week 2 |
| 13 | **GovernanceCanary** | **EXISTS**: `governance/canary.py`. But constructed without `kill_switch_fn` (B17) and cannot actually stop the system. | **HIGH**: Wire kill switch. Add synthetic policy eval health checks. Add ToolGateway dry-run checks. | Day 1 (wire fix) + Day 6 (health checks) |

### Key Architectural Alignments from Your Agent Taxonomy

Your descriptions reinforce several critical design decisions in this spec:

1. **PolicyGateAgent is stateless and deterministic** - Aligns exactly with `PolicyEvaluator` in section 3.3. No side effects. Evaluation uses immutable snapshots referenced by hashes.

2. **ExecutionAgent never bypasses PolicyGate/ApprovalService** - This is the entire point of `ToolGateway` (section 3.2). The agent taxonomy makes this a HARD constraint.

3. **ReconciliationAgent repairs require approvals above threshold** - Validates the decision in section 6.1 to route recon repairs through `ProposedAction -> PolicyEvaluator -> ToolGateway`.

4. **IncidentResponseAgent cannot re-enable trading without T3 approval** - Must be enforced in ApprovalService: kill switch deactivation requires `T3_DUAL_APPROVE`.

5. **DataQualityAgent controls degraded modes for safe trading** - The degraded mode definitions in section 6.4 support this. DQA emits `DegradedModeEnabled` events, PolicyEvaluator enforces the restrictions.

6. **All agents: propose-only vs execute** - The taxonomy makes clear that agents 1-6, 8-9, 11-13 are **propose-only**. Only agent 7 (ExecutionAgent) and agent 10 (ReconciliationAgent, for auto-repairs) execute via ToolGateway. This narrows the attack surface to exactly two agents that need ToolGateway access.

### Revised Implementation Priority (Updated with Agent Taxonomy)

**Week 1 (Control Plane Foundation)**:
- PolicyGateAgent (refactor existing GovernanceGate + PolicyEngine)
- ExecutionAgent (rewire through ToolGateway)
- GovernanceCanary (fix wiring)
- ToolGateway + AuditLog + ApprovalService (new)

**Week 2 (Agent Hardening)**:
- DataQualityAgent (new, stub + core logic)
- IncidentResponseAgent (new, stub + degraded mode logic)
- ReconciliationAgent (rewire through ToolGateway)
- MarketIntelligenceAgent (add lineage/hashes)

**Week 3 (Institutional Completeness)**:
- SurveillanceAgent (new, basic pattern detection)
- ReportingAgent (new, basic reporting)
- SignalAgent (wrap strategies)
- PortfolioAgent (wrap portfolio manager)

---

## DESIGN DECISIONS (CONFIRMED)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | **TP/SL governance** | **T0 fast-path** for reduce-only/protective orders | Emergency SL must not be delayed by approval. Reduce-only TP/SL auto-approved at T0 with full audit trail. Non-protective TP/SL gets normal policy evaluation. |
| D2 | **Recon auto-repair** | **T1_NOTIFY**: auto-approved but audited | Low-latency state sync is critical. Execute immediately, notify operator. Repairs above materiality threshold escalate to T2. |
| D3 | **Existing policy rules** | **Migrate** from `default_policies.py` to new `PolicyEvaluator` format | Preserves existing risk limits (max notional, max leverage, exposure caps). Port rules, don't rewrite. |
| D4 | **Approval notifications** | **Event bus + CLI polling** | Simplest. No external deps. `python3 -m agentic_trading approvals list` and `approve <id>`. |
| D5 | **`scripts/test_live_trade.py`** | **DELETE** | Zero ungoverned exchange access in the repo. No exceptions. |
| D6 | **Backtest mode** | **Leave as-is** | Backtest stays fast and deterministic. Accept sizing divergence. Document the divergence. |

### TP/SL Fast-Path Policy Rule (D1)

```python
# Reduce-only TP/SL: T0 auto-approve with audit trail
# Non-protective TP/SL: normal policy evaluation
PolicyRule(
    rule_id="sys_tp_sl_fast_path",
    description="Reduce-only TP/SL auto-approved at T0",
    field_path="tool_name",
    operator="EQ",
    threshold="set_trading_stop",
    approval_tier="T0_autonomous",
    condition_fields={"is_reduce_only": True},
)
```

### Recon Materiality Thresholds (D2)

```python
RECON_REPAIR_TIERS = {
    "low":    ApprovalTier.T1_NOTIFY,       # Balance drift < 1%, position drift < 2%
    "medium": ApprovalTier.T2_APPROVE,       # Balance drift 1-5%, position drift 2-10%
    "high":   ApprovalTier.T3_DUAL_APPROVE,  # Balance drift > 5%, position drift > 10%
}
```
