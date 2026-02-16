"""Institutional control plane — the sole path for all side effects.

Modules:
    action_types: Core type definitions (ProposedAction, PolicyDecision, etc.)
    audit_log: Append-only audit event journal
    tool_gateway: ToolGateway (sole adapter accessor) — Day 2
    policy_evaluator: Deterministic policy evaluation — Day 3
    approval_service: Tiered approval workflow — Day 3
    state_machine: Order lifecycle FSM — Day 4

Day 3 integration::

    from agentic_trading.control_plane.policy_evaluator import (
        CPPolicyEvaluator, build_default_evaluator,
    )
    from agentic_trading.control_plane.approval_service import CPApprovalService
"""
