"""Backward-compatibility shim — canonical code now lives in ``agentic_trading.policy``.

All public names are re-exported so that existing ``from agentic_trading.governance import X``
continues to work.  Will be removed in PR 16.
"""

from agentic_trading.policy import (  # noqa: F401, F403
    ApprovalDecision,
    ApprovalManager,
    ApprovalRequest,
    ApprovalRule,
    ApprovalStatus,
    ApprovalSummary,
    ApprovalTrigger,
    DriftDetector,
    EscalationLevel,
    ExecutionToken,
    GovernanceCanary,
    GovernanceGate,
    HealthTracker,
    ImpactClassifier,
    MaturityManager,
    PolicyEngine,
    PolicyMode,
    PolicyRule,
    PolicySet,
    PolicyStore,
    TokenManager,
)

__all__ = [
    "ApprovalDecision",
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalRule",
    "ApprovalStatus",
    "ApprovalSummary",
    "ApprovalTrigger",
    "DriftDetector",
    "EscalationLevel",
    "ExecutionToken",
    "GovernanceCanary",
    "GovernanceGate",
    "HealthTracker",
    "ImpactClassifier",
    "MaturityManager",
    "PolicyEngine",
    "PolicyMode",
    "PolicyRule",
    "PolicySet",
    "PolicyStore",
    "TokenManager",
]
