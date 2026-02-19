"""Backward-compatibility shim — see ``agentic_trading.policy.approval_models``.

Will be removed in PR 16.
"""

from agentic_trading.policy.approval_models import *  # noqa: F401, F403
from agentic_trading.policy.approval_models import (  # noqa: F811
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRule,
    ApprovalStatus,
    ApprovalSummary,
    ApprovalTrigger,
    EscalationLevel,
)

__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalRule",
    "ApprovalStatus",
    "ApprovalSummary",
    "ApprovalTrigger",
    "EscalationLevel",
]
