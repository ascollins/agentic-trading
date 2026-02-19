"""Backward-compatibility shim — see ``agentic_trading.policy.governance_gate``.

Will be removed in PR 16.
"""

from agentic_trading.policy.governance_gate import *  # noqa: F401, F403
from agentic_trading.policy.governance_gate import GovernanceGate  # noqa: F811

__all__ = ["GovernanceGate"]
