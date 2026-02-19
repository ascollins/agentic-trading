"""Backward-compatibility shim — see ``agentic_trading.policy.incident_manager``.

Will be removed in PR 16.
"""

from agentic_trading.policy.incident_manager import *  # noqa: F401, F403
from agentic_trading.policy.incident_manager import (  # noqa: F811
    AUTO_TRIAGE_RULES,
    RECOVERY_CRITERIA,
    Incident,
    IncidentManager,
)

__all__ = [
    "AUTO_TRIAGE_RULES",
    "RECOVERY_CRITERIA",
    "Incident",
    "IncidentManager",
]
