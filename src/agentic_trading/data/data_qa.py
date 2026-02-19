"""Backward-compat re-export — canonical location: ``agentic_trading.intelligence.data_qa``.

Will be removed in PR 16.
"""

from agentic_trading.intelligence.data_qa import *  # noqa: F401, F403
from agentic_trading.intelligence.data_qa import (  # noqa: F811
    DataQualityChecker,
    DataQualityIssue,
    Severity,
)

__all__ = ["DataQualityChecker", "DataQualityIssue", "Severity"]
