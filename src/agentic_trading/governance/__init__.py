"""Soteria-inspired governance framework for strategy lifecycle management.

Provides pre-execution admissibility checks that sit above the risk layer:

- **Maturity levels**: Strategies progress from shadow (L0) to autonomous (L4)
- **Health scoring**: Rolling outcome tracker with debt/credit model
- **Canary watchdog**: Independent infrastructure health verification
- **Impact classification**: Per-trade impact tier assessment
- **Drift detection**: Live-vs-backtest metric divergence monitoring
- **Execution tokens**: Scoped, time-limited, revocable trade authorisations
- **Governance gate**: Single orchestrator that composes all checks
- **Policy engine**: Declarative policy-as-code evaluation
- **Approval workflows**: Multi-level approval gates for high-impact actions

Enable via ``settings.governance.enabled = true`` in config.
"""

from agentic_trading.governance.approval_manager import ApprovalManager
from agentic_trading.governance.approval_models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRule,
    ApprovalStatus,
    ApprovalSummary,
    ApprovalTrigger,
    EscalationLevel,
)
from agentic_trading.governance.canary import GovernanceCanary
from agentic_trading.governance.drift_detector import DriftDetector
from agentic_trading.governance.gate import GovernanceGate
from agentic_trading.governance.health_score import HealthTracker
from agentic_trading.governance.impact_classifier import ImpactClassifier
from agentic_trading.governance.maturity import MaturityManager
from agentic_trading.governance.policy_engine import PolicyEngine
from agentic_trading.governance.policy_models import PolicyMode, PolicyRule, PolicySet
from agentic_trading.governance.policy_store import PolicyStore
from agentic_trading.governance.tokens import ExecutionToken, TokenManager

__all__ = [
    "ApprovalDecision",
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalRule",
    "ApprovalStatus",
    "ApprovalSummary",
    "ApprovalTrigger",
    "EscalationLevel",
    "ExecutionToken",
    "GovernanceCanary",
    "GovernanceGate",
    "DriftDetector",
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
