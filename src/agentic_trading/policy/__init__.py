"""Policy-as-code framework for strategy governance and risk management.

Canonical home for all governance / policy logic.  Prior to this move
the code lived under ``agentic_trading.governance``, which now contains
thin re-export shims for backward compatibility (to be removed in PR 16).

Sub-modules:

- **models**: Declarative policy rule, set, and evaluation result models
- **engine**: Policy evaluation engine with dot-path field resolution
- **store**: Versioned policy management with file persistence
- **default_policies**: Pre-trade and post-trade policy set builders
- **governance_gate**: Single orchestrator for all pre-execution checks
- **approval_manager / approval_models**: Multi-level approval workflows
- **maturity**: Strategy maturity level progression (L0–L4)
- **health_score**: Epistemic debt/credit health scoring
- **impact_classifier**: Per-trade impact tier classification
- **drift_detector**: Live-vs-backtest metric divergence monitoring
- **tokens**: Scoped, time-limited execution tokens
- **canary**: Independent infrastructure health watchdog
- **incident_manager**: Incident lifecycle and degraded mode management
- **strategy_lifecycle**: Evidence-gated strategy lifecycle state machine
- **gate**: Unified PolicyGate facade composing all subsystems
"""

from agentic_trading.policy.approval_manager import ApprovalManager
from agentic_trading.policy.approval_models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRule,
    ApprovalStatus,
    ApprovalSummary,
    ApprovalTrigger,
    EscalationLevel,
)
from agentic_trading.policy.canary import GovernanceCanary
from agentic_trading.policy.drift_detector import DriftDetector
from agentic_trading.policy.engine import PolicyEngine
from agentic_trading.policy.governance_gate import GovernanceGate
from agentic_trading.policy.health_score import HealthTracker
from agentic_trading.policy.impact_classifier import ImpactClassifier
from agentic_trading.policy.maturity import MaturityManager
from agentic_trading.policy.models import PolicyMode, PolicyRule, PolicySet
from agentic_trading.policy.store import PolicyStore
from agentic_trading.policy.gate import PolicyGate
from agentic_trading.policy.tokens import ExecutionToken, TokenManager

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
    "PolicyGate",
    "PolicyMode",
    "PolicyRule",
    "PolicySet",
    "PolicyStore",
    "TokenManager",
]
