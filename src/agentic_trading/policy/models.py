"""Declarative policy-as-code models.

Defines the data structures for the policy engine:

- :class:`PolicyRule`: A single declarative check (field, operator, threshold).
- :class:`PolicySet`: A versioned collection of rules with enforcement mode.
- :class:`PolicyEvalResult`: Result of evaluating one rule against a context.
- :class:`PolicyDecision`: Aggregate result of evaluating all rules.

Rules are pure data -- no imperative logic.  The :class:`PolicyEngine`
in ``engine.py`` evaluates rules against an order context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentic_trading.core.enums import GovernanceAction


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PolicyType(str, Enum):
    """Categories of policy rules (spec §4.2)."""

    RISK_LIMIT = "risk_limit"
    EXECUTION_CONSTRAINT = "execution_constraint"
    COMPLIANCE = "compliance"
    VENUE_CONSTRAINT = "venue_constraint"
    CUSTODY = "custody"
    STRATEGY_CONSTRAINT = "strategy_constraint"
    OPERATIONAL = "operational"
    FX_CONSTRAINT = "fx_constraint"
    PRE_TRADE_CONTROL = "pre_trade_control"


class PolicyMode(str, Enum):
    """Whether a policy is actively enforced or in shadow mode."""

    SHADOW = "shadow"        # Log only, never block
    ENFORCED = "enforced"    # Actively block/reduce


class Operator(str, Enum):
    """Comparison operators for rule evaluation."""

    LT = "lt"                # field < threshold
    LE = "le"                # field <= threshold
    GT = "gt"                # field > threshold
    GE = "ge"                # field >= threshold
    EQ = "eq"                # field == threshold
    NE = "ne"                # field != threshold
    IN = "in"                # field in threshold (list)
    NOT_IN = "not_in"        # field not in threshold (list)
    BETWEEN = "between"      # threshold[0] <= field <= threshold[1]


# ---------------------------------------------------------------------------
# Rule model
# ---------------------------------------------------------------------------


class PolicyRule(BaseModel):
    """A single declarative policy check.

    A rule evaluates a named field from the order context against a
    threshold using a comparison operator.  If the check *fails*
    (i.e. the condition is violated), the rule's ``action`` is taken.

    Example -- "single order notional must not exceed $500k"::

        PolicyRule(
            rule_id="max_notional",
            name="Maximum Order Notional",
            field="order_notional_usd",
            operator=Operator.LE,
            threshold=500_000.0,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.RISK_LIMIT,
        )

    The rule *passes* when ``order_notional_usd <= 500_000.0``.
    It *fails* (triggers the action) when the condition is violated.
    """

    rule_id: str
    name: str
    description: str = ""
    field: str                          # dot-path into order context
    operator: Operator
    threshold: Any                      # scalar, list, or [min, max]
    action: GovernanceAction = GovernanceAction.BLOCK
    policy_type: PolicyType = PolicyType.RISK_LIMIT
    severity: str = "high"              # low, medium, high, critical
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Optional scoping -- if set, rule only applies when scope matches
    strategy_ids: list[str] | None = None   # None = all strategies
    symbols: list[str] | None = None        # None = all symbols
    exchanges: list[str] | None = None      # None = all exchanges

    # Regulatory mapping (spec §7.2) -- links rules to external regs
    regulatory_refs: list[str] = Field(default_factory=list)  # e.g. ["SEC-15c3-5"]


# ---------------------------------------------------------------------------
# PolicySet: versioned collection of rules
# ---------------------------------------------------------------------------


class PolicySet(BaseModel):
    """A versioned, named collection of policy rules.

    Policy sets support:
    - **Versioning**: Increment version on changes for audit trail.
    - **Shadow mode**: Log-only evaluation without blocking.
    - **Enforced mode**: Active blocking/sizing reduction.
    - **Metadata**: Arbitrary key-value pairs for UI/audit.

    Example::

        policy_set = PolicySet(
            set_id="pre_trade_risk_v2",
            name="Pre-Trade Risk Limits",
            version=2,
            mode=PolicyMode.ENFORCED,
            rules=[...],
        )
    """

    set_id: str
    name: str
    description: str = ""
    version: int = 1
    mode: PolicyMode = PolicyMode.ENFORCED
    rules: list[PolicyRule] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Regulatory mapping (spec §7.2)
    regulatory_refs: list[str] = Field(default_factory=list)  # set-level refs

    @property
    def active_rules(self) -> list[PolicyRule]:
        """Return only enabled rules."""
        return [r for r in self.rules if r.enabled]

    def rules_for_regulation(self, ref: str) -> list[PolicyRule]:
        """Return all rules tagged with the given regulatory reference."""
        return [r for r in self.rules if ref in r.regulatory_refs]


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------


class PolicyEvalResult(BaseModel):
    """Result of evaluating a single rule against an order context."""

    rule_id: str
    rule_name: str
    passed: bool
    field: str
    field_value: Any = None
    threshold: Any = None
    operator: str = ""
    action: GovernanceAction = GovernanceAction.ALLOW
    reason: str = ""
    policy_type: PolicyType = PolicyType.RISK_LIMIT
    severity: str = "high"
    shadow: bool = False  # True if rule was in shadow mode
    regulatory_refs: list[str] = Field(default_factory=list)


class PolicyDecision(BaseModel):
    """Aggregate result of evaluating all rules in a policy set."""

    set_id: str
    set_name: str
    version: int
    mode: PolicyMode
    all_passed: bool
    action: GovernanceAction = GovernanceAction.ALLOW
    reason: str = ""
    sizing_multiplier: float = 1.0
    results: list[PolicyEvalResult] = Field(default_factory=list)
    failed_rules: list[PolicyEvalResult] = Field(default_factory=list)
    shadow_violations: list[PolicyEvalResult] = Field(default_factory=list)
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
