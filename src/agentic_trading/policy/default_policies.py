"""Default policy sets extracted from hardcoded risk checks.

These functions create :class:`PolicySet` instances that replicate the
logic currently hardcoded in :class:`PreTradeChecker` and
:class:`PostTradeChecker`, but expressed as declarative rules.

Usage::

    from agentic_trading.policy.default_policies import (
        build_pre_trade_policies,
        build_post_trade_policies,
    )
    from agentic_trading.core.config import RiskConfig

    config = RiskConfig()
    pre_trade_ps = build_pre_trade_policies(config)
    # Register with PolicyEngine
"""

from __future__ import annotations

from agentic_trading.core.config import RiskConfig
from agentic_trading.core.enums import GovernanceAction

from .models import (
    Operator,
    PolicyMode,
    PolicyRule,
    PolicySet,
    PolicyType,
)


def build_pre_trade_policies(
    config: RiskConfig | None = None,
    *,
    mode: PolicyMode = PolicyMode.ENFORCED,
    max_notional: float = 500_000.0,
) -> PolicySet:
    """Build a PolicySet replicating the PreTradeChecker logic.

    Maps to the 5 checks in ``risk/pre_trade.py``:
    1. max_position_size → order_position_pct <= max_single_position_pct
    2. max_notional → order_notional_usd <= max_notional
    3. max_leverage → projected_leverage <= max_portfolio_leverage
    4. exposure_limits → projected_exposure_pct <= max_portfolio_leverage
    5. instrument_limits → order_qty >= instrument_min_qty

    The context dict passed to ``PolicyEngine.evaluate()`` must contain
    the pre-computed values (the engine evaluates rules, not computes
    intermediate values).
    """
    cfg = config or RiskConfig()

    rules = [
        PolicyRule(
            rule_id="max_position_size",
            name="Maximum Position Size",
            description=(
                f"Single position must not exceed "
                f"{cfg.max_single_position_pct:.0%} of portfolio equity"
            ),
            field="position_pct_of_equity",
            operator=Operator.LE,
            threshold=cfg.max_single_position_pct,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.RISK_LIMIT,
            severity="high",
        ),
        PolicyRule(
            rule_id="max_order_notional",
            name="Maximum Order Notional",
            description=(
                f"Single order notional must not exceed "
                f"${max_notional:,.0f}"
            ),
            field="order_notional_usd",
            operator=Operator.LE,
            threshold=max_notional,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.RISK_LIMIT,
            severity="high",
        ),
        PolicyRule(
            rule_id="max_leverage",
            name="Maximum Portfolio Leverage",
            description=(
                f"Projected portfolio leverage must not exceed "
                f"{cfg.max_portfolio_leverage:.1f}x"
            ),
            field="projected_leverage",
            operator=Operator.LE,
            threshold=cfg.max_portfolio_leverage,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.RISK_LIMIT,
            severity="high",
        ),
        PolicyRule(
            rule_id="max_gross_exposure",
            name="Maximum Gross Exposure",
            description=(
                f"Gross exposure must not exceed "
                f"{cfg.max_portfolio_leverage:.1f}x equity"
            ),
            field="projected_exposure_pct",
            operator=Operator.LE,
            threshold=cfg.max_portfolio_leverage,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.RISK_LIMIT,
            severity="high",
        ),
        PolicyRule(
            rule_id="min_order_qty",
            name="Minimum Order Quantity",
            description="Order quantity must meet instrument minimum",
            field="order_qty_above_min",
            operator=Operator.EQ,
            threshold=True,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.EXECUTION_CONSTRAINT,
            severity="medium",
        ),
        PolicyRule(
            rule_id="max_daily_loss",
            name="Maximum Daily Loss",
            description=(
                f"Daily loss must not exceed "
                f"{cfg.max_daily_loss_pct:.0%} of equity"
            ),
            field="daily_loss_pct",
            operator=Operator.LE,
            threshold=cfg.max_daily_loss_pct,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.RISK_LIMIT,
            severity="critical",
        ),
        PolicyRule(
            rule_id="max_drawdown",
            name="Maximum Drawdown",
            description=(
                f"Portfolio drawdown must not exceed "
                f"{cfg.max_drawdown_pct:.0%}"
            ),
            field="current_drawdown_pct",
            operator=Operator.LE,
            threshold=cfg.max_drawdown_pct,
            action=GovernanceAction.KILL,
            policy_type=PolicyType.RISK_LIMIT,
            severity="critical",
        ),
        PolicyRule(
            rule_id="max_correlated_exposure",
            name="Maximum Correlated Exposure",
            description=(
                f"Correlated asset exposure must not exceed "
                f"{cfg.max_correlated_exposure_pct:.0%}"
            ),
            field="correlated_exposure_pct",
            operator=Operator.LE,
            threshold=cfg.max_correlated_exposure_pct,
            action=GovernanceAction.REDUCE_SIZE,
            policy_type=PolicyType.RISK_LIMIT,
            severity="medium",
        ),
    ]

    return PolicySet(
        set_id="pre_trade_risk",
        name="Pre-Trade Risk Limits",
        description=(
            "Declarative risk limits extracted from PreTradeChecker. "
            "Evaluates position size, notional, leverage, exposure, "
            "and instrument constraints."
        ),
        version=1,
        mode=mode,
        rules=rules,
    )


def build_post_trade_policies(
    *,
    max_unexpected_loss_pct: float = 0.02,
    max_leverage_after_fill: float = 5.0,
    max_fill_deviation_pct: float = 0.05,
    mode: PolicyMode = PolicyMode.ENFORCED,
) -> PolicySet:
    """Build a PolicySet replicating the PostTradeChecker logic.

    Maps to the 4 checks in ``risk/post_trade.py``:
    1. position_consistency → position exists after fill
    2. pnl_sanity → unexpected loss within threshold
    3. leverage_spike → post-fill leverage within limit
    4. fill_price_deviation → fill price close to mark price
    """
    rules = [
        PolicyRule(
            rule_id="position_consistency",
            name="Position Consistency",
            description="Position must exist after fill (or be fully closed)",
            field="position_exists_or_closed",
            operator=Operator.EQ,
            threshold=True,
            action=GovernanceAction.PAUSE,
            policy_type=PolicyType.OPERATIONAL,
            severity="high",
        ),
        PolicyRule(
            rule_id="pnl_sanity",
            name="PnL Sanity Check",
            description=(
                f"Unexpected single-fill loss must not exceed "
                f"{max_unexpected_loss_pct:.0%} of equity"
            ),
            field="fill_loss_pct_of_equity",
            operator=Operator.LE,
            threshold=max_unexpected_loss_pct,
            action=GovernanceAction.PAUSE,
            policy_type=PolicyType.RISK_LIMIT,
            severity="critical",
        ),
        PolicyRule(
            rule_id="leverage_spike",
            name="Post-Fill Leverage Spike",
            description=(
                f"Portfolio leverage after fill must not exceed "
                f"{max_leverage_after_fill:.1f}x"
            ),
            field="post_fill_leverage",
            operator=Operator.LE,
            threshold=max_leverage_after_fill,
            action=GovernanceAction.PAUSE,
            policy_type=PolicyType.RISK_LIMIT,
            severity="high",
        ),
        PolicyRule(
            rule_id="fill_price_deviation",
            name="Fill Price Deviation",
            description=(
                f"Fill price must be within "
                f"{max_fill_deviation_pct:.0%} of mark price"
            ),
            field="fill_price_deviation_pct",
            operator=Operator.LE,
            threshold=max_fill_deviation_pct,
            action=GovernanceAction.REDUCE_SIZE,
            policy_type=PolicyType.RISK_LIMIT,
            severity="medium",
        ),
    ]

    return PolicySet(
        set_id="post_trade_risk",
        name="Post-Trade Risk Checks",
        description=(
            "Declarative post-trade checks for position consistency, "
            "PnL sanity, leverage spikes, and fill price deviations."
        ),
        version=1,
        mode=mode,
        rules=rules,
    )


def build_strategy_constraint_policies(
    *,
    mode: PolicyMode = PolicyMode.ENFORCED,
) -> PolicySet:
    """Build strategy-level constraint policies.

    These rules enforce per-strategy operating boundaries such as
    allowed symbols, position limits, and trading hours.
    """
    rules = [
        PolicyRule(
            rule_id="symbol_whitelist",
            name="Symbol Whitelist",
            description="Strategy can only trade whitelisted symbols",
            field="symbol_whitelisted",
            operator=Operator.EQ,
            threshold=True,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.STRATEGY_CONSTRAINT,
            severity="high",
        ),
        PolicyRule(
            rule_id="max_open_positions",
            name="Maximum Open Positions",
            description="Strategy cannot exceed max concurrent positions",
            field="open_position_count",
            operator=Operator.LE,
            threshold=10,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.STRATEGY_CONSTRAINT,
            severity="medium",
        ),
    ]

    return PolicySet(
        set_id="strategy_constraints",
        name="Strategy Constraints",
        description="Per-strategy operating boundaries and limits.",
        version=1,
        mode=mode,
        rules=rules,
    )
