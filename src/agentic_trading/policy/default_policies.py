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

from agentic_trading.core.config import FXRiskConfig, RiskConfig
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


def build_fx_policies(
    config: FXRiskConfig | None = None,
    *,
    mode: PolicyMode = PolicyMode.ENFORCED,
) -> PolicySet:
    """Build FX-specific policy rules.

    These rules enforce FX-specific risk constraints: leverage, spread,
    session hours, rollover, slippage, and pair whitelist.  All rules
    follow the same declarative :class:`PolicyRule` pattern.

    The context dict passed to ``PolicyEngine.evaluate()`` must contain
    the pre-computed values listed in each rule's ``field``.
    """
    cfg = config or FXRiskConfig()

    rules = [
        PolicyRule(
            rule_id="fx_max_leverage",
            name="FX Maximum Leverage",
            description=(
                f"FX leverage must not exceed {cfg.max_leverage}x"
            ),
            field="projected_leverage",
            operator=Operator.LE,
            threshold=float(cfg.max_leverage),
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.FX_CONSTRAINT,
            severity="high",
        ),
        PolicyRule(
            rule_id="fx_max_notional",
            name="FX Maximum Order Notional",
            description=(
                f"FX order must not exceed "
                f"${cfg.max_notional_per_order_usd:,.0f}"
            ),
            field="order_notional_usd",
            operator=Operator.LE,
            threshold=cfg.max_notional_per_order_usd,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.FX_CONSTRAINT,
            severity="high",
        ),
        PolicyRule(
            rule_id="fx_max_spread",
            name="FX Maximum Spread",
            description=(
                f"Spread must not exceed {cfg.max_spread_pips} pips"
            ),
            field="current_spread_pips",
            operator=Operator.LE,
            threshold=cfg.max_spread_pips,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.FX_CONSTRAINT,
            severity="medium",
        ),
        PolicyRule(
            rule_id="fx_session_guard",
            name="FX Session Guard",
            description="Orders only during allowed trading sessions",
            field="session_open",
            operator=Operator.EQ,
            threshold=True,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.FX_CONSTRAINT,
            severity="high",
        ),
        PolicyRule(
            rule_id="fx_weekend_guard",
            name="FX Weekend Guard",
            description="Block orders when FX markets are closed",
            field="fx_market_open",
            operator=Operator.EQ,
            threshold=True,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.FX_CONSTRAINT,
            severity="critical",
            enabled=cfg.block_weekend_orders,
        ),
        PolicyRule(
            rule_id="fx_slippage_limit",
            name="FX Slippage Limit",
            description=(
                f"Slippage must not exceed {cfg.max_slippage_pips} pips"
            ),
            field="expected_slippage_pips",
            operator=Operator.LE,
            threshold=cfg.max_slippage_pips,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.FX_CONSTRAINT,
            severity="medium",
        ),
        PolicyRule(
            rule_id="fx_pair_whitelist",
            name="FX Pair Whitelist",
            description="Only trade whitelisted FX pairs",
            field="symbol",
            operator=Operator.IN,
            threshold=cfg.allowed_pairs,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.FX_CONSTRAINT,
            severity="high",
            enabled=cfg.major_pairs_only,
        ),
        PolicyRule(
            rule_id="fx_rollover_limit",
            name="FX Daily Rollover Cost Limit",
            description=(
                f"Daily rollover cost must not exceed "
                f"${cfg.max_daily_rollover_cost_usd:,.0f}"
            ),
            field="daily_rollover_cost_usd",
            operator=Operator.LE,
            threshold=cfg.max_daily_rollover_cost_usd,
            action=GovernanceAction.REDUCE_SIZE,
            policy_type=PolicyType.FX_CONSTRAINT,
            severity="medium",
        ),
    ]

    return PolicySet(
        set_id="fx_risk",
        name="FX Risk Limits",
        description=(
            "FX-specific risk limits: leverage, spread, session, "
            "rollover, slippage, and pair whitelist."
        ),
        version=1,
        mode=mode,
        rules=rules,
    )


def build_pre_trade_control_policies(
    config: RiskConfig | None = None,
    *,
    mode: PolicyMode = PolicyMode.ENFORCED,
) -> PolicySet:
    """Build institutional pre-trade control policies (spec §4.5).

    These declarative rules cover:
    1. Price collars — reject orders deviating beyond a band from ref price.
    2. Self-match prevention — block orders that would cross own resting orders.
    3. Message throttles — rate-limit order submissions per strategy and symbol.

    The context dict passed to ``PolicyEngine.evaluate()`` must contain
    the pre-computed values listed in each rule's ``field``.
    """
    cfg = config or RiskConfig()

    rules = [
        PolicyRule(
            rule_id="price_collar",
            name="Price Collar",
            description=(
                f"Limit order price must not deviate more than "
                f"{cfg.price_collar_bps:.0f} bps from reference price"
            ),
            field="price_deviation_bps",
            operator=Operator.LE,
            threshold=cfg.price_collar_bps,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.PRE_TRADE_CONTROL,
            severity="high",
        ),
        PolicyRule(
            rule_id="self_match_prevention",
            name="Self-Match Prevention",
            description=(
                "Orders must not cross own resting orders on same venue"
            ),
            field="would_self_match",
            operator=Operator.EQ,
            threshold=False,
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.PRE_TRADE_CONTROL,
            severity="critical",
        ),
        PolicyRule(
            rule_id="message_throttle_strategy",
            name="Message Throttle (Strategy)",
            description=(
                f"Strategy message rate must not exceed "
                f"{cfg.max_messages_per_minute_per_strategy} msgs/min"
            ),
            field="strategy_messages_per_minute",
            operator=Operator.LE,
            threshold=float(cfg.max_messages_per_minute_per_strategy),
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.PRE_TRADE_CONTROL,
            severity="high",
        ),
        PolicyRule(
            rule_id="message_throttle_symbol",
            name="Message Throttle (Symbol)",
            description=(
                f"Per-symbol message rate must not exceed "
                f"{cfg.max_messages_per_minute_per_symbol} msgs/min"
            ),
            field="symbol_messages_per_minute",
            operator=Operator.LE,
            threshold=float(cfg.max_messages_per_minute_per_symbol),
            action=GovernanceAction.BLOCK,
            policy_type=PolicyType.PRE_TRADE_CONTROL,
            severity="high",
        ),
    ]

    return PolicySet(
        set_id="pre_trade_controls",
        name="Pre-Trade Controls",
        description=(
            "Institutional pre-trade controls: price collars, self-match "
            "prevention, and message rate throttles."
        ),
        version=1,
        mode=mode,
        rules=rules,
    )
