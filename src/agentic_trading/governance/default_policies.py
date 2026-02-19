"""Backward-compatibility shim — see ``agentic_trading.policy.default_policies``.

Will be removed in PR 16.
"""

from agentic_trading.policy.default_policies import *  # noqa: F401, F403
from agentic_trading.policy.default_policies import (  # noqa: F811
    build_post_trade_policies,
    build_pre_trade_policies,
    build_strategy_constraint_policies,
)

__all__ = [
    "build_post_trade_policies",
    "build_pre_trade_policies",
    "build_strategy_constraint_policies",
]
