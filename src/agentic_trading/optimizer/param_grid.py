"""Parameter grid definitions for strategy optimization.

Defines the search space for each strategy's tunable parameters,
and provides random sampling for efficient search.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

# The 8 CMT strategies that form the platform's core strategy suite.
CMT_STRATEGIES: list[str] = [
    "multi_tf_ma",
    "rsi_divergence",
    "stochastic_macd",
    "bb_squeeze",
    "mean_reversion_enhanced",
    "supply_demand",
    "fibonacci_confluence",
    "obv_divergence",
]

# Default search spaces per strategy
_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "trend_following": {
        "fast_ema": [9, 12],
        "slow_ema": [21, 26],
        "adx_threshold": [15, 20, 25, 30],
        "atr_multiplier": [1.5, 2.0, 2.5, 3.0],
        "min_confidence": [0.3, 0.4, 0.5],
        "signal_cooldown_minutes": [60, 120, 240],
    },
    "mean_reversion": {
        "bb_period": [20],
        "bb_std": [1.5, 2.0, 2.5],
        "rsi_oversold": [20, 25, 30],
        "rsi_overbought": [70, 75, 80],
        "mean_reversion_score_threshold": [0.3, 0.4, 0.5],
        "min_confidence": [0.3, 0.4, 0.5],
        "signal_cooldown_minutes": [60, 120, 240],
        "require_range_regime": [False],
    },
    "breakout": {
        "donchian_period": [15, 20, 25],
        "volume_confirmation_multiplier": [1.2, 1.5, 2.0],
        "min_confidence": [0.3, 0.4, 0.5],
        "signal_cooldown_minutes": [60, 120, 240],
    },
    "bb_squeeze": {
        "squeeze_percentile": [5, 10, 15, 20],
        "adx_threshold": [15, 20, 25, 30],
        "atr_multiplier": [1.5, 2.0, 2.5, 3.0],
        "min_confidence": [0.5, 0.6, 0.65, 0.7],
        "signal_cooldown_minutes": [360, 720, 1440],
    },
    "stochastic_macd": {
        "stoch_oversold": [15, 20, 25],
        "stoch_overbought": [75, 80, 85],
        "volume_gate": [1.0, 1.2, 1.5],
        "confluence_window": [2, 3, 5],
        "atr_multiplier": [1.5, 2.0, 2.5],
        "min_confidence": [0.5, 0.6, 0.65, 0.7],
        "signal_cooldown_minutes": [360, 720, 1440],
    },
    "multi_tf_ma": {
        "fast_ema": [20, 50],
        "slow_ema": [100, 200],
        "pullback_ema": [14, 21],
        "adx_entry": [15, 20, 25],
        "atr_multiplier": [1.5, 2.0, 2.5, 3.0],
        "min_confidence": [0.5, 0.6, 0.65, 0.7],
        "signal_cooldown_minutes": [360, 720, 1440],
    },
    "rsi_divergence": {
        "rsi_oversold": [25, 30, 35],
        "rsi_overbought": [65, 70, 75],
        "lookback_bars": [20, 30, 50],
        "min_divergence_bars": [3, 5, 7],
        "atr_multiplier": [1.5, 2.0, 2.5],
        "min_confidence": [0.5, 0.6, 0.65, 0.7],
        "signal_cooldown_minutes": [360, 720, 1440],
    },
    # --- CMT strategies without prior grids ---
    "mean_reversion_enhanced": {
        "bb_std": [1.5, 2.0, 2.5],
        "rsi_oversold": [25, 30, 35],
        "rsi_overbought": [65, 70, 75],
        "time_stop_bars": [7, 10, 15],
        "atr_multiplier": [2.0, 2.5, 3.0],
        "min_confidence": [0.3, 0.35, 0.5],
        "weekly_trend_filter": [True, False],
    },
    "supply_demand": {
        "max_demand_distance": [0.01, 0.02, 0.03],
        "max_supply_distance": [0.01, 0.02, 0.03],
        "require_bos": [True, False],
        "volume_gate": [0.8, 1.0, 1.5],
        "atr_multiplier": [1.0, 1.5, 2.0],
        "target_rr_ratio": [1.5, 2.0, 3.0],
        "min_confidence": [0.4, 0.5, 0.6],
    },
    "fibonacci_confluence": {
        "swing_lookback": [30, 50, 75],
        "confluence_band_pct": [0.003, 0.005, 0.008],
        "min_confluence_levels": [2, 3],
        "rsi_long_max": [35, 40, 45],
        "rsi_short_min": [55, 60, 65],
        "atr_multiplier": [1.5, 2.0, 2.5],
        "min_confidence": [0.4, 0.5, 0.6],
    },
    "obv_divergence": {
        "lookback_bars": [20, 30, 50],
        "min_divergence_bars": [3, 5, 7],
        "volume_confirmation": [0.8, 1.0, 1.5],
        "atr_multiplier": [1.5, 2.0, 2.5],
        "min_confidence": [0.4, 0.5, 0.6],
    },
}


def build_param_grid(
    strategy_id: str,
    overrides: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the full Cartesian product of parameter combinations.

    Args:
        strategy_id: Which strategy to build the grid for.
        overrides: Optional dict of param_name → list[values] to override
            the default grid.

    Returns:
        List of parameter dicts, one per combination.
    """
    grid = dict(_PARAM_GRIDS.get(strategy_id, {}))
    if overrides:
        grid.update(overrides)

    if not grid:
        return [{}]

    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]

    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo, strict=True)))

    return combos


def random_sample(
    strategy_id: str,
    n: int = 50,
    seed: int = 42,
    overrides: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Randomly sample N parameter combinations from the grid.

    More efficient than full grid search when the space is large.

    Args:
        strategy_id: Which strategy.
        n: Number of samples to draw.
        seed: RNG seed for reproducibility.
        overrides: Optional grid overrides.

    Returns:
        List of N parameter dicts (or all combos if N > total).
    """
    full_grid = build_param_grid(strategy_id, overrides)

    if n >= len(full_grid):
        return full_grid

    rng = random.Random(seed)
    return rng.sample(full_grid, n)


def get_grid_size(strategy_id: str) -> int:
    """Return the total number of combinations for a strategy."""
    return len(build_param_grid(strategy_id))


def list_strategies_with_grids() -> list[str]:
    """Return strategy IDs that have parameter grids defined."""
    return list(_PARAM_GRIDS.keys())


def strategies_missing_grids(strategy_ids: list[str]) -> list[str]:
    """Return strategy IDs from the input list that have no param grid."""
    return [s for s in strategy_ids if s not in _PARAM_GRIDS]
