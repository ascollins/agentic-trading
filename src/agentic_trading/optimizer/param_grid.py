"""Parameter grid definitions for strategy optimization.

Defines the search space for each strategy's tunable parameters,
and provides random sampling for efficient search.
"""

from __future__ import annotations

import itertools
import random
from typing import Any


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
        combos.append(dict(zip(keys, combo)))

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
