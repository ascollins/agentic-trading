"""Slippage models for backtesting.

Three models:
1. FixedBPS: constant basis points of slippage
2. VolatilityBased: slippage scales with current ATR/volatility
3. OrderbookImpact: approximates market impact based on order size vs volume
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class SlippageModel(ABC):
    """Base slippage model."""

    @abstractmethod
    def compute_slippage(
        self,
        price: float,
        qty: float,
        is_buy: bool,
        **kwargs,
    ) -> float:
        """Return the slipped execution price.

        Args:
            price: Intended execution price.
            qty: Order quantity.
            is_buy: True for buy orders.
            **kwargs: Model-specific params (atr, volume, etc.)

        Returns:
            Adjusted price after slippage.
        """
        ...


class FixedBPSSlippage(SlippageModel):
    """Constant basis points slippage."""

    def __init__(self, bps: float = 5.0, seed: int | None = None) -> None:
        self._bps = bps / 10_000.0
        self._rng = random.Random(seed)

    def compute_slippage(
        self, price: float, qty: float, is_buy: bool, **kwargs
    ) -> float:
        # Add small random jitter (±20% of base slippage)
        jitter = self._rng.uniform(0.8, 1.2)
        slip = price * self._bps * jitter

        if is_buy:
            return price + slip  # Buy at higher price
        return price - slip  # Sell at lower price


class VolatilityBasedSlippage(SlippageModel):
    """Slippage proportional to current volatility (ATR)."""

    def __init__(
        self,
        base_bps: float = 2.0,
        vol_multiplier: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._base_bps = base_bps / 10_000.0
        self._vol_mult = vol_multiplier
        self._rng = random.Random(seed)

    def compute_slippage(
        self, price: float, qty: float, is_buy: bool, **kwargs
    ) -> float:
        atr = kwargs.get("atr", 0.0)
        if atr <= 0:
            atr = price * 0.001  # Fallback: 0.1% of price

        # Slippage = base + vol_multiplier * (atr / price)
        vol_component = self._vol_mult * (atr / price) if price > 0 else 0
        total_slip_pct = self._base_bps + vol_component

        jitter = self._rng.uniform(0.7, 1.3)
        slip = price * total_slip_pct * jitter

        if is_buy:
            return price + slip
        return price - slip


class OrderbookImpactSlippage(SlippageModel):
    """Approximates market impact based on order size relative to volume.

    Uses square-root market impact model:
    impact = base_bps + k * sqrt(qty / avg_volume)
    """

    def __init__(
        self,
        base_bps: float = 1.0,
        impact_coefficient: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self._base_bps = base_bps / 10_000.0
        self._impact_k = impact_coefficient / 10_000.0
        self._rng = random.Random(seed)

    def compute_slippage(
        self, price: float, qty: float, is_buy: bool, **kwargs
    ) -> float:
        volume = kwargs.get("volume", qty * 100)  # Default: order is 1% of volume
        if volume <= 0:
            volume = qty * 100

        # Square-root impact
        participation = qty / volume if volume > 0 else 0.01
        impact = self._impact_k * (participation ** 0.5)
        total_slip_pct = self._base_bps + impact

        jitter = self._rng.uniform(0.8, 1.2)
        slip = price * total_slip_pct * jitter

        if is_buy:
            return price + slip
        return price - slip


def create_slippage_model(
    model_name: str,
    seed: int = 42,
    **kwargs,
) -> SlippageModel:
    """Factory for slippage models."""
    if model_name == "fixed_bps":
        return FixedBPSSlippage(bps=kwargs.get("bps", 5.0), seed=seed)
    elif model_name == "volatility_based":
        return VolatilityBasedSlippage(
            base_bps=kwargs.get("base_bps", 2.0),
            vol_multiplier=kwargs.get("vol_multiplier", 0.5),
            seed=seed,
        )
    elif model_name == "impact":
        return OrderbookImpactSlippage(
            base_bps=kwargs.get("base_bps", 1.0),
            impact_coefficient=kwargs.get("impact_coefficient", 10.0),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown slippage model: {model_name}")
