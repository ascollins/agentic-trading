"""Exchange adapter base: re-exports and helper types.

Re-exports ``IExchangeAdapter`` from ``core.interfaces`` so that adapter
implementations can import from a single location.  Also provides helper
configuration types used across adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from pydantic import BaseModel, Field as PydanticField

from agentic_trading.core.enums import AssetClass, Exchange, OrderType, TimeInForce
from agentic_trading.core.interfaces import IExchangeAdapter  # noqa: F401  re-export

__all__ = [
    "IExchangeAdapter",
    "AdapterCapabilities",
    "AdapterConfig",
    "FeeSchedule",
    "SlippageConfig",
]


# ---------------------------------------------------------------------------
# Adapter capabilities declaration
# ---------------------------------------------------------------------------


class AdapterCapabilities(BaseModel):
    """Declares what an adapter supports. Checked at startup."""

    asset_classes: list[AssetClass] = PydanticField(default_factory=list)
    supports_leverage: bool = True
    supports_short: bool = True
    supports_funding: bool = False  # crypto perps only
    supports_rollover: bool = False  # FX only
    supports_batch_orders: bool = False
    supports_amend: bool = False
    supports_trading_stop: bool = False
    position_mode: str = "netting"  # "netting" or "hedging"
    supported_order_types: list[OrderType] = PydanticField(
        default_factory=list
    )
    supported_tif: list[TimeInForce] = PydanticField(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Adapter configuration helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeeSchedule:
    """Maker/taker fee schedule for an exchange or simulation."""

    maker_fee: Decimal = Decimal("0.0002")  # 2 bps
    taker_fee: Decimal = Decimal("0.0004")  # 4 bps

    def fee_for(self, is_maker: bool) -> Decimal:
        """Return the applicable fee rate."""
        return self.maker_fee if is_maker else self.taker_fee


@dataclass(frozen=True)
class SlippageConfig:
    """Configurable slippage model for simulated adapters."""

    # Fixed slippage in basis points applied to each fill.
    fixed_bps: Decimal = Decimal("2")  # 2 bps default

    # Optional random component range [0, max_random_bps) added on top.
    max_random_bps: Decimal = Decimal("0")

    def apply(
        self, price: Decimal, side: str, random_factor: float = 0.0
    ) -> Decimal:
        """Apply slippage to a price.

        Parameters
        ----------
        price:
            The base price before slippage.
        side:
            ``"buy"`` or ``"sell"``.
        random_factor:
            A float in ``[0, 1)`` used to scale the random component.

        Returns
        -------
        The price after slippage.
        """
        total_bps = self.fixed_bps + self.max_random_bps * Decimal(
            str(random_factor)
        )
        slippage_multiplier = total_bps / Decimal("10000")
        if side == "buy":
            return price * (Decimal("1") + slippage_multiplier)
        else:
            return price * (Decimal("1") - slippage_multiplier)


@dataclass
class AdapterConfig:
    """Generic adapter configuration container.

    Exchange-specific adapters can extend or embed this.
    """

    exchange: Exchange = Exchange.BINANCE
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""  # Required by some exchanges
    testnet: bool = False
    sandbox: bool = False
    timeout_ms: int = 30_000
    rate_limit: bool = True
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    extra: dict[str, str] = field(default_factory=dict)
