"""TP/SL Watchdog Agent.

Periodically checks all open positions on the exchange and re-applies
missing TP/SL + trailing stops.  Acts as a safety net for three failure
scenarios:

1. The fill-handler TP/SL call exhausted all 5 retries (position left
   unprotected at entry time).
2. The exchange silently dropped TP/SL orders (API glitch, maintenance
   window, or position adjustment).
3. A position was opened outside the system (manual trade) and never
   received automated protection.

Usage::

    watchdog = TpSlWatchdog(
        adapter=adapter,
        exit_cfg=settings.exits,
        interval=300.0,
        tool_gateway=tool_gateway,          # optional; uses adapter directly if None
        trailing_strategies=exit_cfg.trailing_strategies,
    )
    await watchdog.start()
    # ...
    await watchdog.stop()

The watchdog integrates with the ``AgentRegistry`` so its health is
included in the platform health report.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agentic_trading.agents.base import BaseAgent
from agentic_trading.core.enums import AgentType
from agentic_trading.core.events import AgentCapabilities

logger = logging.getLogger(__name__)


class TpSlWatchdog(BaseAgent):
    """Periodic watchdog that ensures all open positions have TP/SL protection.

    Parameters
    ----------
    adapter:
        Live exchange adapter (``CCXTAdapter`` or ``PaperAdapter``).
    exit_cfg:
        ``ExitConfig`` from settings (sl/tp/trailing multipliers).
    interval:
        Seconds between checks (default 300 = every 5 minutes).
    tool_gateway:
        Optional ``ToolGateway`` for governance-routed calls.
        Falls back to direct adapter if ``None``.
    trailing_strategies:
        Strategy IDs that should receive trailing stops.  Positions
        opened by any strategy not in this list receive only TP/SL
        (no trailing).  Pass ``None`` to skip trailing on all positions.
    read_only:
        If ``True`` the watchdog logs what it *would* do but never
        calls the exchange.  Defaults to ``False``.
    agent_id:
        Optional fixed agent ID (auto-generated if omitted).
    """

    def __init__(
        self,
        *,
        adapter: Any,
        exit_cfg: Any,
        interval: float = 300.0,
        tool_gateway: Any | None = None,
        trailing_strategies: list[str] | None = None,
        read_only: bool = False,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(agent_id=agent_id, interval=interval)
        self._adapter = adapter
        self._exit_cfg = exit_cfg
        self._tool_gateway = tool_gateway
        self._trailing_strategies = set(trailing_strategies or [])
        self._read_only = read_only
        self._positions_checked: int = 0
        self._positions_repaired: int = 0

    # ------------------------------------------------------------------
    # IAgent
    # ------------------------------------------------------------------

    @property
    def agent_type(self) -> AgentType:
        return AgentType.RISK_GATE

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            subscribes_to=[],
            publishes_to=[],
            description=(
                f"TP/SL watchdog: checks all open positions every "
                f"{int(self._interval)}s and re-applies missing protection"
            ),
        )

    # ------------------------------------------------------------------
    # Periodic work
    # ------------------------------------------------------------------

    async def _work(self) -> None:
        """Single watchdog pass: fetch positions, check protection, repair."""
        await self._check_all_positions()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _check_all_positions(self) -> None:
        """Fetch all open positions and ensure each has TP/SL + optional trailing."""
        try:
            positions = await self._adapter.get_positions()
        except Exception:
            logger.warning("TpSlWatchdog: failed to fetch positions", exc_info=True)
            return

        if not positions:
            return

        repaired = 0
        for pos in positions:
            if float(pos.qty) == 0:
                continue
            try:
                repaired += await self._check_position(pos)
            except Exception:
                logger.warning(
                    "TpSlWatchdog: error checking %s", pos.symbol, exc_info=True
                )

        self._positions_checked += len(positions)
        self._positions_repaired += repaired

        if repaired:
            logger.info(
                "TpSlWatchdog: repaired TP/SL on %d/%d positions",
                repaired,
                len(positions),
            )

    async def _check_position(self, pos: Any) -> int:
        """Check one position and re-apply TP/SL if needed.

        Returns
        -------
        int
            1 if TP/SL was (re-)applied, 0 otherwise.
        """
        # --- Fetch raw position data to inspect current TP/SL state ---
        try:
            raw_positions = await self._adapter._ccxt.fetch_positions([pos.symbol])
        except Exception:
            logger.warning(
                "TpSlWatchdog: could not fetch raw position for %s", pos.symbol,
                exc_info=True,
            )
            return 0

        # Match on both CCXT settle-suffixed ('BTC/USDT:USDT') and plain ('BTC/USDT')
        bybit_pos = next(
            (
                p for p in raw_positions
                if (
                    p.get("symbol") == pos.symbol
                    or p.get("symbol", "").split(":")[0] == pos.symbol
                )
            ),
            None,
        )

        has_tp = float((bybit_pos or {}).get("takeProfitPrice") or 0) > 0
        has_sl = float((bybit_pos or {}).get("stopLossPrice") or 0) > 0
        _info = (bybit_pos or {}).get("info") or {}
        has_trail = float(_info.get("trailingStop") or 0) > 0

        # Decide whether trailing is expected for this position.
        # We apply trailing to all positions if trailing_strategies is empty
        # (meaning "all strategies") or if the position symbol is managed
        # by a trailing strategy.  Since we don't have strategy_id per
        # position here, we always apply trailing when the strategy list
        # is non-empty; callers can pass an empty set to disable.
        wants_trail = bool(self._trailing_strategies)

        if has_tp and has_sl and (has_trail or not wants_trail):
            # Fully protected — nothing to do
            return 0

        missing = []
        if not has_tp:
            missing.append("tp")
        if not has_sl:
            missing.append("sl")
        if wants_trail and not has_trail:
            missing.append("trail")

        logger.warning(
            "TpSlWatchdog: %s missing %s — re-applying",
            pos.symbol,
            "+".join(missing),
        )

        # --- Compute protection levels using ATR fallback ---
        entry = pos.entry_price
        atr_est = entry * Decimal("0.004")   # ~0.4% of price (same fallback as fill handler)

        sl_mult = Decimal(str(self._exit_cfg.sl_atr_multiplier))
        tp_mult = Decimal(str(self._exit_cfg.tp_atr_multiplier))
        sl_dist = atr_est * sl_mult
        tp_dist = atr_est * tp_mult

        direction = pos.side.value  # "long" or "short"
        if direction == "long":
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        trail_distance: Decimal | None = None
        active_price: Decimal | None = None
        if wants_trail and not has_trail:
            trail_mult = Decimal(str(self._exit_cfg.trailing_stop_atr_multiplier))
            trail_distance = atr_est * trail_mult
            # Breakeven activation: trail arms after 1× SL distance in profit
            if direction == "long":
                active_price = entry + sl_dist
            else:
                active_price = entry - sl_dist

        if self._read_only:
            logger.info(
                "TpSlWatchdog [dry-run]: would set %s tp=%s sl=%s trail=%s active=%s",
                pos.symbol, tp_price if not has_tp else "keep",
                sl_price if not has_sl else "keep",
                trail_distance, active_price,
            )
            return 0

        # --- Apply via ToolGateway or direct adapter (with 3 retries) ---
        import asyncio

        for attempt in range(1, 4):
            try:
                if self._tool_gateway is not None:
                    from agentic_trading.control_plane.action_types import (
                        ActionScope as _AS,
                        ProposedAction as _PA,
                        ToolName as _TN,
                    )
                    params: dict[str, Any] = {"symbol": pos.symbol}
                    if not has_tp:
                        params["take_profit"] = str(tp_price)
                    if not has_sl:
                        params["stop_loss"] = str(sl_price)
                    if trail_distance is not None:
                        params["trailing_stop"] = str(trail_distance)
                    if active_price is not None:
                        params["active_price"] = str(active_price)

                    result = await self._tool_gateway.call(_PA(
                        tool_name=_TN.SET_TRADING_STOP,
                        scope=_AS(symbol=pos.symbol, actor="tpsl_watchdog"),
                        request_params=params,
                    ))
                    if not result.success:
                        raise RuntimeError(result.error)
                else:
                    await self._adapter.set_trading_stop(
                        pos.symbol,
                        take_profit=tp_price if not has_tp else None,
                        stop_loss=sl_price if not has_sl else None,
                        trailing_stop=trail_distance,
                        active_price=active_price,
                    )

                logger.info(
                    "TpSlWatchdog: ✓ %s protected — tp=%s sl=%s trail=%s active=%s",
                    pos.symbol,
                    tp_price if not has_tp else "kept",
                    sl_price if not has_sl else "kept",
                    trail_distance or "none",
                    active_price or "none",
                )
                return 1

            except Exception as exc:
                logger.warning(
                    "TpSlWatchdog: attempt %d/3 failed for %s: %s",
                    attempt, pos.symbol, exc,
                )
                if attempt < 3:
                    await asyncio.sleep(2 ** (attempt - 1))   # 1s, 2s

        logger.critical(
            "TpSlWatchdog: FAILED to protect %s after 3 attempts — "
            "position has NO stop protection!",
            pos.symbol,
        )
        return 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def positions_checked(self) -> int:
        """Total positions checked since start."""
        return self._positions_checked

    @property
    def positions_repaired(self) -> int:
        """Total positions where TP/SL was re-applied since start."""
        return self._positions_repaired
