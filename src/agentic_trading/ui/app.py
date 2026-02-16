"""Supervision UI — FastAPI + Jinja2 + HTMX application.

Serves the 4-tab supervision dashboard:
  Home | Strategies | Activity | Settings

Uses HTMX for live polling (no websockets needed — polling every 5s
for home cards, 30s for strategies).  Server-rendered Jinja2 templates
with minimal client-side JS.

Data sources:
  - adapter.get_positions() / adapter.get_balances() — live Bybit data
  - journal open/closed trades — trade lifecycle
  - journal strategy stats — per-strategy analytics
  - risk_manager — kill switch, risk limits
  - governance_gate — policy engine status
  - settings — platform configuration

Usage::

    from agentic_trading.ui.app import create_ui_app

    app = create_ui_app(
        journal=journal,
        adapter=adapter,
        settings=settings,
        ...
    )
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_ui_app(
    journal: Any = None,
    agent_registry: Any = None,
    governance_gate: Any = None,
    approval_manager: Any = None,
    incident_manager: Any = None,
    scorecard: Any = None,
    lifecycle_manager: Any = None,
    quality_tracker: Any = None,
    event_bus: Any = None,
    settings: Any = None,
    risk_manager: Any = None,
    adapter: Any = None,
    trading_context: Any = None,
) -> FastAPI:
    """Create the supervision UI FastAPI application.

    All parameters are optional — the UI degrades gracefully when
    components are not available (shows demo data).
    """
    app = FastAPI(
        title="Trading Platform Supervision",
        docs_url=None,
        redoc_url=None,
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Store component references on app state
    app.state.journal = journal
    app.state.registry = agent_registry
    app.state.gate = governance_gate
    app.state.approvals = approval_manager
    app.state.incidents = incident_manager
    app.state.scorecard = scorecard
    app.state.lifecycle = lifecycle_manager
    app.state.quality = quality_tracker
    app.state.bus = event_bus
    app.state.settings = settings
    app.state.risk_manager = risk_manager
    app.state.adapter = adapter
    app.state.trading_context = trading_context

    # Track startup time for equity curve baseline
    app.state.start_time = time.time()
    # Cache for equity snapshots: list of (timestamp, equity) tuples
    app.state.equity_snapshots: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------

    def _ctx(request: Request, **kwargs: Any) -> dict[str, Any]:
        """Build template context with standard fields."""
        return {"request": request, **kwargs}

    # ------------------------------------------------------------------
    # Page routes (full page loads)
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def home_page(request: Request) -> HTMLResponse:
        data = await _build_home_data(app)
        return templates.TemplateResponse("home.html", _ctx(request, **data))

    @app.get("/strategies", response_class=HTMLResponse)
    async def strategies_page(request: Request) -> HTMLResponse:
        data = _build_strategies_data(app)
        return templates.TemplateResponse(
            "strategies.html", _ctx(request, **data),
        )

    @app.get("/activity", response_class=HTMLResponse)
    async def activity_page(request: Request) -> HTMLResponse:
        data = _build_activity_data(app)
        return templates.TemplateResponse(
            "activity.html", _ctx(request, **data),
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request) -> HTMLResponse:
        data = _build_settings_data(app)
        return templates.TemplateResponse(
            "settings.html", _ctx(request, **data),
        )

    # ------------------------------------------------------------------
    # HTMX partial routes (return HTML fragments, not full pages)
    # ------------------------------------------------------------------

    @app.get("/partials/home/scorecard", response_class=HTMLResponse)
    async def partial_scorecard(request: Request) -> HTMLResponse:
        data = _build_scorecard_data(app)
        return templates.TemplateResponse(
            "partials/scorecard_card.html", _ctx(request, **data),
        )

    @app.get("/partials/home/portfolio", response_class=HTMLResponse)
    async def partial_portfolio(request: Request) -> HTMLResponse:
        data = await _build_portfolio_data(app)
        return templates.TemplateResponse(
            "partials/portfolio_card.html", _ctx(request, **data),
        )

    @app.get("/partials/home/positions", response_class=HTMLResponse)
    async def partial_positions(request: Request) -> HTMLResponse:
        data = await _build_positions_data(app)
        return templates.TemplateResponse(
            "partials/positions_card.html", _ctx(request, **data),
        )

    @app.get("/partials/home/system", response_class=HTMLResponse)
    async def partial_system(request: Request) -> HTMLResponse:
        data = _build_system_data(app)
        return templates.TemplateResponse(
            "partials/system_card.html", _ctx(request, **data),
        )

    @app.get("/partials/home/approvals", response_class=HTMLResponse)
    async def partial_approvals(request: Request) -> HTMLResponse:
        data = _build_approvals_data(app)
        return templates.TemplateResponse(
            "partials/approvals_card.html", _ctx(request, **data),
        )

    @app.get("/partials/home/banner", response_class=HTMLResponse)
    async def partial_banner(request: Request) -> HTMLResponse:
        data = _build_banner_data(app)
        return templates.TemplateResponse(
            "partials/banner.html", _ctx(request, **data),
        )

    @app.get("/partials/strategies/list", response_class=HTMLResponse)
    async def partial_strategies_list(request: Request) -> HTMLResponse:
        data = _build_strategies_data(app)
        return templates.TemplateResponse(
            "partials/strategies_list.html", _ctx(request, **data),
        )

    @app.get("/partials/activity/timeline", response_class=HTMLResponse)
    async def partial_activity_timeline(request: Request) -> HTMLResponse:
        filter_type = request.query_params.get("type", "all")
        data = _build_activity_data(app, filter_type=filter_type)
        return templates.TemplateResponse(
            "partials/activity_timeline.html", _ctx(request, **data),
        )

    # ------------------------------------------------------------------
    # Action routes (HTMX POST for approvals, promotions, etc.)
    # ------------------------------------------------------------------

    @app.post("/actions/approve/{request_id}", response_class=HTMLResponse)
    async def action_approve(request: Request, request_id: str) -> HTMLResponse:
        if app.state.approvals:
            try:
                await app.state.approvals.approve(
                    request_id, decided_by="ui_operator",
                )
            except Exception:
                logger.warning("Approval failed: %s", request_id, exc_info=True)
        data = _build_approvals_data(app)
        return templates.TemplateResponse(
            "partials/approvals_card.html", _ctx(request, **data),
        )

    @app.post("/actions/deny/{request_id}", response_class=HTMLResponse)
    async def action_deny(request: Request, request_id: str) -> HTMLResponse:
        if app.state.approvals:
            try:
                await app.state.approvals.reject(
                    request_id,
                    decided_by="ui_operator",
                    reason="Denied via supervision UI",
                )
            except Exception:
                logger.warning("Denial failed: %s", request_id, exc_info=True)
        data = _build_approvals_data(app)
        return templates.TemplateResponse(
            "partials/approvals_card.html", _ctx(request, **data),
        )

    @app.post(
        "/actions/promote/{strategy_id}", response_class=HTMLResponse,
    )
    async def action_promote(
        request: Request, strategy_id: str,
    ) -> HTMLResponse:
        result = {"approved": False, "reason": "lifecycle not configured"}
        if app.state.lifecycle:
            try:
                result = await app.state.lifecycle.request_promotion(
                    strategy_id, operator_id="ui_operator",
                )
            except Exception:
                logger.warning(
                    "Promotion failed: %s", strategy_id, exc_info=True,
                )
        data = _build_strategies_data(app)
        data["promotion_result"] = result
        return templates.TemplateResponse(
            "partials/strategies_list.html", _ctx(request, **data),
        )

    @app.post(
        "/actions/resolve-incident/{incident_id}",
        response_class=HTMLResponse,
    )
    async def action_resolve_incident(
        request: Request, incident_id: str,
    ) -> HTMLResponse:
        if app.state.incidents:
            try:
                await app.state.incidents.resolve_incident(
                    incident_id, resolved_by="ui_operator",
                )
            except Exception:
                logger.warning(
                    "Incident resolve failed: %s", incident_id, exc_info=True,
                )
        data = _build_banner_data(app)
        return templates.TemplateResponse(
            "partials/banner.html", _ctx(request, **data),
        )

    # ------------------------------------------------------------------
    # JSON API (for equity curve chart data)
    # ------------------------------------------------------------------

    @app.get("/api/equity-curve")
    async def api_equity_curve() -> list[dict[str, Any]]:
        """Return equity curve data points for Chart.js."""
        return _build_equity_curve(app)

    # ------------------------------------------------------------------
    # Health endpoint
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


# ======================================================================
# Async helpers: fetch fresh data from the exchange adapter
# ======================================================================

async def _fetch_exchange_state(
    app: FastAPI,
) -> tuple[list[Any], list[Any]]:
    """Fetch positions and balances from the exchange adapter.

    Returns (positions, balances).  Falls back to TradingContext
    cache if the adapter call fails.  Returns empty lists if
    nothing is available.
    """
    adapter = app.state.adapter
    positions: list[Any] = []
    balances: list[Any] = []

    if adapter is not None:
        try:
            positions = await adapter.get_positions()
        except Exception:
            logger.debug("adapter.get_positions() failed", exc_info=True)
        try:
            balances = await adapter.get_balances()
        except Exception:
            logger.debug("adapter.get_balances() failed", exc_info=True)

    # Fallback: use cached TradingContext.portfolio_state
    ctx = app.state.trading_context
    if not positions and not balances and ctx is not None:
        ps = ctx.portfolio_state
        if ps is not None:
            positions = list(ps.positions.values())
            balances = list(ps.balances.values())

    return positions, balances


# ======================================================================
# Data builders (extract data from platform components for templates)
# ======================================================================


async def _build_home_data(app: FastAPI) -> dict[str, Any]:
    """Build all data for the home page (initial full load)."""
    portfolio = await _build_portfolio_data(app)
    positions = await _build_positions_data(app)
    return {
        **_build_banner_data(app),
        **_build_scorecard_data(app),
        **portfolio,
        **positions,
        **_build_system_data(app),
        **_build_approvals_data(app),
    }


def _build_banner_data(app: FastAPI) -> dict[str, Any]:
    """Build critical banner data."""
    banner = None

    # Check kill switch
    risk_mgr = app.state.risk_manager
    if risk_mgr is not None and hasattr(risk_mgr, "kill_switch"):
        try:
            if risk_mgr.kill_switch.is_active:
                banner = {
                    "mode": "killed",
                    "incident_count": 1,
                    "incidents": [{"title": "Kill switch activated"}],
                }
        except Exception:
            pass

    # Check incident manager
    if banner is None and app.state.incidents:
        try:
            mode = app.state.incidents.current_mode
            if hasattr(mode, "value"):
                mode = mode.value
            if mode != "normal":
                active = app.state.incidents.get_active_incidents()
                banner = {
                    "mode": mode,
                    "incident_count": len(active),
                    "incidents": active[:3],
                }
        except Exception:
            pass

    return {"banner": banner}


def _build_scorecard_data(app: FastAPI) -> dict[str, Any]:
    """Build daily effectiveness scorecard from journal strategy stats.

    Computes 4 sub-scores:
      - edge_quality:  win_rate and profit_factor
      - execution_quality:  avg management_efficiency
      - risk_discipline:  drawdown, loss streaks
      - operational_integrity:  DLQ count, system health
    """
    # Use external scorecard if available
    if app.state.scorecard is not None:
        try:
            scores = app.state.scorecard.last_scores
            if scores is None:
                scores = app.state.scorecard.compute()
            return {"scorecard": scores}
        except Exception:
            pass

    # Compute from journal stats
    journal = app.state.journal
    if journal is None:
        return {"scorecard": _default_scorecard()}

    try:
        all_stats = journal.get_all_strategy_stats()
        if not all_stats:
            return {"scorecard": _default_scorecard()}

        # Aggregate across strategies
        total_trades = sum(s.get("total_trades", 0) for s in all_stats.values())
        if total_trades == 0:
            return {"scorecard": _default_scorecard()}

        total_wins = sum(s.get("wins", 0) for s in all_stats.values())
        total_losses = sum(s.get("losses", 0) for s in all_stats.values())
        gross_wins = sum(s.get("avg_winner", 0) * s.get("wins", 0) for s in all_stats.values())
        gross_losses = sum(s.get("avg_loser", 0) * s.get("losses", 0) for s in all_stats.values())

        # Edge quality: win_rate and profit_factor, scale to 0-10
        agg_wr = total_wins / total_trades if total_trades > 0 else 0
        agg_pf = (gross_wins / gross_losses) if gross_losses > 0 else 2.0
        edge = min(10.0, (agg_wr * 10) * 0.5 + min(agg_pf, 3.0) / 3.0 * 10 * 0.5)

        # Execution quality: avg management efficiency
        efficiencies = []
        for s in all_stats.values():
            eff = s.get("avg_management_efficiency", 0)
            if eff > 0:
                efficiencies.append(eff)
        avg_eff = sum(efficiencies) / len(efficiencies) if efficiencies else 0.5
        execution = min(10.0, avg_eff * 10)

        # Risk discipline: penalise loss streaks, reward low drawdown
        max_loss_streak = max(
            (s.get("max_loss_streak", 0) for s in all_stats.values()), default=0,
        )
        risk = max(0.0, 10.0 - max_loss_streak * 0.5)

        # Operational integrity: DLQ count
        dlq = 0
        if app.state.bus:
            try:
                dlq = len(getattr(app.state.bus, "dead_letters", []))
            except Exception:
                pass
        ops = max(0.0, 10.0 - dlq * 2.0)

        total_score = round(edge * 0.35 + execution * 0.25 + risk * 0.25 + ops * 0.15, 1)

        return {
            "scorecard": {
                "total": total_score,
                "edge_quality": round(edge, 1),
                "execution_quality": round(execution, 1),
                "risk_discipline": round(risk, 1),
                "operational_integrity": round(ops, 1),
            },
        }
    except Exception:
        logger.debug("Scorecard computation failed", exc_info=True)
        return {"scorecard": _default_scorecard()}


def _default_scorecard() -> dict[str, float]:
    """Return a neutral scorecard when no data is available."""
    return {
        "total": 5.0,
        "edge_quality": 5.0,
        "execution_quality": 5.0,
        "risk_discipline": 5.0,
        "operational_integrity": 5.0,
    }


async def _build_portfolio_data(app: FastAPI) -> dict[str, Any]:
    """Build portfolio summary from live Bybit data."""
    positions, balances = await _fetch_exchange_state(app)

    # Compute equity from balances + unrealized PnL
    total_equity = 0.0
    for b in balances:
        total_equity += float(b.total)
    for p in positions:
        total_equity += float(p.unrealized_pnl)

    # Count open positions with non-zero qty
    open_count = sum(1 for p in positions if float(p.qty) > 0)

    # Compute daily P&L from journal closed trades
    pnl_today = 0.0
    journal = app.state.journal
    if journal is not None:
        try:
            today_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            closed = journal.get_closed_trades()
            for t in closed:
                if hasattr(t, "closed_at") and t.closed_at is not None:
                    if t.closed_at >= today_start:
                        pnl_today += float(t.net_pnl)
        except Exception:
            pass

    # Add unrealized P&L from open positions to today's running P&L
    for p in positions:
        pnl_today += float(p.unrealized_pnl)

    pnl_pct = (pnl_today / total_equity * 100) if total_equity > 0 else 0.0

    # Track equity snapshot for the curve
    if total_equity > 0:
        now = time.time()
        snaps = app.state.equity_snapshots
        # Only add a new snapshot every 30 seconds
        if not snaps or (now - snaps[-1][0]) >= 30:
            snaps.append((now, total_equity))
            # Keep last 8 hours of snapshots (~960 points at 30s intervals)
            max_snaps = 960
            if len(snaps) > max_snaps:
                app.state.equity_snapshots = snaps[-max_snaps:]

    portfolio = {
        "equity": round(total_equity, 2),
        "pnl_today": round(pnl_today, 2),
        "pnl_today_pct": round(pnl_pct, 2),
        "open_positions": open_count,
        "pnl_positive": pnl_today >= 0,
    }
    return {"portfolio": portfolio}


async def _build_positions_data(app: FastAPI) -> dict[str, Any]:
    """Build open positions from live Bybit data."""
    positions_raw, _ = await _fetch_exchange_state(app)

    # Build journal index for strategy cross-reference
    journal_map: dict[str, str] = {}  # symbol -> strategy_id
    journal = app.state.journal
    if journal is not None:
        try:
            for t in journal.get_open_trades():
                journal_map[t.symbol] = t.strategy_id
        except Exception:
            pass

    positions: list[dict[str, Any]] = []
    for p in positions_raw:
        if float(p.qty) == 0:
            continue

        side_val = p.side.value if hasattr(p.side, "value") else str(p.side)
        upnl = float(p.unrealized_pnl)

        # Normalise symbol: strip ":USDT" settle suffix
        display_sym = p.symbol.split(":")[0] if ":" in p.symbol else p.symbol

        # Cross-reference strategy from journal
        strat = journal_map.get(display_sym, "")
        if not strat:
            strat = journal_map.get(p.symbol, "")

        positions.append({
            "symbol": display_sym,
            "side": side_val.upper(),
            "qty": str(p.qty),
            "entry_price": f"{float(p.entry_price):,.2f}",
            "mark_price": f"{float(p.mark_price):,.2f}",
            "unrealized_pnl": f"{'+' if upnl >= 0 else ''}${upnl:,.2f}",
            "pnl_positive": upnl >= 0,
            "strategy": strat,
        })

    # Fallback: journal open trades if adapter returned nothing
    if not positions and journal is not None:
        try:
            for t in journal.get_open_trades():
                positions.append({
                    "symbol": t.symbol,
                    "side": t.direction.upper(),
                    "qty": str(t.remaining_qty),
                    "entry_price": f"{float(t.avg_entry_price):,.2f}",
                    "mark_price": "",
                    "unrealized_pnl": "$0.00",
                    "pnl_positive": True,
                    "strategy": t.strategy_id,
                })
        except Exception:
            pass

    return {"positions": positions}


def _build_system_data(app: FastAPI) -> dict[str, Any]:
    """Build system status data."""
    system: dict[str, Any] = {
        "agents_total": 0,
        "agents_healthy": 0,
        "all_healthy": True,
        "last_candle_seconds": 0,
        "event_bus_ok": True,
        "dlq_count": 0,
    }

    # Agent registry health
    if app.state.registry:
        try:
            health = app.state.registry.health_check_all()
            total = len(health)
            healthy = sum(1 for h in health.values() if h.healthy)
            system["agents_total"] = total
            system["agents_healthy"] = healthy
            system["all_healthy"] = healthy == total
        except Exception:
            pass

    # Event bus dead-letter queue
    if app.state.bus:
        try:
            dlq = getattr(app.state.bus, "dead_letters", [])
            system["dlq_count"] = len(dlq)
            system["event_bus_ok"] = len(dlq) == 0

            # Messages processed
            processed = getattr(app.state.bus, "messages_processed", 0)
            system["messages_processed"] = processed
        except Exception:
            pass

    # Error counts from event bus
    if app.state.bus:
        try:
            errors = app.state.bus.get_error_counts()
            system["error_counts"] = dict(errors) if errors else {}
        except Exception:
            pass

    # Adapter connectivity check
    system["adapter_connected"] = app.state.adapter is not None

    return {"system": system}


def _build_approvals_data(app: FastAPI) -> dict[str, Any]:
    """Build pending approvals data."""
    approvals: list[dict[str, Any]] = []
    if app.state.approvals:
        try:
            pending = app.state.approvals.get_pending()
            approvals = [
                {
                    "request_id": r.request_id,
                    "strategy_id": r.strategy_id,
                    "symbol": r.symbol,
                    "trigger": (
                        r.trigger.value if hasattr(r.trigger, "value") else str(r.trigger)
                    ),
                    "escalation_level": (
                        r.escalation_level.value
                        if hasattr(r.escalation_level, "value")
                        else str(r.escalation_level)
                    ),
                    "notional_usd": r.notional_usd,
                    "created_at": (
                        r.created_at.strftime("%H:%M")
                        if hasattr(r, "created_at") and r.created_at
                        else ""
                    ),
                }
                for r in pending
            ]
        except Exception:
            pass
    return {"approvals": approvals}


def _build_strategies_data(app: FastAPI) -> dict[str, Any]:
    """Build strategies list from journal stats + settings."""
    strategies: list[dict[str, Any]] = []
    journal = app.state.journal
    settings = app.state.settings

    # Get all strategy stats from the journal
    if journal is not None:
        try:
            all_stats = journal.get_all_strategy_stats()
            for sid, stats in all_stats.items():
                total = stats.get("total_trades", 0)
                if total == 0:
                    continue

                wr = round(stats.get("win_rate", 0) * 100)
                pf = stats.get("profit_factor", 0)
                sharpe = stats.get("sharpe", 0)

                # Quality score: composite of win_rate, profit_factor, sharpe
                quality = _compute_quality_score(stats)

                # Stage: from lifecycle manager or maturity config
                stage = "live"
                if app.state.lifecycle:
                    try:
                        s = app.state.lifecycle._stages.get(sid)
                        if s is not None:
                            stage = s.value if hasattr(s, "value") else str(s)
                    except Exception:
                        pass

                strategies.append({
                    "strategy_id": sid,
                    "stage": stage,
                    "quality_score": quality,
                    "quality_grade": _score_to_grade(quality),
                    "win_rate": wr,
                    "total_trades": total,
                    "pnl_today": round(stats.get("total_pnl", 0), 2),
                    "pnl_positive": stats.get("total_pnl", 0) >= 0,
                })
        except Exception:
            logger.debug("Failed to build strategies from journal", exc_info=True)

    # If journal had no data, show configured strategies from settings
    if not strategies and settings is not None:
        try:
            for strat_cfg in settings.strategies:
                if strat_cfg.enabled:
                    strategies.append({
                        "strategy_id": strat_cfg.strategy_id,
                        "stage": "configured",
                        "quality_score": 0,
                        "quality_grade": "-",
                        "win_rate": 0,
                        "total_trades": 0,
                        "pnl_today": 0,
                        "pnl_positive": True,
                    })
        except Exception:
            pass

    return {"strategies": strategies}


def _build_activity_data(
    app: FastAPI, filter_type: str = "all",
) -> dict[str, Any]:
    """Build activity timeline from journal trades."""
    events: list[dict[str, Any]] = []
    journal = app.state.journal

    if journal is not None:
        try:
            # Open trades (entry events)
            if filter_type in ("all", "trades"):
                for t in journal.get_open_trades():
                    ts = ""
                    if hasattr(t, "opened_at") and t.opened_at is not None:
                        ts = t.opened_at.strftime("%H:%M")
                    elif t.entry_fills:
                        ts = t.entry_fills[0].timestamp.strftime("%H:%M")

                    events.append({
                        "time": ts,
                        "type": "fill",
                        "icon": "fill",
                        "title": f"OPEN {t.symbol} {t.direction.upper()} qty={t.remaining_qty}",
                        "subtitle": f"{t.strategy_id} | conf {t.signal_confidence:.2f}",
                        "_sort_ts": t.entry_fills[0].timestamp if t.entry_fills else datetime.min.replace(tzinfo=timezone.utc),
                    })

            # Closed trades (completed trade lifecycle)
            if filter_type in ("all", "trades"):
                closed = journal.get_closed_trades(last_n=30)
                for t in reversed(closed):
                    ts = ""
                    if hasattr(t, "closed_at") and t.closed_at is not None:
                        ts = t.closed_at.strftime("%H:%M")
                    elif t.exit_fills:
                        ts = t.exit_fills[-1].timestamp.strftime("%H:%M")

                    pnl = float(t.net_pnl)
                    outcome = t.outcome.value if hasattr(t.outcome, "value") else str(t.outcome)
                    r_mult = t.r_multiple

                    events.append({
                        "time": ts,
                        "type": "fill",
                        "icon": "fill",
                        "title": f"CLOSED {t.symbol} {t.direction.upper()} P&L ${pnl:+,.2f}",
                        "subtitle": f"{t.strategy_id} | {outcome} | R={r_mult:.1f} | hold {t.hold_duration_seconds:.0f}s",
                        "_sort_ts": t.exit_fills[-1].timestamp if t.exit_fills else datetime.min.replace(tzinfo=timezone.utc),
                    })
        except Exception:
            logger.debug("Failed to build activity from journal", exc_info=True)

    # Sort by timestamp descending (most recent first)
    events.sort(key=lambda e: e.get("_sort_ts", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    # Strip sort key from output
    for e in events:
        e.pop("_sort_ts", None)

    return {"events": events, "filter_type": filter_type}


def _build_settings_data(app: FastAPI) -> dict[str, Any]:
    """Build settings page data from live Settings object."""
    settings = app.state.settings
    risk_mgr = app.state.risk_manager

    # Defaults
    data: dict[str, Any] = {
        "mode": "paper",
        "exchange": "Not configured",
        "exchange_connected": False,
        "risk_limits": {
            "max_position_pct": "-",
            "max_daily_loss": "-",
            "max_drawdown": "-",
            "kill_switch": False,
        },
        "governance": {
            "policy_engine": False,
            "policy_rules_count": 0,
            "approval_workflow": False,
            "auto_approve_l1": False,
            "shadow_mode": False,
        },
        "agents": {"total": 0, "running": 0},
        "system": {
            "event_bus": "Unknown",
            "metrics": "Not configured",
            "database": "Not configured",
        },
    }

    if settings is not None:
        try:
            mode_val = settings.mode.value if hasattr(settings.mode, "value") else str(settings.mode)
            data["mode"] = mode_val

            # Exchange info
            if settings.exchanges:
                exc = settings.exchanges[0]
                name = exc.name.value if hasattr(exc.name, "value") else str(exc.name)
                suffix = ""
                if exc.demo:
                    suffix = " (demo)"
                elif exc.testnet:
                    suffix = " (testnet)"
                data["exchange"] = f"{name.title()}{suffix}"
                data["exchange_connected"] = app.state.adapter is not None

            # Risk limits
            data["risk_limits"] = {
                "max_position_pct": f"{settings.risk.max_single_position_pct * 100:.0f}%",
                "max_daily_loss": f"-{settings.risk.max_daily_loss_pct * 100:.1f}%",
                "max_drawdown": f"-{settings.risk.max_drawdown_pct * 100:.1f}%",
                "kill_switch": False,
            }
            if risk_mgr is not None and hasattr(risk_mgr, "kill_switch"):
                data["risk_limits"]["kill_switch"] = risk_mgr.kill_switch.is_active

            # Governance
            gov = settings.governance
            data["governance"] = {
                "policy_engine": gov.enabled,
                "policy_rules_count": 0,
                "approval_workflow": app.state.approvals is not None,
                "auto_approve_l1": False,
                "shadow_mode": False,
            }

            # Event bus type
            bus_type = "Redis Streams" if mode_val != "backtest" else "MemoryEventBus"
            data["system"] = {
                "event_bus": bus_type,
                "metrics": f"Prometheus :{settings.observability.metrics_port}",
                "database": "Postgres",
            }
        except Exception:
            logger.debug("Failed to build settings from config", exc_info=True)

    # Agent registry health
    if app.state.registry:
        try:
            health = app.state.registry.health_check_all()
            total = len(health)
            running = sum(1 for h in health.values() if h.healthy)
            data["agents"] = {"total": total, "running": running}
        except Exception:
            pass

    # Policy engine rule count from gate
    if app.state.gate:
        try:
            pe = app.state.gate.policy_engine
            if pe:
                data["governance"]["policy_rules_count"] = len(
                    getattr(pe, "registered_sets", [])
                )
        except Exception:
            pass

    return data


def _build_equity_curve(app: FastAPI) -> list[dict[str, Any]]:
    """Build equity curve from recorded snapshots.

    Each HTMX portfolio poll records a snapshot.  The equity curve
    endpoint returns these for Chart.js rendering.
    """
    snaps = app.state.equity_snapshots
    if not snaps:
        # Fallback: compute from journal closed trades
        journal = app.state.journal
        if journal is not None:
            try:
                return _equity_curve_from_journal(journal)
            except Exception:
                pass
        return []

    points = []
    for ts, eq in snaps:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        points.append({
            "time": dt.strftime("%H:%M"),
            "equity": round(eq, 2),
        })
    return points


def _equity_curve_from_journal(journal: Any) -> list[dict[str, Any]]:
    """Build equity curve from journal closed trade P&L history."""
    closed = journal.get_closed_trades()
    if not closed:
        return []

    # Sort by close time
    sorted_trades = sorted(
        closed,
        key=lambda t: (
            t.exit_fills[-1].timestamp if t.exit_fills
            else datetime.min.replace(tzinfo=timezone.utc)
        ),
    )

    # Start from a baseline (first trade's pre-PnL equity)
    equity = 100_000.0  # Base value — will be offset
    points = []
    for t in sorted_trades:
        equity += float(t.net_pnl)
        ts = ""
        if t.exit_fills:
            ts = t.exit_fills[-1].timestamp.strftime("%H:%M")
        points.append({
            "time": ts,
            "equity": round(equity, 2),
        })

    return points


# ======================================================================
# Utility functions
# ======================================================================


def _compute_quality_score(stats: dict[str, Any]) -> int:
    """Compute a 0-100 quality score from strategy stats."""
    wr = stats.get("win_rate", 0)  # 0..1
    pf = stats.get("profit_factor", 0)
    sharpe = stats.get("sharpe", 0)
    avg_eff = stats.get("avg_management_efficiency", 0)

    # Win rate: 0-25 points (50% WR = 12.5, 70% = 17.5)
    wr_score = min(25, wr * 25 / 0.5) if wr > 0 else 0

    # Profit factor: 0-25 points (1.5 PF = 12.5, 3.0 = 25)
    pf_score = min(25, pf * 25 / 3.0) if pf > 0 else 0

    # Sharpe: 0-25 points (1.0 = 12.5, 2.0 = 25)
    sharpe_score = min(25, max(0, sharpe) * 25 / 2.0)

    # Efficiency: 0-25 points
    eff_score = min(25, avg_eff * 25)

    return round(wr_score + pf_score + sharpe_score + eff_score)


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A+"
    elif score >= 85:
        return "A"
    elif score >= 80:
        return "A-"
    elif score >= 75:
        return "B+"
    elif score >= 70:
        return "B"
    elif score >= 65:
        return "B-"
    elif score >= 60:
        return "C+"
    elif score >= 55:
        return "C"
    elif score >= 50:
        return "C-"
    elif score >= 45:
        return "D"
    else:
        return "F"
