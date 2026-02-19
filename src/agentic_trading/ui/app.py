"""Supervision UI — FastAPI + Jinja2 + HTMX application.

Serves the 5-tab supervision dashboard:
  Overview | Strategies | Activity | Risk & Controls | Settings

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

import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def _toast_header(message: str, toast_type: str = "success") -> dict[str, str]:
    """Build an HX-Trigger header value that fires a showToast event."""
    return {
        "HX-Trigger": json.dumps(
            {"showToast": {"message": message, "type": toast_type}}
        )
    }


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
    # Backtest jobs: job_id → {status, lines, started_at, finished_at}
    app.state.backtest_jobs: dict[str, dict[str, Any]] = {}

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
        data = {**_build_strategies_data(app), **_build_status_bar_data(app)}
        return templates.TemplateResponse(
            "strategies.html", _ctx(request, **data),
        )

    @app.get("/activity", response_class=HTMLResponse)
    async def activity_page(request: Request) -> HTMLResponse:
        data = {**_build_activity_data(app), **_build_status_bar_data(app)}
        return templates.TemplateResponse(
            "activity.html", _ctx(request, **data),
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request) -> HTMLResponse:
        data = {**_build_settings_data(app), **_build_status_bar_data(app)}
        return templates.TemplateResponse(
            "settings.html", _ctx(request, **data),
        )

    @app.get("/risk", response_class=HTMLResponse)
    async def risk_page(request: Request) -> HTMLResponse:
        data = {
            **(await _build_risk_controls_data(app)),
            **_build_circuit_breakers_data(app),
            **_build_status_bar_data(app),
        }
        return templates.TemplateResponse(
            "risk.html", _ctx(request, **data),
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
        data = await _build_banner_data(app)
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

    @app.get("/partials/status-bar", response_class=HTMLResponse)
    async def partial_status_bar(request: Request) -> HTMLResponse:
        data = _build_status_bar_data(app)
        return templates.TemplateResponse(
            "partials/status_bar.html", _ctx(request, **data),
        )

    @app.get("/partials/risk/gauges", response_class=HTMLResponse)
    async def partial_risk_gauges(request: Request) -> HTMLResponse:
        data = _build_risk_gauges_data(app)
        return templates.TemplateResponse(
            "partials/risk_gauges.html", _ctx(request, **data),
        )

    @app.get("/partials/risk/kill-switch", response_class=HTMLResponse)
    async def partial_kill_switch(request: Request) -> HTMLResponse:
        data = await _build_risk_controls_data(app)
        return templates.TemplateResponse(
            "partials/kill_switch_card.html", _ctx(request, **data),
        )

    @app.get("/partials/risk/circuit-breakers", response_class=HTMLResponse)
    async def partial_circuit_breakers(request: Request) -> HTMLResponse:
        data = _build_circuit_breakers_data(app)
        return templates.TemplateResponse(
            "partials/circuit_breakers_card.html", _ctx(request, **data),
        )

    # ------------------------------------------------------------------
    # Action routes (HTMX POST for approvals, promotions, etc.)
    # ------------------------------------------------------------------

    @app.post("/actions/approve/{request_id}", response_class=HTMLResponse)
    async def action_approve(request: Request, request_id: str) -> HTMLResponse:
        toast = _toast_header("Approval granted", "success")
        if app.state.approvals:
            try:
                await app.state.approvals.approve(
                    request_id, decided_by="ui_operator",
                )
            except Exception:
                logger.warning("Approval failed: %s", request_id, exc_info=True)
                toast = _toast_header("Approval failed — check logs", "error")
        else:
            toast = _toast_header("Approval manager not available", "warning")
        data = _build_approvals_data(app)
        resp = templates.TemplateResponse(
            "partials/approvals_card.html", _ctx(request, **data),
        )
        resp.headers.update(toast)
        return resp

    @app.post("/actions/deny/{request_id}", response_class=HTMLResponse)
    async def action_deny(request: Request, request_id: str) -> HTMLResponse:
        toast = _toast_header("Trade denied", "success")
        if app.state.approvals:
            try:
                await app.state.approvals.reject(
                    request_id,
                    decided_by="ui_operator",
                    reason="Denied via supervision UI",
                )
            except Exception:
                logger.warning("Denial failed: %s", request_id, exc_info=True)
                toast = _toast_header("Denial failed — check logs", "error")
        else:
            toast = _toast_header("Approval manager not available", "warning")
        data = _build_approvals_data(app)
        resp = templates.TemplateResponse(
            "partials/approvals_card.html", _ctx(request, **data),
        )
        resp.headers.update(toast)
        return resp

    @app.post(
        "/actions/promote/{strategy_id}", response_class=HTMLResponse,
    )
    async def action_promote(
        request: Request, strategy_id: str,
    ) -> HTMLResponse:
        result = {"approved": False, "reason": "lifecycle not configured"}
        toast = _toast_header("Lifecycle manager not available", "warning")
        if app.state.lifecycle:
            try:
                result = await app.state.lifecycle.request_promotion(
                    strategy_id, operator_id="ui_operator",
                )
                if result.get("approved"):
                    toast = _toast_header(
                        f"Promotion approved for {strategy_id}", "success",
                    )
                else:
                    toast = _toast_header(
                        f"Promotion denied: {result.get('reason', 'unknown')}",
                        "warning",
                    )
            except Exception:
                logger.warning(
                    "Promotion failed: %s", strategy_id, exc_info=True,
                )
                toast = _toast_header("Promotion failed — check logs", "error")
        data = _build_strategies_data(app)
        data["promotion_result"] = result
        resp = templates.TemplateResponse(
            "partials/strategies_list.html", _ctx(request, **data),
        )
        resp.headers.update(toast)
        return resp

    @app.post(
        "/actions/resolve-incident/{incident_id}",
        response_class=HTMLResponse,
    )
    async def action_resolve_incident(
        request: Request, incident_id: str,
    ) -> HTMLResponse:
        toast = _toast_header("Incident resolved", "success")
        if app.state.incidents:
            try:
                await app.state.incidents.resolve_incident(
                    incident_id, resolved_by="ui_operator",
                )
            except Exception:
                logger.warning(
                    "Incident resolve failed: %s", incident_id, exc_info=True,
                )
                toast = _toast_header("Incident resolve failed — check logs", "error")
        else:
            toast = _toast_header("Incident manager not available", "warning")
        data = await _build_banner_data(app)
        resp = templates.TemplateResponse(
            "partials/banner.html", _ctx(request, **data),
        )
        resp.headers.update(toast)
        return resp

    @app.post("/actions/kill-switch/activate", response_class=HTMLResponse)
    async def action_kill_switch_activate(request: Request) -> HTMLResponse:
        toast = _toast_header("Kill switch activated — all orders halted", "warning")
        risk_mgr = app.state.risk_manager
        if risk_mgr is not None and hasattr(risk_mgr, "kill_switch"):
            try:
                await risk_mgr.kill_switch.activate(reason="UI operator activated")
                logger.warning("Kill switch ACTIVATED via UI")
            except Exception:
                logger.warning("Kill switch activate failed", exc_info=True)
                toast = _toast_header("Kill switch activation failed", "error")
        else:
            toast = _toast_header("Risk manager not available", "warning")
        data = await _build_risk_controls_data(app)
        from fastapi.responses import HTMLResponse as _HTML
        ks = data["risk_controls"]
        active = ks.get("kill_switch_active", False)
        html = _build_kill_switch_html(active)
        resp = _HTML(content=html)
        resp.headers.update(toast)
        return resp

    @app.post("/actions/kill-switch/deactivate", response_class=HTMLResponse)
    async def action_kill_switch_deactivate(request: Request) -> HTMLResponse:
        toast = _toast_header("Trading resumed — kill switch deactivated", "success")
        risk_mgr = app.state.risk_manager
        if risk_mgr is not None and hasattr(risk_mgr, "kill_switch"):
            try:
                await risk_mgr.kill_switch.deactivate()
                logger.warning("Kill switch DEACTIVATED via UI")
            except Exception:
                logger.warning("Kill switch deactivate failed", exc_info=True)
                toast = _toast_header("Kill switch deactivation failed", "error")
        else:
            toast = _toast_header("Risk manager not available", "warning")
        data = await _build_risk_controls_data(app)
        ks = data["risk_controls"]
        active = ks.get("kill_switch_active", False)
        from fastapi.responses import HTMLResponse as _HTML
        html = _build_kill_switch_html(active)
        resp = _HTML(content=html)
        resp.headers.update(toast)
        return resp

    @app.post("/actions/positions/close/{symbol:path}", response_class=HTMLResponse)
    async def action_close_position(request: Request, symbol: str) -> HTMLResponse:
        """Close an open position by placing a market order in the opposite direction."""
        adapter = app.state.adapter
        error_msg: str | None = None

        # Safety: respect kill switch — if active, route to exchange directly
        risk_mgr = app.state.risk_manager
        if risk_mgr is not None and hasattr(risk_mgr, "kill_switch"):
            if getattr(risk_mgr.kill_switch, "is_active", False):
                error_msg = "Kill switch active — close positions directly on exchange"

        if error_msg is None and adapter is None:
            error_msg = "No adapter available in this mode"

        if error_msg is None:
            try:
                import uuid
                from decimal import Decimal

                from agentic_trading.core.enums import Exchange, OrderType, Side
                from agentic_trading.core.events import OrderIntent

                # Fetch all positions; match by exact symbol or base symbol
                all_positions = await adapter.get_positions()
                matched = [
                    p for p in all_positions
                    if p.symbol == symbol or p.symbol.split(":")[0] == symbol
                ]

                if not matched or float(matched[0].qty) == 0:
                    error_msg = f"No open position found for {symbol}"
                else:
                    pos = matched[0]
                    pos_side = pos.side.value if hasattr(pos.side, "value") else str(pos.side)
                    close_side = Side.SELL if pos_side.upper() == "LONG" else Side.BUY

                    # Determine exchange from settings, default to BYBIT
                    exchange = Exchange.BYBIT
                    settings = app.state.settings
                    if settings is not None:
                        try:
                            exc = settings.exchanges[0].name
                            exchange = exc if isinstance(exc, Exchange) else Exchange(
                                exc.value if hasattr(exc, "value") else exc
                            )
                        except Exception:
                            pass

                    intent = OrderIntent(
                        dedupe_key=f"ui_close_{symbol}_{uuid.uuid4().hex[:8]}",
                        strategy_id="ui_operator",
                        symbol=pos.symbol,
                        exchange=exchange,
                        side=close_side,
                        order_type=OrderType.MARKET,
                        qty=Decimal(str(pos.qty)),
                        reduce_only=True,
                    )
                    await adapter.submit_order(intent)
                    logger.info(
                        "Position close submitted via UI: symbol=%s side=%s qty=%s",
                        symbol, close_side.value, pos.qty,
                    )
            except Exception:
                logger.warning("Position close failed: symbol=%s", symbol, exc_info=True)
                error_msg = "Close order failed — check logs"

        data = await _build_positions_data(app)
        if error_msg:
            data["close_error"] = error_msg
            toast = _toast_header(error_msg, "error")
        else:
            toast = _toast_header(f"Close order submitted for {symbol}", "success")
        resp = templates.TemplateResponse(
            "partials/positions_card.html", _ctx(request, **data),
        )
        resp.headers.update(toast)
        return resp

    # ------------------------------------------------------------------
    # Backtest launcher
    # ------------------------------------------------------------------

    @app.get("/backtest", response_class=HTMLResponse)
    async def backtest_page(request: Request) -> HTMLResponse:
        data = _build_status_bar_data(app)
        # Provide strategy options from config
        strategies_opts: list[str] = []
        cfg_settings = app.state.settings
        if cfg_settings is not None:
            try:
                strategies_opts = [
                    s.strategy_id for s in cfg_settings.strategies if s.enabled
                ]
            except Exception:
                pass
        if not strategies_opts:
            strategies_opts = [
                "trend_following", "mean_reversion", "breakout",
                "multi_tf_ma", "rsi_divergence", "stochastic_macd",
            ]
        return templates.TemplateResponse(
            "backtest.html", _ctx(request,
                strategies_opts=strategies_opts,
                recent_jobs=list(app.state.backtest_jobs.values())[-5:],
                **data,
            ),
        )

    @app.post("/actions/backtest/run", response_class=HTMLResponse)
    async def action_backtest_run(
        request: Request,
        symbols: str = Form(default="BTC/USDT,ETH/USDT"),
        start_date: str = Form(default="2024-01-01"),
        end_date: str = Form(default="2024-06-30"),
        strategy: str = Form(default=""),
    ) -> HTMLResponse:
        """Launch a backtest subprocess and return the job status fragment."""
        job_id = uuid.uuid4().hex[:8]
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        app.state.backtest_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "strategy": strategy or "all",
            "lines": [],
            "started_at": now_str,
            "finished_at": None,
        }

        async def _run_backtest(jid: str) -> None:
            job = app.state.backtest_jobs.get(jid)
            if job is None:
                return
            cmd = [
                sys.executable, "-m", "agentic_trading", "backtest",
                "--config", "configs/backtest.toml",
                "--symbols", symbols,
                "--start", start_date,
                "--end", end_date,
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(Path(__file__).parent.parent.parent.parent),  # project root
                )
                assert proc.stdout is not None
                async for raw_line in proc.stdout:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                    job["lines"].append(line)
                    if len(job["lines"]) > 500:  # cap memory usage
                        job["lines"] = job["lines"][-500:]
                await proc.wait()
                job["status"] = "success" if proc.returncode == 0 else "failed"
            except Exception as exc:
                job["lines"].append(f"ERROR: {exc}")
                job["status"] = "failed"
            job["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            logger.info("Backtest job %s finished: %s", jid, job["status"])

        asyncio.create_task(_run_backtest(job_id), name=f"backtest-{job_id}")

        return templates.TemplateResponse(
            "partials/backtest_job.html",
            _ctx(request, job=app.state.backtest_jobs[job_id]),
        )

    @app.get("/api/backtest/status/{job_id}", response_class=HTMLResponse)
    async def api_backtest_status(request: Request, job_id: str) -> HTMLResponse:
        """Return updated job status fragment (polled by HTMX)."""
        job = app.state.backtest_jobs.get(job_id)
        if job is None:
            return templates.TemplateResponse(
                "partials/backtest_job.html",
                _ctx(request, job={
                    "job_id": job_id, "status": "failed",
                    "lines": ["Job not found"], "symbols": "", "started_at": "",
                    "finished_at": "", "start_date": "", "end_date": "", "strategy": "all",
                }),
            )
        return templates.TemplateResponse(
            "partials/backtest_job.html", _ctx(request, job=job),
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

    # ------------------------------------------------------------------
    # Error pages (404 / 500)
    # ------------------------------------------------------------------

    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException,
    ) -> HTMLResponse:
        if exc.status_code == 404:
            ctx = _ctx(request, **_build_status_bar_data(app))
            return templates.TemplateResponse(
                "errors/404.html", ctx, status_code=404,
            )
        if exc.status_code == 500:
            ctx = _ctx(request, **_build_status_bar_data(app))
            return templates.TemplateResponse(
                "errors/500.html", ctx, status_code=500,
            )
        # Fallback for other HTTP errors
        return HTMLResponse(
            content=f"<h1>{exc.status_code}</h1><p>{exc.detail}</p>",
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception,
    ) -> HTMLResponse:
        logger.exception("Unhandled exception on %s", request.url.path)
        try:
            ctx = _ctx(request, **_build_status_bar_data(app))
            return templates.TemplateResponse(
                "errors/500.html", ctx, status_code=500,
            )
        except Exception:
            return HTMLResponse(
                content="<h1>500</h1><p>Internal Server Error</p>",
                status_code=500,
            )

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
        **(await _build_banner_data(app)),
        **_build_scorecard_data(app),
        **portfolio,
        **positions,
        **_build_system_data(app),
        **_build_approvals_data(app),
        **_build_status_bar_data(app),
    }


async def _build_banner_data(app: FastAPI) -> dict[str, Any]:
    """Build critical banner data."""
    banner = None

    # Check kill switch
    risk_mgr = app.state.risk_manager
    if risk_mgr is not None and hasattr(risk_mgr, "kill_switch"):
        try:
            if await risk_mgr.kill_switch.is_active():
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

    # Compute P&L from journal closed trades across multiple periods
    pnl_today = 0.0
    pnl_7d = 0.0
    pnl_30d = 0.0
    journal = app.state.journal
    if journal is not None:
        try:
            now_utc = datetime.now(timezone.utc)
            today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_7d = now_utc - timedelta(days=7)
            cutoff_30d = now_utc - timedelta(days=30)
            closed = journal.get_closed_trades()
            for t in closed:
                closed_at = getattr(t, "closed_at", None)
                if closed_at is None:
                    continue
                pnl = float(t.net_pnl)
                if closed_at >= today_start:
                    pnl_today += pnl
                if closed_at >= cutoff_7d:
                    pnl_7d += pnl
                if closed_at >= cutoff_30d:
                    pnl_30d += pnl
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

    pnl_7d_pct = (pnl_7d / total_equity * 100) if total_equity > 0 else 0.0
    pnl_30d_pct = (pnl_30d / total_equity * 100) if total_equity > 0 else 0.0

    portfolio = {
        "equity": round(total_equity, 2),
        "pnl_today": round(pnl_today, 2),
        "pnl_today_pct": round(pnl_pct, 2),
        "pnl_positive": pnl_today >= 0,
        "pnl_7d": round(pnl_7d, 2),
        "pnl_7d_pct": round(pnl_7d_pct, 2),
        "pnl_7d_positive": pnl_7d >= 0,
        "pnl_30d": round(pnl_30d, 2),
        "pnl_30d_pct": round(pnl_30d_pct, 2),
        "pnl_30d_positive": pnl_30d >= 0,
        "open_positions": open_count,
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
    agent_details: list[dict[str, Any]] = []
    if app.state.registry:
        try:
            health = app.state.registry.health_check_all()
            total = len(health)
            healthy = sum(1 for h in health.values() if h.healthy)
            system["agents_total"] = total
            system["agents_healthy"] = healthy
            system["all_healthy"] = healthy == total

            # Build per-agent detail list by cross-referencing health + summary
            summary_list = app.state.registry.summary()
            # Create lookup: full_id -> summary dict (match by 8-char prefix)
            summary_by_prefix: dict[str, dict[str, Any]] = {}
            for s in summary_list:
                summary_by_prefix[s["agent_id"]] = s

            for full_id, report in health.items():
                prefix = full_id[:8]
                summary = summary_by_prefix.get(prefix, {})
                last_work = ""
                if report.last_work_at is not None:
                    try:
                        delta = (
                            datetime.now(timezone.utc) - report.last_work_at
                        ).total_seconds()
                        if delta < 60:
                            last_work = f"{int(delta)}s ago"
                        elif delta < 3600:
                            last_work = f"{int(delta / 60)}m ago"
                        else:
                            last_work = f"{int(delta / 3600)}h ago"
                    except Exception:
                        last_work = str(report.last_work_at)

                agent_details.append({
                    "name": summary.get("name", prefix),
                    "type": summary.get("type", "unknown"),
                    "healthy": report.healthy,
                    "message": report.message or ("OK" if report.healthy else "Unhealthy"),
                    "error_count": report.error_count,
                    "last_work": last_work or "—",
                    "running": summary.get("running", False),
                })
        except Exception:
            logger.warning("Failed to build agent details", exc_info=True)

    system["agent_details"] = agent_details

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
                quality_data = _compute_quality_score(stats)
                quality = quality_data["total"]

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
                    "quality_breakdown": quality_data,
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
                    _empty_breakdown: dict[str, Any] = {
                        "total": 0,
                        "win_rate_score": 0.0, "win_rate_pct": 0.0,
                        "profit_factor_score": 0.0, "profit_factor_val": 0.0,
                        "sharpe_score": 0.0, "sharpe_val": 0.0,
                        "efficiency_score": 0.0, "efficiency_pct": 0.0,
                    }
                    strategies.append({
                        "strategy_id": strat_cfg.strategy_id,
                        "stage": "configured",
                        "quality_score": 0,
                        "quality_grade": "-",
                        "quality_breakdown": _empty_breakdown,
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
    """Build activity timeline from journal trades, signals, and governance events."""
    events: list[dict[str, Any]] = []
    journal = app.state.journal

    if journal is not None:
        try:
            # Open trades (entry fill events)
            if filter_type in ("all", "trades"):
                for t in journal.get_open_trades():
                    ts = ""
                    if hasattr(t, "opened_at") and t.opened_at is not None:
                        ts = t.opened_at.strftime("%H:%M")
                    elif t.entry_fills:
                        ts = t.entry_fills[0].timestamp.strftime("%H:%M")

                    conf = getattr(t, "signal_confidence", 0.0) or 0.0
                    conf_pct = int(conf * 100) if conf <= 1 else int(conf)
                    detail = f"Entry fill · Signal confidence: {conf_pct}%"
                    if t.entry_fills:
                        fill = t.entry_fills[0]
                        detail += f" · Fill price: {float(fill.price):,.4f} · Qty: {float(fill.qty)}"

                    events.append({
                        "time": ts,
                        "type": "fill",
                        "icon": "fill",
                        "title": f"Trade Open — {t.symbol} {t.direction.upper()}",
                        "subtitle": f"{t.strategy_id} · qty {t.remaining_qty}",
                        "detail": detail,
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
                    hold_s = getattr(t, "hold_duration_seconds", 0) or 0

                    detail_parts = [
                        f"Outcome: {outcome}",
                        f"R-multiple: {r_mult:.2f}",
                        f"Hold time: {int(hold_s // 60)}m {int(hold_s % 60)}s",
                    ]
                    if t.exit_fills:
                        fill = t.exit_fills[-1]
                        detail_parts.append(f"Exit price: {float(fill.price):,.4f}")

                    events.append({
                        "time": ts,
                        "type": "fill",
                        "icon": "fill",
                        "title": f"Trade Closed — {t.symbol} {t.direction.upper()} P&L ${pnl:+,.2f}",
                        "subtitle": f"{t.strategy_id} · {outcome}",
                        "detail": " · ".join(detail_parts),
                        "_sort_ts": t.exit_fills[-1].timestamp if t.exit_fills else datetime.min.replace(tzinfo=timezone.utc),
                    })
        except Exception:
            logger.debug("Failed to build activity from journal", exc_info=True)

    # Signal events from governance gate / event bus history
    if filter_type in ("all", "signals"):
        gate = app.state.gate
        if gate is not None:
            try:
                recent_signals = getattr(gate, "recent_signals", [])
                for sig in list(recent_signals)[-30:]:
                    ts = ""
                    sort_ts = datetime.min.replace(tzinfo=timezone.utc)
                    if hasattr(sig, "timestamp"):
                        ts = sig.timestamp.strftime("%H:%M")
                        sort_ts = sig.timestamp
                    conf = getattr(sig, "confidence", 0.0) or 0.0
                    conf_pct = int(conf * 100) if conf <= 1 else int(conf)
                    direction = getattr(sig, "direction", "")
                    symbol = getattr(sig, "symbol", "")
                    strategy = getattr(sig, "strategy_id", "")
                    reasons = getattr(sig, "reasons", [])
                    detail = f"Confidence: {conf_pct}%"
                    if reasons:
                        detail += f" · Indicators: {', '.join(str(r) for r in reasons[:3])}"
                    events.append({
                        "time": ts,
                        "type": "signal",
                        "icon": "signal",
                        "title": f"Signal — {symbol} {str(direction).upper()}",
                        "subtitle": f"{strategy} · confidence {conf_pct}%",
                        "detail": detail,
                        "_sort_ts": sort_ts,
                    })
            except Exception:
                logger.debug("Failed to build signal events", exc_info=True)

    # Governance decisions from governance gate
    if filter_type in ("all", "governance"):
        gate = app.state.gate
        if gate is not None:
            try:
                recent_decisions = getattr(gate, "recent_decisions", [])
                for dec in list(recent_decisions)[-30:]:
                    ts = ""
                    sort_ts = datetime.min.replace(tzinfo=timezone.utc)
                    if hasattr(dec, "timestamp"):
                        ts = dec.timestamp.strftime("%H:%M")
                        sort_ts = dec.timestamp
                    allowed = getattr(dec, "allowed", True)
                    symbol = getattr(dec, "symbol", "")
                    strategy = getattr(dec, "strategy_id", "")
                    reason = getattr(dec, "reason", "")
                    sizing_adj = getattr(dec, "sizing_adjustment", None)
                    title = f"Trade {'Approved' if allowed else 'Blocked'} by Policy — {symbol}"
                    subtitle = strategy
                    detail_parts = []
                    if reason:
                        detail_parts.append(f"Reason: {reason}")
                    if sizing_adj is not None and sizing_adj != 1.0:
                        detail_parts.append(f"Size adjustment: ×{sizing_adj:.2f}")
                    events.append({
                        "time": ts,
                        "type": "governance",
                        "icon": "governance",
                        "title": title,
                        "subtitle": subtitle,
                        "detail": " · ".join(detail_parts) if detail_parts else None,
                        "_sort_ts": sort_ts,
                    })
            except Exception:
                logger.debug("Failed to build governance events", exc_info=True)

    # Approval events
    if filter_type in ("all", "approvals"):
        approvals = app.state.approvals
        if approvals is not None:
            try:
                # Show recent resolved + pending approvals
                all_requests = []
                if hasattr(approvals, "get_pending"):
                    all_requests.extend(approvals.get_pending())
                if hasattr(approvals, "get_recent_resolved"):
                    all_requests.extend(approvals.get_recent_resolved(last_n=20))
                for req in all_requests:
                    ts = ""
                    sort_ts = datetime.min.replace(tzinfo=timezone.utc)
                    created = getattr(req, "created_at", None)
                    if created:
                        ts = created.strftime("%H:%M")
                        sort_ts = created
                    status = getattr(req, "status", "pending")
                    status_str = status.value if hasattr(status, "value") else str(status)
                    symbol = getattr(req, "symbol", "")
                    strategy = getattr(req, "strategy_id", "")
                    trigger = getattr(req, "trigger", "")
                    trigger_str = trigger.value if hasattr(trigger, "value") else str(trigger)
                    notional = getattr(req, "notional_usd", None)
                    subtitle = f"{strategy}"
                    if notional:
                        subtitle += f" · ${float(notional):,.0f} notional"
                    detail = f"Trigger: {trigger_str}"
                    if notional:
                        detail += f" · Notional: ${float(notional):,.2f}"
                    events.append({
                        "time": ts,
                        "type": "approval",
                        "icon": "approval",
                        "title": f"Approval {status_str.title()} — {symbol}",
                        "subtitle": subtitle,
                        "detail": detail,
                        "_sort_ts": sort_ts,
                    })
            except Exception:
                logger.debug("Failed to build approval events", exc_info=True)

    # Incident events
    if filter_type in ("all", "incidents"):
        incidents = app.state.incidents
        if incidents is not None:
            try:
                recent = getattr(incidents, "get_recent", lambda n: [])(20)
                for inc in recent:
                    ts = ""
                    sort_ts = datetime.min.replace(tzinfo=timezone.utc)
                    created = getattr(inc, "created_at", None)
                    if created:
                        ts = created.strftime("%H:%M")
                        sort_ts = created
                    severity = getattr(inc, "severity", "")
                    severity_str = severity.value if hasattr(severity, "value") else str(severity)
                    title_str = getattr(inc, "title", "Incident")
                    description = getattr(inc, "description", "")
                    events.append({
                        "time": ts,
                        "type": "incident",
                        "icon": "incident",
                        "title": f"Incident — {title_str}",
                        "subtitle": severity_str,
                        "detail": description if description else None,
                        "_sort_ts": sort_ts,
                    })
            except Exception:
                logger.debug("Failed to build incident events", exc_info=True)

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
# New data builders (UX redesign)
# ======================================================================


def _build_status_bar_data(app: FastAPI) -> dict[str, Any]:
    """Build data for the persistent status bar."""
    settings = app.state.settings
    risk_mgr = app.state.risk_manager

    mode = "paper"
    exchange_name = "Unknown"
    exchange_connected = app.state.adapter is not None

    if settings is not None:
        try:
            mode = settings.mode.value if hasattr(settings.mode, "value") else str(settings.mode)
            if settings.exchanges:
                exc = settings.exchanges[0]
                exchange_name = exc.name.value if hasattr(exc.name, "value") else str(exc.name)
                exchange_name = exchange_name.title()
        except Exception:
            pass

    # Agent health
    agents_total = 0
    agents_healthy = 0
    if app.state.registry:
        try:
            health = app.state.registry.health_check_all()
            agents_total = len(health)
            agents_healthy = sum(1 for h in health.values() if h.healthy)
        except Exception:
            pass

    # Last candle age (seconds)
    candle_age_seconds = 0
    trading_ctx = app.state.trading_context
    if trading_ctx is not None:
        try:
            last_ts = getattr(trading_ctx, "last_candle_timestamp", None)
            if last_ts is not None:
                now_ts = datetime.now(timezone.utc)
                delta = now_ts - last_ts
                candle_age_seconds = int(delta.total_seconds())
        except Exception:
            pass

    # Pending approvals count
    pending_approvals = 0
    if app.state.approvals:
        try:
            pending_approvals = len(app.state.approvals.get_pending())
        except Exception:
            pass

    return {
        "status_bar": {
            "mode": mode,
            "exchange_name": exchange_name,
            "exchange_connected": exchange_connected,
            "candle_age_seconds": candle_age_seconds,
            "agents_total": agents_total,
            "agents_healthy": agents_healthy,
            "pending_approvals": pending_approvals,
        },
        "pending_approvals_count": pending_approvals,
    }


async def _build_risk_controls_data(app: FastAPI) -> dict[str, Any]:
    """Build data for the Risk & Controls page."""
    settings = app.state.settings
    risk_mgr = app.state.risk_manager

    kill_switch_active = False
    if risk_mgr is not None and hasattr(risk_mgr, "kill_switch"):
        try:
            kill_switch_active = await risk_mgr.kill_switch.is_active()
        except Exception:
            pass

    policy_engine_enabled = False
    policy_rules_count = 0
    approval_workflow = app.state.approvals is not None
    auto_approve_l1 = False
    shadow_mode = False

    if settings is not None:
        try:
            gov = settings.governance
            policy_engine_enabled = gov.enabled
        except Exception:
            pass

    if app.state.gate:
        try:
            pe = app.state.gate.policy_engine
            if pe:
                policy_rules_count = len(getattr(pe, "registered_sets", []))
        except Exception:
            pass

    pending_approvals = 0
    if app.state.approvals:
        try:
            pending_approvals = len(app.state.approvals.get_pending())
        except Exception:
            pass

    risk_controls = {
        "kill_switch_active": kill_switch_active,
        "policy_engine_enabled": policy_engine_enabled,
        "policy_rules_count": policy_rules_count,
        "approval_workflow": approval_workflow,
        "auto_approve_l1": auto_approve_l1,
        "shadow_mode": shadow_mode,
        "pending_approvals": pending_approvals,
    }
    gauges = _build_risk_gauges_data(app)
    return {"risk_controls": risk_controls, **gauges}


def _build_risk_gauges_data(app: FastAPI) -> dict[str, Any]:
    """Build risk utilisation gauge data."""
    settings = app.state.settings
    risk_mgr = app.state.risk_manager

    # Defaults
    daily_loss_limit_pct = 0.05
    drawdown_limit_pct = 0.15
    position_size_limit_pct = 0.10

    if settings is not None:
        try:
            daily_loss_limit_pct = settings.risk.max_daily_loss_pct
            drawdown_limit_pct = settings.risk.max_drawdown_pct
            position_size_limit_pct = settings.risk.max_single_position_pct
        except Exception:
            pass

    # Current values
    daily_loss_current = 0.0
    drawdown_current = 0.0
    largest_position_pct = 0.0

    if risk_mgr is not None:
        try:
            dm = getattr(risk_mgr, "drawdown", None)
            if dm is not None:
                daily_loss_current = float(getattr(dm, "daily_loss", 0))
                drawdown_current = float(getattr(dm, "peak_drawdown", 0))
        except Exception:
            pass

    # Calculate utilisation percentages (as 0-100 scale)
    def pct_used(current: float, limit: float) -> float:
        if limit <= 0:
            return 0.0
        return min(100.0, abs(current) / limit * 100.0)

    daily_loss_pct_used = pct_used(daily_loss_current, daily_loss_limit_pct)
    drawdown_pct_used = pct_used(drawdown_current, drawdown_limit_pct)
    position_pct_used = pct_used(largest_position_pct, position_size_limit_pct)

    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    return {
        "risk_gauges": {
            "daily_loss_pct_used": daily_loss_pct_used,
            "daily_loss_limit": f"-{daily_loss_limit_pct * 100:.1f}%",
            "daily_loss_current": f"${daily_loss_current:,.2f}",
            "drawdown_pct_used": drawdown_pct_used,
            "drawdown_limit": f"-{drawdown_limit_pct * 100:.1f}%",
            "drawdown_current": f"${drawdown_current:,.2f}",
            "position_size_pct_used": position_pct_used,
            "position_size_limit": f"{position_size_limit_pct * 100:.0f}%",
            "position_size_current": f"{largest_position_pct * 100:.1f}%",
            "updated_at": now_str,
        },
    }


def _build_circuit_breakers_data(app: FastAPI) -> dict[str, Any]:
    """Build circuit breaker status data for the Risk & Controls page."""
    risk_mgr = app.state.risk_manager
    breakers_list: list[dict[str, Any]] = []

    if risk_mgr is not None and hasattr(risk_mgr, "circuit_breakers"):
        try:
            cb_mgr = risk_mgr.circuit_breakers
            now_mono = time.monotonic()
            # Friendly display names for breaker types
            _names: dict[str, str] = {
                "volatility": "Volatility",
                "spread": "Spread",
                "liquidity": "Liquidity",
                "staleness": "Data Staleness",
                "error_rate": "Error Rate",
                "clock_skew": "Clock Skew",
            }
            for (bt, symbol), breaker in cb_mgr._breakers.items():
                name = _names.get(bt.value, bt.value.replace("_", " ").title())
                if symbol:
                    name = f"{name} ({symbol})"

                cooldown_remaining = 0
                if breaker.tripped and breaker.last_trip_time > 0:
                    elapsed = now_mono - breaker.last_trip_time
                    remaining = breaker.cooldown_seconds - elapsed
                    cooldown_remaining = max(0, int(remaining))

                breakers_list.append({
                    "name": name,
                    "type": bt.value,
                    "symbol": symbol,
                    "tripped": breaker.tripped,
                    "trip_count": breaker.trip_count,
                    "threshold": breaker.threshold,
                    "cooldown_seconds": int(breaker.cooldown_seconds),
                    "cooldown_remaining": cooldown_remaining,
                    "window_seconds": int(breaker.window_seconds),
                    "hysteresis": breaker.hysteresis,
                })
        except Exception:
            logger.warning("Failed to read circuit breakers", exc_info=True)

    return {"circuit_breakers": breakers_list}


def _build_kill_switch_html(active: bool) -> str:
    """Build minimal kill switch card HTML for HTMX swap."""
    if active:
        return """<div class="kill-switch-card active" id="kill-switch-card"
             hx-get="/partials/risk/kill-switch" hx-trigger="every 10s" hx-swap="outerHTML">
  <div class="kill-switch-info">
    <h3 style="color: var(--color-critical);">🛑 Emergency Stop — ACTIVE</h3>
    <p>All new orders are halted. Existing positions remain open.</p>
  </div>
  <button class="btn-kill-switch deactivate" onclick="showKillSwitchModal('deactivate')">Resume Trading</button>
</div>"""
    return """<div class="kill-switch-card" id="kill-switch-card"
         hx-get="/partials/risk/kill-switch" hx-trigger="every 10s" hx-swap="outerHTML">
  <div class="kill-switch-info">
    <h3>Emergency Stop</h3>
    <p>Activating will halt all new orders immediately. Use in emergencies only.</p>
  </div>
  <button class="btn-kill-switch activate" onclick="showKillSwitchModal('activate')">Activate Emergency Stop</button>
</div>"""


# ======================================================================
# Utility functions
# ======================================================================


def _compute_quality_score(stats: dict[str, Any]) -> dict[str, Any]:
    """Compute a 0-100 quality score from strategy stats.

    Returns a dict with ``total`` (int, 0-100) and individual sub-scores so
    callers can surface breakdowns in tooltips and grade cards.
    """
    wr = stats.get("win_rate", 0)  # 0..1
    pf = stats.get("profit_factor", 0)
    sharpe = stats.get("sharpe", 0)
    avg_eff = stats.get("avg_management_efficiency", 0)

    # Win rate: 0-25 points (50% WR = 12.5, 70% = 17.5)
    wr_score = min(25.0, wr * 25 / 0.5) if wr > 0 else 0.0

    # Profit factor: 0-25 points (1.5 PF = 12.5, 3.0 = 25)
    pf_score = min(25.0, pf * 25 / 3.0) if pf > 0 else 0.0

    # Sharpe: 0-25 points (1.0 = 12.5, 2.0 = 25)
    sharpe_score = min(25.0, max(0.0, sharpe) * 25 / 2.0)

    # Efficiency: 0-25 points
    eff_score = min(25.0, avg_eff * 25)

    return {
        "total": round(wr_score + pf_score + sharpe_score + eff_score),
        "win_rate_score": round(wr_score, 1),
        "win_rate_pct": round(wr * 100, 1),
        "profit_factor_score": round(pf_score, 1),
        "profit_factor_val": round(pf, 2),
        "sharpe_score": round(sharpe_score, 1),
        "sharpe_val": round(sharpe, 2),
        "efficiency_score": round(eff_score, 1),
        "efficiency_pct": round(avg_eff * 100, 1),
    }


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
