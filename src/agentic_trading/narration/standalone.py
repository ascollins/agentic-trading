"""Standalone narration server for development and testing.

Run directly to start the narration HTTP server without the full trading platform:

    python -m agentic_trading.narration.standalone

This starts:
  - The narration HTTP server on port 8099
  - A mock Tavus adapter (or real if TAVUS_API_KEY is set and --real-tavus flag used)
  - Pre-populated with a sample narration item so you can immediately test

Endpoints:
  GET  http://localhost:8099/narration/latest
  GET  http://localhost:8099/narration/health
  GET  http://localhost:8099/avatar/watch
  POST http://localhost:8099/avatar/briefing
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone

from .schema import (
    ConsideredSetup,
    DecisionExplanation,
    NarrationItem,
    PositionSnapshot,
    RiskSummary,
)
from .server import start_narration_server
from .service import NarrationService, Verbosity
from .store import NarrationStore
from .tavus import MockTavusAdapter, TavusAdapterHttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _seed_store(store: NarrationStore, service: NarrationService) -> None:
    """Pre-populate store with sample narrations for immediate testing."""
    samples = [
        DecisionExplanation(
            symbol="BTC/USDT",
            timeframe="5m",
            market_summary="Bitcoin has been climbing steadily over the past hour with solid buying pressure.",
            active_strategy="trend_following",
            active_regime="trend",
            regime_confidence=0.85,
            considered_setups=[
                ConsideredSetup(name="momentum breakout", direction="long", confidence=0.82, status="triggered"),
            ],
            action="ENTER",
            reasons=["Strong upward momentum confirmed by rising volume", "Price above key support level"],
            reason_confidences=[0.85, 0.78],
            risk=RiskSummary(
                intended_size_pct=0.03,
                stop_invalidation="Below 96,500 support",
                health_score=0.92,
                maturity_level="L3_constrained",
            ),
            position=PositionSnapshot(
                open_positions=1,
                gross_exposure_usd=5000.0,
                unrealized_pnl_usd=120.0,
                available_balance_usd=95000.0,
            ),
            trace_id="demo-trace-001",
        ),
        DecisionExplanation(
            symbol="ETH/USDT",
            timeframe="15m",
            market_summary="Ethereum is moving sideways in a tight range with declining volume.",
            active_strategy="mean_reversion",
            active_regime="range",
            action="NO_TRADE",
            reasons=["No clear directional edge"],
            why_not=["Spread above acceptable threshold", "Volume too low for confident entry"],
            what_would_change=["Volume picks up above average", "Price breaks out of the range"],
            risk=RiskSummary(
                active_blocks=["spread_circuit_breaker"],
                health_score=0.70,
            ),
            trace_id="demo-trace-002",
        ),
        DecisionExplanation(
            symbol="BTC/USDT",
            timeframe="5m",
            market_summary="Bitcoin holding near session highs with momentum intact.",
            active_strategy="trend_following",
            active_regime="trend",
            action="HOLD",
            reasons=["Trend still intact, no exit signal"],
            position=PositionSnapshot(
                open_positions=1,
                gross_exposure_usd=5200.0,
                unrealized_pnl_usd=200.0,
                available_balance_usd=94800.0,
            ),
            trace_id="demo-trace-003",
        ),
    ]

    for exp in samples:
        item = service.generate(exp, force=True)
        if item is not None:
            item.metadata = {
                "action": exp.action,
                "symbol": exp.symbol,
                "regime": exp.active_regime,
            }
            store.add(item)

    logger.info("Seeded narration store with %d sample items", store.count)


async def run_standalone(
    port: int = 8099,
    use_real_tavus: bool = False,
) -> None:
    """Start the standalone narration server."""
    service = NarrationService(
        verbosity=Verbosity.NORMAL,
        heartbeat_seconds=60.0,
        dedupe_window_seconds=0,  # Allow all samples through
    )
    store = NarrationStore(max_items=200)

    # Seed with sample data
    _seed_store(store, service)

    # Create Tavus adapter
    if use_real_tavus and os.environ.get("TAVUS_API_KEY"):
        logger.info("Using real Tavus adapter")
        tavus = TavusAdapterHttp()
    else:
        logger.info("Using mock Tavus adapter")
        tavus = MockTavusAdapter(base_url=f"http://localhost:{port}")

    # Start server
    runner = await start_narration_server(
        store=store,
        tavus=tavus,
        port=port,
    )

    logger.info("=" * 60)
    logger.info("Narration server running!")
    logger.info("  Text stream:  http://localhost:%d/narration/latest", port)
    logger.info("  Health check: http://localhost:%d/narration/health", port)
    logger.info("  Avatar watch: http://localhost:%d/avatar/watch", port)
    logger.info("  Briefing:     POST http://localhost:%d/avatar/briefing", port)
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop.")

    # Wait for shutdown signal
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    # Cleanup
    await runner.cleanup()
    await tavus.close()
    logger.info("Narration server stopped.")


def main() -> None:
    """CLI entry point."""
    use_real = "--real-tavus" in sys.argv
    port = 8099

    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])

    asyncio.run(run_standalone(port=port, use_real_tavus=use_real))


if __name__ == "__main__":
    main()
