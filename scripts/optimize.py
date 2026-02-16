#!/usr/bin/env python3
"""Strategy parameter optimizer.

Runs automated parameter search with walk-forward validation to find
optimal strategy parameters. Optionally fetches live Bybit market context.

Usage:
    python3 scripts/optimize.py --strategy trend_following --start 2024-01-01 --end 2024-01-31
    python3 scripts/optimize.py --all --start 2024-01-01 --end 2024-01-31 --samples 30
    python3 scripts/optimize.py --strategy breakout --samples 20 --write-config
    python3 scripts/optimize.py --all --live-context
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@click.command()
@click.option("--strategy", default=None, help="Strategy ID to optimize (e.g. trend_following)")
@click.option("--all", "optimize_all", is_flag=True, help="Optimize all strategies")
@click.option("--start", default="2024-01-01", help="Backtest start date (YYYY-MM-DD)")
@click.option("--end", default="2024-01-31", help="Backtest end date (YYYY-MM-DD)")
@click.option("--symbols", default="BTC/USDT,ETH/USDT", help="Comma-separated symbols")
@click.option("--samples", default=30, type=int, help="Number of parameter samples")
@click.option("--live-context", is_flag=True, help="Fetch live Bybit market context")
@click.option("--write-config", is_flag=True, help="Write optimized params to config files")
@click.option("--wf-folds", default=3, type=int, help="Walk-forward validation folds")
def main(
    strategy: str | None,
    optimize_all: bool,
    start: str,
    end: str,
    symbols: str,
    samples: int,
    live_context: bool,
    write_config: bool,
    wf_folds: int,
) -> None:
    """Run strategy parameter optimization."""
    asyncio.run(
        _run(
            strategy=strategy,
            optimize_all=optimize_all,
            start=start,
            end=end,
            symbols=symbols.split(","),
            samples=samples,
            live_context=live_context,
            write_config=write_config,
            wf_folds=wf_folds,
        )
    )


async def _run(
    strategy: str | None,
    optimize_all: bool,
    start: str,
    end: str,
    symbols: list[str],
    samples: int,
    live_context: bool,
    write_config: bool,
    wf_folds: int,
) -> None:
    from agentic_trading.core.enums import Exchange, Timeframe
    from agentic_trading.data.historical import HistoricalDataLoader
    from agentic_trading.features.engine import FeatureEngine
    from agentic_trading.optimizer.engine import ParameterOptimizer
    from agentic_trading.optimizer.param_grid import list_strategies_with_grids
    from agentic_trading.optimizer.report import (
        OptimizationReport,
        print_summary,
        to_full_toml,
    )

    # Determine which strategies to optimize
    if optimize_all:
        strategy_ids = list_strategies_with_grids()
    elif strategy:
        strategy_ids = [strategy]
    else:
        click.echo("Error: specify --strategy <name> or --all")
        sys.exit(1)

    click.echo(f"\n{'=' * 70}")
    click.echo(f"TRADING AGENT OPTIMIZER")
    click.echo(f"{'=' * 70}")
    click.echo(f"  Strategies:  {', '.join(strategy_ids)}")
    click.echo(f"  Symbols:     {', '.join(symbols)}")
    click.echo(f"  Period:      {start} to {end}")
    click.echo(f"  Samples:     {samples}")
    click.echo(f"  WF Folds:    {wf_folds}")

    # Load historical data
    click.echo(f"\nLoading historical data...")
    loader = HistoricalDataLoader(data_dir="data/historical")
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    candles_by_symbol = {}
    for symbol in symbols:
        candles = loader.load_candles(
            exchange=Exchange.BINANCE,
            symbol=symbol.strip(),
            timeframe=Timeframe.M1,
            start=start_dt,
            end=end_dt,
        )
        if candles:
            candles_by_symbol[symbol.strip()] = candles
            click.echo(f"  {symbol}: {len(candles)} candles loaded")
        else:
            click.echo(f"  {symbol}: NO DATA FOUND")

    if not candles_by_symbol:
        click.echo("Error: no historical data found. Run download_historical.py first.")
        sys.exit(1)

    # Optional: live market context
    if live_context:
        click.echo(f"\nFetching live market context from Bybit...")
        try:
            from agentic_trading.optimizer.market_context import MarketContextFetcher

            fetcher = MarketContextFetcher()
            for symbol in symbols:
                # Convert BTC/USDT → BTCUSDT for Bybit
                bybit_symbol = symbol.strip().replace("/", "")
                ctx = fetcher.fetch_context(bybit_symbol)
                fetcher.print_context(ctx)
        except Exception as e:
            click.echo(f"  Warning: Failed to fetch live context: {e}")

    # Run optimization for each strategy
    reports: list[OptimizationReport] = []

    for sid in strategy_ids:
        click.echo(f"\n{'─' * 70}")
        click.echo(f"Optimizing: {sid} ({samples} samples)...")
        click.echo(f"{'─' * 70}")

        feature_engine = FeatureEngine()

        optimizer = ParameterOptimizer(
            strategy_id=sid,
            candles_by_symbol=candles_by_symbol,
            feature_engine=feature_engine,
        )

        report = await optimizer.run(
            n_samples=samples,
            top_n_for_wf=3,
            wf_folds=wf_folds,
        )

        reports.append(report)
        print_summary(report)

    # Write optimized config
    if write_config and reports:
        toml_content = to_full_toml(reports)
        click.echo(f"\n--- GENERATED TOML ---")
        click.echo(toml_content)
        click.echo(f"\nTo apply, copy the above into configs/backtest.toml and configs/live.toml")

    click.echo(f"\nOptimization complete.")


if __name__ == "__main__":
    main()
