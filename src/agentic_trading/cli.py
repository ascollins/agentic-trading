"""CLI entry point for the trading platform."""

from __future__ import annotations

import click

from .core.enums import Mode


@click.group()
def main() -> None:
    """Agentic Trading Platform."""


@main.command()
@click.option("--config", default="configs/backtest.toml", help="Config file path")
@click.option("--symbols", default=None, help="Comma-separated symbols override")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
def backtest(config: str, symbols: str | None, start: str | None, end: str | None) -> None:
    """Run a backtest."""
    import asyncio

    from .main import run

    overrides: dict = {"mode": Mode.BACKTEST.value}
    if symbols:
        overrides["symbols"] = {"symbols": symbols.split(",")}
    if start:
        overrides.setdefault("backtest", {})["start_date"] = start
    if end:
        overrides.setdefault("backtest", {})["end_date"] = end

    asyncio.run(run(config_path=config, overrides=overrides))


@main.command()
@click.option("--config", default="configs/paper.toml", help="Config file path")
@click.option("--symbols", default=None, help="Comma-separated symbols override")
def paper(config: str, symbols: str | None) -> None:
    """Run paper trading."""
    import asyncio

    from .main import run

    overrides: dict = {"mode": Mode.PAPER.value}
    if symbols:
        overrides["symbols"] = {"symbols": symbols.split(",")}

    asyncio.run(run(config_path=config, overrides=overrides))


@main.command()
@click.option("--config", default="configs/live.toml", help="Config file path")
@click.option("--live", "live_flag", is_flag=True, required=True, help="Confirm live trading")
@click.option("--symbols", default=None, help="Comma-separated symbols override")
def live(config: str, live_flag: bool, symbols: str | None) -> None:
    """Run live trading. Requires --live flag AND I_UNDERSTAND_LIVE_TRADING=true."""
    import asyncio

    from .main import run

    overrides: dict = {"mode": Mode.LIVE.value, "live_flag": live_flag}
    if symbols:
        overrides["symbols"] = {"symbols": symbols.split(",")}

    asyncio.run(run(config_path=config, overrides=overrides))


@main.command("walk-forward")
@click.option("--config", default="configs/backtest.toml", help="Config file path")
@click.option("--symbols", default=None, help="Comma-separated symbols override")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--folds", default=5, type=int, help="Number of walk-forward folds")
def walk_forward(config: str, symbols: str | None, start: str | None, end: str | None, folds: int) -> None:
    """Run walk-forward validation."""
    import asyncio

    from .main import run_walk_forward

    overrides: dict = {"mode": Mode.BACKTEST.value}
    if symbols:
        overrides["symbols"] = {"symbols": symbols.split(",")}
    if start:
        overrides.setdefault("backtest", {})["start_date"] = start
    if end:
        overrides.setdefault("backtest", {})["end_date"] = end

    asyncio.run(run_walk_forward(config_path=config, overrides=overrides, n_folds=folds))


@main.command()
@click.option("--config", default="configs/backtest.toml", help="Config file path")
@click.option("--symbols", default=None, help="Comma-separated symbols override")
@click.option("--strategy", default=None, help="Single strategy to optimize (default: all configured)")
@click.option("--samples", default=30, type=int, help="Number of parameter combinations to test")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
def optimize(
    config: str,
    symbols: str | None,
    strategy: str | None,
    samples: int,
    start: str | None,
    end: str | None,
) -> None:
    """Run one-off parameter optimization for strategies."""
    import asyncio

    from .main import run_optimize

    overrides: dict = {"mode": Mode.BACKTEST.value}
    if symbols:
        overrides["symbols"] = {"symbols": symbols.split(",")}
    if start:
        overrides.setdefault("backtest", {})["start_date"] = start
    if end:
        overrides.setdefault("backtest", {})["end_date"] = end

    asyncio.run(
        run_optimize(
            config_path=config,
            overrides=overrides,
            strategy_id=strategy,
            n_samples=samples,
        )
    )


@main.command("optimizer-results")
@click.option("--results-dir", default="data/optimizer_results", help="Results directory")
@click.option("--all", "show_all", is_flag=True, help="Show all results (not just latest)")
def optimizer_results(results_dir: str, show_all: bool) -> None:
    """View optimizer results."""
    from .optimizer.scheduler import OptimizerScheduler

    if show_all:
        all_results = OptimizerScheduler.load_all_results(results_dir)
        if not all_results:
            click.echo("No optimizer results found.")
            return
        for i, result in enumerate(all_results):
            _print_optimizer_result(result, i + 1, len(all_results))
    else:
        latest = OptimizerScheduler.load_latest_results(results_dir)
        if latest is None:
            click.echo("No optimizer results found.")
            return
        _print_optimizer_result(latest)


def _print_optimizer_result(
    result: dict, index: int | None = None, total: int | None = None
) -> None:
    """Print a formatted optimizer result."""
    header = "OPTIMIZER RESULTS"
    if index is not None and total is not None:
        header = f"OPTIMIZER RESULTS ({index}/{total})"

    click.echo(f"\n{'=' * 70}")
    click.echo(header)
    click.echo(f"{'=' * 70}")
    click.echo(f"  Run #:           {result.get('run_number', '?')}")
    click.echo(f"  Started:         {result.get('started_at', '?')}")
    click.echo(f"  Completed:       {result.get('completed_at', '?')}")
    click.echo(f"  Duration:        {result.get('duration_seconds', 0):.1f}s")

    strategies = result.get("strategies", {})
    for strat_id, strat_data in strategies.items():
        click.echo(f"\n--- {strat_id} ---")

        if "error" in strat_data and not strat_data.get("best_params"):
            click.echo(f"  Error: {strat_data['error']}")
            continue

        click.echo(f"  Best Sharpe:     {strat_data.get('best_sharpe', 0):.4f}")
        click.echo(f"  Best Return:     {strat_data.get('best_return', 0):+.2f}%")
        click.echo(f"  Samples Tested:  {strat_data.get('samples_tested', 0)}")
        click.echo(f"  Data Period:     {strat_data.get('data_period', '?')}")
        click.echo(f"  Overfit:         {'YES' if strat_data.get('is_overfit') else 'No'}")

        best_params = strat_data.get("best_params", {})
        if best_params:
            click.echo(f"\n  Best Parameters:")
            for k, v in sorted(best_params.items()):
                click.echo(f"    {k:35s}: {v}")

        wf = strat_data.get("walk_forward")
        if wf:
            click.echo(f"\n  Walk-Forward Validation:")
            click.echo(f"    Avg Train Sharpe:  {wf.get('avg_train_sharpe', 0):.4f}")
            click.echo(f"    Avg Test Sharpe:   {wf.get('avg_test_sharpe', 0):.4f}")
            click.echo(f"    Overfit Score:     {wf.get('overfit_score', 0):.4f}")
            click.echo(f"    Degradation:       {wf.get('degradation_pct', 0):.1f}%")

        top_results = strat_data.get("top_results", [])
        if top_results:
            click.echo(f"\n  Top Results:")
            click.echo(
                f"    {'#':<4} {'Sharpe':>8} {'Return':>9} {'MaxDD':>8} "
                f"{'Trades':>7} {'WinRate':>8}"
            )
            click.echo(f"    {'-' * 50}")
            for i, r in enumerate(top_results, 1):
                click.echo(
                    f"    {i:<4} {r.get('sharpe', 0):>8.4f} "
                    f"{r.get('return_pct', 0):>+8.2f}% "
                    f"{r.get('max_dd', 0):>+7.2f}% "
                    f"{r.get('trades', 0):>7} "
                    f"{r.get('win_rate', 0):>7.1f}%"
                )

    click.echo(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
