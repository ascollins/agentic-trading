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


if __name__ == "__main__":
    main()
