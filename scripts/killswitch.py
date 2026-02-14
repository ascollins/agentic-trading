#!/usr/bin/env python3
"""Emergency kill switch CLI.

Usage:
  python scripts/killswitch.py --activate --reason "Manual halt"
  python scripts/killswitch.py --deactivate
  python scripts/killswitch.py --status
"""

import asyncio
import sys

import click

sys.path.insert(0, "src")


@click.command()
@click.option("--activate", is_flag=True, help="Activate kill switch")
@click.option("--deactivate", is_flag=True, help="Deactivate kill switch")
@click.option("--status", "check_status", is_flag=True, help="Check kill switch status")
@click.option("--reason", default="CLI manual activation", help="Reason for activation")
@click.option("--redis-url", default="redis://localhost:6379/0", help="Redis URL")
def main(
    activate: bool,
    deactivate: bool,
    check_status: bool,
    reason: str,
    redis_url: str,
) -> None:
    """Kill switch management."""
    asyncio.run(_run(activate, deactivate, check_status, reason, redis_url))


async def _run(
    activate: bool,
    deactivate: bool,
    check_status: bool,
    reason: str,
    redis_url: str,
) -> None:
    import redis.asyncio as aioredis

    r = aioredis.from_url(redis_url, decode_responses=True)

    try:
        if activate:
            await r.set("kill_switch:active", "1")
            await r.set("kill_switch:reason", reason)
            click.echo(f"KILL SWITCH ACTIVATED: {reason}")

        elif deactivate:
            await r.delete("kill_switch:active")
            await r.delete("kill_switch:reason")
            click.echo("Kill switch deactivated")

        elif check_status:
            active = await r.get("kill_switch:active")
            reason_str = await r.get("kill_switch:reason") or ""
            if active:
                click.echo(f"Kill switch: ACTIVE (reason: {reason_str})")
            else:
                click.echo("Kill switch: INACTIVE")

        else:
            click.echo("Specify --activate, --deactivate, or --status")

    finally:
        await r.aclose()


if __name__ == "__main__":
    main()
