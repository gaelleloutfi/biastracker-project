from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(help="CLI entrypoint for protperties.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Print the installed protperties version.",
    ),
) -> None:
    """CLI entrypoint for protperties."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    
    # If no command and no --version flag, show version by default
    if ctx.invoked_subcommand is None:
        typer.echo(__version__)


if __name__ == "__main__":
    app()
  
  
# NOTE:
# This CLI is intentionally minimal.
# If protperties grows a full pipeline interface,
# this file can be refactored into a factory-based CLI.
