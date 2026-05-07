import typer
from typing import Optional
from biastracker import __version__

app = typer.Typer()

def version_callback(value: bool):
    if value:
        typer.echo(f"BiasTracker version: {__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Print the BiasTracker version.",
    ),
):
    """BiasTracker CLI."""
    pass

@app.command()
def version_cmd():
    """Print the BiasTracker version."""
    typer.echo(f"BiasTracker version: {__version__}")

@app.command()
def check():
    """Check if the protperties package is correctly installed and accessible."""
    from biastracker.dataset import check_protperties_available
    try:
        check_protperties_available()
        typer.echo("Success: protperties is correctly installed and accessible.")
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
