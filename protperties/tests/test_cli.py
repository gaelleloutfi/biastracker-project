"""
Tests for CLI module (cli.py).

The CLI is intentionally minimal, showing version information by default.
"""
from __future__ import annotations

from typer.testing import CliRunner

from protperties import __version__
from protperties.cli import app


runner = CliRunner()


def test_default_shows_version():
    """Test that invoking with no arguments shows the version."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_flag():
    """Test that the --version flag shows the version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_short_flag():
    """Test that the -v flag shows the version."""
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_command():
    """Test that the --help flag works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "protperties" in result.stdout.lower()
