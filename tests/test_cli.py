"""Tests for SpecRAG CLI using Typer CliRunner."""

from typer.testing import CliRunner

from specrag.cli import app

runner = CliRunner()


def test_cli_help():
    """Verify that root CLI help renders correctly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "SpecRAG" in result.stdout
    assert "benchmark" in result.stdout
    assert "ingest" in result.stdout
    assert "query" in result.stdout


def test_cli_query_help():
    """Verify that query command help renders correctly."""
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "--mode" in result.stdout
    assert "--top-k" in result.stdout


def test_cli_ingest_missing_file():
    """Verify that ingest handles non-existent files gracefully."""
    result = runner.invoke(app, ["ingest", "non_existent_file.pdf"])
    assert result.exit_code != 0
    assert "Error" in result.stdout or "not found" in result.stdout
