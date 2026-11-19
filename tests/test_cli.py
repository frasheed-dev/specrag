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


def test_cli_benchmark_dataset_options():
    """Verify that benchmark CLI command exposes --dataset option."""
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.stdout or "-d" in result.stdout


def test_cli_ui_help():
    """Verify that ui CLI command exposes --port and --host options."""
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.stdout
    assert "--host" in result.stdout


def test_cli_ui_missing_streamlit(monkeypatch):
    """Verify that ui exits cleanly with install hint if streamlit is missing."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    result = runner.invoke(app, ["ui"])
    assert result.exit_code == 1
    assert ".[ui]" in result.stdout
