"""Tests for offline 20-sample benchmark packs integrity."""

import json
from pathlib import Path


def test_benchmark_packs_exist_and_valid():
    """Verify that both T2-RAGBench and IBM TechReport benchmark packs exist and have 20 items."""
    benchmarks_dir = Path("data/benchmarks")
    t2_path = benchmarks_dir / "t2_ragbench_20.json"
    ibm_path = benchmarks_dir / "ibm_techreport_20.json"

    assert t2_path.exists(), "T2-RAGBench 20 pack missing"
    assert ibm_path.exists(), "IBM TechReport 20 pack missing"

    t2_data = json.loads(t2_path.read_text(encoding="utf-8"))
    assert len(t2_data) == 20
    assert "query" in t2_data[0]
    assert "ground_truth_answer" in t2_data[0]
    assert "expected_doc_id" in t2_data[0]

    ibm_data = json.loads(ibm_path.read_text(encoding="utf-8"))
    assert len(ibm_data) == 20
    assert "query" in ibm_data[0]
    assert "ground_truth_answer" in ibm_data[0]
    assert "expected_doc_id" in ibm_data[0]


def test_benchmark_specs_markdown_exist():
    """Verify that the underlying markdown documents for the benchmarks exist."""
    specs_dir = Path("data/benchmarks/specs")
    assert specs_dir.exists()
    markdown_files = list(specs_dir.glob("*.md"))
    assert len(markdown_files) >= 2
