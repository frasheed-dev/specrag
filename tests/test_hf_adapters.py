"""Unit tests for Hugging Face dataset adapters and normalizers."""

from specrag.ingestion.hf_adapters import normalize_benchmark_item


def test_normalize_t2_ragbench_item():
    """Verify normalization of T2-RAGBench text-and-table queries."""
    raw = {
        "question": "What was the operating margin in 2022?",
        "context": "| Year | Margin |\n|---|---|\n| 2022 | 18.5% |",
        "answer": "18.5%",
        "table_id": "table_fin_01",
    }
    normalized = normalize_benchmark_item(raw, "t2_ragbench")
    assert normalized["query"] == "What was the operating margin in 2022?"
    assert normalized["ground_truth_answer"] == "18.5%"
    assert normalized["expected_doc_id"] == "t2_ragbench_table_fin_01.md"
    assert "t2_ragbench" in normalized["query_id"]


def test_normalize_ibm_techreport_item():
    """Verify normalization of IBM TechReport specification queries."""
    raw = {
        "query": "What is the maximum interface clock frequency for Module X?",
        "positive_passage": "Table 4. Interface Timing Specifications: Max Clock is 800 MHz.",
        "answer": "800 MHz",
        "doc_id": "ibm_power10_overview.pdf",
        "section": "Table 4. Interface Timing Specifications",
    }
    normalized = normalize_benchmark_item(raw, "ibm_techreport")
    assert normalized["query"] == "What is the maximum interface clock frequency for Module X?"
    assert normalized["ground_truth_answer"] == "800 MHz"
    assert normalized["expected_doc_id"] == "ibm_power10_overview.md"
    assert normalized["expected_section_or_clause"] == "Table 4. Interface Timing Specifications"
