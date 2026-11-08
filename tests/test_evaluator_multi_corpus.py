"""Tests for multi-corpus evaluator and normalized answer correctness scoring."""

from specrag.evaluator import BenchmarkRunner


def test_evaluate_answer_correctness():
    """Verify numeric, percentage, and currency normalization during answer evaluation."""
    runner = BenchmarkRunner()

    # Direct numerical match with comma formatting
    assert runner.evaluate_answer_correctness("1,080 MHz", "1080 MHz") == 1.0

    # Currency match
    assert runner.evaluate_answer_correctness("$42.5 million", "$42.5M") == 1.0
    assert runner.evaluate_answer_correctness("33,088", "33088") == 1.0

    # Percentage match
    assert runner.evaluate_answer_correctness("5.2%", "5.2 %") == 1.0

    # Negative / Mismatch
    assert runner.evaluate_answer_correctness("310 W", "400 W") == 0.0
    assert runner.evaluate_answer_correctness("TSMC", "Intel") == 0.0


def test_load_dataset_by_name():
    """Verify loading different benchmark suites by name."""
    runner = BenchmarkRunner()

    # Local hardware & legal specs
    docs_local, golden_local = runner.load_benchmark_data("local")
    assert len(golden_local) == 15
    assert len(docs_local) >= 3

    # T2-RAGBench
    docs_t2, golden_t2 = runner.load_benchmark_data("t2")
    assert len(golden_t2) == 20
    assert len(docs_t2) >= 1

    # IBM TechReport
    docs_ibm, golden_ibm = runner.load_benchmark_data("ibm")
    assert len(golden_ibm) == 20
    assert len(docs_ibm) >= 1

