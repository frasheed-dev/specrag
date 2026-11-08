# Multi-Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate two established Hugging Face benchmarks (`G4KMU/t2-ragbench` and `ibm-research/REAL-MM-RAG_TechReport`) into SpecRAG with 20-query offline sample packs, ground-truth answer verification, and a Google Colab GPU harness.

**Architecture:** A modular adapter layer (`hf_adapters.py`) normalizes diverse Hugging Face schemas into SpecRAG's `DocumentChunk` and `GoldenEvalItem`. The benchmark runner (`evaluator.py`) executes multi-corpus evaluations measuring both deterministic retrieval (Hit@5, MRR against target table keys) and answer correctness (normalized numeric match + LLM-as-a-judge). A Google Colab notebook provides zero-cost GPU execution for full splits.

**Tech Stack:** Python 3.12, Hugging Face `datasets`, `pydantic`, `qdrant-client`, `rank-bm25`, `sentence-transformers`, `typer`, `rich`, `pytest`.

**Spec:** `specrag/docs/specs/2026-11-08-multi-benchmark-suite-design.md`

## Global Constraints

- Standalone Git repo isolation: All code, tests, and data packs live in `specrag/`.
- No modification of `SECOND_BRAIN` files from this workspace.
- Deterministic offline execution: Tests and local CLI must execute using cached JSON packs without requiring live internet on every run.
- Line length: 100 characters per `pyproject.toml` ruff settings.

## Review Focus

1. Hugging Face dataset download timeouts or schema changes -> Graceful fallbacks and static JSON evaluation fixtures.
2. Table markdown format differences in `T²-RAGBench` -> Handle both raw text tables and Markdown `|---|---|` strings.
3. Multi-page document IDs in `ibm-research` -> Unique chunk IDs scoped to `doc_id` and page index.
4. Numerical variance in ground-truth answers (e.g. `"$42.5M"` vs `"42,500,000"` vs `"42.5"`) -> String and numerical normalization before Exact Match calculation.
5. CPU memory pressure during multi-corpus Qdrant indexing -> Recreate ephemeral collections in Qdrant between corpus runs.

---

### Task 1: Hugging Face Dataset Adapters (`hf_adapters.py`)

**Files:**
- Create: `src/specrag/ingestion/hf_adapters.py`
- Test: `tests/test_hf_adapters.py`

**Interfaces:**
- Produces:
  - `load_t2_ragbench(sample_size: int = 20) -> tuple[dict[str, str], list[dict]]`
  - `load_ibm_techreport(sample_size: int = 20) -> tuple[dict[str, str], list[dict]]`
  - `normalize_benchmark_item(raw_item: dict, benchmark_name: str) -> dict`

- [ ] **Step 1: Write failing test in `tests/test_hf_adapters.py`**

```python
from specrag.ingestion.hf_adapters import normalize_benchmark_item


def test_normalize_t2_ragbench_item():
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hf_adapters.py -v`  
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `src/specrag/ingestion/hf_adapters.py`**

Implement `normalize_benchmark_item` and loaders for `G4KMU/t2-ragbench` and `ibm-research/REAL-MM-RAG_TechReport` with fallback mock data when offline.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hf_adapters.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/specrag/ingestion/hf_adapters.py tests/test_hf_adapters.py
git commit -m "feat: add Hugging Face dataset adapters for T2-RAGBench and IBM TechReport"
```

---

### Task 2: Generate Offline 20-Sample Benchmark Packs

**Files:**
- Create: `scripts/generate_benchmark_packs.py`
- Create: `data/benchmarks/t2_ragbench_20.json`
- Create: `data/benchmarks/ibm_techreport_20.json`
- Create: `data/benchmarks/specs/` (converted markdown documents for the benchmarks)
- Test: `tests/test_benchmark_packs.py`

**Interfaces:**
- Consumes: `src/specrag/ingestion/hf_adapters.py`
- Produces: Static JSON files with 20 queries, ground-truth context references, and canonical answers.

- [ ] **Step 1: Write test verifying benchmark packs integrity**

```python
from pathlib import Path
import json


def test_benchmark_packs_exist_and_valid():
    benchmarks_dir = Path("data/benchmarks")
    t2_path = benchmarks_dir / "t2_ragbench_20.json"
    ibm_path = benchmarks_dir / "ibm_techreport_20.json"

    assert t2_path.exists(), "T2-RAGBench 20 pack missing"
    assert ibm_path.exists(), "IBM TechReport 20 pack missing"

    t2_data = json.loads(t2_path.read_text(encoding="utf-8"))
    assert len(t2_data) == 20
    assert "query" in t2_data[0]
    assert "ground_truth_answer" in t2_data[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_packs.py -v`  
Expected: FAIL with `AssertionError: T2-RAGBench 20 pack missing`.

- [ ] **Step 3: Implement `scripts/generate_benchmark_packs.py` and generate the packs**

Script downloads or synthesizes the top 20 representative queries and contexts from `t2_ragbench` and `ibm_techreport` and saves them in `data/benchmarks/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_packs.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/benchmarks/ scripts/generate_benchmark_packs.py tests/test_benchmark_packs.py
git commit -m "feat: add 20-query offline benchmark packs for T2-RAGBench and IBM TechReport"
```

---

### Task 3: Evaluator Extension for Multi-Corpus & Answer Verification

**Files:**
- Modify: `src/specrag/evaluator.py`
- Test: `tests/test_evaluator_multi_corpus.py`

**Interfaces:**
- Modifies:
  - `BenchmarkRunner.run_benchmark(dataset: str = "local", run_llm_generation: bool = False) -> dict`
  - `BenchmarkRunner.evaluate_answer_correctness(predicted: str, ground_truth: str) -> float`

- [ ] **Step 1: Write failing test in `tests/test_evaluator_multi_corpus.py`**

```python
from specrag.evaluator import BenchmarkRunner


def test_evaluate_answer_correctness():
    runner = BenchmarkRunner()
    # Direct numerical match
    assert runner.evaluate_answer_correctness("1,080 MHz", "1080 MHz") == 1.0
    # Currency match
    assert runner.evaluate_answer_correctness("$42.5 million", "$42.5M") == 1.0
    # Mismatch
    assert runner.evaluate_answer_correctness("310 W", "400 W") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluator_multi_corpus.py -v`  
Expected: FAIL with `AttributeError: 'BenchmarkRunner' object has no attribute 'evaluate_answer_correctness'`.

- [ ] **Step 3: Implement `evaluate_answer_correctness` and dataset parameter in `evaluator.py`**

Add regex normalization for numbers, percentages, and currencies, and support `--dataset local|t2|ibm|all`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluator_multi_corpus.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/specrag/evaluator.py tests/test_evaluator_multi_corpus.py
git commit -m "feat: support multi-corpus evaluation and answer correctness in evaluator"
```

---

### Task 4: CLI Integration & Google Colab GPU Harness

**Files:**
- Modify: `src/specrag/cli.py`
- Create: `notebooks/SpecRAG_Colab_GPU_Benchmark.ipynb`
- Test: `tests/test_cli.py`

**Interfaces:**
- Modifies: `specrag.cli:benchmark` command to accept `--dataset [local|t2|ibm|all]`.

- [ ] **Step 1: Write test for CLI dataset option in `tests/test_cli.py`**

```python
def test_cli_benchmark_dataset_options():
    result = runner.invoke(app, ["benchmark", "--help"])
    assert "--dataset" in result.stdout or "-d" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_cli_benchmark_dataset_options -v`  
Expected: FAIL.

- [ ] **Step 3: Update `cli.py` with `--dataset` argument and build Colab Notebook**

Add Typer option for `--dataset` and construct the Google Colab Notebook with T4 GPU setup commands.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_cli_benchmark_dataset_options -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/specrag/cli.py notebooks/SpecRAG_Colab_GPU_Benchmark.ipynb tests/test_cli.py
git commit -m "feat: add CLI dataset selector and Google Colab GPU benchmark notebook"
```

---

### Task 5: End-to-End Execution, Linting, & Benchmark Results Update

**Files:**
- Modify: `BENCHMARK_RESULTS.md`
- Modify: `README.md`

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`  
Expected: 100% PASS rate.

- [ ] **Step 2: Run ruff check and format**

Run: `ruff check --fix . && ruff format .`  
Expected: Zero lint or formatting errors.

- [ ] **Step 3: Execute multi-dataset benchmark and update `BENCHMARK_RESULTS.md`**

Run: `python -m specrag.cli benchmark --dataset all`  
Verify `BENCHMARK_RESULTS.md` logs comparative Hit@5, MRR, and Answer Correctness across all three corpora.

- [ ] **Step 4: Final Git commit**

```bash
git add .
git commit -m "docs: publish multi-benchmark evaluation results across Hardware, IBM, and T2-RAGBench"
```

