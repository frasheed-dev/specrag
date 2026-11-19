# SpecRAG Local Streamlit Chat UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight, optional local Streamlit web app (`specrag ui`) where users upload technical documents and chat with them using SpecRAG's structure-aware hybrid RAG pipeline with grounded citations.

**Architecture:** Extract a UI-agnostic `SpecRAGPipeline` service in `src/specrag/pipeline.py` backed by in-memory Qdrant (`storage_path=":memory:"`) and injectable/cached embedding and cross-encoder models. Build a thin Streamlit chat interface in `src/specrag/ui/app.py` exposed via a new `specrag ui` CLI command and an optional `.[ui]` dependency extra.

**Tech Stack:** Python 3.11+, Streamlit (`>=1.35.0`, optional extra), Typer, Pydantic, Qdrant (`:memory:`), BM25Okapi, SentenceTransformers (`BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`), PyMuPDF4LLM + MarkItDown.

**Spec:** `docs/superpowers/specs/2026-11-19-streamlit-chat-ui-design.md`

## Global Constraints

- Reuse existing `specrag` components (`DocumentConverter`, `StructureAwareMarkdownChunker`, `QdrantDenseRetriever`, `BM25SparseRetriever`, `HybridRRFRetriever`, `CrossEncoderReranker`, `RAGGenerator`, `LLMRouter`) without duplicating retrieval logic.
- Keep `streamlit` strictly optional under `[project.optional-dependencies].ui` so core CLI/benchmark users are unaffected.
- Never reload embedding or cross-encoder models on Streamlit reruns; cache them with `@st.cache_resource`.
- Always close `QdrantClient` cleanly on `clear()` / `close()` to avoid Python 3.13 teardown errors.
- Follow `ruff` linting rules (`line-length = 100`, target `py311`) and ensure all existing tests continue to pass.

## Review Focus

1. **Querying before any document is indexed:** `SpecRAGPipeline.query()` must return an unanswerable `QueryResult` with a helpful message instead of crashing on an empty Qdrant/BM25 index.
2. **No LLM API key configured and Ollama unavailable:** `SpecRAGPipeline.query()` must catch generation failures gracefully and still return the retrieved chunks/tables plus an actionable setup hint.
3. **Multiple Streamlit sessions or rapid re-indexing on Windows:** Using `storage_path=":memory:"` prevents local Qdrant disk-lock collisions (`RuntimeError: Storage folder is already accessed`).
4. **Uploaded file with unsupported or empty content:** `ingest_bytes()` must clean up its temporary file in a `finally` block and raise or report a clear error rather than leaving orphan temp files or indexing zero-chunk documents silently.
5. **Running `specrag ui` without `.[ui]` installed:** The CLI command must detect the missing `streamlit` import and exit cleanly with an install command (`uv pip install -e ".[ui]"`).

---

### Task 1: In-Memory Qdrant Option & `SpecRAGPipeline` Service

**Files:**
- Modify: `src/specrag/retriever.py:49-90` and `src/specrag/retriever.py:236-252`
- Create: `src/specrag/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `DocumentConverter`, `StructureAwareMarkdownChunker`, `QdrantDenseRetriever`, `BM25SparseRetriever`, `HybridRRFRetriever`, `CrossEncoderReranker`, `RAGGenerator`, `SpecAnswer`, `DocumentChunk`.
- Produces:
  - `QdrantDenseRetriever(..., storage_path=":memory:", embedding_model: SentenceTransformer | None = None)`
  - `CrossEncoderReranker(..., reranker_model: CrossEncoder | None = None)`
  - `QueryResult(BaseModel)` with fields: `answer: SpecAnswer`, `provider: str`, `retrieved_chunks: list[DocumentChunk]`, `latency_ms: float`, `generation_error: str | None = None`
  - `SpecRAGPipeline` with methods:
    - `ingest_markdown(doc_id: str, markdown_text: str) -> int`
    - `ingest_bytes(filename: str, data: bytes) -> int`
    - `query(query_text: str, top_k: int = 5, use_reranker: bool = True) -> QueryResult`
    - `clear() -> None`
    - `close() -> None`
    - Properties: `document_names: list[str]`, `chunk_count: int`

- [ ] **Step 1: Write failing unit tests in `tests/test_pipeline.py`**

```python
from unittest.mock import MagicMock

from specrag.generator import SpecAnswer
from specrag.pipeline import QueryResult, SpecRAGPipeline


SAMPLE_MD = """# H100 Datasheet

## Power Specifications

| Parameter | Value |
| :--- | :--- |
| Total Board Power | 400 W |
| SKU | P1010 SKU 210 |
"""


def test_query_on_empty_index_returns_unanswerable():
    pipeline = SpecRAGPipeline(dense_retriever=MagicMock(), generator=MagicMock())
    result = pipeline.query("What is TDP?")
    assert isinstance(result, QueryResult)
    assert result.answer.is_answerable is False
    assert result.retrieved_chunks == []


def test_ingest_markdown_and_query_with_reranker_and_fallback():
    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = []
    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = (
        SpecAnswer(
            answer="400 W",
            cited_clause_ids=["Power Specifications"],
            verbatim_evidence=["| Total Board Power | 400 W |"],
            confidence=0.98,
            is_answerable=True,
        ),
        "groq",
    )
    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda q, cands, top_k=5: [(c, 0.99) for c in cands[:top_k]]

    pipeline = SpecRAGPipeline(
        dense_retriever=mock_dense,
        reranker=mock_reranker,
        generator=mock_gen,
    )
    added = pipeline.ingest_markdown("h100.md", SAMPLE_MD)
    assert added >= 1
    assert pipeline.document_names == ["h100.md"]
    assert pipeline.chunk_count == added

    res = pipeline.query("P1010 SKU 210", top_k=3, use_reranker=True)
    assert res.answer.answer == "400 W"
    assert res.provider == "groq"
    assert len(res.retrieved_chunks) >= 1
    mock_reranker.rerank.assert_called_once()


def test_query_gracefully_handles_llm_failure():
    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = []
    mock_gen = MagicMock()
    mock_gen.generate_answer.side_effect = RuntimeError("No API key configured")

    pipeline = SpecRAGPipeline(dense_retriever=mock_dense, generator=mock_gen)
    pipeline.ingest_markdown("h100.md", SAMPLE_MD)

    res = pipeline.query("Total Board Power", top_k=3, use_reranker=False)
    assert res.generation_error is not None
    assert "No API key" in res.generation_error
    assert len(res.retrieved_chunks) >= 1


def test_ingest_bytes_cleans_up_temp_file_and_clear_resets_state():
    mock_dense = MagicMock()
    pipeline = SpecRAGPipeline(dense_retriever=mock_dense, generator=MagicMock())
    count = pipeline.ingest_bytes("spec.md", SAMPLE_MD.encode("utf-8"))
    assert count >= 1
    pipeline.clear()
    assert pipeline.chunk_count == 0
    assert pipeline.document_names == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'specrag.pipeline'`.

- [ ] **Step 3: Update `src/specrag/retriever.py` for `:memory:` Qdrant and optional preloaded models**

In `QdrantDenseRetriever.__init__`, accept `embedding_model: SentenceTransformer | None = None`; if `self.storage_path == ":memory:"`, initialize `QdrantClient(location=":memory:")` instead of `QdrantClient(path=self.storage_path)`, and set `self._model = embedding_model`.
In `CrossEncoderReranker.__init__`, accept `reranker_model: CrossEncoder | None = None` and assign `self._reranker = reranker_model`.

- [ ] **Step 4: Implement `src/specrag/pipeline.py`**

Implement `QueryResult` and `SpecRAGPipeline` using `StructureAwareMarkdownChunker(max_prose_chunk_size=1000)`, `DocumentConverter`, `QdrantDenseRetriever(collection_name="specrag_session", storage_path=":memory:", ...)`, `BM25SparseRetriever`, `HybridRRFRetriever`, `CrossEncoderReranker`, and `RAGGenerator`.
- `ingest_bytes(filename: str, data: bytes) -> int` writes bytes to a `tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False)`, converts via `DocumentConverter().convert_file(tmp_path)`, removes the temp file in `finally`, and delegates to `ingest_markdown(filename, md_text)`.
- `ingest_markdown` replaces any prior chunks for the same `doc_id`, appends new chunks, and re-indexes both dense and sparse retrievers.
- `query` runs `HybridRRFRetriever.retrieve(query_text, top_k=max(20, top_k * 3))`, optionally reranks via `CrossEncoderReranker.rerank(..., top_k=top_k)`, and calls `RAGGenerator.generate_answer(...)` inside a `try/except` that populates `generation_error` and returns a fallback `SpecAnswer` without losing `retrieved_chunks`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py tests/test_hybrid_rrf.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/specrag/retriever.py src/specrag/pipeline.py tests/test_pipeline.py
git commit -m "feat: add SpecRAGPipeline service and in-memory Qdrant support"
```

---

### Task 2: Streamlit Local Chat UI

**Files:**
- Create: `src/specrag/ui/__init__.py`
- Create: `src/specrag/ui/app.py`

**Interfaces:**
- Consumes: `SpecRAGPipeline`, `QueryResult`, `get_available_providers`, `settings`, `SentenceTransformer`, `CrossEncoder`.
- Produces: Runnable Streamlit script `src/specrag/ui/app.py` (invoked via `streamlit run`).

- [ ] **Step 1: Create `src/specrag/ui/__init__.py` and `src/specrag/ui/app.py`**

Implement `src/specrag/ui/app.py` with:
- `@st.cache_resource` helpers `load_embedding_model()` and `load_reranker_model()` so heavy Hugging Face weights load once per server process.
- `get_pipeline() -> SpecRAGPipeline` stored in `st.session_state["pipeline"]`, initialized with cached models and `:memory:` Qdrant.
- Sidebar (`st.sidebar`):
  - Multi-file uploader (`type=["pdf", "md", "txt", "docx", "html"]`).
  - "Index Documents" button that calls `pipeline.ingest_bytes(file.name, file.getvalue())` with `st.spinner` and reports total indexed files and chunks.
  - "Clear Index & Chat" button that calls `pipeline.clear()` and resets `st.session_state["messages"]`.
  - Retrieval settings: `"Use Cross-Encoder Reranker"` checkbox (default `True`, with caption noting ~0.5s GPU / ~15–20s CPU) and `"Top-K Context Chunks"` slider (`1`–`10`, default `5`).
  - Active LLM providers list via `get_available_providers()` showing which providers are ready from `.env` / Ollama.
- Main area:
  - Title + one-line description ("Upload technical specs and chat with grounded, verbatim citations.").
  - Chat message history rendered with `st.chat_message`. For each assistant turn, render:
    - `result.answer.answer` (or warning if `result.generation_error` is set).
    - Compact metadata caption: confidence score, answerability status, LLM provider, and latency in ms.
    - Collapsible `st.expander("Citations & Retrieved Evidence")` displaying `cited_clause_ids`, `verbatim_evidence` blockquotes, and each retrieved `DocumentChunk` (with `[TABLE]`/`[PROSE]` badge, doc name, breadcrumbs, and full Markdown text so tables render cleanly).
  - `st.chat_input("Ask a question about your uploaded documents...")` wired to `pipeline.query(prompt, top_k=top_k, use_reranker=use_reranker)`.

- [ ] **Step 2: Syntax & lint verification**

Run: `.\.venv\Scripts\python.exe -m py_compile src/specrag/ui/app.py`
Run: `.\.venv\Scripts\python.exe -m ruff check src/specrag/ui/app.py src/specrag/pipeline.py src/specrag/retriever.py`
Expected: Zero syntax or lint errors.

- [ ] **Step 3: Commit**

```bash
git add src/specrag/ui/__init__.py src/specrag/ui/app.py
git commit -m "feat(ui): add lightweight Streamlit document chat app"
```

---

### Task 3: Optional `.[ui]` Packaging, `specrag ui` CLI Command, and README Update

**Files:**
- Modify: `pyproject.toml:28-35`
- Modify: `src/specrag/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md` (Quickstart section)

**Interfaces:**
- Consumes: `src/specrag/ui/app.py` path resolved via `Path(__file__).resolve().parent / "ui" / "app.py"`.
- Produces:
  - Optional extra `ui = ["streamlit>=1.35.0"]` in `pyproject.toml`.
  - CLI command `specrag ui [--port 8501] [--host localhost]`.

- [ ] **Step 1: Add failing CLI tests in `tests/test_cli.py`**

```python
def test_cli_ui_help():
    """Verify that ui command help renders correctly."""
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.stdout
    assert "--host" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli.py::test_cli_ui_help -v`
Expected: FAIL with "No such command 'ui'".

- [ ] **Step 3: Add `ui` optional extra to `pyproject.toml` and `ui` command to `src/specrag/cli.py`**

- In `pyproject.toml`, add `ui = ["streamlit>=1.35.0"]` under `[project.optional-dependencies]`.
- In `src/specrag/cli.py`, add `@app.command() def ui(port: int = ..., host: str = ...)`:
  - Check `importlib.util.find_spec("streamlit")`; if `None`, print an actionable install hint (`uv pip install -e ".[ui]"`) and `raise typer.Exit(code=1)`.
  - Otherwise launch `subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port), "--server.address", host], check=False)`.

- [ ] **Step 4: Update `README.md` Quickstart**

Add a concise "Local Chat UI" subsection under Quickstart showing:
```bash
uv pip install -e ".[ui]"
python -m specrag.cli ui
```

- [ ] **Step 5: Run full test suite & linter**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
Run: `.\.venv\Scripts\python.exe -m ruff check src/ tests/`
Expected: All tests PASS and zero ruff errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/specrag/cli.py tests/test_cli.py README.md
git commit -m "feat(cli): add specrag ui launcher, optional ui extra, and docs"
```
