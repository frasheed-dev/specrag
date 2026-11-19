# Design: SpecRAG Local Streamlit Chat UI

**Date:** 2026-11-19  
**Size:** Small feature  
**Status:** Draft (awaiting approval)

## Intent & Why
SpecRAG currently exposes a CLI (`specrag ingest`, `specrag query`, `specrag benchmark`) and a Google Colab benchmark notebook. Adding a lightweight local web UI lets anyone who clones the repository upload their own technical documents and chat with them using SpecRAG's structure-aware hybrid pipeline, with grounded answers and verbatim citations.

## Goals
- Let users upload one or more local documents (PDF, Markdown, DOCX, TXT, HTML) and index them into the SpecRAG hybrid pipeline.
- Provide a clean chat interface where every assistant reply displays:
  - The synthesized answer from `RAGGenerator` (`SpecAnswer.answer`).
  - Grounding metadata: confidence score, answerability indicator, and LLM provider used.
  - Collapsible citation panel showing `cited_clause_ids`, `verbatim_evidence`, and the retrieved source chunks (preserving Markdown tables and breadcrumbs).
- Keep the UI optional (`uv pip install -e ".[ui]"`) so core CLI and benchmark dependencies stay minimal.
- Launch easily with a single command (`specrag ui` or `streamlit run src/specrag/ui/app.py`).

## Non-Goals
- Multi-user authentication, persistent database hosting, or public SaaS deployment.
- Side-by-side Naive vs. Hybrid benchmark comparison inside the chat app (that stays in the CLI and Colab notebook).
- Multi-turn conversational query rewriting (each question runs directly through SpecRAG's retrieval pipeline; chat history is displayed in the UI for readability).

## Anti-Goals
- Re-loading heavy embedding (`bge-small-en-v1.5`) or cross-encoder (`bge-reranker-base`) models on every Streamlit rerun.
- Leaving unclosed local `QdrantClient` instances that trigger Python teardown errors.
- Failing silently when no LLM API key or local Ollama instance is configured — if generation fails, still surface the retrieved evidence chunks with a clear setup hint.

## Constraints
- Must reuse existing `specrag` components (`DocumentConverter`, `StructureAwareMarkdownChunker`, `QdrantDenseRetriever`, `BM25SparseRetriever`, `HybridRRFRetriever`, `CrossEncoderReranker`, `RAGGenerator`, `LLMRouter`) without duplicating retrieval logic.
- Must run smoothly on CPU laptops as well as GPU machines: Cross-Encoder reranking takes ~0.5s on GPU but ~15–20s on CPU, so the sidebar should include a simple "Enable Cross-Encoder Reranking" toggle (enabled by default, with a brief note if running on CPU).
- Must stay 100% Python within `specrag`.

## Approaches Considered
1. **Streamlit (Selected):** Single Python UI module + optional `.[ui]` extra + `specrag ui` CLI entry point. Native chat widgets, multi-file upload, collapsible expanders, Markdown table rendering, and `@st.cache_resource` for model caching. *Trade-off:* Adds `streamlit` to an optional extra; layout uses standard Streamlit primitives.
2. **Gradio:** Single Python script with easy Hugging Face Spaces portability. *Trade-off:* Custom per-message citation cards and Markdown table expanders are clunkier to style and maintain.
3. **FastAPI + Single-File HTML/JS:** Minimal Python dependencies and full layout control. *Trade-off:* Introduces HTML/CSS/JS frontend maintenance into an otherwise pure-Python repo.

**Why Streamlit:** Keeps the codebase 100% Python, renders Markdown tables and collapsible citations cleanly out of the box, and stays completely optional for non-UI users.

## Key Design Decisions & Defaults (Proposed Calls)
- **Module layout:** Extract a small, UI-agnostic `SpecRAGSession` (or `SpecRAGPipeline`) helper in `src/specrag/pipeline.py` that owns document ingestion, indexing, hybrid retrieval, optional reranking, and generation — making it cleanly unit-testable without Streamlit. Put the thin Streamlit UI in `src/specrag/ui/app.py`.
- **CLI integration:** Add `specrag ui` command in `src/specrag/cli.py` that checks whether `streamlit` is installed (printing an install hint if not) and invokes `streamlit run`.
- **Sidebar controls (minimal):**
  - File uploader (PDF, MD, TXT, DOCX, HTML) + "Index Documents" button + indexed file/chunk count summary + "Clear Index" button.
  - "Use Cross-Encoder Reranker" checkbox (default: on) + Top-K slider (default: 5).
  - Active LLM provider status badge (detected from `.env` / Ollama).
- **Graceful fallback without LLM keys:** If no cloud API key is set and Ollama isn't reachable, still display the Top-K retrieved chunks/tables with a clear message explaining how to configure `.env` or Ollama.

## Left to the Builder
- Exact Streamlit visual styling, spacing, and helper text.
- Temp file handling for uploaded bytes passed into `DocumentConverter.convert_file`.
- Unit tests for `SpecRAGPipeline` and `specrag ui --help`.
- README Quickstart section update showing how to install `.[ui]` and launch `specrag ui`.
