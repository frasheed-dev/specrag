"""Interactive SpecRAG session pipeline for UI and programmatic chat workflows."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from specrag.chunkers import StructureAwareMarkdownChunker
from specrag.generator import RAGGenerator, SpecAnswer
from specrag.ingestion.converter import DocumentConverter
from specrag.retriever import (
    BM25SparseRetriever,
    CrossEncoderReranker,
    HybridRRFRetriever,
    QdrantDenseRetriever,
)
from specrag.schemas import DocumentChunk

logger = logging.getLogger("specrag.pipeline")


class QueryResult(BaseModel):
    """Structured result returned by SpecRAGPipeline.query()."""

    answer: SpecAnswer
    provider: str
    retrieved_chunks: list[DocumentChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    generation_error: str | None = None


class SpecRAGPipeline:
    """Stateful, UI-agnostic pipeline wrapping SpecRAG's structure-aware hybrid RAG stack."""

    def __init__(
        self,
        embedding_model: Any | None = None,
        reranker_model: Any | None = None,
        dense_retriever: QdrantDenseRetriever | None = None,
        sparse_retriever: BM25SparseRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        generator: RAGGenerator | None = None,
        converter: DocumentConverter | None = None,
        max_prose_chunk_size: int = 1000,
    ):
        self.chunker = StructureAwareMarkdownChunker(max_prose_chunk_size=max_prose_chunk_size)
        self.converter = converter or DocumentConverter()
        self.dense = dense_retriever or QdrantDenseRetriever(
            collection_name="specrag_session",
            storage_path=":memory:",
            embedding_model=embedding_model,
        )
        self.sparse = sparse_retriever or BM25SparseRetriever()
        self.hybrid = HybridRRFRetriever(dense_retriever=self.dense, sparse_retriever=self.sparse)
        self.reranker = reranker or CrossEncoderReranker(reranker_model=reranker_model)
        self.generator = generator or RAGGenerator()

        self._documents: dict[str, list[DocumentChunk]] = {}
        self._chunks: list[DocumentChunk] = []

    @property
    def document_names(self) -> list[str]:
        return list(self._documents.keys())

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _rebuild_indices(self) -> None:
        """Flatten all stored document chunks and index them in dense + sparse retrievers."""
        all_chunks: list[DocumentChunk] = []
        for doc_chunks in self._documents.values():
            all_chunks.extend(doc_chunks)
        self._chunks = all_chunks
        if all_chunks:
            self.hybrid.index_chunks(all_chunks)

    def ingest_markdown(self, doc_id: str, markdown_text: str) -> int:
        """Chunk markdown text and index it under doc_id. Returns chunks added."""
        if not markdown_text or not markdown_text.strip():
            raise ValueError(f"Document '{doc_id}' produced no extractable text.")

        new_chunks = self.chunker.chunk_document(markdown_text, doc_id=doc_id)
        if not new_chunks:
            raise ValueError(f"Document '{doc_id}' produced zero chunks.")

        self._documents[doc_id] = new_chunks
        self._rebuild_indices()
        return len(new_chunks)

    def ingest_bytes(self, filename: str, data: bytes) -> int:
        """Convert raw uploaded bytes (PDF, MD, DOCX, TXT, HTML) and index them."""
        if not data:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        suffix = Path(filename).suffix.lower() or ".txt"
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)

            if suffix in {".md", ".markdown", ".txt"}:
                md_text = tmp_path.read_text(encoding="utf-8", errors="replace")
            else:
                md_text = self.converter.convert_file(tmp_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return self.ingest_markdown(doc_id=filename, markdown_text=md_text)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        use_reranker: bool = True,
        router: Any | None = None,
    ) -> QueryResult:
        """Retrieve relevant chunks via Hybrid RRF (+ optional Cross-Encoder) and synthesize answer."""
        t0 = time.perf_counter()

        if not self._chunks:
            return QueryResult(
                answer=SpecAnswer(
                    answer="No documents are indexed yet. Upload and index at least one document first.",
                    confidence=0.0,
                    is_answerable=False,
                ),
                provider="none",
                retrieved_chunks=[],
                latency_ms=0.0,
            )

        candidate_pool_size = max(20, top_k * 3) if use_reranker else top_k
        candidates = self.hybrid.retrieve(query_text, top_k=candidate_pool_size)

        if use_reranker and candidates:
            reranked = self.reranker.rerank(
                query_text, [c for c, _ in candidates], top_k=top_k
            )
            retrieved_chunks = [c for c, _ in reranked]
        else:
            retrieved_chunks = [c for c, _ in candidates[:top_k]]

        if router is not None:
            self.generator.router = router

        generation_error: str | None = None
        try:
            answer, provider = self.generator.generate_answer(query_text, retrieved_chunks)
            if provider == "fallback" and answer.answer.startswith("Generation error:"):
                generation_error = answer.answer.removeprefix("Generation error:").strip()
                answer = SpecAnswer(
                    answer=(
                        "Retrieved the most relevant passages below, but LLM synthesis failed. "
                        "Select a provider (e.g. Gemini with an API key, or Ollama with a running "
                        "model such as `glm-5:cloud`) in the sidebar or configure `.env`."
                    ),
                    confidence=0.0,
                    is_answerable=False,
                )
                provider = "unavailable"
        except Exception as exc:
            logger.warning(f"LLM synthesis failed: {exc}")
            generation_error = str(exc)
            answer = SpecAnswer(
                answer=(
                    "Retrieved the most relevant passages below, but LLM synthesis failed. "
                    "Select a provider (e.g. Gemini with an API key, or Ollama with a running "
                    "model such as `glm-5:cloud`) in the sidebar or configure `.env`."
                ),
                confidence=0.0,
                is_answerable=False,
            )
            provider = "unavailable"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return QueryResult(
            answer=answer,
            provider=provider,
            retrieved_chunks=retrieved_chunks,
            latency_ms=latency_ms,
            generation_error=generation_error,
        )

    def clear(self) -> None:
        """Remove all indexed documents and reset sparse state while keeping the session alive."""
        self._documents.clear()
        self._chunks.clear()
        self.sparse.bm25 = None
        self.sparse.chunks = []

    def close(self) -> None:
        """Close underlying Qdrant client cleanly."""
        self.hybrid.close()
