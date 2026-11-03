"""Retrieval engines for SpecRAG.

Includes:
- BaseRetriever: Abstract interface for vector/sparse stores
- QdrantDenseRetriever: Local Qdrant dense vector index (BAAI/bge-small-en-v1.5)
- BM25SparseRetriever: Lexical index (BM25Okapi) for exact part numbers/clauses
- HybridRRFRetriever: Reciprocal Rank Fusion fusing Dense + BM25
- CrossEncoderReranker: Cross-Encoder (BAAI/bge-reranker-base) for Top-5 reranking
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from specrag.config import settings
from specrag.schemas import DocumentChunk

logger = logging.getLogger("specrag.retriever")


class BaseRetriever(ABC):
    """Abstract retrieval interface enabling future comparative benchmarking across Vector DBs."""

    @abstractmethod
    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Index a list of DocumentChunks."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Retrieve top_k chunks matching query with relevance score."""
        pass


class QdrantDenseRetriever(BaseRetriever):
    """Dense vector retriever powered by local Qdrant and BAAI/bge-small-en-v1.5."""

    def __init__(
        self,
        collection_name: str = "specrag_chunks",
        storage_path: str | Path | None = None,
        embedding_model_name: str | None = None,
    ):
        self.collection_name = collection_name
        self.storage_path = str(storage_path or settings.qdrant_storage_path)
        self.model_name = embedding_model_name or settings.embedding_model_name

        # Initialize Qdrant Client (local disk persistence)
        self.client = QdrantClient(path=self.storage_path)

        # Lazy loaded model
        self._model: SentenceTransformer | None = None
        self._chunks_by_id: dict[str, DocumentChunk] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading dense embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Create collection and insert document chunks with dense vector embeddings."""
        if not chunks:
            return

        vector_size = self.model.get_embedding_dimension()

        # Recreate collection cleanly
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        texts_to_embed = [c.contextualized_text for c in chunks]
        embeddings = self.model.encode(
            texts_to_embed, show_progress_bar=False, normalize_embeddings=True
        )

        points: list[PointStruct] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            self._chunks_by_id[chunk.chunk_id] = chunk
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "is_table": chunk.is_table,
                        "section_breadcrumbs": chunk.section_breadcrumbs,
                        "metadata": chunk.metadata,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(
            f"Indexed {len(points)} chunks into Qdrant collection '{self.collection_name}'."
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Dense cosine similarity search in Qdrant."""
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        results: list[tuple[DocumentChunk, float]] = []
        for hit in response.points:
            payload = hit.payload or {}
            chunk = DocumentChunk(
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                is_table=payload.get("is_table", False),
                section_breadcrumbs=payload.get("section_breadcrumbs", []),
                metadata=payload.get("metadata", {}),
            )
            results.append((chunk, float(hit.score)))

        return results


def tokenize_technical_text(text: str) -> list[str]:
    """Tokenize technical and legal text preserving section symbols, numbers, and register identifiers."""
    return re.findall(r"[\w§]+|[^\w\s]", text.lower())


class BM25SparseRetriever(BaseRetriever):
    """Lexical sparse search powered by BM25Okapi for exact SKU, error codes, and clauses."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[DocumentChunk] = []

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        tokenized_corpus = [tokenize_technical_text(c.contextualized_text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Indexed {len(chunks)} chunks in BM25 lexical index.")

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = tokenize_technical_text(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[idx], float(score)) for idx, score in scored_indices]


class HybridRRFRetriever(BaseRetriever):
    """Fuses Dense (Qdrant) and Sparse (BM25) search using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: QdrantDenseRetriever,
        sparse_retriever: BM25SparseRetriever,
        k: int = 60,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.k = k

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        self.dense.index_chunks(chunks)
        self.sparse.index_chunks(chunks)

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[DocumentChunk, float]]:
        """Execute dual retrieval and merge with RRF: score = sum(1 / (k + rank))."""
        dense_hits = self.dense.retrieve(query, top_k=top_k)
        sparse_hits = self.sparse.retrieve(query, top_k=top_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, DocumentChunk] = {}

        for rank, (chunk, _) in enumerate(dense_hits, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (self.k + rank)
            )

        for rank, (chunk, _) in enumerate(sparse_hits, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (self.k + rank)
            )

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(chunk_map[cid], score) for cid, score in sorted_chunks]


class CrossEncoderReranker:
    """Scores (query, document) candidate pairs using BAAI/bge-reranker-base."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model_name
        self._reranker: CrossEncoder | None = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            logger.info(f"Loading Cross-Encoder model: {self.model_name}")
            self._reranker = CrossEncoder(self.model_name)
        return self._reranker

    def rerank(
        self, query: str, candidates: list[DocumentChunk], top_k: int = 5
    ) -> list[tuple[DocumentChunk, float]]:
        """Rerank candidates and return the Top-k highest scoring chunks."""
        if not candidates:
            return []

        pairs = [[query, c.contextualized_text] for c in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(candidates, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
