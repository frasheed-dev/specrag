from unittest.mock import MagicMock

from specrag.retriever import (
    BM25SparseRetriever,
    HybridRRFRetriever,
    tokenize_technical_text,
)
from specrag.schemas import DocumentChunk


def test_tokenize_technical_text():
    text = "NVIDIA H100 NVL P1010 SKU 210 with § 4 ArbZG and Device ID 0x2321"
    tokens = tokenize_technical_text(text)

    assert "p1010" in tokens
    assert "sku" in tokens
    assert "210" in tokens
    assert "§" in tokens
    assert "0x2321" in tokens


def test_bm25_exact_identifier_retrieval():
    chunk1 = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        text="The general compute architecture supports standard memory buffers.",
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        doc_id="d2",
        text="Product SKU: P1010 SKU 210 NVPN: 699-21010-0210-xxx Device ID 0x2321",
    )
    chunk3 = DocumentChunk(
        chunk_id="c3",
        doc_id="d3",
        text="Additional auxiliary power connection instructions and thermal envelopes.",
    )

    bm25 = BM25SparseRetriever()
    bm25.index_chunks([chunk1, chunk2, chunk3])

    hits = bm25.retrieve("P1010 SKU 210", top_k=1)
    assert len(hits) == 1
    assert hits[0][0].chunk_id == "c2"


def test_rrf_scoring_math():
    mock_dense = MagicMock()
    mock_sparse = MagicMock()

    chunk_a = DocumentChunk(chunk_id="A", doc_id="d1", text="Chunk A")
    chunk_b = DocumentChunk(chunk_id="B", doc_id="d2", text="Chunk B")

    # Dense: A at rank 1, B at rank 2
    mock_dense.retrieve.return_value = [(chunk_a, 0.9), (chunk_b, 0.8)]
    # Sparse: B at rank 1, A at rank 2
    mock_sparse.retrieve.return_value = [(chunk_b, 15.0), (chunk_a, 5.0)]

    hybrid = HybridRRFRetriever(dense_retriever=mock_dense, sparse_retriever=mock_sparse, k=60)
    results = hybrid.retrieve("query", top_k=2)

    assert len(results) == 2
    # Both A and B appeared at rank 1 and rank 2 across the two modalities:
    # Expected score for each = 1/(60+1) + 1/(60+2)
    expected_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert abs(results[0][1] - expected_score) < 1e-6
    assert abs(results[1][1] - expected_score) < 1e-6
