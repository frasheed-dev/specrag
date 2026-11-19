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


def test_ingest_markdown_and_query_with_reranker():
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
    mock_dense.close.assert_not_called()
    pipeline.close()
    mock_dense.close.assert_called_once()
