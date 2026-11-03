from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents an ingested chunk of technical text or table data."""

    chunk_id: str
    doc_id: str
    text: str
    is_table: bool = False
    section_breadcrumbs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def breadcrumb_prefix(self) -> str:
        if not self.section_breadcrumbs:
            return f"[Doc: {self.doc_id}]"
        return f"[Doc: {self.doc_id} > {' > '.join(self.section_breadcrumbs)}]"

    @property
    def contextualized_text(self) -> str:
        """Text prefixed with hierarchical section breadcrumbs."""
        return f"{self.breadcrumb_prefix}\n{self.text}"


class Citation(BaseModel):
    """Specific technical citation linking answers to exact sections or tables."""

    doc_id: str
    section_breadcrumb: str
    quote_or_table_row: str


class RAGResponse(BaseModel):
    """Structured QA response from the generator."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_answerable: bool = True
    latency_ms: float = 0.0
    retrieved_chunks: list[DocumentChunk] = Field(default_factory=list)
    pipeline_mode: str = "hybrid_rerank"


class GoldenEvalItem(BaseModel):
    """Standardized schema for benchmark ground truth items."""

    query_id: str
    dataset: str
    query: str
    ground_truth_answer: str
    expected_doc_id: str
    expected_section_or_table: str
    query_type: Literal["table_lookup", "exact_sku", "multi_section", "general_tech"]


class EvalMetricResult(BaseModel):
    """Metric results for an individual evaluation query."""

    query_id: str
    dataset: str
    pipeline: str
    context_recall_at_5: float
    mrr: float
    table_hit: bool
    faithfulness: float | None = None
    answer_accuracy: float | None = None
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0


class BenchmarkSummary(BaseModel):
    """Aggregated benchmark statistics across a dataset."""

    dataset: str
    pipeline: str
    total_queries: int
    mean_recall_at_5: float
    mean_mrr: float
    table_hit_rate: float
    mean_faithfulness: float | None = None
    mean_accuracy: float | None = None
    mean_latency_ms: float = 0.0
