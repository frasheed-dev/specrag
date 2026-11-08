"""Hugging Face dataset adapters for multi-corpus RAG benchmarking.

Normalizes diverse third-party public benchmark schemas:
- G4KMU/t2-ragbench (Financial and numerical text-and-table reasoning)
- ibm-research/REAL-MM-RAG_TechReport (Enterprise technical specifications & reports)

Converts them into standard SpecRAG DocumentChunk and GoldenEvalItem representations.
"""

import uuid
from pathlib import Path
from typing import Any


def normalize_benchmark_item(raw_item: dict[str, Any], benchmark_name: str) -> dict[str, Any]:
    """Normalize a raw benchmark record from Hugging Face into SpecRAG GoldenEval format.

    Args:
        raw_item: Raw dictionary from Hugging Face dataset.
        benchmark_name: Name of the source benchmark ('t2_ragbench' or 'ibm_techreport').

    Returns:
        Standardized dictionary compatible with SpecRAG evaluator golden set.
    """
    if benchmark_name == "t2_ragbench":
        query = raw_item.get("question") or raw_item.get("query", "")
        answer = str(raw_item.get("answer", ""))
        table_id = raw_item.get("table_id") or raw_item.get("id") or str(uuid.uuid4())[:8]
        expected_doc_id = f"t2_ragbench_{table_id}.md"

        return {
            "query_id": f"t2_ragbench_{table_id}",
            "domain": "finance_numerical_tables",
            "query": query,
            "ground_truth_answer": answer,
            "expected_doc_id": expected_doc_id,
            "expected_section_or_clause": raw_item.get("table_name", "Table"),
            "query_type": "table_lookup",
            "context": raw_item.get("context", ""),
        }

    elif benchmark_name == "ibm_techreport":
        query = raw_item.get("query") or raw_item.get("question", "")
        answer = str(raw_item.get("answer", ""))
        raw_doc_id = raw_item.get("doc_id") or "ibm_tech_doc"
        clean_doc_id = Path(raw_doc_id).stem + ".md"

        section = raw_item.get("section") or raw_item.get("table_caption", "Specifications")
        query_id = raw_item.get("query_id") or f"ibm_{uuid.uuid4().hex[:8]}"

        return {
            "query_id": query_id,
            "domain": "enterprise_tech_specs",
            "query": query,
            "ground_truth_answer": answer,
            "expected_doc_id": clean_doc_id,
            "expected_section_or_clause": section,
            "query_type": "exact_sku" if "ID" in query or "SKU" in query else "table_lookup",
            "context": raw_item.get("positive_passage") or raw_item.get("context", ""),
        }

    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")
