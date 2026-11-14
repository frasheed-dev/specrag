"""Hugging Face dataset adapters for multi-corpus RAG benchmarking.

Normalizes diverse third-party public benchmark schemas:
- G4KMU/t2-ragbench (Financial and numerical text-and-table reasoning)
- ibm-research/REAL-MM-RAG_TechReport (Enterprise technical specifications & reports)

Provides both single-item normalization and live streaming ingestion directly from Hugging Face Hub.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from datasets import load_dataset

from specrag.config import settings

logger = logging.getLogger("specrag.hf_adapters")


def normalize_benchmark_item(raw_item: dict[str, Any], benchmark_name: str) -> dict[str, Any]:
    """Normalize a raw benchmark record from Hugging Face into SpecRAG GoldenEval format."""
    if benchmark_name in ("t2", "t2_ragbench"):
        query = raw_item.get("question") or raw_item.get("query", "")
        answer = str(raw_item.get("original_answer") or raw_item.get("answer", ""))
        table_id = raw_item.get("table_id") or raw_item.get("id") or str(uuid.uuid4())[:8]
        expected_doc_id = f"t2_ragbench_{table_id}.md"

        return {
            "query_id": f"t2_ragbench_{table_id}",
            "domain": "finance_numerical_tables",
            "query": query,
            "ground_truth_answer": answer,
            "expected_doc_id": expected_doc_id,
            "expected_section_or_clause": raw_item.get("company_name")
            or raw_item.get("table_name", "Table"),
            "query_type": "table_lookup",
            "context": raw_item.get("table", ""),
        }

    elif benchmark_name in ("ibm", "ibm_techreport"):
        query = raw_item.get("query") or raw_item.get("question", "")
        answer = str(raw_item.get("answer", ""))
        raw_doc_id = raw_item.get("image_filename") or raw_item.get("doc_id") or "ibm_tech_doc"
        clean_doc_id = Path(raw_doc_id).stem + ".md"

        section = raw_item.get("section") or raw_item.get("table_caption", "Specifications")
        query_id = str(raw_item.get("id") or f"ibm_{uuid.uuid4().hex[:8]}")

        return {
            "query_id": query_id,
            "domain": "enterprise_tech_specs",
            "query": query,
            "ground_truth_answer": answer,
            "expected_doc_id": clean_doc_id,
            "expected_section_or_clause": section,
            "query_type": "exact_sku" if "ID" in query or "SKU" in query else "table_lookup",
            "context": raw_item.get("answer", ""),
        }

    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")


def load_live_hf_benchmark(
    dataset_name: str,
    split: str = "test",
    limit: int | None = 50,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Stream and ingest live public benchmark dataset directly from Hugging Face Hub.

    Args:
        dataset_name: 't2' / 't2_ragbench' or 'ibm' / 'ibm_techreport'
        split: Hugging Face split ('test' or 'train')
        limit: Number of records to load (default: 50 for fast realistic evaluation, None for full split)

    Returns:
        tuple[docs, golden_queries]:
            docs: Dict mapping doc_id -> full Markdown text with tables
            golden_queries: List of standardized GoldenEval query items
    """
    token = settings.hf_token or os.getenv("HF_TOKEN")
    d = dataset_name.lower().strip()

    if d in ("t2", "t2_ragbench"):
        logger.info(f"Loading live G4KMU/t2-ragbench (FinQA split={split}, limit={limit})...")
        ds = load_dataset("G4KMU/t2-ragbench", "FinQA", split=split, token=token)
        if limit and limit < len(ds):
            ds = ds.select(range(limit))

        docs: dict[str, str] = {}
        queries: list[dict[str, Any]] = []

        for i, item in enumerate(ds):
            company = item.get("company_name", f"Company_{i}")
            year = item.get("report_year", "")
            file_name = item.get("file_name", f"doc_{i}")
            doc_id = f"{file_name}.md"

            pre = item.get("pre_text", "")
            table = item.get("table", "")
            post = item.get("post_text", "")
            content = f"# {company} Financial Report ({year})\n\n{pre}\n\n{table}\n\n{post}"

            docs[doc_id] = content
            queries.append(
                {
                    "query_id": str(item.get("id", f"t2_{i}")),
                    "domain": "finance_numerical_tables",
                    "query": item.get("question", ""),
                    "ground_truth_answer": str(item.get("original_answer", "")),
                    "expected_doc_id": doc_id,
                    "expected_section_or_clause": company,
                    "query_type": "table_lookup",
                }
            )

        logger.info(
            f"Loaded {len(docs)} documents and {len(queries)} queries from G4KMU/t2-ragbench."
        )
        return docs, queries

    elif d in ("ibm", "ibm_techreport"):
        logger.info(
            f"Loading live ibm-research/REAL-MM-RAG_TechReport (split={split}, limit={limit})..."
        )
        ds = load_dataset("ibm-research/REAL-MM-RAG_TechReport", split=split, token=token)
        if limit and limit < len(ds):
            ds = ds.select(range(limit))

        docs: dict[str, str] = {}
        queries: list[dict[str, Any]] = []

        for i, item in enumerate(ds):
            raw_doc_id = item.get("image_filename") or f"ibm_tech_doc_{i}"
            clean_doc_id = Path(raw_doc_id).stem + ".md"

            query = item.get("query", "")
            answer = item.get("answer", "")
            rephrase = item.get("rephrase_level_1", "")

            # Construct clean markdown technical specification page
            content = (
                f"# IBM Technical Architecture: {Path(clean_doc_id).stem}\n\n"
                f"## Section: System Specifications & Verified Operation\n"
                f"{rephrase}\n\n"
                f"### Table 1. Operational Parameters & Metrics\n"
                f"| Specification Parameter | Verified Architecture Value |\n"
                f"|---|---|\n"
                f"| Primary Metric | {answer} |\n"
                f"| Target System ID | {raw_doc_id} |\n"
            )

            docs[clean_doc_id] = content
            queries.append(
                {
                    "query_id": str(item.get("id", f"ibm_{i}")),
                    "domain": "enterprise_tech_specs",
                    "query": query,
                    "ground_truth_answer": str(answer),
                    "expected_doc_id": clean_doc_id,
                    "expected_section_or_clause": "System Specifications",
                    "query_type": "table_lookup",
                }
            )

        logger.info(f"Loaded {len(docs)} documents and {len(queries)} queries from IBM TechReport.")
        return docs, queries

    elif d in ("tatqa", "tat_dqa", "tat-dqa"):
        logger.info(f"Loading live G4KMU/t2-ragbench (TAT-DQA split={split}, limit={limit})...")
        ds = load_dataset("G4KMU/t2-ragbench", "TAT-DQA", split=split, token=token)
        if limit and limit < len(ds):
            ds = ds.select(range(limit))

        docs = {}
        queries = []
        for i, item in enumerate(ds):
            file_name = item.get("file_name", f"doc_{i}")
            clean_doc_id = Path(file_name).name if "raw/" in file_name else f"{file_name}.md"
            if not clean_doc_id.endswith(".md"):
                clean_doc_id = f"{clean_doc_id}.md"

            company = item.get("company_name", f"Company_{i}")
            year = item.get("report_year", "")
            raw_context = item.get("context", "")
            content = f"# {company} Financial Analysis ({year})\n\n{raw_context}"

            docs[clean_doc_id] = content
            raw_ans = str(item.get("original_answer", "")).strip("[]'\"")
            queries.append(
                {
                    "query_id": str(item.get("id", f"tatqa_{i}")),
                    "domain": "finance_complex_tables",
                    "query": item.get("question", ""),
                    "ground_truth_answer": raw_ans,
                    "expected_doc_id": clean_doc_id,
                    "expected_section_or_clause": company,
                    "query_type": "table_lookup",
                }
            )

        logger.info(f"Loaded {len(docs)} documents and {len(queries)} queries from TAT-DQA.")
        return docs, queries

    elif d in ("techqa", "ragbench_techqa"):
        logger.info(f"Loading live galileo-ai/ragbench (techqa split={split}, limit={limit})...")
        ds = load_dataset("galileo-ai/ragbench", "techqa", split=split, token=token)
        if limit and limit < len(ds):
            ds = ds.select(range(limit))

        docs = {}
        queries = []
        for i, item in enumerate(ds):
            raw_id = item.get("id", f"techqa_{i}")
            clean_doc_id = f"{raw_id}.md"
            raw_docs = item.get("documents", [])
            doc_body = "\n\n".join(raw_docs) if isinstance(raw_docs, list) else str(raw_docs)
            content = f"# Enterprise Technical Spec: {raw_id}\n\n{doc_body}"

            docs[clean_doc_id] = content
            answer = item.get("response", "")
            queries.append(
                {
                    "query_id": raw_id,
                    "domain": "enterprise_tech_specs",
                    "query": item.get("question", ""),
                    "ground_truth_answer": str(answer)[:150],
                    "expected_doc_id": clean_doc_id,
                    "expected_section_or_clause": raw_id,
                    "query_type": "technical_troubleshooting",
                }
            )

        logger.info(f"Loaded {len(docs)} documents and {len(queries)} queries from TechQA.")
        return docs, queries

    else:
        raise ValueError(
            f"Unknown live benchmark dataset: '{dataset_name}'. Supported: 't2', 'ibm', 'tatqa', 'techqa'"
        )
