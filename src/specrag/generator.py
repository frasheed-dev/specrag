"""Structured generator for SpecRAG enforcing Pydantic schemas and strict citations."""

import json
import logging

from pydantic import BaseModel, Field

from specrag.llm_factory import LLMRouter, get_llm_router
from specrag.schemas import DocumentChunk

logger = logging.getLogger("specrag.generator")


class SpecAnswer(BaseModel):
    """Structured response schema enforcing exact clause citations and grounded evidence."""

    answer: str = Field(
        description="Direct, factual answer to the query based exclusively on context."
    )
    cited_clause_ids: list[str] = Field(
        default_factory=list,
        description="List of specific clause headers, section numbers, or table names cited (e.g. 'Table 1', '§ 4 ArbZG').",
    )
    verbatim_evidence: list[str] = Field(
        default_factory=list,
        description="Exact verbatim sentences or table rows copied directly from context that prove the answer.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 (uncertain) to 1.0 (fully proven).",
    )
    is_answerable: bool = Field(
        default=True,
        description="False if the provided context does not contain sufficient facts to answer.",
    )


class RAGGenerator:
    """Generates structured, citation-grounded answers using the Zero-Cost Multi-Provider LLM Router."""

    def __init__(self, router: LLMRouter | None = None):
        self.router = router or get_llm_router()

    def generate_answer(self, query: str, chunks: list[DocumentChunk]) -> tuple[SpecAnswer, str]:
        """Generate a structured SpecAnswer from retrieved chunks.

        Returns:
            tuple[SpecAnswer, provider_name]
        """
        if not chunks:
            return (
                SpecAnswer(
                    answer="No context was retrieved to answer this query.",
                    confidence=0.0,
                    is_answerable=False,
                ),
                "abstain",
            )

        context_str = "\n\n---\n\n".join(
            [f"[Chunk {i + 1} | {c.breadcrumb_prefix}]\n{c.text}" for i, c in enumerate(chunks)]
        )

        system_prompt = (
            "You are a Principal Hardware & Legal Compliance AI Architect. "
            "Your task is to answer the user query STRICTLY based on the provided technical context.\n"
            "Rules:\n"
            "1. Ground every statement in verbatim quotes or exact table numbers from the context.\n"
            "2. Never extrapolate or hallucinate specifications.\n"
            "3. If the context does not contain the answer, set is_answerable to false.\n"
            "4. Return your output strictly as a valid JSON object matching this schema:\n"
            "{\n"
            '  "answer": "string",\n'
            '  "cited_clause_ids": ["string"],\n'
            '  "verbatim_evidence": ["string"],\n'
            '  "confidence": 1.0,\n'
            '  "is_answerable": true\n'
            "}"
        )

        user_prompt = (
            f"Technical Specifications Context:\n{context_str}\n\nQuery: {query}\n\nJSON Output:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content, provider = self.router.chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=800,
            )

            # Extract JSON block
            json_str = content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            data = json.loads(json_str)
            return SpecAnswer(**data), provider
        except Exception as e:
            logger.warning(f"Structured JSON parsing failed ({e}). Returning fallback SpecAnswer.")
            return (
                SpecAnswer(
                    answer=content if "content" in locals() else f"Generation error: {e}",
                    confidence=0.5,
                    is_answerable=True,
                ),
                provider if "provider" in locals() else "fallback",
            )
