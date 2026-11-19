"""SpecRAG Local Document Chat UI (Streamlit).

Launch via:
    specrag ui
or:
    streamlit run src/specrag/ui/app.py
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

from specrag.config import settings
from specrag.llm_factory import PROVIDER_ENDPOINTS, create_llm_router, get_available_providers
from specrag.pipeline import QueryResult, SpecRAGPipeline


@st.cache_resource(show_spinner="Loading dense embedding model...")
def load_embedding_model(model_name: str) -> SentenceTransformer:
    token = settings.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return SentenceTransformer(model_name, token=token)


@st.cache_resource(show_spinner="Loading cross-encoder reranker...")
def load_reranker_model(model_name: str) -> CrossEncoder:
    token = settings.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    try:
        return CrossEncoder(model_name, token=token)
    except TypeError:
        return CrossEncoder(model_name)


def get_pipeline(use_reranker: bool) -> SpecRAGPipeline:
    """Return the session-scoped SpecRAGPipeline, reusing cached model weights."""
    if "pipeline" not in st.session_state:
        embedding_model = load_embedding_model(settings.embedding_model_name)
        reranker_model = (
            load_reranker_model(settings.reranker_model_name) if use_reranker else None
        )
        st.session_state["pipeline"] = SpecRAGPipeline(
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
    pipeline: SpecRAGPipeline = st.session_state["pipeline"]
    if use_reranker and pipeline.reranker._reranker is None:
        pipeline.reranker._reranker = load_reranker_model(settings.reranker_model_name)
    return pipeline


def render_assistant_result(result: QueryResult) -> None:
    """Render an assistant answer along with grounding metadata and collapsible citations."""
    if result.generation_error:
        st.warning(result.answer.answer)
        st.caption(f"LLM error detail: {result.generation_error}")
    elif not result.answer.is_answerable:
        st.info(result.answer.answer)
    else:
        st.markdown(result.answer.answer)

    st.caption(
        f"Confidence: **{result.answer.confidence:.2f}** · "
        f"Provider: `{result.provider}` · "
        f"Latency: **{result.latency_ms:.0f} ms**"
    )

    if result.retrieved_chunks or result.answer.verbatim_evidence:
        label = f"Citations & Retrieved Evidence ({len(result.retrieved_chunks)} chunks)"
        with st.expander(label, expanded=False):
            if result.answer.cited_clause_ids:
                clauses = ", ".join(f"`{c}`" for c in result.answer.cited_clause_ids)
                st.markdown(f"**Cited Sections / Tables:** {clauses}")

            if result.answer.verbatim_evidence:
                st.markdown("**Verbatim Evidence:**")
                for quote in result.answer.verbatim_evidence:
                    st.markdown(f"> {quote}")

            if result.retrieved_chunks:
                st.markdown("**Retrieved Source Chunks:**")
                for idx, chunk in enumerate(result.retrieved_chunks, start=1):
                    kind = "TABLE" if chunk.is_table else "PROSE"
                    breadcrumb = f" · {chunk.breadcrumb_prefix}" if chunk.breadcrumb_prefix else ""
                    st.markdown(f"**#{idx} [{kind}]** `{chunk.doc_id}`{breadcrumb}")
                    st.markdown(chunk.text)
                    if idx < len(result.retrieved_chunks):
                        st.divider()


def main() -> None:
    st.set_page_config(
        page_title="SpecRAG — Document Chat",
        page_icon="📄",
        layout="wide",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload technical specifications",
            type=["pdf", "md", "txt", "docx", "html"],
            accept_multiple_files=True,
            help="PDFs are parsed with PyMuPDF4LLM + MarkItDown to preserve tables.",
        )

        st.divider()
        st.header("Retrieval Settings")
        use_reranker = st.checkbox(
            "Use Cross-Encoder Reranker",
            value=True,
            help="Improves Top-1 table precision (~0.5s on GPU, ~15-20s on CPU).",
        )
        top_k = st.slider("Top-K Context Chunks", min_value=1, max_value=10, value=5)

        pipeline = get_pipeline(use_reranker=use_reranker)

        col1, col2 = st.columns(2)
        with col1:
            index_clicked = st.button(
                "Index Files",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_files,
            )
        with col2:
            clear_clicked = st.button("Clear All", use_container_width=True)

        if index_clicked and uploaded_files:
            errors: list[str] = []
            with st.spinner("Parsing and indexing documents..."):
                for uf in uploaded_files:
                    try:
                        pipeline.ingest_bytes(uf.name, uf.getvalue())
                    except Exception as exc:
                        errors.append(f"{uf.name}: {exc}")
            if errors:
                for err in errors:
                    st.error(err)
            else:
                st.success(
                    f"Indexed {len(pipeline.document_names)} file(s) "
                    f"({pipeline.chunk_count} chunks)."
                )

        if clear_clicked:
            pipeline.clear()
            st.session_state["messages"] = []
            st.rerun()

        if pipeline.document_names:
            st.caption(
                f"**Active Index:** {len(pipeline.document_names)} file(s), "
                f"{pipeline.chunk_count} chunks"
            )
            for name in pipeline.document_names:
                st.markdown(f"- `{name}`")
        else:
            st.caption("No documents indexed yet.")

        st.divider()
        st.header("LLM Settings")
        provider_options = ["groq", "auto", "ollama", "gemini", "nvidia", "openrouter", "github"]
        selected_provider = st.selectbox(
            "Provider",
            options=provider_options,
            index=0,
            help=(
                "Recommended: 'groq' with a free API key from console.groq.com/keys. "
                "Use 'auto' for .env cascade or 'ollama' for local models."
            ),
        )

        custom_model: str | None = None
        custom_api_key: str | None = None
        if selected_provider != "auto":
            default_model = PROVIDER_ENDPOINTS[selected_provider][1]
            custom_model = st.text_input(
                "Model",
                value=default_model,
                help=(
                    "For Groq: e.g. `openai/gpt-oss-120b`. "
                    "For Ollama: any local tag from `ollama list` (e.g. `llama3.2`)."
                ),
            )
            if selected_provider != "ollama":
                custom_api_key = st.text_input(
                    f"{selected_provider.upper()} API Key (optional if in .env)",
                    type="password",
                    placeholder="Paste API key and press Enter...",
                )
                if selected_provider == "groq" and not custom_api_key:
                    st.caption("Free Groq API key: [console.groq.com/keys](https://console.groq.com/keys)")

        providers = get_available_providers()
        provider_names = ", ".join(p.name for p in providers) if providers else "none"
        st.caption(f"**Configured in .env:** `{provider_names}`")

    st.title("SpecRAG Document Chat")
    st.caption(
        "Structure-aware hybrid retrieval (BM25 + Qdrant RRF + Cross-Encoder) "
        "with verbatim citations."
    )

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant_result(msg["result"])

    prompt = st.chat_input("Ask a question about your uploaded specifications...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        router = create_llm_router(
            provider=selected_provider,
            api_key=custom_api_key,
            model=custom_model,
        )
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and synthesizing grounded answer..."):
                result: QueryResult = pipeline.query(
                    prompt,
                    top_k=top_k,
                    use_reranker=use_reranker,
                    router=router,
                )
            render_assistant_result(result)

        assistant_entry: dict[str, Any] = {"role": "assistant", "result": result}
        st.session_state["messages"].append(assistant_entry)


if __name__ == "__main__":
    main()
