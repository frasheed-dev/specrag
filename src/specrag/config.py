from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for SpecRAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys for LLM Cascade
    groq_api_key: str | None = None
    nvidia_api_key: str | None = None
    openrouter_api_key: str | None = None
    github_token: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"

    # Default Provider selection
    specrag_llm_provider: Literal[
        "auto", "groq", "nvidia", "openrouter", "github", "gemini", "ollama"
    ] = "auto"

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    qdrant_storage_path: Path = project_root / "qdrant_db"

    # Retrieval Models
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    reranker_model_name: str = "BAAI/bge-reranker-base"

    # Hugging Face Authentication & Download Cache
    hf_token: str | None = None

    # Supported Benchmark Corpora
    corpora: tuple[str, ...] = ("t2_ragbench", "ibm_techreport")


settings = Settings()
