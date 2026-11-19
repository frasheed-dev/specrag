import logging
from collections.abc import Generator
from typing import Any

from openai import APIConnectionError, OpenAI, RateLimitError
from pydantic import BaseModel

from specrag.config import settings

logger = logging.getLogger("specrag.llm")


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key: str
    default_model: str


def get_available_providers() -> list[ProviderConfig]:
    """Inspect environment configuration and return configured providers in cascade order."""
    providers: list[ProviderConfig] = []

    # 1. Groq (Fastest ~500 tok/s free tier)
    if settings.groq_api_key:
        providers.append(
            ProviderConfig(
                name="groq",
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.groq_api_key,
                default_model=settings.groq_model,
            )
        )

    # 2. NVIDIA NIM
    if settings.nvidia_api_key:
        providers.append(
            ProviderConfig(
                name="nvidia",
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=settings.nvidia_api_key,
                default_model="meta/llama-3.1-70b-instruct",
            )
        )

    # 3. OpenRouter Free Tier
    if settings.openrouter_api_key:
        providers.append(
            ProviderConfig(
                name="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
                default_model="meta-llama/llama-3.3-70b-instruct:free",
            )
        )

    # 4. GitHub Models
    if settings.github_token:
        providers.append(
            ProviderConfig(
                name="github",
                base_url="https://models.inference.ai.azure.com",
                api_key=settings.github_token,
                default_model="gpt-4o-mini",
            )
        )

    # 5. Gemini OpenAI-compatible endpoint
    if settings.gemini_api_key:
        providers.append(
            ProviderConfig(
                name="gemini",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=settings.gemini_api_key,
                default_model=settings.gemini_model,
            )
        )

    # 6. Local Ollama (supports both local models and Ollama cloud models like glm-5:cloud)
    providers.append(
        ProviderConfig(
            name="ollama",
            base_url=settings.ollama_base_url,
            api_key="ollama",
            default_model=settings.ollama_model,
        )
    )

    return providers


PROVIDER_ENDPOINTS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", settings.groq_model),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "meta/llama-3.1-70b-instruct"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free"),
    "github": ("https://models.inference.ai.azure.com", "gpt-4o-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", settings.gemini_model),
    "ollama": (settings.ollama_base_url, settings.ollama_model),
}


def create_llm_router(
    provider: str = "auto",
    api_key: str | None = None,
    model: str | None = None,
) -> "LLMRouter":
    """Create an LLMRouter with optional runtime provider, API key, and model overrides."""
    router = LLMRouter(preferred_provider=provider)
    if provider == "auto" or provider not in PROVIDER_ENDPOINTS:
        return router

    base_url, default_model = PROVIDER_ENDPOINTS[provider]
    existing = next((p for p in router.providers if p.name == provider), None)
    resolved_key = (
        api_key.strip()
        if api_key and api_key.strip()
        else (existing.api_key if existing else ("ollama" if provider == "ollama" else ""))
    )
    resolved_model = (
        model.strip()
        if model and model.strip()
        else (existing.default_model if existing else default_model)
    )

    if resolved_key:
        override = ProviderConfig(
            name=provider,
            base_url=existing.base_url if existing else base_url,
            api_key=resolved_key,
            default_model=resolved_model,
        )
        router.providers = [override]
    else:
        router.providers = []
    return router


class LLMRouter:
    """Multi-Provider LLM client cascade with automatic fallback on rate-limits (429)."""

    def __init__(self, preferred_provider: str | None = None):
        self.preferred_provider = preferred_provider or settings.specrag_llm_provider
        self.providers = get_available_providers()

    def _get_provider_order(self) -> Generator[ProviderConfig, None, None]:
        if self.preferred_provider != "auto":
            # Put requested provider first if configured
            chosen = [p for p in self.providers if p.name == self.preferred_provider]
            others = [p for p in self.providers if p.name != self.preferred_provider]
            ordered = chosen + others
        else:
            ordered = self.providers

        yield from ordered

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Execute chat completion with automatic failover across providers.

        Returns:
            tuple[content, provider_name]
        """
        ordered_providers = list(self._get_provider_order())
        if not ordered_providers:
            raise RuntimeError(
                f"No API key configured for provider '{self.preferred_provider}'. "
                "Enter an API key in the sidebar or set it in `.env`."
            )

        errors: list[str] = []

        for provider in ordered_providers:
            client = OpenAI(
                base_url=provider.base_url,
                api_key=provider.api_key,
                timeout=30.0,
            )
            target_model = model or provider.default_model

            try:
                logger.info(
                    f"Attempting inference with provider: {provider.name} (model: {target_model})"
                )
                response = client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                content = response.choices[0].message.content or ""
                return content, provider.name
            except (RateLimitError, APIConnectionError) as e:
                logger.warning(
                    f"Provider {provider.name} failed with {type(e).__name__}: {e}. Falling back..."
                )
                errors.append(f"{provider.name} ({target_model}): {e}")
                continue
            except Exception as e:
                # E.g. ModelNotFound or 401 Unauthorized
                logger.warning(f"Provider {provider.name} encountered error: {e}. Falling back...")
                errors.append(f"{provider.name} ({target_model}): {e}")
                continue

        raise RuntimeError("All configured LLM providers failed: " + " | ".join(errors))


def get_llm_router() -> LLMRouter:
    """Factory helper to obtain an initialized router."""
    return LLMRouter()
