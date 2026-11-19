from unittest.mock import MagicMock, patch

from specrag.llm_factory import (
    LLMRouter,
    ProviderConfig,
    create_llm_router,
    get_available_providers,
)


def test_available_providers_contains_ollama_fallback():
    providers = get_available_providers()
    assert len(providers) >= 1
    # Ollama is always present as fallback
    assert any(p.name == "ollama" for p in providers)


def test_llm_router_provider_ordering():
    router = LLMRouter(preferred_provider="ollama")
    order = list(router._get_provider_order())
    assert order[0].name == "ollama"


def test_llm_router_fallback_mechanism():
    router = LLMRouter()
    # Mock two providers: first fails, second succeeds
    prov1 = ProviderConfig(
        name="test_fail", base_url="https://fail.api", api_key="k1", default_model="m1"
    )
    prov2 = ProviderConfig(
        name="test_success", base_url="https://ok.api", api_key="k2", default_model="m2"
    )

    router.providers = [prov1, prov2]
    router.preferred_provider = "auto"

    with patch("specrag.llm_factory.OpenAI") as mock_openai:
        mock_client_fail = MagicMock()
        mock_client_fail.chat.completions.create.side_effect = Exception("Rate limit 429")

        mock_client_success = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Success response from provider 2"
        mock_client_success.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        # Return mock_client_fail on first instantiation, mock_client_success on second
        mock_openai.side_effect = [mock_client_fail, mock_client_success]

        content, provider = router.chat_completion(messages=[{"role": "user", "content": "Hello"}])

        assert provider == "test_success"
        assert content == "Success response from provider 2"


def test_create_llm_router_gemini_override():
    router = create_llm_router(
        provider="gemini",
        api_key="test-gemini-key",
        model="gemini-2.5-flash",
    )
    order = list(router._get_provider_order())
    assert order[0].name == "gemini"
    assert order[0].api_key == "test-gemini-key"
    assert order[0].default_model == "gemini-2.5-flash"


def test_create_llm_router_ollama_custom_model():
    router = create_llm_router(provider="ollama", model="glm-5:cloud")
    order = list(router._get_provider_order())
    assert order[0].name == "ollama"
    assert order[0].default_model == "glm-5:cloud"
