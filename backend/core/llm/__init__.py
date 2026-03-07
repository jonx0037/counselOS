from core.config import settings
from core.llm.base import BaseLLMProvider


def get_llm_provider() -> BaseLLMProvider:
    """
    Factory: returns the configured LLM provider.
    Add new providers here — agent code never changes.
    """
    provider = settings.llm_provider.lower()

    if provider == "anthropic":
        from core.llm.anthropic import AnthropicProvider
        return AnthropicProvider()

    # Example: add OpenAI support by uncommenting and implementing:
    # if provider == "openai":
    #     from core.llm.openai import OpenAIProvider
    #     return OpenAIProvider()

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Set LLM_PROVIDER in .env (e.g. 'anthropic')."
    )
