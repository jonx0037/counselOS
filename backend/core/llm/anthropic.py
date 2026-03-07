import anthropic

from core.config import settings
from core.llm.base import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude implementation of the LLM provider interface."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        message = self._client.messages.create(**kwargs)

        return LLMResponse(
            content=message.content[0].text,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )

    def model_name(self) -> str:
        return self._model
