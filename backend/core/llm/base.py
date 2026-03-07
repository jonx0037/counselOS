"""
Abstract LLM provider interface.

All agents interact exclusively with this interface.
Swapping providers means implementing a new subclass and
updating LLM_PROVIDER in .env — no agent code changes required.
"""
from abc import ABC, abstractmethod


class LLMResponse:
    def __init__(self, content: str, model: str, input_tokens: int, output_tokens: int):
        self.content = content
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model!r}, "
            f"tokens={self.input_tokens}+{self.output_tokens})"
        )


class BaseLLMProvider(ABC):
    """Provider-agnostic LLM interface. All agents call this."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        """
        Send a prompt and return a structured response.

        Args:
            prompt:  The user-turn message.
            system:  Optional system prompt (persona, constraints, output schema).

        Returns:
            LLMResponse with content and token metadata.
        """
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...
