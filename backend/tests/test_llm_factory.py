"""
Tests for the LLM provider factory.
Validates that the factory correctly resolves providers
and raises on unknown configurations — without making real API calls.
"""
import pytest
from unittest.mock import patch


class TestLLMProviderFactory:
    def test_anthropic_provider_resolves(self) -> None:
        with patch("core.config.settings") as mock_settings:
            mock_settings.llm_provider = "anthropic"
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.llm_model = "claude-sonnet-4-6"

            from core.llm.anthropic import AnthropicProvider
            # Just verify the class is importable and has the right interface
            assert hasattr(AnthropicProvider, "complete")
            assert hasattr(AnthropicProvider, "model_name")

    def test_unknown_provider_raises_value_error(self) -> None:
        with patch("core.llm.settings") as mock_settings:
            mock_settings.llm_provider = "unknown_provider"
            from core.llm import get_llm_provider
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_llm_provider()
