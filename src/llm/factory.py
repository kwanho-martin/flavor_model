from src.llm.openai_client import OpenAIClient
from src.llm.gemini_client import GeminiClient
from src.llm.anthropic_client import AnthropicClient

class LLMFactory:
    _providers = {
        "openai": OpenAIClient,
        "gemini": GeminiClient,
        "anthropic": AnthropicClient
    }

    @classmethod
    def create(cls, provider_name: str, **kwargs):
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider_class(**kwargs)
