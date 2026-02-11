import anthropic
from src.llm.base import LLMProvider
import os

class AnthropicClient(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20240620"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.default_model = model

    def generate(self, prompt: str, system_prompt: str = "You are a flavor scientist.", **kwargs) -> str:
        response = self.client.messages.create(
            model=kwargs.get("model", self.default_model),
            max_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.7),
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def get_embedding(self, text: str) -> list[float]:
        # 현재 Anthropic은 공식 Embedding API를 제공하지 않으므로 구현과정중에 에러를 보기위해서 NotImplementedError를 발생시킵니다.
        raise NotImplementedError("Anthropic does not provide an embedding API yet.")
