import google.generativeai as genai
from src.llm.base import LLMProvider
import os

class GeminiClient(LLMProvider):
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-pro"):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.default_config = {
            "temperature": 0.7,
            "max_output_tokens": 1000,
        }

    def generate(self, prompt: str, system_prompt: str = "You are a flavor expert.", **kwargs) -> str:
        model = genai.GenerativeModel(
            model_name=kwargs.get("model", self.model_name),
            system_instruction=system_prompt
        )
        
        config = self.default_config.copy()
        config.update({
            "temperature": kwargs.get("temperature", 0.7),
            "max_output_tokens": kwargs.get("max_tokens", 1000)
        })

        response = model.generate_content(prompt, generation_config=config)
        return response.text

    def get_embedding(self, text: str, model: str = "models/text-embedding-004") -> list[float]:
        result = genai.embed_content(model=model, content=text)
        return result['embedding']