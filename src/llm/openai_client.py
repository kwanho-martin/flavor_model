import openai
import os
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.base import LLMProvider
from src.llm import prompt_templates as prompt_temp


class OpenAIClient(LLMProvider):
    def __init__(self, api_key, model_name='gpt-4-turbo'):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_name = model_name

    @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model_name),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperatur=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        return response.choices[0].message.content
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model)['data'][0]['embedding']
