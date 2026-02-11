from abc import ABC, abstractmethod
from typing import Dict, Any

from src.llm import prompt_templates as prompt_temp


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """각 API별 실제 호출부 (자식 클래스에서 구현)"""
        pass

    def get_embedding(self, text: str, model: str = None) -> list[float]:
        """각 API별 임베딩 호출부 (자식 클래스에서 구현)"""
        pass

    def run_task(self, task_name: str, params: dict, **override_kwargs) -> str:
        """템플릿과 설정을 결합하여 실행하는 공통 메서드"""
        
        task = prompt_temp.TASKS.get(task_name)
        if not task:
            raise ValueError(f"정의되지 않은 태스크: {task_name}")

        system_content = prompt_temp.SYSTEM_PROMPTS.get(task["system_key"])
        user_template = prompt_temp.USER_PROMPTS.get(task["user_key"])
        
        # 1. 태스크 기본 설정 로드
        llm_config = task["config"].copy()
        # 2. 런타임에서 전달된 값으로 덮어쓰기 (예: 특정 호출에서만 온도를 낮추고 싶을 때)
        llm_config.update(override_kwargs)

        # 3. 구현된 자식 클래스의 generate 호출
        return self.generate(
            prompt=user_template.format(**params),
            system_prompt=system_content,
            **llm_config
        )
