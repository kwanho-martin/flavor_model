from typing import TypeDict, Dict, Any


class TaskConfig(TypeDict):
    system_key: str
    user_key: str
    config: Dict[str, Any]

SYSTEM_PROMPTS = {
    "flavor_scientist": "You are a flavor scientist.",
    "sommelier_expert": "You are an expert sommelier with deep knowledge of wine flavors.",
}

USER_PROMPTS = {
    "extract_compounds": (
        "Extract key flavor compounds from the following text and return them in JSON format.\n"
        "Text: {text}"
    ),
}

TASKS: Dict[str, TaskConfig] = {
    "flavor_compound_extraction": {
        "system_key": "flavor_scientist",
        "user_key": "extract_compounds",
        "config": {
            "temperature": 0.3,
            "max_tokens": 500,
        },
    }
}