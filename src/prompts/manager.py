import os
from typing import Optional

class PromptManager:
    def __init__(self, prompts_dir: str = "src/prompts"):
        self.prompts_dir = prompts_dir

    def load_template(self, name: str) -> str:
        path = os.path.join(self.prompts_dir, f"{name}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt template {name} not found at {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def build_prompt(self, template_name: str, input_text: str, **kwargs) -> str:
        template = self.load_template(template_name)
        prompt = template.replace("{INPUT}", input_text)
        for k, v in kwargs.items():
            prompt = prompt.replace(f"{{{k}}}", str(v))
        return prompt
