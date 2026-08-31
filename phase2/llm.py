"""
LLM integration: calls Ollama with structured prompts.
Designed to be swappable — change provider in config to use Claude/GPT-4o.
"""

import ollama
from typing import Optional
from phase2.prompt import SYSTEM_PROMPT


class LLMClient:
    def __init__(self, config: dict):
        llm_config = config['llm']
        self.provider = llm_config['provider']
        self.model = llm_config['model']
        self.max_tokens = llm_config.get('max_tokens', 1000)
        self.temperature = llm_config.get('temperature', 0.1)
        self.base_url = llm_config.get('base_url', 'http://localhost:11434')

        print(f"LLM client initialized: {self.provider} / {self.model}")

    def call(self, prompt: str) -> str:
        """Send prompt to LLM and return response text."""
        if self.provider == "ollama":
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. "
                             f"Use 'ollama' for now.")

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )
            return response['message']['content']

        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}\n"
                               f"Make sure ollama serve is running.")