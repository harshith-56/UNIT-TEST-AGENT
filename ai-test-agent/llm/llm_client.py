from __future__ import annotations

from dataclasses import dataclass

import requests

from agent.config import AgentConfig
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str


class LLMClient:
    def __init__(self, config: AgentConfig) -> None:
        if not config.llm_api_url or not config.llm_api_key or not config.llm_model:
            raise RuntimeError("LLM_API_URL, LLM_API_KEY, and LLM_MODEL must be configured")
        self._config = config

    def generate(self, prompt: str) -> LLMResponse:
        trimmed_prompt = prompt[: self._config.llm_max_prompt_chars]
        payload = {
            "model": self._config.llm_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Generate only test code."},
                {"role": "user", "content": trimmed_prompt},
            ],
        }
        response = requests.post(
            self._config.llm_api_url,
            headers={
                "Authorization": f"Bearer {self._config.llm_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._config.llm_timeout_seconds,
        )
        if response.status_code >= 400:
            LOGGER.error("LLM URL:%s",self.api_url)
            LOGGER.error("llm_request_failed status=%s body=%s", response.status_code, response.text[:1000])
            raise RuntimeError(f"LLM request failed with status {response.status_code}")
        body = response.json()
        content = _extract_content(body)
        if not content.strip():
            raise RuntimeError("LLM returned empty content")
        return LLMResponse(content=content, model=self._config.llm_model)


def _extract_content(body: dict) -> str:
    choices = body.get("choices") or []
    if not choices:
        return body.get("output_text", "")
    message = choices[0].get("message", {})
    if isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(message.get("content"), list):
        return "".join(part.get("text", "") for part in message["content"] if isinstance(part, dict))
    return choices[0].get("text", "")
