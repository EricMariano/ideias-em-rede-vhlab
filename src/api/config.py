"""Configuração da API via variáveis de ambiente."""

import os
from dataclasses import dataclass
from functools import lru_cache


DEFAULT_SERVED_MODEL = "knowledge-lab-auditor"


@dataclass(frozen=True)
class Settings:
    """Parâmetros de runtime da API HTTP."""

    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_temperature: float
    llm_max_tokens: int
    llm_request_timeout: float
    agent_max_iterations: int
    api_host: str
    api_port: int
    api_bearer_token: str
    chroma_path: str | None
    served_model: str


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Lê as variáveis de ambiente uma vez e devolve Settings imutável."""
    chroma_path = os.getenv("CHROMA_PATH")
    if chroma_path == "":
        chroma_path = None

    return Settings(
        llm_base_url=_env("LLM_BASE_URL", "http://192.168.68.120:1234/v1"),
        llm_model=_env("LLM_MODEL", "qwen/qwen3.6-35b-a3b"),
        llm_api_key=_env("LLM_API_KEY", "lm-studio"),
        llm_temperature=float(_env("LLM_TEMPERATURE", "0.0")),
        llm_max_tokens=int(_env("LLM_MAX_TOKENS", "2048")),
        llm_request_timeout=float(_env("LLM_REQUEST_TIMEOUT", "90.0")),
        agent_max_iterations=int(_env("AGENT_MAX_ITERATIONS", "5")),
        api_host=_env("API_HOST", "0.0.0.0"),
        api_port=int(_env("API_PORT", "8000")),
        api_bearer_token=_env("API_BEARER_TOKEN", ""),
        chroma_path=chroma_path,
        served_model=_env("SERVED_MODEL", DEFAULT_SERVED_MODEL),
    )
