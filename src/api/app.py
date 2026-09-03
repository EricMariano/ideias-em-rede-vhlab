"""Servidor FastAPI OpenAI-compatible para o Agente RAG."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import Settings, get_settings
from src.api.openai_compat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    MappingError,
    ModelsListResponse,
    build_chat_completion_response,
    extract_agent_output,
    messages_to_agent_input,
)

logger = logging.getLogger(__name__)

EXPECTED_COLLECTIONS = ("transcricoes_db", "noticias_db", "nli_metadados_db")


def _check_bearer(authorization: Optional[str], settings: Settings) -> None:
    token = settings.api_bearer_token
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


async def ping_llm(base_url: str, api_key: str, timeout: float = 5.0) -> bool:
    """Tenta GET {base_url}/models no endpoint OpenAI-compatible da LLM."""
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            return response.status_code < 500
    except Exception as exc:
        logger.warning("[api] Ping da LLM falhou (%s): %s", url, exc)
        return False


def check_chroma(chroma_path: Optional[str]) -> dict:
    """Verifica se o ChromaDB abre e lista coleções esperadas."""
    try:
        from src.vector_store import get_chroma_client

        client = get_chroma_client(chroma_path)
        existing = {item.name for item in client.list_collections()}
        return {
            "ok": True,
            "collections": sorted(existing),
            "missing": [name for name in EXPECTED_COLLECTIONS if name not in existing],
        }
    except Exception as exc:
        logger.warning("[api] Health ChromaDB falhou: %s", exc)
        return {"ok": False, "error": str(exc), "collections": [], "missing": list(EXPECTED_COLLECTIONS)}


def create_rag_agent_from_settings(settings: Settings):
    from src.agent import create_rag_agent

    return create_rag_agent(
        model_name=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        request_timeout=settings.llm_request_timeout,
        max_iterations=settings.agent_max_iterations,
        verbose=False,
        api_key=settings.llm_api_key,
    )


def create_app(
    *,
    agent: Any = None,
    initialize_agent: bool = True,
    settings: Optional[Settings] = None,
) -> FastAPI:
    """
    Fábrica da aplicação.

    - agent: injeta um AgentExecutor (ou fake) para testes.
    - initialize_agent=False: não cria o agente no lifespan (chat devolve 503).
    """
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.agent = agent
        if initialize_agent and app.state.agent is None:
            try:
                logger.info("[api] Inicializando Agente RAG...")
                app.state.agent = create_rag_agent_from_settings(resolved_settings)
            except Exception as exc:
                logger.error("[api] Falha ao criar o Agente RAG no startup: %s", exc)
                app.state.agent = None
        yield

    application = FastAPI(
        title="knowledge-lab auditor API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.agent = agent

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_optional_bearer(
        authorization: Optional[str] = Header(default=None),
    ) -> None:
        _check_bearer(authorization, resolved_settings)

    @application.get("/health")
    async def health():
        chroma = check_chroma(resolved_settings.chroma_path)
        llm_ok = await ping_llm(resolved_settings.llm_base_url, resolved_settings.llm_api_key)
        agent_ready = getattr(application.state, "agent", None) is not None
        return {
            "status": "ok" if chroma["ok"] else "degraded",
            "chroma": chroma,
            "llm": llm_ok,
            "agent": agent_ready,
            "model": resolved_settings.served_model,
        }

    @application.get("/v1/models", response_model=ModelsListResponse)
    async def list_models(_: None = Depends(require_optional_bearer)):
        return ModelsListResponse(
            data=[
                {
                    "id": resolved_settings.served_model,
                    "object": "model",
                    "owned_by": "knowledge-lab",
                }
            ]
        )

    @application.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        _: None = Depends(require_optional_bearer),
    ):
        if payload.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming não é suportado. Envie stream=false.",
            )

        current_agent = getattr(request.app.state, "agent", None)
        if current_agent is None:
            raise HTTPException(
                status_code=503,
                detail="Agente RAG indisponível. Verifique LLM_BASE_URL e o LM Studio.",
            )

        try:
            agent_payload = messages_to_agent_input(payload.messages)
        except MappingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            result = current_agent.invoke(agent_payload)
        except Exception as exc:
            logger.exception("[api] Erro ao invocar o agente")
            raise HTTPException(status_code=502, detail=f"Falha no Agente RAG: {exc}") from exc

        output = extract_agent_output(result)
        model_name = payload.model or resolved_settings.served_model
        return build_chat_completion_response(output, model_name)

    return application


app = create_app()
