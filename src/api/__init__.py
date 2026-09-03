"""API HTTP OpenAI-compatible para o Agente RAG de auditoria parlamentar."""

from src.api.app import create_app, app

__all__ = ["create_app", "app"]
