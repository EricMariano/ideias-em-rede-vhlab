"""Modelos Pydantic e mapeamento OpenAI chat completions → AgentExecutor."""

from __future__ import annotations

import time
import uuid
from typing import Any, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """Mensagem no formato OpenAI Chat Completions."""

    model_config = ConfigDict(extra="ignore")

    role: str
    content: Union[str, List[Any], None] = None


class ChatCompletionRequest(BaseModel):
    """Subset do POST /v1/chat/completions usado pelo RIDE/vhtoolkit."""

    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: List[dict]


def message_content_to_text(content: Union[str, List[Any], None]) -> str:
    """Normaliza o campo content (string ou lista de parts) para texto plano."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "text" and "text" in item:
                parts.append(str(item["text"]))
            elif "text" in item:
                parts.append(str(item["text"]))
        else:
            text_attr = getattr(item, "text", None)
            if text_attr is not None:
                parts.append(str(text_attr))
    return "\n".join(parts)


class MappingError(ValueError):
    """Payload OpenAI inválido para invocação do agente."""


def messages_to_agent_input(messages: List[ChatMessage]) -> dict:
    """
    Converte messages OpenAI em {input, chat_history} para AgentExecutor.invoke.

    - role=system é ignorado (o SYSTEM_PROMPT de auditoria permanece no agente).
    - A última mensagem user vira input.
    - user/assistant anteriores viram chat_history (HumanMessage / AIMessage).
    """
    conversational: List[tuple[int, ChatMessage]] = [
        (index, message)
        for index, message in enumerate(messages)
        if message.role in ("user", "assistant")
    ]

    last_user_pos = None
    for pos in range(len(conversational) - 1, -1, -1):
        if conversational[pos][1].role == "user":
            last_user_pos = pos
            break

    if last_user_pos is None:
        raise MappingError("É necessária ao menos uma mensagem com role=user.")

    last_user = conversational[last_user_pos][1]
    user_input = message_content_to_text(last_user.content).strip()
    if not user_input:
        raise MappingError("A última mensagem user não pode ter content vazio.")

    history: List[BaseMessage] = []
    for pos in range(last_user_pos):
        message = conversational[pos][1]
        text = message_content_to_text(message.content)
        if message.role == "user":
            history.append(HumanMessage(content=text))
        else:
            history.append(AIMessage(content=text))

    payload: dict = {"input": user_input}
    if history:
        payload["chat_history"] = history
    return payload


def build_chat_completion_response(content: str, model: str) -> ChatCompletionResponse:
    """Monta o envelope OpenAI a partir do texto final do agente."""
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(),
    )


def extract_agent_output(result: Any) -> str:
    """Extrai a string de resposta de AgentExecutor.invoke (ou dict equivalente)."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        output = result.get("output")
        if output is None:
            return str(result)
        if isinstance(output, str):
            return output
        return str(output)
    return str(result)
