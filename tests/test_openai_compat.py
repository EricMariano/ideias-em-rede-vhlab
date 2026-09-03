from langchain_core.messages import AIMessage, HumanMessage

from src.api.openai_compat import (
    ChatMessage,
    MappingError,
    extract_agent_output,
    message_content_to_text,
    messages_to_agent_input,
)
import pytest


def test_message_content_to_text_string():
    assert message_content_to_text("olá") == "olá"


def test_message_content_to_text_parts():
    content = [
        {"type": "text", "text": "parte 1"},
        {"type": "text", "text": "parte 2"},
    ]
    assert message_content_to_text(content) == "parte 1\nparte 2"


def test_messages_to_agent_input_single_turn():
    messages = [
        ChatMessage(role="system", content="ignorar"),
        ChatMessage(role="user", content="Qual o doc_id 1?"),
    ]
    payload = messages_to_agent_input(messages)
    assert payload == {"input": "Qual o doc_id 1?"}


def test_messages_to_agent_input_multi_turn():
    messages = [
        ChatMessage(role="system", content="persona unity"),
        ChatMessage(role="user", content="Pergunta 1"),
        ChatMessage(role="assistant", content="Resposta 1"),
        ChatMessage(role="user", content="Pergunta 2"),
    ]
    payload = messages_to_agent_input(messages)
    assert payload["input"] == "Pergunta 2"
    history = payload["chat_history"]
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "Pergunta 1"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "Resposta 1"


def test_messages_to_agent_input_requires_user():
    messages = [ChatMessage(role="system", content="oi"), ChatMessage(role="assistant", content="x")]
    with pytest.raises(MappingError):
        messages_to_agent_input(messages)


def test_messages_to_agent_input_empty_user():
    messages = [ChatMessage(role="user", content="   ")]
    with pytest.raises(MappingError):
        messages_to_agent_input(messages)


def test_extract_agent_output_dict():
    assert extract_agent_output({"output": "texto"}) == "texto"
