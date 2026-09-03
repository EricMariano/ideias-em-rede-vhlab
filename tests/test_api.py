from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.config import Settings, DEFAULT_SERVED_MODEL


class FakeAgent:
    def __init__(self, output: str = "parecer de auditoria"):
        self.output = output
        self.last_payload = None

    def invoke(self, payload):
        self.last_payload = payload
        return {"output": self.output}


def _test_settings(**overrides) -> Settings:
    values = dict(
        llm_base_url="http://127.0.0.1:9/v1",
        llm_model="fake-llm",
        llm_api_key="test",
        llm_temperature=0.0,
        llm_max_tokens=128,
        llm_request_timeout=1.0,
        agent_max_iterations=2,
        api_host="127.0.0.1",
        api_port=8000,
        api_bearer_token="",
        chroma_path=None,
        served_model=DEFAULT_SERVED_MODEL,
    )
    values.update(overrides)
    return Settings(**values)


def test_chat_completions_multi_turn_openai_envelope():
    agent = FakeAgent(output="resposta auditada")
    app = create_app(agent=agent, initialize_agent=False, settings=_test_settings())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "knowledge-lab-auditor",
                "stream": False,
                "messages": [
                    {"role": "system", "content": "prompt do unity"},
                    {"role": "user", "content": "Pergunta 1"},
                    {"role": "assistant", "content": "Resposta 1"},
                    {"role": "user", "content": "Pergunta 2"},
                ],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "resposta auditada"
    assert body["usage"]["total_tokens"] == 0
    assert agent.last_payload["input"] == "Pergunta 2"
    assert len(agent.last_payload["chat_history"]) == 2


def test_chat_completions_stream_true_returns_400():
    app = create_app(agent=FakeAgent(), initialize_agent=False, settings=_test_settings())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "oi"}],
                "stream": True,
            },
        )
    assert response.status_code == 400
    assert "stream" in response.json()["detail"].lower()


def test_chat_completions_without_user_returns_400():
    app = create_app(agent=FakeAgent(), initialize_agent=False, settings=_test_settings())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "system", "content": "x"}]},
        )
    assert response.status_code == 400


def test_chat_completions_without_agent_returns_503():
    app = create_app(agent=None, initialize_agent=False, settings=_test_settings())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "oi"}]},
        )
    assert response.status_code == 503


def test_list_models():
    app = create_app(agent=FakeAgent(), initialize_agent=False, settings=_test_settings())
    with TestClient(app) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == DEFAULT_SERVED_MODEL


def test_health():
    chroma = {"ok": True, "collections": ["transcricoes_db"], "missing": ["noticias_db", "nli_metadados_db"]}
    app = create_app(agent=FakeAgent(), initialize_agent=False, settings=_test_settings())
    with patch("src.api.app.check_chroma", return_value=chroma):
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["chroma"] == chroma
    assert body["llm"] is False
    assert body["agent"] is True


def test_bearer_token_required_when_configured():
    settings = _test_settings(api_bearer_token="secret")
    app = create_app(agent=FakeAgent(), initialize_agent=False, settings=settings)
    with TestClient(app) as client:
        denied = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "oi"}]},
        )
        allowed = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "oi"}]},
            headers={"Authorization": "Bearer secret"},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200
