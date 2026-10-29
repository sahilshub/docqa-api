from fastapi.testclient import TestClient

from app.llm.client import LLMRateLimited, LLMUnavailable
from app.main import app

client = TestClient(app)


def _raises(exc):
    async def fake(q):
        raise exc
    return fake


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask(monkeypatch):
    async def fake(q):
        return "stub"

    monkeypatch.setattr("app.main.complete", fake)
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 200 and r.json() == {"answer": "stub"}


def test_ask_rate_limited(monkeypatch):
    monkeypatch.setattr("app.main.complete", _raises(LLMRateLimited("7")))
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"
    assert r.json()["detail"] == "Rate limited, try again shortly"


def test_ask_provider_unavailable(monkeypatch):
    monkeypatch.setattr("app.main.complete", _raises(LLMUnavailable()))
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 503
    assert r.json()["detail"] == "Model provider unavailable"


def test_ask_missing_question():
    assert client.post("/ask", json={}).status_code == 422