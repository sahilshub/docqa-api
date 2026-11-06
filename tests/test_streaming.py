import asyncio
import json
from types import SimpleNamespace

import httpx2 as httpx
import openai
from fastapi.testclient import TestClient

from app.llm import client
from app.llm.client import LLMRateLimited, LLMUnavailable, stream_complete
from app.main import app

REQUEST = httpx.Request("POST", "https://api.example.test/chat/completions")
api = TestClient(app)


def chunk(text=None, finish=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text),
                finish_reason=finish,
            )
        ]
    )


def patch_create(monkeypatch, *, chunks=(), raise_before=None, raise_after=None):
    """Make the provider's create(stream=True) return a fake async stream."""

    async def fake_stream():
        for c in chunks:
            yield c
        if raise_after:
            raise raise_after

    async def fake_create(**kwargs):
        assert kwargs.get("stream") is True
        if raise_before:
            raise raise_before
        return fake_stream()

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)


def collect():
    async def run():
        return [item async for item in stream_complete("hi")]

    return asyncio.run(run())


def patch_stream(monkeypatch, items=(), raise_after=None):
    """Replace stream_complete as seen by the API layer."""

    async def fake(question):
        for item in items:
            yield item
        if raise_after:
            raise raise_after

    monkeypatch.setattr("app.main.stream_complete", fake)


def parse_sse(body):
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def rate_limit_error():
    response = httpx.Response(429, headers={"retry-after": "7"}, request=REQUEST)
    return openai.RateLimitError("slow down", response=response, body=None)


# ---------- client level: stream_complete ----------

def test_stream_yields_deltas_then_done(monkeypatch):
    patch_create(monkeypatch, chunks=[chunk("Hel"), chunk("lo"), chunk(None, "stop")])
    items = collect()
    assert [i["text"] for i in items if i["type"] == "delta"] == ["Hel", "lo"]
    assert items[-1] == {"type": "done", "truncated": False}


def test_stream_flags_truncation(monkeypatch):
    patch_create(monkeypatch, chunks=[chunk("part"), chunk(None, "length")])
    assert collect()[-1] == {"type": "done", "truncated": True}


def test_stream_skips_empty_and_choiceless_chunks(monkeypatch):
    patch_create(
        monkeypatch,
        chunks=[SimpleNamespace(choices=[]), chunk(None), chunk("ok"), chunk(None, "stop")],
    )
    items = collect()
    assert [i["text"] for i in items if i["type"] == "delta"] == ["ok"]


def test_stream_rate_limit_before_first_token(monkeypatch):
    patch_create(monkeypatch, raise_before=rate_limit_error())
    try:
        collect()
    except LLMRateLimited as e:
        assert e.retry_after == "7"
    else:
        raise AssertionError("expected LLMRateLimited")


def test_stream_connection_error_before_first_token(monkeypatch):
    patch_create(monkeypatch, raise_before=openai.APIConnectionError(request=REQUEST))
    try:
        collect()
    except LLMUnavailable:
        pass
    else:
        raise AssertionError("expected LLMUnavailable")


def test_stream_connection_drop_mid_stream(monkeypatch):
    patch_create(
        monkeypatch,
        chunks=[chunk("par")],
        raise_after=openai.APIConnectionError(request=REQUEST),
    )
    seen = []

    async def run():
        async for item in stream_complete("hi"):
            seen.append(item)

    try:
        asyncio.run(run())
    except LLMUnavailable:
        pass
    else:
        raise AssertionError("expected LLMUnavailable")
    assert seen == [{"type": "delta", "text": "par"}]


# ---------- API level: POST /ask/stream ----------

def test_ask_stream_returns_sse(monkeypatch):
    patch_stream(
        monkeypatch,
        [
            {"type": "delta", "text": "Hel"},
            {"type": "delta", "text": "lo"},
            {"type": "done", "truncated": False},
        ],
    )
    r = api.post("/ask/stream", json={"question": "hi"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e[0] for e in events] == ["delta", "delta", "done"]
    assert "".join(e[1]["text"] for e in events if e[0] == "delta") == "Hello"
    assert events[-1][1]["truncated"] is False


def test_ask_stream_reports_truncation(monkeypatch):
    patch_stream(
        monkeypatch,
        [{"type": "delta", "text": "part"}, {"type": "done", "truncated": True}],
    )
    events = parse_sse(api.post("/ask/stream", json={"question": "hi"}).text)
    assert events[-1] == ("done", {"type": "done", "truncated": True})


def test_ask_stream_rate_limit_before_stream_is_http_429(monkeypatch):
    patch_stream(monkeypatch, raise_after=LLMRateLimited("7"))
    r = api.post("/ask/stream", json={"question": "hi"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"


def test_ask_stream_unavailable_before_stream_is_http_503(monkeypatch):
    patch_stream(monkeypatch, raise_after=LLMUnavailable())
    r = api.post("/ask/stream", json={"question": "hi"})
    assert r.status_code == 503
    assert r.json()["detail"] == "Model provider unavailable"


def test_ask_stream_failure_mid_stream_emits_error_event(monkeypatch):
    patch_stream(
        monkeypatch,
        [{"type": "delta", "text": "par"}],
        raise_after=LLMUnavailable(),
    )
    r = api.post("/ask/stream", json={"question": "hi"})
    assert r.status_code == 200  # status was already sent when the failure happened
    events = parse_sse(r.text)
    assert events[0][0] == "delta"
    assert events[-1][0] == "error"
    assert "done" not in [e[0] for e in events]


def test_ask_stream_missing_question():
    assert api.post("/ask/stream", json={}).status_code == 422
