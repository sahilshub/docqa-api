import json
import logging
from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.llm import client as llm_client
from app.main import app

api = TestClient(app)
PROVIDER_REQUEST = httpx.Request("POST", "https://api.example.test/chat/completions")


# ---------- fakes and helpers ----------

def fake_response(content, finish_reason, prompt_tokens=11, completion_tokens=7):
    """A non-streaming chat completion, shaped like the provider's response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def chunk(text=None, finish=None):
    """One streamed chunk."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)
        ]
    )


@pytest.fixture
def provider(monkeypatch):
    """Replace the provider call. Returns a function that sets what it does.

    response:    returned by a non-streaming call
    chunks:      yielded by a streaming call
    error:       raised instead of returning anything
    error_after: raised after the chunks have been yielded
    """

    def configure(*, response=None, chunks=None, error=None, error_after=None):
        async def fake_stream():
            for c in chunks or ():
                yield c
            if error_after:
                raise error_after

        async def fake_create(**kwargs):
            if error:
                raise error
            return fake_stream() if kwargs.get("stream") else response

        monkeypatch.setattr(llm_client._client.chat.completions, "create", fake_create)

    return configure


def rate_limit_error(retry_after="7"):
    response = httpx.Response(429, headers={"retry-after": retry_after}, request=PROVIDER_REQUEST)
    return openai.RateLimitError("slow down", response=response, body=None)


def server_error():
    response = httpx.Response(503, request=PROVIDER_REQUEST)
    return openai.InternalServerError("boom", response=response, body=None)


def timeout_error():
    return openai.APITimeoutError(request=PROVIDER_REQUEST)


def connection_error():
    return openai.APIConnectionError(request=PROVIDER_REQUEST)


def parse_sse(body):
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def logged(caplog, event):
    return [r for r in caplog.records if r.getMessage() == event]


def ask(question="hi", **kwargs):
    return api.post("/ask", json={"question": question}, **kwargs)


def ask_stream(question="hi"):
    return api.post("/ask/stream", json={"question": question})


# ---------- GET /health ----------

def test_health():
    assert api.get("/health").json() == {"status": "ok"}


# ---------- POST /ask ----------

def test_ask_returns_the_answer(provider):
    provider(response=fake_response("full answer", "stop"))
    r = ask()
    assert r.status_code == 200
    assert r.json() == {"answer": "full answer", "truncated": False}


def test_ask_flags_an_answer_cut_off_at_the_token_limit(provider):
    provider(response=fake_response("partial ans", "length"))
    assert ask().json() == {"answer": "partial ans", "truncated": True}


def test_ask_returns_an_empty_answer_when_the_model_sends_no_content(provider):
    provider(response=fake_response(None, "length"))
    r = ask()
    assert r.status_code == 200
    assert r.json() == {"answer": "", "truncated": True}


def test_ask_rate_limited_returns_429_with_retry_after(provider):
    provider(error=rate_limit_error("7"))
    r = ask()
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"
    assert r.json()["detail"] == "Rate limited, try again shortly"


@pytest.mark.parametrize("make_error", [timeout_error, connection_error, server_error])
def test_ask_provider_failure_returns_503(provider, make_error):
    provider(error=make_error())
    r = ask()
    assert r.status_code == 503
    assert r.json()["detail"] == "Model provider unavailable"


def test_ask_without_a_question_returns_422():
    assert api.post("/ask", json={}).status_code == 422


# ---------- POST /ask/stream ----------

def test_stream_sends_text_pieces_then_done(provider):
    provider(chunks=[chunk("Hel"), chunk("lo"), chunk(None, "stop")])
    r = ask_stream()
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [name for name, _ in events] == ["delta", "delta", "done"]
    assert "".join(data["text"] for name, data in events if name == "delta") == "Hello"
    assert events[-1][1]["truncated"] is False


def test_stream_flags_an_answer_cut_off_at_the_token_limit(provider):
    provider(chunks=[chunk("part"), chunk(None, "length")])
    events = parse_sse(ask_stream().text)
    assert events[-1] == ("done", {"type": "done", "truncated": True})


def test_stream_rate_limit_before_the_first_token_returns_429(provider):
    provider(error=rate_limit_error("7"))
    r = ask_stream()
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"


def test_stream_provider_failure_before_the_first_token_returns_503(provider):
    provider(error=connection_error())
    r = ask_stream()
    assert r.status_code == 503
    assert r.json()["detail"] == "Model provider unavailable"


def test_stream_failure_mid_stream_sends_an_error_event(provider):
    provider(chunks=[chunk("par")], error_after=connection_error())
    r = ask_stream()
    assert r.status_code == 200  # the status was already sent when the failure happened
    names = [name for name, _ in parse_sse(r.text)]
    assert names == ["delta", "error"]


def test_stream_without_a_question_returns_422():
    assert api.post("/ask/stream", json={}).status_code == 422


# ---------- request ids and logging ----------

def test_a_request_id_is_generated_and_returned(provider):
    provider(response=fake_response("answer", "stop"))
    assert len(ask().headers["x-request-id"]) == 32


def test_a_callers_request_id_is_reused(provider):
    provider(response=fake_response("answer", "stop"))
    r = ask(headers={"X-Request-ID": "trace-42"})
    assert r.headers["x-request-id"] == "trace-42"


def test_an_unsafe_request_id_is_replaced(provider):
    provider(response=fake_response("answer", "stop"))
    r = ask(headers={"X-Request-ID": "bad id!{}"})
    assert r.headers["x-request-id"] != "bad id!{}"


def test_ask_logs_the_request_and_the_model_call(provider, caplog):
    provider(response=fake_response("answer", "stop", prompt_tokens=11, completion_tokens=7))
    with caplog.at_level(logging.INFO, logger="docqa"):
        ask("my secret question")

    (request,) = logged(caplog, "request")
    assert (request.method, request.path, request.status) == ("POST", "/ask", 200)
    assert request.latency_ms >= 0

    (call,) = logged(caplog, "llm_call")
    assert (call.prompt_tokens, call.completion_tokens, call.total_tokens) == (11, 7, 18)
    assert call.finish_reason == "stop" and call.truncated is False
    assert call.question_chars == len("my secret question")
    # the question text itself must never be logged
    assert "my secret question" not in json.dumps([r.__dict__ for r in caplog.records], default=str)


def test_a_provider_failure_is_logged_with_its_error_type(provider, caplog):
    provider(error=timeout_error())
    with caplog.at_level(logging.INFO, logger="docqa"):
        ask()
    (failure,) = logged(caplog, "llm_unavailable")
    assert failure.error_type == "APITimeoutError"
    (request,) = logged(caplog, "request")
    assert request.status == 503


def test_stream_logs_the_time_to_the_first_token(provider, caplog):
    provider(chunks=[chunk("Hel"), chunk("lo"), chunk(None, "stop")])
    with caplog.at_level(logging.INFO, logger="docqa"):
        ask_stream()
    (stream,) = logged(caplog, "llm_stream")
    assert stream.finish_reason == "stop" and stream.truncated is False
    assert stream.first_token_ms is not None and stream.latency_ms >= stream.first_token_ms


def test_an_unexpected_error_is_logged_with_a_traceback(provider, caplog):
    provider(error=RuntimeError("unexpected"))
    with caplog.at_level(logging.INFO, logger="docqa"), pytest.raises(RuntimeError):
        ask()
    (failure,) = logged(caplog, "request_failed")
    assert failure.levelname == "ERROR" and failure.path == "/ask"
    assert failure.exc_info[0] is RuntimeError
