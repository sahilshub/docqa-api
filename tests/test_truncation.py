import asyncio
from types import SimpleNamespace

from app.llm import client
from app.llm.client import complete


def fake_response(content, finish_reason, total_tokens=25):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def patch_create(monkeypatch, response):
    async def fake_create(**kwargs):
        return response

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)


def test_complete_answer_is_not_truncated(monkeypatch):
    patch_create(monkeypatch, fake_response("full answer", "stop"))
    result = asyncio.run(complete("hi"))
    assert result.text == "full answer"
    assert result.truncated is False


def test_length_finish_reason_is_flagged_truncated(monkeypatch):
    patch_create(monkeypatch, fake_response("partial ans", "length"))
    result = asyncio.run(complete("hi"))
    assert result.text == "partial ans"
    assert result.truncated is True


def test_empty_content_becomes_empty_string(monkeypatch):
    patch_create(monkeypatch, fake_response(None, "length"))
    result = asyncio.run(complete("hi"))
    assert result.text == ""
    assert result.truncated is True
    