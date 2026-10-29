import asyncio

import httpx
import openai
import pytest

from app.llm import client
from app.llm.client import LLMRateLimited, LLMUnavailable, complete

REQUEST = httpx.Request("POST", "https://api.example.test/chat/completions")


def _patch_create(monkeypatch, exc):
    async def fake_create(**kwargs):
        raise exc

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)


def test_rate_limit_maps_to_llm_rate_limited(monkeypatch):
    response = httpx.Response(429, headers={"retry-after": "5"}, request=REQUEST)
    _patch_create(monkeypatch, openai.RateLimitError("slow down", response=response, body=None))
    with pytest.raises(LLMRateLimited) as info:
        asyncio.run(complete("hi"))
    assert info.value.retry_after == "5"


def test_timeout_maps_to_llm_unavailable(monkeypatch):
    _patch_create(monkeypatch, openai.APITimeoutError(request=REQUEST))
    with pytest.raises(LLMUnavailable):
        asyncio.run(complete("hi"))


def test_connection_error_maps_to_llm_unavailable(monkeypatch):
    _patch_create(monkeypatch, openai.APIConnectionError(request=REQUEST))
    with pytest.raises(LLMUnavailable):
        asyncio.run(complete("hi"))


def test_server_error_maps_to_llm_unavailable(monkeypatch):
    response = httpx.Response(503, request=REQUEST)
    _patch_create(monkeypatch, openai.InternalServerError("boom", response=response, body=None))
    with pytest.raises(LLMUnavailable):
        asyncio.run(complete("hi"))
