import logging
import time
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from app.config import settings
from app.schemas.client import Completion

log = logging.getLogger("docqa.llm")

_client = AsyncOpenAI(
    base_url=settings.groq_base_url,
    api_key=settings.groq_api_key,
    timeout=settings.request_timeout_seconds,
    max_retries=3,
)

SYSTEM_PROMPT = "You are a helpful assistant."


class LLMUnavailable(Exception): ...


class LLMRateLimited(Exception):
    def __init__(self, retry_after=None):
        self.retry_after = retry_after


def _model() -> str:
    return settings.llm_model or "llama-3.3-70b-versatile"


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _log_error(event: str, exc: Exception, start: float) -> None:
    log.warning(
        event,
        extra={
            "model": _model(),
            "error_type": type(exc).__name__,
            "latency_ms": _elapsed_ms(start),
        },
    )


async def complete(question: str) -> Completion:
    start = time.perf_counter()
    try:
        resp = await _client.chat.completions.create(
            model=_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_completion_tokens=settings.max_completion_tokens,
        )

    except openai.RateLimitError as e:
        _log_error("llm_rate_limited", e, start)
        raise LLMRateLimited(e.response.headers.get("retry-after")) from e

    except (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    ) as e:
        _log_error("llm_unavailable", e, start)
        raise LLMUnavailable() from e

    choice = resp.choices[0]
    truncated = choice.finish_reason == "length"
    usage = resp.usage
    log.log(
        logging.WARNING if truncated else logging.INFO,
        "llm_call",
        extra={
            "model": _model(),
            "latency_ms": _elapsed_ms(start),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "finish_reason": choice.finish_reason,
            "truncated": truncated,
            "question_chars": len(question),  # never log the question text itself
        },
    )
    return Completion(text=choice.message.content or "", truncated=truncated)


async def stream_complete(question: str) -> AsyncIterator[dict]:
    start = time.perf_counter()
    first_token_ms = None
    finish_reason = None
    usage = None
    try:
        stream = await _client.chat.completions.create(
            model=_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_completion_tokens=settings.max_completion_tokens,
            stream=True,
        )
        async for chunk in stream:
            usage = getattr(chunk, "usage", None) or usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                if first_token_ms is None:
                    first_token_ms = _elapsed_ms(start)
                yield {"type": "delta", "text": choice.delta.content}
            if choice.finish_reason:
                finish_reason = choice.finish_reason

    except openai.RateLimitError as e:
        _log_error("llm_rate_limited", e, start)
        raise LLMRateLimited(e.response.headers.get("retry-after")) from e

    except (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    ) as e:
        _log_error("llm_unavailable", e, start)
        raise LLMUnavailable() from e

    truncated = finish_reason == "length"
    log.log(
        logging.WARNING if truncated else logging.INFO,
        "llm_stream",
        extra={
            "model": _model(),
            "latency_ms": _elapsed_ms(start),
            "first_token_ms": first_token_ms,
            "total_tokens": getattr(usage, "total_tokens", None),
            "finish_reason": finish_reason,
            "truncated": truncated,
            "question_chars": len(question),
        },
    )
    yield {"type": "done", "truncated": truncated}
