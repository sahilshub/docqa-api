import logging

import openai
from openai import AsyncOpenAI

from app.config import settings
from app.schemas.client import Completion

log = logging.getLogger("uvicorn.error")

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


async def complete(question: str) -> Completion:
    try:
        resp = await _client.chat.completions.create(
            model=settings.llm_model or "llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_completion_tokens=settings.max_completion_tokens,
        )

    except openai.RateLimitError as e:
        log.warning("429 after retries")
        raise LLMRateLimited(e.response.headers.get("retry-after")) from e
    
    except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
        log.warning("provider unavailable: %s", type(e).__name__)
        raise LLMUnavailable() from e

    choice = resp.choices[0]
    truncated = choice.finish_reason == "length"
    log.info("tokens=%s finish_reason=%s", resp.usage.total_tokens, choice.finish_reason)
    
    if truncated:
        log.warning("response truncated at token limit")

    return Completion(text=choice.message.content or "", truncated=truncated)
