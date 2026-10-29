import logging

import openai
from openai import AsyncOpenAI

from app.config import settings

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


async def complete(question: str) -> str:
    try:
        resp = await _client.chat.completions.create(
            model=settings.llm_model or "llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_completion_tokens=1000,
        )

    except openai.RateLimitError as e:
        log.warning("429 after retries")
        raise LLMRateLimited(e.response.headers.get("retry-after")) from e
    
    except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
        log.warning("provider unavailable: %s", type(e).__name__)
        raise LLMUnavailable() from e
    
    log.info("tokens: %s", resp.usage.total_tokens)
    log.info("finish_reason: %s", resp.choices[0].finish_reason)

    return resp.choices[0].message.content
