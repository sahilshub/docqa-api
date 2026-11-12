import json
import logging
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.llm.client import LLMRateLimited, LLMUnavailable, complete, stream_complete
from app.logging_config import request_id_var, setup_logging
from app.schemas.api import AskRequest, AskResponse

setup_logging()
log = logging.getLogger("docqa.api")

app = FastAPI(title="docqa-api")

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Give every request an id and write one log line when it finishes.

    A caller's X-Request-ID is reused if it is short and plain; otherwise a new
    one is generated. The id is returned in the X-Request-ID response header.
    """
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    fields = {"method": request.method, "path": request.url.path}
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request_failed", extra={**fields, "latency_ms": _elapsed_ms(start)})
        raise
    else:
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            extra={**fields, "status": response.status_code, "latency_ms": _elapsed_ms(start)},
        )
        return response
    finally:
        request_id_var.reset(token)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        result = await complete(req.question)
        return AskResponse(answer=result.text, truncated=result.truncated)
    
    except LLMRateLimited as e:
        headers = {"Retry-After": e.retry_after} if e.retry_after else {}
        raise HTTPException(429, "Rate limited, try again shortly", headers=headers)
    
    except LLMUnavailable:
        raise HTTPException(503, "Model provider unavailable")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    gen = stream_complete(req.question)
    try:
        first = await gen.__anext__()      # errors before the first token still map to 429/503
    except StopAsyncIteration:
        first = None
    except LLMRateLimited as e:
        headers = {"Retry-After": e.retry_after} if e.retry_after else {}
        raise HTTPException(429, "Rate limited, try again shortly", headers=headers)
    except LLMUnavailable:
        raise HTTPException(503, "Model provider unavailable")

    async def events():
        try:
            if first:
                yield _sse(first["type"], first)
            async for item in gen:
                yield _sse(item["type"], item)
        except (LLMRateLimited, LLMUnavailable):
            yield _sse("error", {"detail": "Model provider failed mid-stream"})

    return StreamingResponse(events(), media_type="text/event-stream")
