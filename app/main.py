import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.llm.client import LLMRateLimited, LLMUnavailable, complete, stream_complete
from app.schemas.api import AskRequest, AskResponse

app = FastAPI(title="docqa-api")


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
