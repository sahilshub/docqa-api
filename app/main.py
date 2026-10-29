from fastapi import FastAPI, HTTPException

from app.llm.client import LLMRateLimited, LLMUnavailable, complete
from app.schemas.api import AskRequest, AskResponse

app = FastAPI(title="docqa-api")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        return AskResponse(answer=await complete(req.question))
    
    except LLMRateLimited as e:
        headers = {"Retry-After": e.retry_after} if e.retry_after else {}
        raise HTTPException(429, "Rate limited, try again shortly", headers=headers)
    
    except LLMUnavailable:
        raise HTTPException(503, "Model provider unavailable")
