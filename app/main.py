from fastapi import FastAPI

app = FastAPI(title="docqa-api")


@app.get("/health")
def health():
    return {"status": "ok"}
