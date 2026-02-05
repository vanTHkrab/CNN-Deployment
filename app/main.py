from fastapi import FastAPI

app = FastAPI(title="CNN Deployment API")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}

