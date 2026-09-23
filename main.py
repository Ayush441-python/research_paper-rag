from fastapi import FastAPI



from apis import health
from apis import upload
from apis import ingestion
from apis import retriever
from apis import chat


app = FastAPI(
    title="Research RAG API",
    description="Production-ready Retrieval-Augmented Generation API",
    version="1.0.0"
)


app.include_router(
    health.router,
    prefix="/api")

app.include_router(
    upload.router,
    prefix="/api")

app.include_router(
    ingestion.router,
    prefix="/api")

app.include_router(
    retriever.router,
    prefix="/api"
)

app.include_router(
    chat.router,
    prefix="/api"
)


@app.get("/")
def root():
    return {
        "message": "Research RAG API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )