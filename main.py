from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from src.apis import health
from src.apis import upload
from src.apis import ingestion
from src.apis import retriever
from src.apis import chat


app = FastAPI(
    title="Research RAG API",
    description="Production-ready Retrieval-Augmented Generation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(ingestion.router, prefix="/api")
app.include_router(retriever.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Research RAG API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )