from fastapi import APIRouter, FastAPI

from ingestion.loader import load_pdf
from ingestion.splitter import docs_splitter

from apis import upload

router = APIRouter()
app = FastAPI

@router.post("/ingestion")
def ingestion_pipleline():
    docs = app.include_router(upload.router)
    chunks= docs_splitter(docs)
    return {
        "message": "PDF ingested successfully",
        "pages": len(docs),
        "chunks": len(chunks)
    }
