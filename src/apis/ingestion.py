import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.ingestion.loader import load_pdf
from src.ingestion.splitter import docs_splitter
from src.embedding.model import get_embedding
from src.vectorstore.redis import create_vectorstore

router = APIRouter()


class IngestionRequest(BaseModel):
    filename: str


@router.post("/ingestion")
def ingestion_pipeline(request: IngestionRequest):
    try:
        file_path = os.path.join("data/uploads", request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File '{request.filename}' not found in uploads"
            )

        docs = load_pdf(file_path)
        chunks = docs_splitter(docs)
        embeddings = get_embedding()
        create_vectorstore(chunks, embeddings)

        return {
            "message": "PDF ingested successfully",
            "filename": request.filename,
            "pages": len(docs),
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )