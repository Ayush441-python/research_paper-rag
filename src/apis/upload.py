import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

from src.ingestion.loader import load_pdf
from src.ingestion.splitter import docs_splitter
from src.embedding.model import get_embedding
from src.vectorstore.redis import create_vectorstore

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        docs = load_pdf(file_path)
        chunks = docs_splitter(docs)
        embeddings = get_embedding()
        create_vectorstore(chunks, embeddings)

        return {
            "message": "PDF uploaded and ingested successfully",
            "filename": file.filename,
            "pages": len(docs),
            "chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )