from fastapi import APIRouter, UploadFile, File, HTTPException

from ingestion.loader import load_pdf
from ingestion.splitter import docs_splitter

router = APIRouter()


@router.post("/ingestion")
async def ingestion_pipeline(file: UploadFile = File(...)):

    try:
        docs = await load_pdf(file)
        chunks = docs_splitter(docs)

        return {
            "message": "PDF ingested successfully",
            "filename": file.filename,
            "pages": len(docs),
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )