import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException


from src.ingestion.loader import load_pdf

router = APIRouter()



UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
def upload_pdf(file: UploadFile=File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename
    }