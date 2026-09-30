from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def Health():
    print("Everything is fine")





