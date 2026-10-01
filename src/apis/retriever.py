from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from embedding.model import get_embedding
from src.vectorstore.redis import create_vectorstore
from src.retriever.mqr import create_mqr
from src.retriever.keyword import create_keyword_retriever
from src.retriever.hybrid import create_hybrid_retriever


router = APIRouter()


class RetrieverRequest(BaseModel):
    query: str
    k: int = 5

@router.post("/retrieve")
def retrieve_documents(request: RetrieverRequest):

    try:
        embeddings = get_embedding()

        vectorstore = create_vectorstore(
            embeddings
        )


        mqr_retriever = create_mqr(
            vectorstore
        )

        documents = vectorstore.similarity_search(
            request.query,
            k=20
        )

        keyword_retriever = create_keyword_retriever(
            documents,
            k=request.k
        )

        hybrid_retriever = create_hybrid_retriever(
            mqr_retriever=mqr_retriever,
            keyword_retriever=keyword_retriever,
            k=request.k,
            rrf_k=60
        )

        results = hybrid_retriever(
            request.query
        )

        return {
            "query": request.query,
            "count": len(results),
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )