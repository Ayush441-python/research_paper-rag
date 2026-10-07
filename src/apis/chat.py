from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.retriever.mqr import create_mqr
from src.vectorstore.redis import get_vectorstore
from src.embedding.model import get_embedding
from src.llm.chain import get_chain

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    k: int = 5


@router.post("/chat")
def chat(request: ChatRequest):
    try:
        embeddings = get_embedding()
        vectorstore = get_vectorstore(embeddings)

        try:
            retriever = create_mqr(vectorstore, k=request.k)
            documents = retriever.invoke(request.question)
        except Exception:
            documents = vectorstore.similarity_search(request.question, k=request.k)

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        chain = get_chain()
        response = chain.invoke({
            "question": request.question,
            "context": context
        })

        return {
            "question": request.question,
            "answer": response,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )