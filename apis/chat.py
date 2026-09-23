from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from retriever.mqr import create_mqr
from vectorstore.redis import create_vectorstore
from embedding.model import get_embedding
from llm.chain import get_chain

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    k: int = 5


@router.post("/chat")
def chat(request: ChatRequest):

    try:
        embeddings = get_embedding()

        vectorstore = create_vectorstore(embeddings)

        retriever = create_mqr(
            vectorstore.as_retriever(
                search_kwargs={
                    "k": request.k
                }
            )
        )

        documents = retriever.invoke(request.question)

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