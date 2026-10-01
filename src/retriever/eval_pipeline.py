from langchain_core.runnables import RunnableLambda

from src.embedding.model import get_embedding
from src.vectorstore.redis import get_vectorstore
from src.retriever.keyword import create_keyword_retriever
from src.retriever.hybrid import create_hybrid_retriever


def create_eval_retriever(k=5):

    embeddings = get_embedding()

    vectorstore = get_vectorstore(
        embeddings
    )

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    documents = vectorstore.similarity_search(
        "transformer attention encoder decoder",
        k=20
    )

    keyword_retriever = create_keyword_retriever(
        documents,
        k=k
    )

    return create_hybrid_retriever(
        mqr_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        k=k,
        rrf_k=60
    )