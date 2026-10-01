from src.embedding.model import get_embedding
from src.vectorstore.redis import get_vectorstore
from src.retriever.mqr import create_mqr
from src.retriever.keyword import create_keyword_retriever
from src.retriever.hybrid import create_hybrid_retriever


def create_retriever(k=5):

    embeddings = get_embedding()

    vectorstore = get_vectorstore(
        embeddings
    )

    mqr_retriever = create_mqr(
        vectorstore
    )

    documents = vectorstore.similarity_search(
        "transformer attention encoder decoder",
        k=20
    )

    keyword_retriever = create_keyword_retriever(
        documents,
        k=k
    )

    hybrid_retriever = create_hybrid_retriever(
        mqr_retriever=mqr_retriever,
        keyword_retriever=keyword_retriever,
        k=k,
        rrf_k=60
    )

    return hybrid_retriever