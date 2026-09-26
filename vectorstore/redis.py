from langchain_redis import RedisVectorStore
import os


def create_vectorstore(documents, embeddings):
    vector_store = RedisVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        redis_url=os.getenv("REDIS_URL"),
        index_name="pdf_rag"
    )

    return vector_store


def get_vectorstore(embeddings):
    vector_store = RedisVectorStore(
        embeddings=embeddings,
        redis_url=os.getenv("REDIS_URL"),
        index_name="pdf_rag"
    )

    return vector_store