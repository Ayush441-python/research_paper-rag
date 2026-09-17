from langchain_redis import RedisVectorStore
import os

## docker run -d --name redis-rag -p 6379:6379 -p 8001:8001 redis/redis-stack:latest


def create_vectorstore(documents, embeddings):

    vector_store = RedisVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        redis_url=os.getenv("REDIS_URL"),
        index_name="pdf_rag"
    )

    return vector_store   