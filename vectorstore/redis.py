from langchain_redis import RedisVectorStore



def create_vector_store(documents, embeddings):

    vector_store = RedisVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        redis_url=os.getenv("REDIS_URL"),
        index_name="pdf_rag"
    )

    return vector_store   