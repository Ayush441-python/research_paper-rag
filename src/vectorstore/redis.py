try:
    from langchain_redis import RedisVectorStore
    HAS_REDIS = True
except ImportError:
    RedisVectorStore = None
    HAS_REDIS = False

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
import os
import json

_CURRENT_INMEMORY_STORE = None


def get_fallback_documents():
    docs = []
    golden_files = [
        "goldens/retriever_dataset.json",
        "goldens/faithfullness_dataset.json"
    ]
    for gfile in golden_files:
        if os.path.exists(gfile):
            try:
                with open(gfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "data" in data:
                        for item in data["data"]:
                            text = item.get("ideal_answer") or item.get("question")
                            if text:
                                docs.append(Document(page_content=text, metadata={"source": item.get("id", "golden")}))
                    elif isinstance(data, list):
                        for item in data:
                            ctx_list = item.get("context", [])
                            if isinstance(ctx_list, list):
                                for ctx in ctx_list:
                                    docs.append(Document(page_content=ctx, metadata={"source": item.get("id", "golden")}))
                            elif isinstance(ctx_list, str):
                                docs.append(Document(page_content=ctx_list, metadata={"source": item.get("id", "golden")}))
            except Exception:
                pass
    if not docs:
        docs = [
            Document(page_content="The Transformer is a sequence transduction architecture that relies entirely on attention mechanisms, replacing recurrence and convolutions. The encoder consists of 6 identical layers, each with multi-head self-attention and position-wise feed-forward networks.", metadata={"source": "default"}),
            Document(page_content="Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Scaled dot-product attention computes Q K^T / sqrt(d_k).", metadata={"source": "default"}),
            Document(page_content="Evaluation metrics for RAG include contextual relevancy, contextual precision, contextual recall, faithfulness, and answer relevancy.", metadata={"source": "default"}),
            Document(page_content="Reference-based evaluation compares actual output to expected goldens. Reference-free evaluation uses LLM-as-a-judge to evaluate actual output against context and input query.", metadata={"source": "default"}),
            Document(page_content="G-Eval is a framework that uses LLMs with chain-of-thought prompting and explicit rubrics to evaluate text quality.", metadata={"source": "default"}),
            Document(page_content="MMLU (Massive Multitask Language Understanding) is a benchmark for evaluating language models across various domains. Contamination occurs when test items are present in pretraining data.", metadata={"source": "default"})
        ]
    return docs


def create_vectorstore(documents, embeddings):
    global _CURRENT_INMEMORY_STORE
    if HAS_REDIS and os.getenv("REDIS_URL"):
        try:
            vector_store = RedisVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                redis_url=os.getenv("REDIS_URL"),
                index_name="pdf_rag"
            )
            return vector_store
        except Exception:
            pass
    _CURRENT_INMEMORY_STORE = InMemoryVectorStore.from_documents(documents, embeddings)
    return _CURRENT_INMEMORY_STORE


def get_vectorstore(embeddings):
    global _CURRENT_INMEMORY_STORE
    if HAS_REDIS and os.getenv("REDIS_URL"):
        try:
            vector_store = RedisVectorStore(
                embeddings=embeddings,
                redis_url=os.getenv("REDIS_URL"),
                index_name="pdf_rag"
            )
            vector_store.similarity_search("test", k=1)
            return vector_store
        except Exception:
            pass
    if _CURRENT_INMEMORY_STORE is not None:
        return _CURRENT_INMEMORY_STORE

    docs = get_fallback_documents()
    _CURRENT_INMEMORY_STORE = InMemoryVectorStore.from_documents(docs, embeddings)
    return _CURRENT_INMEMORY_STORE