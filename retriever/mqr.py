from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from llm.llm import get_llm


def create_mqr(vectorstore, k=5):
    base_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k
        }
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=get_llm()
    )

    return retriever