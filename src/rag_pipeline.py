from src.retriever.pipeline import create_retriever
from src.llm.generator import generate
from langsmith import traceable


class RagPipeline:
    def __init__(self):
        self.retriever = create_retriever()

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        docs = self.retriever.invoke(query)

        context = [
            doc.page_content
            for doc in docs
        ]

        answer = generate(
            query=query,
            context=context
        )

        return {
            "query": query,
            "context": context,
            "answer": answer
        }


if __name__ == "__main__":
    rag = RagPipeline()

    query = "Why do we need golden datasets?"
    result = rag.invoke(query)

    print("QUERY:", result["query"])
    print("ANSWER:", result["answer"])

    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"[{i}] {chunk[:300]}...")