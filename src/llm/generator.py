from src.llm.chain import get_chain

chain = get_chain()

def generate(query: str, context: list[str]) -> str:
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})

def generate_stream(query: str, context: list[str]):
    context_text = "\n\n".join(context)
    return chain.stream({"question": query, "context": context_text})