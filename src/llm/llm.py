import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


@lru_cache(maxsize=1)
def get_llm():
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    llm = ChatGroq(
        model=model_name,
        temperature=0,
        max_retries=3
    )
    return llm