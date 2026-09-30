import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm():

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL"),
        temperature=0
    )

    return llm