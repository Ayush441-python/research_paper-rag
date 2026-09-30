from src.llm.llm import get_llm
from src.llm.prompt import get_prompt
from langchain_core.output_parsers import StrOutputParser

llm = get_llm()
prompt = get_prompt()



def get_chain():
    chain = prompt | llm | StrOutputParser()

    return chain