from ingestion.loader import load_pdf
from ingestion.splitter import docs_splitter
from vectorstore.redis import create_vectorstore
from embedding.model import get_embedding
from retriever.mqr import create_mqr
from llm.chain import get_chain

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from dotenv import load_dotenv


load_dotenv()

docs= load_pdf("./data/attention_is_all_you_need.pdf")
chunks= docs_splitter(docs)
embedded = get_embedding()
vector = create_vectorstore(chunks, embedded)
mqr = create_mqr(vector)
chain = get_chain()



ques = input("Enter your query here: ")

retrieved_docs = mqr.invoke(ques)

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

output = chain.invoke({"question":ques,"context":context})


print(output)