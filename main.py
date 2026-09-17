from ingestion.loader import load_pdf
from ingestion.splitter import docs_splitter
from vectorstore.redis import create_vectorstore
from embedding.model import get_embedding
from retriever.mqr import create_mqr


from dotenv import load_dotenv


load_dotenv()

docs= load_pdf("./data/attention_is_all_you_need.pdf")
chunks= docs_splitter(docs)
embedded = get_embedding()
vector = create_vectorstore(chunks, embedded)
mqr = create_mqr(vector)




query = input("Enter your query here: ")

retrieve_docs = mqr.invoke(query)

for i, doc in enumerate(retrieve_docs):
    print(f"\n--- Document {i+1} ---\n\n\n")
    print(doc.page_content)
