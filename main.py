from ingestion.loader import load_pdf
from ingestion.splitter import docs_splitter
from vectorstore.redis import create_vectorstore
from embedding.model import get_embedding

from dotenv import load_dotenv

load_dotenv()

docs= load_pdf("./data/attention_is_all_you_need.pdf")
chunks= docs_splitter(docs)
embedded = get_embedding()

vector = create_vectorstore(docs,embedded)


