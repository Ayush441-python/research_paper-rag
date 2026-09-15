from ingestion.loader import load_pdf
from ingestion.splitter import docs_splitter

docs= load_pdf("./data/attention_is_all_you_need.pdf")
chunks= docs_splitter(docs)




print(chunks[0])
