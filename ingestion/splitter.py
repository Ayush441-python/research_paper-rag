from langchain_text_splitters import RecursiveCharacterTextSplitter

def docs_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    return chunks