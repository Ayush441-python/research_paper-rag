from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file):

    loader = PyPDFLoader(file)
    doc = loader.load()
    return doc
