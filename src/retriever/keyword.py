from rank_bm25 import BM25Okapi


def create_keyword_retriever(documents, k=5):

    # Tokenize documents
    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in documents
    ]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)

    def keyword_search(query):

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get top-k documents
        results = bm25.get_top_n(
            tokenized_query,
            documents,
            n=k
        )

        return results

    return keyword_search
