try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def create_keyword_retriever(documents, k=5):
    if HAS_BM25:
        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in documents
        ]
        bm25 = BM25Okapi(tokenized_docs)

        def keyword_search(query):
            tokenized_query = query.lower().split()
            return bm25.get_top_n(tokenized_query, documents, n=k)

        return keyword_search

    def fallback_keyword_search(query):
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            score = len(query_words.intersection(doc_words))
            scored_docs.append((score, doc))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]

    return fallback_keyword_search

