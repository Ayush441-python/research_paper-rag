def create_hybrid_retriever(
    mqr_retriever,
    keyword_retriever,
    k=5,
    rrf_k=60
):

    def hybrid_search(query):
        mqr_results = mqr_retriever.invoke(query)
        keyword_results = keyword_retriever(query)

        scores = {}
        documents = {}

        for rank, doc in enumerate(mqr_results):

            doc_id = doc.page_content

            documents[doc_id] = doc
            score = 1 / (rrf_k + rank + 1)

            scores[doc_id] = scores.get(doc_id, 0) + score

        # BM25 results
        for rank, doc in enumerate(keyword_results):

            doc_id = doc.page_content

            documents[doc_id] = doc

            score = 1 / (rrf_k + rank + 1)

            scores[doc_id] = scores.get(doc_id, 0) + score


        ranked_documents = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            documents[doc_id]
            for doc_id, score in ranked_documents[:k]
        ]

    return hybrid_search