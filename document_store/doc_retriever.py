from document_store.vector_store import get_vector_store


def retrieve_from_documents(query: str, top_k: int = 2) -> dict:
    
    store = get_vector_store()
    
    if not store.documents:
        return {
            "documents": [],
            "formatted_context": "No documents in vector store.",
            "doc_count": 0
        }
    
    results = store.search(query, top_k=top_k)
    
    MIN_SCORE = 0.5
    relevant = [r for r in results if r["score"] >= MIN_SCORE]
    
    if not relevant:
        return {
            "documents": [],
            "formatted_context": "No relevant movie descriptions found.",
            "doc_count": 0
        }
    
    formatted = format_document_context(relevant)
    
    return {
        "documents": relevant,
        "formatted_context": formatted,
        "doc_count": len(relevant)
    }


def format_document_context(documents: list[dict]) -> str:
    """Format retrieved documents for the LLM."""
    lines = ["=== MOVIE DESCRIPTIONS ==="]
    
    for doc in documents:
        meta = doc["metadata"]
        score = doc["score"]
        
        lines.append(f"\n📽️ {meta['title']} ({meta['year']}) [similarity: {score:.2f}]")
        lines.append("-" * 40)
        
        text = doc["text"]
        if len(text) > 1500:
            text = text[:1500] + "..."
        lines.append(text)
    
    return "\n".join(lines)
