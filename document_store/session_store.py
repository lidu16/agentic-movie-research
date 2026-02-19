"""
Session Vector Store
A lightweight, session-based vector store for uploaded documents.
Lives in memory (Streamlit Session State) and uses Gemini embeddings.
"""

import numpy as np
from core.config import config
from document_store.vector_store import generate_embedding, generate_query_embedding, cosine_similarity

class SessionVectorStore:
    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.name = "Session Store"
        
    def add_documents(self, docs: list[dict]):
        """
        Add processed documents to the store.
        Docs expected to have 'id', 'text', 'metadata'.
        """
        print(f"Adding {len(docs)} documents to Session Store...")
        for doc in docs:
            embedding = generate_embedding(doc["text"])
            self.documents[doc["id"]] = {
                "text": doc["text"],
                "embedding": np.array(embedding),
                "metadata": doc["metadata"]
            }
            
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for relevant chunks."""
        if not self.documents:
            return []
            
        query_embedding = np.array(generate_query_embedding(query))
        
        results = []
        for doc_id, doc in self.documents.items():
            score = cosine_similarity(query_embedding, doc["embedding"])
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "score": float(score),
                "metadata": doc["metadata"]
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def clear(self):
        """Clear the store."""
        self.documents = {}
