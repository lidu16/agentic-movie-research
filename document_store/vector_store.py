"""
Vector Store - Simple in-memory vector store with cosine similarity.
Uses Gemini embeddings for semantic search.
"""

import numpy as np
import google.generativeai as genai
from core.config import config

genai.configure(api_key=config.GEMINI_API_KEY)

EMBEDDING_MODEL = "models/text-embedding-004"


def generate_embedding(text: str) -> list[float]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]


def generate_query_embedding(text: str) -> list[float]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result["embedding"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class VectorStore:
    
    def __init__(self):
        self.documents: dict[str, dict] = {}  
        self._initialized = False
    
    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        embedding = generate_embedding(text)
        self.documents[doc_id] = {
            "text": text,
            "embedding": np.array(embedding),
            "metadata": metadata or {}
        }
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
       
        if not self.documents:
            return []
        
        query_embedding = np.array(generate_query_embedding(query))
        
        # Calculate similarities
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
    
    def is_initialized(self) -> bool:
        return self._initialized
    
    def mark_initialized(self):
        self._initialized = True


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get the vector store instance."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def initialize_vector_store():
    """Initialize the vector store with movie documents."""
    from document_store.movie_documents import get_document_texts, get_all_documents
    
    store = get_vector_store()
    
    if store.is_initialized():
        return store
    
    print("📚 Loading movie documents into vector store...")
    
    all_docs = {d["id"]: d for d in get_all_documents()}
    
    for doc_id, text in get_document_texts():
        metadata = {
            "title": all_docs[doc_id]["title"],
            "year": all_docs[doc_id]["year"],
            "genre": all_docs[doc_id]["genre"]
        }
        store.add_document(doc_id, text, metadata)
        print(f"  ✓ Added: {all_docs[doc_id]['title']}")
    
    store.mark_initialized()
    print(f"✅ Vector store ready ({len(store.documents)} documents)")
    
    return store
