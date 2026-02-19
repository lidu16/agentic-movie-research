import numpy as np
import time
from core.config import config
from langchain_google_genai import GoogleGenerativeAIEmbeddings

EMBEDDING_MODEL = config.GEMINI_EMBEDDING_MODEL
_embeddings_client: GoogleGenerativeAIEmbeddings | None = None


def _get_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=config.GEMINI_API_KEY,
        )
    return _embeddings_client


def _embed_with_retry(content: str, task_type: str, max_attempts: int = 3, base_delay: float = 0.8) -> list[float]:
    """Call Gemini embedding endpoint with bounded retry/backoff."""
    embeddings_client = _get_embeddings_client()
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return embeddings_client.embed_query(
                content,
                task_type=task_type,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(base_delay * attempt)
    raise RuntimeError(f"Embedding request failed after {max_attempts} attempts: {last_error}")


def generate_embedding(text: str) -> list[float]:
    return _embed_with_retry(text, "retrieval_document")


def generate_query_embedding(text: str) -> list[float]:
    return _embed_with_retry(text, "retrieval_query")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
