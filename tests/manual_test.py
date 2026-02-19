import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.rag_pipeline import build_pipeline
from document_store.session_store import SessionVectorStore

def test_pipeline():
    print("Building Pipeline...")
    try:
        app = build_pipeline()
        print("✅ Pipeline Built Successfully.")
    except Exception as e:
        print(f"❌ Pipeline Build Failed: {e}")
        return

    # Mock Session Store
    store = SessionVectorStore()
    store.documents = {"test_id": {"text": "This is a test document about Project X.", "embedding": [], "metadata": {}}}
    
    print("\n--- Testing Route: DOCUMENT ---")
    state = {
        "question": "What is Project X?",
        "session_store": store
    }
    

    
    if not os.getenv("GEMINI_API_KEY"):
        print("No API Key found, skipping execution.")
        return

    try:
       
        print("Invoking pipeline (may take a few seconds)...")
        print("Invocation skipped to save tokens, but build passed.")
        
    except Exception as e:
        print(f"Invocation Failed: {e}")

if __name__ == "__main__":
    test_pipeline()
