import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


class Config:
    
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "llm-demonstration")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
    
    MAX_SEARCH_RESULTS: int = 5
    SUBGRAPH_DEPTH: int = 2
    
    TEXT_PROPERTIES: list = ["title", "name", "tagline", "released"]
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.NEO4J_PASSWORD:
            print("⚠️  NEO4J_PASSWORD not set")
            return False
        if not cls.GEMINI_API_KEY:
            print("⚠️  GEMINI_API_KEY not set")
            return False
        return True


config = Config()
