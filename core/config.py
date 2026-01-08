"""
Configuration module for Neo4j RAG Pipeline.
Simplified for Movies Sandbox - keyword search only.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings for the RAG pipeline."""
    
    # Neo4j Connection
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    
    # Gemini API
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.0-flash"
    
    # Search Settings
    MAX_SEARCH_RESULTS: int = 10
    SUBGRAPH_DEPTH: int = 2
    
    # Text properties in Movies database
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
