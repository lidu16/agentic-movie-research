"""
CLI for Neo4j Movies RAG Pipeline.
"""

import sys
from core.config import config
from core.langsmith_setup import setup_langsmith
from knowledge_graph.neo4j_reader import get_reader
from core.rag_pipeline import build_pipeline, get_pipeline_diagram


def main():
    print("\n" + "=" * 50)
    print("  🎬 Movies RAG Pipeline (Graph + Uploads)")
    print("=" * 50)
    
    if not config.validate():
        print("\n❌ Check your .env file!")
        sys.exit(1)

    langsmith_status = setup_langsmith()
    if langsmith_status.get("enabled"):
        print(f"LangSmith tracing enabled (project: {langsmith_status.get('project')})")
    else:
        print(f"ℹLangSmith tracing disabled: {langsmith_status.get('reason', 'Not configured')}")
    
    print("\nDatabase Schema:")
    print("-" * 40)
    try:
        reader = get_reader()
        print(reader.get_schema_summary())
    except Exception as e:
        print(f" Neo4j connection failed: {e}")
        sys.exit(1)
    print("-" * 40)
    
    print("\n Building pipeline...")
    pipeline = build_pipeline()
    print(" Ready!")
    
    print("\n Pipeline Structure (Mermaid):")
    print("-" * 40)
    print(get_pipeline_diagram())
    print("-" * 40)
    
    print("\n Ask about movies! (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break
            
            # Run pipeline
            print("Thinking...")
            result = pipeline.invoke({"question": question})
            
            print("-" * 40)
            print(f"📈 Graph: {result.get('node_count', 0)} nodes, {result.get('relationship_count', 0)} relations")
            print(f"📄 Uploaded Docs: {result.get('session_count', 0)} matched")
            print("-" * 40)
            print(f"💬 {result.get('answer', 'No answer')}")
            
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break


if __name__ == "__main__":
    main()
