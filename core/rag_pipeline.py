"""
LangGraph RAG Pipeline for Movies Sandbox.
Integrates Neo4j Graph RAG + Document Vector RAG.
"""

from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from core.config import config
from knowledge_graph.neo4j_reader import get_reader
from document_store.doc_retriever import retrieve_from_documents


class RAGState(TypedDict):
    """Pipeline state."""
    question: str
    
    # Neo4j Data
    schema_summary: str
    graph_plan: dict  # New: Agent's Plan (Search vs Query)
    graph_context: str
    node_count: int
    relationship_count: int
    
    # Document Data
    doc_context: str
    doc_count: int
    
    # Result
    answer: str
    error: str | None


# =============================================================================
# Nodes
# =============================================================================

def introspect_graph(state: RAGState) -> dict:
    """Discover graph schema."""
    try:
        reader = get_reader()
        return {"schema_summary": reader.get_schema_summary(), "error": None}
    except Exception as e:
        return {"schema_summary": "Unknown", "error": str(e)}


def identify_entities(state: RAGState) -> dict:
    """
    [AGENT] Analyze question to decide retrieval strategy.
    Output: {"strategy": "search", "keywords": ...} OR {"strategy": "query", "cypher": ...}
    """
    try:
        from knowledge_graph.retriever import generate_graph_plan
        plan = generate_graph_plan(
            state["question"], 
            schema=state.get("schema_summary", "")
        )
        return {"graph_plan": plan}
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"graph_plan": {}}


def lookup_graph(state: RAGState) -> dict:
    """
    [TOOL] Execute the Graph Plan.
    """
    try:
        from knowledge_graph.retriever import execute_graph_plan
        plan = state.get("graph_plan", {})
        result = execute_graph_plan(plan)
        
        return {
            "graph_context": result["formatted_context"],
            "node_count": result["node_count"],
            "relationship_count": result["relationship_count"]
        }
    except Exception as e:
        return {"graph_context": "", "node_count": 0, "relationship_count": 0, "error": str(e)}


def retrieve_document_context(state: RAGState) -> dict:
    """Retrieve relevant movie descriptions from vector store."""
    try:
        result = retrieve_from_documents(state["question"])
        return {
            "doc_context": result["formatted_context"],
            "doc_count": result["doc_count"]
        }
    except Exception as e:
        return {"doc_context": "", "doc_count": 0, "error": str(e)}


def generate_answer(state: RAGState) -> dict:
    """Generate answer combining Graph and Document context."""
    if state.get("error"):
        print(f"⚠️ Pipeline Warning: {state['error']}")
    
    graph_context = state.get("graph_context", "")
    doc_context = state.get("doc_context", "")
    
    # If both are empty, we can't answer (unless it's general chat, but for RAG we stop)
    if (not graph_context or "No specific" in graph_context) and \
       (not doc_context or "No relevant" in doc_context):
        # We try to answer anyway using LLM knowledge if RAG fails, 
        # but let's be explicit about what we found.
        pass
    
    system_prompt = f"""You are a helpful assistant answering questions about movies using a Knowledge Graph and Movie Documents.

RULES:
1. Combine information from both sources to give a complete answer.
2. If info is missing in one source, rely on the other.
3. Be concise and factual.

GRAPH SCHEMA:
{state.get('schema_summary', 'Unknown')}

SEARCH INSIGHTS (Agent Plan):
{state.get('graph_plan', 'None')}

=== SOURCE 1: KNOWLEDGE GRAPH (Relationships & Facts) ===
{graph_context}

=== SOURCE 2: MOVIE DOCUMENTS (Plot & Details) ===
{doc_context}
"""
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.3
        )
        
        response = llm.invoke([
            ("system", system_prompt),
            ("human", state["question"])
        ])
        return {"answer": response.content}
        
    except Exception as e:
        return {"answer": f"Failed to generate answer: {e}"}



def build_pipeline():
    """Build LangGraph workflow."""
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("introspect", introspect_graph)
    graph.add_node("identify_entities", identify_entities) # AGENT
    graph.add_node("lookup_graph", lookup_graph)           # TOOL 
    graph.add_node("retrieve_docs", retrieve_document_context)
    graph.add_node("answer", generate_answer)
    
  
    graph.add_edge(START, "introspect")
    
    graph.add_edge("introspect", "identify_entities")
    graph.add_edge("identify_entities", "lookup_graph")
    graph.add_edge("lookup_graph", "answer")
    
    graph.add_edge("introspect", "retrieve_docs")
    graph.add_edge("retrieve_docs", "answer")
    
    graph.add_edge("answer", END)
    
    return graph.compile()


def get_pipeline_diagram():
    """Get the Mermaid diagram of the pipeline."""
    graph = build_pipeline()
    return graph.get_graph().draw_mermaid()
