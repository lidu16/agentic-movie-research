
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from document_store.session_store import SessionVectorStore
from core.agent_hub import (
    aggregate_answer,
    introspect_graph,
    neo4j_agent,
    rag_agent,
    router_node,
    verify_agent,
    web_agent,
)


class RAGState(TypedDict, total=False):
    question: str
    session_store: SessionVectorStore | None
    route: Literal["graph", "document", "search", "multi"]
    selected_agents: list[Literal["neo4j", "rag", "web"]]
    schema_summary: str
    graph_context: str
    node_count: int
    relationship_count: int
    session_context: str
    session_count: int
    web_context: str
    web_error: str
    answer: str
    verification: str
    error: str | None

def build_pipeline():
    graph = StateGraph(RAGState)
    
    graph.add_node("introspect", introspect_graph)
    graph.add_node("router", router_node)
    
    graph.add_node("neo4j_agent", neo4j_agent)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("web_agent", web_agent)
    graph.add_node("answer", aggregate_answer)
    graph.add_node("verify", verify_agent)
    

    graph.add_edge(START, "introspect")
    graph.add_edge("introspect", "router")

    graph.add_edge("router", "neo4j_agent")
    graph.add_edge("neo4j_agent", "rag_agent")
    graph.add_edge("rag_agent", "web_agent")
    graph.add_edge("web_agent", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)
    
    return graph.compile()

def get_pipeline_diagram():
    graph = build_pipeline()
    return graph.get_graph().draw_mermaid()
