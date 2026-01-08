from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import config
from knowledge_graph.neo4j_reader import get_reader


import json
import re

def generate_graph_plan(question: str, schema: str = "") -> dict[str, Any]:
    """
    [AGENT] Decides whether to:
    1. SEARCH (keyword search for specific entities) - e.g. "Who is Neo?"
    2. QUERY (Cypher query for aggregation/filtering) - e.g. "How many movies?"
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.0
        )
        
        system_ctx = f"\nGraph Schema: {schema}" if schema else ""
        
        prompt = f"""
        You are a Neo4j Graph Expert. Analyze the user question and decide the best retrieval strategy.
        
        STRATEGIES:
        1. "search": Use this for questions about specific entities (e.g., "Cast of Matrix", "Who is Keanu?").
           Return specific keywords people/movie names.
        2. "query": Use this for aggregations, counting, filtering, or multi-hop logic (e.g., "How many movies?", "Movies released in 1999").
           Return a valid READ-ONLY Cypher query.
           
        {system_ctx}
        
        OUTPUT FORMAT (JSON):
        {{
            "strategy": "search",
            "keywords": ["entity1", "entity2"]
        }}
        OR
        {{
            "strategy": "query",
            "cypher": "MATCH (m:Movie)... RETURN ..."
        }}
        
        Do not output markdown code blocks. Just the JSON string.
        
        Question: "{question}"
        Plan:
        """
        
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean markdown if present
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        return json.loads(content)
        
    except Exception as e:
        print(f"⚠️ Agent Planning Error: {e}")
        # Fallback to simple keyword extraction
        return {"strategy": "search", "keywords": re.findall(r'\b[a-zA-Z0-9]+\b', question.lower())}


def execute_graph_plan(plan: dict) -> dict[str, Any]:
    """
    [TOOL] Execute the plan (Search or Query).
    """
    reader = get_reader()
    
    strategy = plan.get("strategy", "search")
    
    # CASE 1: KEYWORD SEARCH
    if strategy == "search":
        keywords = plan.get("keywords", [])
        if not keywords:
             return _empty_result("No keywords found.")
             
        nodes = reader.keyword_search(keywords)
        relationships = []
        if nodes:
            node_ids = [n["_id"] for n in nodes]
            subgraph = reader.expand_subgraph(node_ids)
            # Merge nodes
            existing_ids = {n["_id"] for n in nodes}
            for node in subgraph["nodes"]:
                if node["_id"] not in existing_ids:
                    nodes.append(node)
                    existing_ids.add(node["_id"])
            relationships = subgraph["relationships"]
            
        return _format_result(nodes, relationships)

    # CASE 2: CYPHER QUERY
    elif strategy == "query":
        cypher = plan.get("cypher", "")
        if not cypher:
            return _empty_result("Empty Cypher query.")
            
        try:
            results = reader.execute_cypher(cypher)
            # For Cypher, we might get raw data, not just nodes/rels.
            # We treat them as generic records.
            formatted = f"Cypher Execution Result:\n{json.dumps(results, indent=2, default=str)}"
            return {
                "nodes": [], 
                "relationships": [], 
                "formatted_context": formatted,
                "node_count": len(results),
                "relationship_count": 0
            }
        except Exception as e:
            return _empty_result(f"Cypher Error: {e}")
            
    return _empty_result("Unknown strategy.")


def _empty_result(msg: str):
    return {
        "nodes": [], 
        "relationships": [], 
        "formatted_context": msg,
        "node_count": 0, 
        "relationship_count": 0
    }

def _format_result(nodes, relationships):
     formatted = format_context(nodes, relationships)
     return {
        "nodes": nodes,
        "relationships": relationships,
        "formatted_context": formatted,
        "node_count": len(nodes),
        "relationship_count": len(relationships)
    }


def format_context(nodes: list[dict], relationships: list[dict]) -> str:
    """Format nodes and relationships as text for Gemini."""
    if not nodes:
        return "No relevant information found in the graph."
    
    lines = []
    node_names = {}  

    lines.append("=== NODES ===")
    for node in nodes:
        labels = node.get("_labels", [])
        label_str = ":".join(labels) if labels else "Node"
        
        name = node.get("name") or node.get("title") or "Unknown"
        node_names[node["_id"]] = name
        
        lines.append(f"• [{label_str}] {name}")
        
        props = []
        if node.get("released"):
            props.append(f"released: {node['released']}")
        if node.get("tagline"):
            props.append(f"tagline: {node['tagline'][:80]}")
        if node.get("born"):
            props.append(f"born: {node['born']}")
        if props:
            lines.append(f"  {', '.join(props)}")
    
    if relationships:
        lines.append("")
        lines.append("=== RELATIONSHIPS ===")
        for rel in relationships:
            start = node_names.get(rel["start"], "?")
            end = node_names.get(rel["end"], "?")
            lines.append(f"• ({start})-[:{rel['type']}]->({end})")
    
    return "\n".join(lines)
