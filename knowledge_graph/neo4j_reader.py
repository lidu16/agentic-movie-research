"""
Neo4j Reader - Read-only operations for Movies Sandbox.
Simplified: keyword search only, no vector search.
"""

from typing import Any
from neo4j import GraphDatabase

from core.config import config


class Neo4jReader:
    """Read-only Neo4j database interface."""
    
    def __init__(self):
        self._driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        
    def close(self):
        self._driver.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    

    def get_labels(self) -> list[str]:
        """Get all node labels."""
        with self._driver.session() as session:
            result = session.run("CALL db.labels()")
            return [record["label"] for record in result]
    
    def get_relationship_types(self) -> list[str]:
        """Get all relationship types."""
        with self._driver.session() as session:
            result = session.run("CALL db.relationshipTypes()")
            return [record["relationshipType"] for record in result]
    
    def get_schema_summary(self) -> str:
        """Get schema summary."""
        labels = self.get_labels()
        rel_types = self.get_relationship_types()
        
        return (
            f"📊 Node Labels: {', '.join(labels)}\n"
            f"🔗 Relationships: {', '.join(rel_types)}"
        )
    
  
    def keyword_search(self, keywords: list[str], limit: int = None) -> list[dict]:
        """
        Search nodes by keywords in text properties.
        Case-insensitive CONTAINS matching.
        """
        if limit is None:
            limit = config.MAX_SEARCH_RESULTS
            
        if not keywords:
            return []
        
        conditions = []
        for prop in config.TEXT_PROPERTIES:
            for i, _ in enumerate(keywords):
                conditions.append(f"toLower(toString(n.{prop})) CONTAINS toLower($kw_{i})")
        
        where_clause = " OR ".join(conditions)
        params = {f"kw_{i}": kw for i, kw in enumerate(keywords)}
        params["limit"] = limit
        
        query = f"""
            MATCH (n)
            WHERE {where_clause}
            RETURN DISTINCT n, labels(n) as labels, elementId(n) as nodeId
            LIMIT $limit
        """
        
        with self._driver.session() as session:
            result = session.run(query, params)
            nodes = []
            for record in result:
                node_data = dict(record["n"])
                node_data["_labels"] = record["labels"]
                node_data["_id"] = record["nodeId"]
                nodes.append(node_data)
            return nodes
    
   
    def expand_subgraph(self, node_ids: list[str]) -> dict[str, Any]:
        """Expand 1-2 hops from matched nodes."""
        if not node_ids:
            return {"nodes": [], "relationships": []}
        
        query = """
            MATCH (n)
            WHERE elementId(n) IN $node_ids
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN DISTINCT 
                n, labels(n) as n_labels, elementId(n) as n_id,
                r, type(r) as r_type,
                m, labels(m) as m_labels, elementId(m) as m_id
            LIMIT 100
        """
        
        nodes_map = {}
        relationships = []
        
        with self._driver.session() as session:
            result = session.run(query, {"node_ids": node_ids})
            for record in result:
                if record["n"]:
                    n_id = record["n_id"]
                    if n_id not in nodes_map:
                        node_data = dict(record["n"])
                        node_data["_labels"] = record["n_labels"]
                        node_data["_id"] = n_id
                        nodes_map[n_id] = node_data
                
                if record["r"] and record["m"]:
                    m_id = record["m_id"]
                    if m_id not in nodes_map:
                        node_data = dict(record["m"])
                        node_data["_labels"] = record["m_labels"]
                        node_data["_id"] = m_id
                        nodes_map[m_id] = node_data
                    
                    relationships.append({
                        "type": record["r_type"],
                        "start": record["n_id"],
                        "end": m_id
                    })
        
        return {
            "nodes": list(nodes_map.values()),
            "relationships": relationships
        }

    # =========================================================================
    # Safe Cypher Execution
    # =========================================================================
    
    def validate_cypher(self, query: str) -> bool:
        """Ensure query is read-only."""
        unsafe_keywords = [
            "CREATE", "DELETE", "DETACH", "MERGE", "SET", "REMOVE", 
            "DROP", "INSERT", "ALTER"
        ]
        upper_query = query.upper()
        
        # Must start with a read clause or WITH/CALL
        if not any(upper_query.strip().startswith(x) for x in ["MATCH", "RETURN", "WITH", "CALL", "UNWIND"]):
            return False
            
        # Must not contain write keywords
        for kw in unsafe_keywords:
            if f" {kw} " in f" {upper_query} " or f"\n{kw} " in f"\n{upper_query} ":
                return False
        
        return True

    def execute_cypher(self, query: str) -> list[dict]:
        """
        Execute raw Cypher query safely.
        """
        if not self.validate_cypher(query):
            raise ValueError("Unsafe or Invalid Cypher Query Rejected.")
            
        with self._driver.session() as session:
            result = session.run(query)
            # Convert result keys to list of dicts
            return [dict(record) for record in result]


# Singleton
_reader: Neo4jReader | None = None

def get_reader() -> Neo4jReader:
    global _reader
    if _reader is None:
        _reader = Neo4jReader()
    return _reader
