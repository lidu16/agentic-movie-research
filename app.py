"""
Streamlit Frontend for Neo4j Movies RAG.
"""

import streamlit as st
import time
import os
from core.config import config
from knowledge_graph.neo4j_reader import get_reader
from document_store.vector_store import initialize_vector_store
from core.rag_pipeline import build_pipeline, get_pipeline_diagram
import streamlit.components.v1 as components

# ... (Previous imports match)

# Mermaid Renderer
# Mermaid Renderer
def mermaid(code: str):
    # Strip LangGraph frontmatter usually formatted as:
    # ---
    # config: ...
    # ---
    if "---" in code:
        parts = code.split("---")
        if len(parts) >= 3:
            code = parts[-1]
    
    # Clean up
    code = code.strip()
    
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});
            </script>
        </head>
        <body>
            <pre class="mermaid">
                {code}
            </pre>
        </body>
        </html>
        """,
        height=500,
        scrolling=True
    )

with st.sidebar:
    st.title("🎬 Configuration")
    
    st.subheader("System Status")
    try:
        reader = get_reader()
        summary = reader.get_schema_summary()
        st.success("Connected to Neo4j ✅")
        st.info(f"Gemini Model: `{config.GEMINI_MODEL}`")
        
        tab1, tab2 = st.tabs(["📊 Schema", "🕸️ Pipeline"])
        
        with tab1:
            st.text(summary)
            
        with tab2:
            st.caption("LangGraph Workflow")
            # Fallback to static image if dynamic rendering fails
            if os.path.exists("graph-diagram.png"):
                st.image("graph-diagram.png", use_container_width=True)
            else:
                st.warning("Pipeline diagram not found.")
            
    except Exception as e:
        st.error(f"Neo4j Error: {e}")

    st.markdown("---")
    res = st.toggle("Show Retrieval Details", value=True)


@st.cache_resource
def init_resources():
    """Initialize expensive resources once."""
    if not config.validate():
        st.error("Missing configuration! Check your .env file.")
        st.stop()
    
    with st.spinner("📚 Loading Movie Documents..."):
        initialize_vector_store()
    
    pipeline = build_pipeline()
    return pipeline

pipeline = None
try:
    pipeline = init_resources()
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()
    
if pipeline is None:
    st.error("Pipeline failed to load.")
    st.stop()    



st.title("🎬 Movies Knowledge Assistant")
st.markdown("Ask about movies, actors, plots, and more! Integrates **Knowledge Graph** + **Vector Search**.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        
        if "retrieval_info" in message:
            with st.expander("🔍 Retrieval Verification"):
                info = message["retrieval_info"]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 🕸️ Knowledge Graph")
                    st.metric("Nodes", info["node_count"])
                    st.metric("Relations", info["rel_count"])
                    if info["graph_context"]:
                        st.caption("Excerpt:")
                        st.code(info["graph_context"][:300] + "...")
                        
                with c2:
                    st.markdown("### 📚 Document Store")
                    st.metric("Documents", info["doc_count"])
                    if info["doc_context"]:
                        st.caption("Excerpt:")
                        st.code(info["doc_context"][:300] + "...")


st.markdown("### 💡 Suggested Questions")
c1, c2, c3 = st.columns(3)
suggestion = None

if c1.button("🕸️ Who directed The Matrix?", use_container_width=True):
    suggestion = "Who directed The Matrix?"
if c2.button("📚 Plot of Inception?", use_container_width=True):
    suggestion = "What is the plot of Inception?"
if c3.button("✨ Director & Plot of Forrest Gump?", use_container_width=True):
    suggestion = "Who directed Forrest Gump and what is the movie about?"

prompt = st.chat_input("Ask a question (e.g., 'Who is Neo?')")

if suggestion:
    prompt = suggestion

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            start = time.time()
            
            result = pipeline.invoke({"question": prompt})
            
            answer = result.get("answer", "No answer generated.")
            
            retrieval_info = {
                "node_count": result.get("node_count", 0),
                "rel_count": result.get("relationship_count", 0),
                "doc_count": result.get("doc_count", 0),
                "graph_context": result.get("graph_context", ""),
                "doc_context": result.get("doc_context", "")
            }
            
            duration = time.time() - start
            
            st.markdown(answer)
            st.caption(f"⏱️ Generated in {duration:.2f}s")
            
            if res:
                with st.expander("🔍 Live Retrieval Context", expanded=True):
                    c1, c2 = st.columns(2)
                    c1.markdown("#### 🕸️ Graph Context")
                    if retrieval_info["graph_context"]:
                        c1.text(retrieval_info["graph_context"])
                    else:
                        c1.warning("No graph matches.")
                        
                    c2.markdown("#### 📚 Document Context")
                    if retrieval_info["doc_context"]:
                        c2.text(retrieval_info["doc_context"])
                    else:
                        c2.warning("No document matches.")
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "retrieval_info": retrieval_info
    })
