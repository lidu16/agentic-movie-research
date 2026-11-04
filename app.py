import streamlit as st
import time
import os
from core.config import config
from core.langsmith_setup import setup_langsmith
from knowledge_graph.neo4j_reader import get_reader
from document_store.session_store import SessionVectorStore
from core.doc_processor import process_uploaded_file
from core.rag_pipeline import build_pipeline, get_pipeline_diagram
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Intelligent Movie Researcher",
    page_icon="🎬",
    layout="wide",
)

AGENT_DISPLAY_NAMES = {
    "neo4j": "Knowledge graph",
    "rag": "Uploaded documents",
    "web": "Web search",
}

ROUTE_DISPLAY_NAMES = {
    "graph": "Knowledge graph",
    "document": "Uploaded documents",
    "search": "Web search",
    "multi": "Hybrid research",
}


def mermaid(code: str):
  
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


def render_retrieval_verification(info: dict, title: str = "🔍 Retrieval Verification", expanded: bool = False):
    with st.expander(title, expanded=expanded):
        selected_agents = info.get("selected_agents", [])
        if selected_agents:
            sources = [
                AGENT_DISPLAY_NAMES.get(agent, agent)
                for agent in selected_agents
            ]
            st.caption(f"Sources searched: {', '.join(sources)}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🕸️ Knowledge Graph")
            st.metric("Nodes", info.get("node_count", 0))
            st.metric("Relations", info.get("rel_count", 0))
            graph_context = info.get("graph_context", "")
            if graph_context:
                st.caption("Excerpt:")
                st.code(graph_context[:500] + ("..." if len(graph_context) > 500 else ""))
            else:
                st.info("No graph matches.")

        with c2:
            st.markdown("### 📄 Document Store")
            st.metric("Documents", info.get("session_count", 0))
            session_context = info.get("session_context", "")
            if session_context:
                st.caption("Excerpt:")
                st.code(session_context[:500] + ("..." if len(session_context) > 500 else ""))
            else:
                st.info("No uploaded document matches.")

        web_context = info.get("web_context", "")
        if web_context:
            st.markdown("### 🌐 Web Context")
            st.code(web_context[:900] + ("..." if len(web_context) > 900 else ""))

        web_error = info.get("web_error", "")
        if web_error:
            st.warning(f"Web search status: {web_error}")

if "session_store" not in st.session_state:
    st.session_state.session_store = SessionVectorStore()

langsmith_status = setup_langsmith()

with st.sidebar:
    st.title("🎬 Configuration")
    
    st.subheader("System Status")
    if langsmith_status.get("enabled"):
        st.success(f"LangSmith tracing ON ({langsmith_status.get('project')})")
    else:
        st.info(f"LangSmith tracing OFF: {langsmith_status.get('reason', 'Not configured')}")
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
                st.image("graph-diagram.png", width="stretch")
            else:
                st.warning("Pipeline diagram not found.")
            
    except Exception as e:
        st.error(f"Neo4j Error: {e}")

    st.markdown("---")
    res = st.toggle("Show Retrieval Details", value=True)
    
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
    
    if uploaded_file:
        file_key = f"processed_{uploaded_file.name}"
        if file_key not in st.session_state:
            with st.spinner("Processing Document..."):
                docs = process_uploaded_file(uploaded_file)
                st.session_state.session_store.clear() # Clear old docs for now (simple mode)
                st.session_state.session_store.add_documents(docs)
                st.session_state[file_key] = True
            st.success(f"Loaded {len(docs)} chunks!")
        else:
            st.info("Document ready.")


@st.cache_resource
def init_resources():
    """Initialize expensive resources once."""
    if not config.validate():
        st.error("Missing configuration! Check your .env file.")
        st.stop()
    
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



st.title("🎬 Intelligent Movie Researcher")
st.markdown("Ask about movies, uploaded scripts, or browse the web for latest info.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        
        if "retrieval_info" in message:
            info = message["retrieval_info"]
            route = ROUTE_DISPLAY_NAMES.get(
                info.get("route", ""),
                "Unknown",
            )
            render_retrieval_verification(
                info,
                title=f"Retrieval details · {route}",
                expanded=False,
            )


st.markdown("### 💡 Suggested Questions")
c1, c2, c3 = st.columns(3)
suggestion = None

if c1.button("🕸️ Who directed The Matrix?", width="stretch"):
    suggestion = "Who directed The Matrix?"
if c2.button("📄 Summarize uploaded file", width="stretch"):
    suggestion = "Summarize the uploaded document."
if c3.button("🌐 Gladiator 2 box office?", width="stretch"):
    suggestion = "What is the current box office for Gladiator 2?"

prompt = st.chat_input("Ask a question...")

if suggestion:
    prompt = suggestion

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking & Routing..."):
            
            start = time.time()
            
            # INVOKE PIPELINE WITH SESSION STORE
            result = pipeline.invoke({
                "question": prompt, 
                "session_store": st.session_state.session_store
            })
            
            answer = result.get("answer", "No answer generated.")
            route = result.get("route", "unknown")
            route_label = ROUTE_DISPLAY_NAMES.get(route, "Research")
            selected_agents = result.get("selected_agents", [])
            
            retrieval_info = {
                "route": route,
                "selected_agents": selected_agents,
                "node_count": result.get("node_count", 0),
                "rel_count": result.get("relationship_count", 0),
                "session_count": result.get("session_count", 0),
                "graph_context": result.get("graph_context", ""),
                "session_context": result.get("session_context", ""),
                "web_context": result.get("web_context", ""),
                "web_error": result.get("web_error", "")
            }
            
            duration = time.time() - start
            
            st.markdown(answer)
            if selected_agents:
                sources = ", ".join(
                    AGENT_DISPLAY_NAMES.get(agent, agent)
                    for agent in selected_agents
                )
                st.caption(
                    f"⏱️ {duration:.2f}s · {route_label} · "
                    f"Sources: **{sources}**"
                )
            else:
                st.caption(
                    f"⏱️ {duration:.2f}s · {route_label}"
                )
            
            if res:
                render_retrieval_verification(
                    retrieval_info,
                    title=f"Retrieval details · {route_label}",
                    expanded=True,
                )
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "retrieval_info": retrieval_info
    })
