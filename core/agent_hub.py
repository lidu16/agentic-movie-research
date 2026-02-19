import re
import time
from typing import Literal

from duckduckgo_search import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from core.config import config
from document_store.session_store import SessionVectorStore
from knowledge_graph.neo4j_reader import get_reader
from prompt.agent_prompts import (
    AGGREGATOR_PROMPT,
    NEO4J_AGENT_PROMPT,
    RAG_AGENT_PROMPT,
    ROUTER_PROMPT,
    WEB_AGENT_PROMPT,
)
from core.verification import verify_answer


AgentLabel = Literal["neo4j", "rag", "web"]
RouteLabel = Literal["graph", "document", "search", "multi"]


def _build_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=temperature,
    )


def _invoke_with_retry(
    llm: ChatGoogleGenerativeAI,
    prompt,
    max_attempts: int = 3,
    base_delay: float = 0.8,
):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(base_delay * attempt)
    raise RuntimeError(f"LLM invocation failed after {max_attempts} attempts: {last_error}")


def _normalize_agents(items: list[str], has_upload: bool) -> list[AgentLabel]:
    allowed = {"NEO4J": "neo4j", "RAG": "rag", "WEB": "web"}
    selected: list[AgentLabel] = []
    for item in items:
        key = re.sub(r"[^A-Za-z0-9]", "", item).strip().upper()
        if key in allowed:
            value = allowed[key]
            if value == "rag" and not has_upload:
                continue
            selected.append(value)
    return list(dict.fromkeys(selected))


def route_from_agents(selected_agents: list[AgentLabel]) -> RouteLabel:
    if len(selected_agents) > 1:
        return "multi"
    if selected_agents == ["rag"]:
        return "document"
    if selected_agents == ["web"]:
        return "search"
    return "graph"


def introspect_graph(state: dict) -> dict:
    try:
        reader = get_reader()
        return {"schema_summary": reader.get_schema_summary(), "error": None}
    except Exception as exc:
        return {"schema_summary": "Unknown", "error": str(exc)}


def router_node(state: dict) -> dict:
    print("--- ROUTING ---")
    llm = _build_llm(temperature=0.0)

    session_store = state.get("session_store")
    has_upload = bool(session_store and getattr(session_store, "documents", {}))

    prompt = ROUTER_PROMPT.format(
        question=state["question"],
        has_upload=has_upload,
    )

    try:
        response = _invoke_with_retry(llm, prompt)
        raw_choice = str(response.content).strip()
        selected_agents = _normalize_agents(re.split(r"[,\n]", raw_choice), has_upload)
    except Exception:
        selected_agents = ["neo4j"]

    if not selected_agents:
        selected_agents = ["neo4j"]

    return {
        "selected_agents": selected_agents,
        "route": route_from_agents(selected_agents),
    }


def neo4j_agent(state: dict) -> dict:
    if "neo4j" not in state.get("selected_agents", []):
        return {"graph_context": "", "node_count": 0, "relationship_count": 0}

    return _run_neo4j_retrieval(state)


def _run_neo4j_retrieval(state: dict) -> dict:
    try:
        llm = _build_llm(temperature=0.0)
        planner_prompt = NEO4J_AGENT_PROMPT.format(
            question=state["question"],
            schema_summary=state.get("schema_summary", ""),
        )
        planner_response = _invoke_with_retry(llm, planner_prompt)
        graph_question = str(planner_response.content).strip() or state["question"]

        from knowledge_graph.retriever import execute_graph_plan
        from knowledge_graph.retriever import generate_graph_plan

        plan = generate_graph_plan(graph_question, schema=state.get("schema_summary", ""))
        result = execute_graph_plan(plan)
        return {
            "graph_context": result["formatted_context"],
            "node_count": result["node_count"],
            "relationship_count": result["relationship_count"],
        }
    except Exception as exc:
        return {"graph_context": "", "error": str(exc)}


def rag_agent(state: dict) -> dict:
    if "rag" not in state.get("selected_agents", []):
        return {"session_context": "", "session_count": 0}

    store: SessionVectorStore | None = state.get("session_store")
    if not store:
        return {"session_context": "", "session_count": 0}

    llm = _build_llm(temperature=0.0)
    rag_prompt = RAG_AGENT_PROMPT.format(question=state["question"])
    try:
        rag_response = _invoke_with_retry(llm, rag_prompt)
        retrieval_query = str(rag_response.content).strip() or state["question"]
    except Exception:
        retrieval_query = state["question"]

    results = store.search(retrieval_query)
    if not results:
        return {"session_context": "", "session_count": 0}

    formatted_chunks = []
    for idx, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source = metadata.get("source", "uploaded_file")
        chunk_id = metadata.get("chunk_id", "-")
        score = result.get("score", 0.0)
        preview = re.sub(r"\s+", " ", result.get("text", "")).strip()
        if len(preview) > 600:
            preview = preview[:600] + "..."
        formatted_chunks.append(
            f"[{idx}] Source: {source} | Chunk: {chunk_id} | Similarity: {score:.2f}\n{preview}"
        )

    context = "\n\n".join(formatted_chunks)
    return {"session_context": context, "session_count": len(results)}


def web_agent(state: dict) -> dict:
    if "web" not in state.get("selected_agents", []):
        return {"web_context": ""}

    print("--- RUNNING WEB SEARCH ---")
    try:
        llm = _build_llm(temperature=0.0)
        web_prompt = WEB_AGENT_PROMPT.format(question=state["question"])
        web_response = _invoke_with_retry(llm, web_prompt)
        web_query = str(web_response.content).strip() or state["question"]

        if config.TAVILY_API_KEY:
            try:
                client = TavilyClient(api_key=config.TAVILY_API_KEY)
                results = client.search(query=web_query, search_depth="basic")
                entries = _format_tavily_entries(results)
                if entries:
                    return {
                        "web_context": "\n\n".join(entries),
                        "web_error": "",
                    }
            except Exception as exc:
                error_name = type(exc).__name__
                error_text = str(exc)
                is_forbidden = error_name == "ForbiddenError" or "forbidden" in error_text.lower()
                if not is_forbidden:
                    return {
                        "web_context": "Web search failed. See error message.",
                        "web_error": f"{type(exc).__name__}: {exc!r}",
                    }

        ddg_entries = _search_with_duckduckgo(web_query)
        if ddg_entries:
            warning = ""
            if config.TAVILY_API_KEY:
                warning = "Tavily unavailable; used DuckDuckGo fallback."
            else:
                warning = "TAVILY_API_KEY not set; used DuckDuckGo fallback."
            return {
                "web_context": "\n\n".join(ddg_entries),
                "web_error": warning,
            }

        return {
            "web_context": f"No web results found for query: {web_query}",
            "web_error": "Web search returned no usable entries.",
        }
    except Exception as exc:
        return {
            "web_context": "Web search failed. See error message.",
            "web_error": f"{type(exc).__name__}: {exc!r}",
        }


def _format_tavily_entries(results: dict) -> list[str]:
    if not isinstance(results, dict) or "results" not in results:
        return []

    entries: list[str] = []
    for res in results.get("results", [])[: config.MAX_SEARCH_RESULTS]:
        url = res.get("url", "")
        content = res.get("content", "")
        if url or content:
            entries.append(f"Source: {url}\nContent: {content}")
    return entries


def _search_with_duckduckgo(query: str) -> list[str]:
    entries: list[str] = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=config.MAX_SEARCH_RESULTS)
        for res in results or []:
            url = res.get("href", "")
            title = res.get("title", "")
            body = res.get("body", "")
            if url or body or title:
                content = f"{title}\n{body}".strip()
                entries.append(f"Source: {url}\nContent: {content}")
    return entries


def aggregate_answer(state: dict) -> dict:
    graph_context = state.get("graph_context", "")
    session_context = state.get("session_context", "")
    web_context = state.get("web_context", "")
    selected_agents = ", ".join(state.get("selected_agents", [])) or "none"

    system_prompt = AGGREGATOR_PROMPT.format(
        selected_agents=selected_agents,
        graph_context=graph_context,
        session_context=session_context,
        web_context=web_context,
    )

    llm = _build_llm(temperature=0.2)
    try:
        response = _invoke_with_retry(llm, [("system", system_prompt), ("human", state["question"])])
        return {"answer": response.content, "error": None}
    except Exception as exc:
        return {
            "answer": "I couldn't generate a final answer at the moment. Please try again.",
            "error": str(exc),
        }


def verify_agent(state: dict) -> dict:
    selected_agents = state.get("selected_agents", [])
    if not any(agent in selected_agents for agent in ("neo4j", "rag")):
        return {}

    draft = state.get("answer", "")
    if not draft:
        return {}

    result = verify_answer(
        question=state["question"],
        draft=draft,
        graph_context=state.get("graph_context", ""),
        session_context=state.get("session_context", ""),
    )
    return {
        "answer": result.get("answer", draft),
        "verification": result.get("verification", ""),
    }
