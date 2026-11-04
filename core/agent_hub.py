
import re
import time
from typing import Literal

from ddgs import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from core.config import config
from core.llm_utils import normalize_llm_text
from document_store.session_store import SessionVectorStore
from knowledge_graph.neo4j_reader import get_reader
from prompt.agent_prompts import (
    AGGREGATOR_PROMPT,
    NEO4J_AGENT_PROMPT,
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
    max_attempts: int = 2,
    base_delay: float = 0.8,
):
    """Invoke Gemini with a limited number of retries."""
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            last_error = exc

            if attempt < max_attempts:
                time.sleep(base_delay * attempt)

    raise RuntimeError(
        f"LLM invocation failed after {max_attempts} attempts: {last_error}"
    )


def _normalize_agents(
    items: list[str],
    has_upload: bool,
) -> list[AgentLabel]:

    allowed = {
        "NEO4J": "neo4j",
        "RAG": "rag",
        "WEB": "web",
    }

    selected: list[AgentLabel] = []

    for item in items:
        key = re.sub(r"[^A-Za-z0-9]", "", str(item)).upper()

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


def _select_agents(question: str, has_upload: bool) -> list[AgentLabel]:
    """Select agents using simple rules. General questions use web search."""

    q = question.lower()

    document_keywords = [
        "uploaded document",
        "uploaded file",
        "uploaded pdf",
        "my document",
        "my pdf",
        "this document",
        "this pdf",
        "the document",
        "the pdf",
        "the uploaded file",
        "according to the document",
        "according to the pdf",
        "in the document",
        "in the pdf",
        "from my file",
        "from the uploaded",
        "my notes",
        "this script",
        "the script",
        "my script",
        "uploaded script",
        "movie script",
        "screenplay",
        "summarize this",
        "summarize the script",
    ]

    web_keywords = [
        "search the web",
        "search online",
        "on the internet",
        "latest",
        "current",
        "recent",
        "today",
        "this week",
        "this month",
        "box office",
        "opening weekend",
        "streaming availability",
        "latest information",
        "current information",
        "recent information",
        "online sources",
        "web sources",
        "compare with online",
        "compare with the web",
    ]

    graph_keywords = [
        "neo4j",
        "knowledge graph",
        "graph database",
        "in the graph",
        "from the graph database",
        "database records",
    ]

    wants_document = (
        has_upload
        and any(keyword in q for keyword in document_keywords)
    )

    wants_web = any(keyword in q for keyword in web_keywords)
    wants_graph = any(keyword in q for keyword in graph_keywords)

    selected: list[AgentLabel] = []

    if wants_document:
        selected.append("rag")

    if wants_web:
        selected.append("web")

    if wants_graph:
        selected.append("neo4j")

    if selected:
        return selected

    # This assistant is movie-focused: use the graph for established facts
    # and web search to fill gaps or provide source links.
    return ["neo4j", "web"]


def introspect_graph(state: dict) -> dict:
    try:
        reader = get_reader()

        return {
            "schema_summary": reader.get_schema_summary(),
            "error": None,
        }

    except Exception as exc:
        return {
            "schema_summary": "Unknown",
            "error": str(exc),
        }


def router_node(state: dict) -> dict:
    """Route questions without making an unnecessary Gemini request."""

    print("--- ROUTING ---")

    session_store = state.get("session_store")

    has_upload = bool(
        session_store
        and getattr(session_store, "documents", {})
    )

    question = str(state.get("question", "")).strip()

    selected_agents = _select_agents(
        question=question,
        has_upload=has_upload,
    )

    route = route_from_agents(selected_agents)

    print(f"Selected agents: {selected_agents}")
    print(f"Selected route: {route}")

    return {
        "selected_agents": selected_agents,
        "route": route,
    }


def neo4j_agent(state: dict) -> dict:
    if "neo4j" not in state.get("selected_agents", []):
        return {
            "graph_context": "",
            "node_count": 0,
            "relationship_count": 0,
        }

    return _run_neo4j_retrieval(state)


def _run_neo4j_retrieval(state: dict) -> dict:
    try:
        llm = _build_llm(temperature=0.0)

        planner_prompt = NEO4J_AGENT_PROMPT.format(
            question=state["question"],
            schema_summary=state.get("schema_summary", ""),
        )

        planner_response = _invoke_with_retry(
            llm,
            planner_prompt,
        )

        graph_question = (
            normalize_llm_text(planner_response.content).strip()
            or state["question"]
        )

        from knowledge_graph.retriever import (
            execute_graph_plan,
            generate_graph_plan,
        )

        plan = generate_graph_plan(
            graph_question,
            schema=state.get("schema_summary", ""),
        )

        result = execute_graph_plan(plan)

        return {
            "graph_context": result.get("formatted_context", ""),
            "node_count": result.get("node_count", 0),
            "relationship_count": result.get("relationship_count", 0),
        }

    except Exception as exc:
        return {
            "graph_context": "",
            "node_count": 0,
            "relationship_count": 0,
            "error": str(exc),
        }


def rag_agent(state: dict) -> dict:
    if "rag" not in state.get("selected_agents", []):
        return {
            "session_context": "",
            "session_count": 0,
        }

    store: SessionVectorStore | None = state.get("session_store")

    if not store:
        return {
            "session_context": "",
            "session_count": 0,
        }

    try:
        results = store.search(state["question"])

        if not results:
            return {
                "session_context": "",
                "session_count": 0,
            }

        formatted_chunks = []

        for idx, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})

            source = metadata.get("source", "uploaded_file")
            chunk_id = metadata.get("chunk_id", "-")
            score = result.get("score", 0.0)

            preview = re.sub(
                r"\s+",
                " ",
                result.get("text", ""),
            ).strip()

            if len(preview) > 600:
                preview = preview[:600] + "..."

            formatted_chunks.append(
                f"[{idx}] Source: {source} | "
                f"Chunk: {chunk_id} | "
                f"Similarity: {score:.2f}\n{preview}"
            )

        return {
            "session_context": "\n\n".join(formatted_chunks),
            "session_count": len(results),
        }

    except Exception as exc:
        return {
            "session_context": "",
            "session_count": 0,
            "error": f"Document retrieval failed: {exc}",
        }


def web_agent(state: dict) -> dict:
    if "web" not in state.get("selected_agents", []):
        return {
            "web_context": "",
            "web_error": "",
        }

    print("--- RUNNING WEB SEARCH ---")

    web_query = str(state.get("question", "")).strip()

    if not web_query:
        return {
            "web_context": "",
            "web_error": "The search question is empty.",
        }

    search_warning = ""

    # First try Tavily when an API key is configured.
    if config.TAVILY_API_KEY:
        try:
            client = TavilyClient(
                api_key=config.TAVILY_API_KEY
            )

            results = client.search(
                query=web_query,
                search_depth="basic",
            )

            entries = _format_tavily_entries(results)

            if entries:
                print(f"Tavily returned {len(entries)} results.")

                return {
                    "web_context": "\n\n".join(entries),
                    "web_error": "",
                }

            search_warning = "Tavily returned no usable results."

        except Exception as exc:
            search_warning = (
                f"Tavily unavailable ({type(exc).__name__}). "
            )

    else:
        search_warning = "Tavily API key not configured. "

    # Use Bing through the updated ddgs package.
    try:
        bing_entries = _search_with_bing(web_query)

        if bing_entries:
            print(f"Bing returned {len(bing_entries)} results.")

            return {
                "web_context": "\n\n".join(bing_entries),
                "web_error": "",
            }

        return {
            "web_context": "",
            "web_error": (
                search_warning
                + "Bing returned no usable search results."
            ),
        }

    except Exception as exc:
        print(f"Bing search failed: {exc}")

        return {
            "web_context": "",
            "web_error": (
                search_warning
                + f"Bing search failed: {type(exc).__name__}: {exc}"
            ),
        }


def _format_tavily_entries(results: dict) -> list[str]:
    if not isinstance(results, dict):
        return []

    entries: list[str] = []

    for res in results.get("results", [])[
        :config.MAX_SEARCH_RESULTS
    ]:
        if not isinstance(res, dict):
            continue

        url = str(res.get("url", "")).strip()
        title = str(res.get("title", "")).strip()
        content = str(res.get("content", "")).strip()

        if not (url or title or content):
            continue

        entries.append(
            f"Title: {title or 'Untitled'}\n"
            f"Source: {url or 'Not provided'}\n"
            f"Content: {content}"
        )

    return entries


def _search_with_bing(query: str) -> list[str]:
    """Search Bing using ddgs with a bounded timeout."""

    entries: list[str] = []

    with DDGS(timeout=15) as ddgs:
        results = ddgs.text(
            query,
            max_results=config.MAX_SEARCH_RESULTS,
            backend="bing",
        )

        for res in results or []:
            if not isinstance(res, dict):
                continue

            url = str(res.get("href", "")).strip()
            title = str(res.get("title", "")).strip()
            body = str(res.get("body", "")).strip()

            # Ignore empty or invalid search-result placeholders.
            if not title and not body:
                continue

            if not url.startswith(("http://", "https://")):
                continue

            entries.append(
                f"Title: {title or 'Untitled'}\n"
                f"Source: {url}\n"
                f"Content: {body or 'No description provided.'}"
            )

    return entries


def aggregate_answer(state: dict) -> dict:
    graph_context = state.get("graph_context", "")
    session_context = state.get("session_context", "")
    web_context = state.get("web_context", "")

    selected_agents = ", ".join(
        state.get("selected_agents", [])
    ) or "none"

    # Do not ask Gemini to invent an answer when retrieval failed.
    if not any([
        graph_context.strip(),
        session_context.strip(),
        web_context.strip(),
    ]):
        error = state.get("web_error", "")

        answer = (
            "I couldn't find usable information from the selected "
            "sources, so I can't provide a reliable answer yet."
        )

        if error:
            answer += f"\n\nSearch details: {error}"

        return {
            "answer": answer,
            "error": error or None,
        }

    system_prompt = AGGREGATOR_PROMPT.format(
        selected_agents=selected_agents,
        graph_context=graph_context,
        session_context=session_context,
        web_context=web_context,
    )

    llm = _build_llm(temperature=0.2)

    try:
        response = _invoke_with_retry(
            llm,
            [
                ("system", system_prompt),
                ("human", state["question"]),
            ],
        )

        return {
            "answer": normalize_llm_text(response.content),
            "error": None,
        }

    except Exception as exc:
        return {
            "answer": (
                "I couldn't generate a final answer at the moment. "
                "Please try again."
            ),
            "error": str(exc),
        }


def verify_agent(state: dict) -> dict:
    selected_agents = state.get("selected_agents", [])

    if not any(
        agent in selected_agents
        for agent in ("neo4j", "rag")
    ):
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
