import time
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import config


class VerificationState(TypedDict, total=False):
    question: str
    draft: str
    graph_context: str
    session_context: str
    critique: str
    final_answer: str


llm = ChatGoogleGenerativeAI(
    model=config.GEMINI_MODEL,
    google_api_key=config.GEMINI_API_KEY,
    temperature=0.2,
)


def _invoke_llm_with_retry(prompt, max_attempts: int = 3, base_delay: float = 0.8):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(base_delay * attempt)
    raise RuntimeError(f"Verification LLM call failed after {max_attempts} attempts: {last_error}")


def critique_node(state: VerificationState) -> dict:
    prompt = f"""
    Question: {state['question']}
    Draft Answer: {state['draft']}

    Evidence:
    --- NEO4J GRAPH CONTEXT ---
    {state.get('graph_context', '')}

    --- UPLOADED DOCUMENT CONTEXT ---
    {state.get('session_context', '')}

    Critique the draft for accuracy against the evidence above.
    If it is correct, say so. If not, explain what is wrong or missing.
    """
    try:
        response = _invoke_llm_with_retry(prompt)
        return {"critique": response.content}
    except Exception as exc:
        return {"critique": f"Verification failed: {exc}"}


def revise_node(state: VerificationState) -> dict:
    prompt = f"""
    Question: {state['question']}
    Draft Answer: {state['draft']}
    Critique: {state.get('critique', '')}

    Produce a corrected, concise final answer using the evidence.
    If the draft is already correct, return it unchanged.
    """
    try:
        response = _invoke_llm_with_retry(prompt)
        return {"final_answer": response.content}
    except Exception as exc:
        return {"final_answer": f"I couldn't verify the answer due to an error: {exc}"}


def verify_answer(question: str, draft: str, graph_context: str, session_context: str) -> dict:
    state: VerificationState = {
        "question": question,
        "draft": draft,
        "graph_context": graph_context,
        "session_context": session_context,
    }
    critique = critique_node(state)
    state.update(critique)
    revised = revise_node(state)
    return {
        "verification": state.get("critique", ""),
        "answer": revised.get("final_answer", draft),
    }
