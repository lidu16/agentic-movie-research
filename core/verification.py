
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import config
from core.llm_utils import normalize_llm_text


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


def _invoke_llm(prompt):
    """Make one LLM request without unnecessary retries."""
    return llm.invoke(prompt)


def critique_node(state: VerificationState) -> dict:
    prompt = f"""
Question: {state.get('question', '')}

Draft Answer:
{state.get('draft', '')}

Evidence:
--- NEO4J GRAPH CONTEXT ---
{state.get('graph_context', '')}

--- UPLOADED DOCUMENT CONTEXT ---
{state.get('session_context', '')}

Check whether the draft is accurate based on the evidence.

If correct, say APPROVED.
If incorrect, briefly explain what needs correction.
"""

    try:
        response = _invoke_llm(prompt)
        critique = normalize_llm_text(response.content)

        return {"critique": critique}

    except Exception as exc:
        return {
            "critique": f"Verification unavailable: {exc}"
        }


def revise_node(state: VerificationState) -> dict:
    prompt = f"""
Question: {state.get('question', '')}

Draft Answer:
{state.get('draft', '')}

Critique:
{normalize_llm_text(state.get('critique', ''))}

Produce a corrected, concise final answer using the available evidence.

If the draft is already correct, return it unchanged.
Do not invent unsupported facts.
"""

    try:
        response = _invoke_llm(prompt)

        return {
            "final_answer": normalize_llm_text(response.content)
        }

    except Exception:
        # Preserve the original answer if verification is unavailable.
        return {
            "final_answer": state.get("draft", "")
        }


def verify_answer(
    question: str,
    draft: str,
    graph_context: str,
    session_context: str
) -> dict:
    """Verify a draft answer and revise it when necessary."""

    state: VerificationState = {
        "question": question,
        "draft": draft,
        "graph_context": graph_context,
        "session_context": session_context,
    }

    # Step 1: Check the draft answer.
    critique_result = critique_node(state)

    # Step 2: Normalize the critique before using it.
    critique_text = normalize_llm_text(
        critique_result.get("critique", "")
    )

    state["critique"] = critique_text

    # Step 3: Avoid another Gemini request if verification failed.
    if critique_text.startswith("Verification unavailable:"):
        return {
            "verification": critique_text,
            "answer": draft,
        }

    # Step 4: Revise the answer using the critique.
    revised_result = revise_node(state)

    final_answer = normalize_llm_text(
        revised_result.get("final_answer", draft)
    )

    return {
        "verification": critique_text,
        "answer": final_answer or draft,
    }
