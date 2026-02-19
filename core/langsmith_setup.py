import os
from typing import Any

from langsmith import Client

from core.config import config


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def setup_langsmith() -> dict[str, Any]:
    tracing_enabled = _is_truthy(config.LANGSMITH_TRACING)
    api_key = config.LANGSMITH_API_KEY.strip()

    if not tracing_enabled:
        return {"enabled": False, "reason": "LANGSMITH_TRACING is disabled."}

    if not api_key:
        return {"enabled": False, "reason": "LANGSMITH_API_KEY is missing."}

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = config.LANGSMITH_ENDPOINT

    try:
        Client(api_key=api_key, api_url=config.LANGSMITH_ENDPOINT)
        return {
            "enabled": True,
            "project": config.LANGSMITH_PROJECT,
            "endpoint": config.LANGSMITH_ENDPOINT,
        }
    except Exception as exc:
        return {
            "enabled": False,
            "reason": f"LangSmith client init failed: {type(exc).__name__}: {exc}",
        }
