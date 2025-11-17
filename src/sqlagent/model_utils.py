"""Utils for model management."""

import ollama


def is_model_available(model_name: str) -> bool:
    """Check if the specified Ollama model is available locally."""
    try:
        models = ollama.list()["models"]
        return any(m["model"] == model_name for m in models)
    except Exception:
        return False
