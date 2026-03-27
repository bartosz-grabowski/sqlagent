"""Utils for model management."""

import ollama


def is_model_available(model_name: str, host: str | None = None) -> bool:
    """Check if the specified Ollama model is available on an Ollama host."""
    if not model_name:
        return False

    try:
        client = ollama.Client(host=host)
        models = client.list().get("models", [])
        return any(
            model.get("model") == model_name or model.get("name") == model_name
            for model in models
        )
    except Exception:
        return False
