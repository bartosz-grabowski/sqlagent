"""Unit tests for model_utils module."""

from sqlagent.model_utils import is_model_available


class MockClient:
    """Minimal Ollama client stub used by the tests."""

    def __init__(self, host=None):
        self.host = host

    def list(self):
        return {"models": [{"model": "gpt-oss:20b"}, {"name": "other-model"}]}


def test_is_model_available_existing_model(monkeypatch):
    """Test that an existing model is reported as available."""
    monkeypatch.setattr("ollama.Client", MockClient)
    model_name = "gpt-oss:20b"
    assert is_model_available(model_name, host="http://ollama:11434") is True


def test_is_model_available_non_existing_model(monkeypatch):
    """Test that a non-existing model is reported as unavailable."""
    monkeypatch.setattr("ollama.Client", MockClient)
    model_name = "nonexistent"
    assert is_model_available(model_name) is False


def test_is_model_available_invalid_input(monkeypatch):
    """Test that invalid input is handled gracefully."""
    monkeypatch.setattr("ollama.Client", MockClient)
    model_name = ""
    assert is_model_available(model_name) is False


def test_is_model_available_exception(monkeypatch):
    """Test that exceptions in ollama.list are handled gracefully."""

    class FailingClient:
        def __init__(self, host=None):
            self.host = host

        def list(self):
            raise Exception("Mocked exception")

    monkeypatch.setattr("ollama.Client", FailingClient)
    model_name = "gpt-oss:20b"
    assert is_model_available(model_name) is False
