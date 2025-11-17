"""Unit tests for model_utils module."""

from sqlagent.model_utils import is_model_available


def mock_get():
    return {"models": [{"model": "gpt-oss:20b"}, {"model": "other-model"}]}


def test_is_model_available_existing_model(monkeypatch):
    """Test that an existing model is reported as available."""
    monkeypatch.setattr("ollama.list", mock_get)
    model_name = "gpt-oss:20b"
    assert is_model_available(model_name) is True


def test_is_model_available_non_existing_model(monkeypatch):
    """Test that a non-existing model is reported as unavailable."""
    monkeypatch.setattr("ollama.list", mock_get)
    model_name = "nonexistent"
    assert is_model_available(model_name) is False


def test_is_model_available_invalid_input(monkeypatch):
    """Test that invalid input is handled gracefully."""
    monkeypatch.setattr("ollama.list", mock_get)
    model_name = ""
    assert is_model_available(model_name) is False


def test_is_model_available_exception(monkeypatch):
    """Test that exceptions in ollama.list are handled gracefully."""

    def mock_list_exception():
        raise Exception("Mocked exception")

    monkeypatch.setattr("ollama.list", mock_list_exception)
    model_name = "gpt-oss:20b"
    assert is_model_available(model_name) is False
