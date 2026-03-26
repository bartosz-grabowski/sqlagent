"""Unit tests for SQLAgent runtime configuration."""

from pathlib import Path

import pytest

from sqlagent.config import (
    DEFAULT_OLLAMA_BASE_URL,
    load_settings,
    normalize_ollama_base_url,
    read_mysql_password,
)


def test_normalize_ollama_base_url_uses_default_for_empty_value():
    """An empty Ollama URL should fall back to the local Docker default."""
    assert normalize_ollama_base_url("") == DEFAULT_OLLAMA_BASE_URL


def test_normalize_ollama_base_url_adds_http_scheme():
    """A bare host:port Ollama endpoint should gain an HTTP scheme."""
    assert (
        normalize_ollama_base_url("ollama.internal:11434")
        == "http://ollama.internal:11434"
    )


def test_load_settings_uses_explicit_ollama_base_url(tmp_path: Path):
    """OLLAMA_BASE_URL should be used as the configured endpoint."""
    password_file = tmp_path / "passwd.txt"
    password_file.write_text("secret", encoding="utf-8")

    settings = load_settings(
        {
            "OLLAMA_BASE_URL": "https://ollama.example.com/",
            "MYSQL_ROOT_PASSWORD_FILE": str(password_file),
        }
    )

    assert settings.ollama_base_url == "https://ollama.example.com"


def test_read_mysql_password_prefers_inline_secret():
    """A direct password environment variable should take precedence."""
    assert (
        read_mysql_password(
            {
                "MYSQL_ROOT_PASSWORD": "inline-secret",
                "MYSQL_ROOT_PASSWORD_FILE": "/not-used",
            }
        )
        == "inline-secret"
    )


def test_read_mysql_password_reads_secret_file(tmp_path: Path):
    """The configured secret file should be read without trailing newlines."""
    password_file = tmp_path / "passwd.txt"
    password_file.write_text("secret\n", encoding="utf-8")

    assert (
        read_mysql_password({"MYSQL_ROOT_PASSWORD_FILE": str(password_file)})
        == "secret"
    )


def test_read_mysql_password_requires_secret():
    """The app should fail fast when no DB password is configured."""
    with pytest.raises(ValueError):
        read_mysql_password({})


def test_database_uri_escapes_password(tmp_path: Path):
    """Special characters in passwords should be URL-encoded."""
    password_file = tmp_path / "passwd.txt"
    password_file.write_text("p@ss word", encoding="utf-8")

    settings = load_settings({"MYSQL_ROOT_PASSWORD_FILE": str(password_file)})

    assert (
        settings.database_uri == "mysql+pymysql://root:p%40ss+word@db:3306/sqlagent_db"
    )
