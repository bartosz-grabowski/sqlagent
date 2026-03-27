"""Runtime configuration helpers for SQLAgent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from urllib.parse import quote_plus

DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
DEFAULT_DB_HOST = "db"
DEFAULT_DB_PORT = 3306
DEFAULT_DB_NAME = "sqlagent_db"
DEFAULT_DB_USER = "root"
DEFAULT_TOP_K = 5


def normalize_ollama_base_url(base_url: str | None) -> str:
    """Return an Ollama base URL with a scheme and no trailing slash."""
    candidate = (base_url or "").strip()
    if not candidate:
        return DEFAULT_OLLAMA_BASE_URL
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def _read_secret(env: Mapping[str, str], *value_names: str) -> str | None:
    """Read a secret value directly from the environment."""
    for name in value_names:
        value = env.get(name)
        if value:
            return value
    return None


def _read_secret_file(env: Mapping[str, str], *file_names: str) -> str | None:
    """Read a secret from a file referenced by the environment."""
    for name in file_names:
        path = env.get(name)
        if path:
            return Path(path).read_text(encoding="utf-8").rstrip("\r\n")
    return None


def read_mysql_password(env: Mapping[str, str]) -> str:
    """Return the MySQL password from an env var or a mounted secret file."""
    password = _read_secret(env, "MYSQL_ROOT_PASSWORD", "MYSQL_PASSWORD")
    if password is not None:
        return password

    password = _read_secret_file(
        env,
        "MYSQL_ROOT_PASSWORD_FILE",
        "MYSQL_PASSWORD_FILE",
    )
    if password is not None:
        return password

    raise ValueError(
        "MySQL password is not configured. Set MYSQL_ROOT_PASSWORD or "
        "MYSQL_ROOT_PASSWORD_FILE."
    )


@dataclass(frozen=True)
class Settings:
    """Resolved application settings."""

    mysql_user: str
    mysql_password: str
    mysql_host: str
    mysql_port: int
    mysql_database: str
    ollama_model: str
    ollama_base_url: str
    top_k: int

    @property
    def database_uri(self) -> str:
        """Return a SQLAlchemy-compatible database URI."""
        encoded_password = quote_plus(self.mysql_password)
        return (
            "mysql+pymysql://"
            f"{self.mysql_user}:{encoded_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build settings from the environment."""
    env = os.environ if env is None else env
    ollama_base_url = normalize_ollama_base_url(env.get("OLLAMA_BASE_URL"))

    return Settings(
        mysql_user=env.get("MYSQL_USER", DEFAULT_DB_USER),
        mysql_password=read_mysql_password(env),
        mysql_host=env.get("MYSQL_HOST", DEFAULT_DB_HOST),
        mysql_port=int(env.get("MYSQL_PORT", str(DEFAULT_DB_PORT))),
        mysql_database=env.get("MYSQL_DATABASE", DEFAULT_DB_NAME),
        ollama_model=env.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        ollama_base_url=ollama_base_url,
        top_k=int(env.get("SQLAGENT_TOP_K", str(DEFAULT_TOP_K))),
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings for the current process."""
    return load_settings()
