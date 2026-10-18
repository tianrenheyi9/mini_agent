from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_seconds: float
    tool_timeout_seconds: float
    max_tool_steps: int
    context_token_budget: int
    max_history_messages: int
    keep_recent_messages: int
    summary_max_chars: int
    data_dir: Path

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(_env_str("DATA_DIR", "./data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    return Settings(
        llm_base_url=_env_str("LLM_BASE_URL"),
        llm_api_key=_env_str("LLM_API_KEY"),
        llm_model=_env_str("LLM_MODEL"),
        llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
        tool_timeout_seconds=_env_float("TOOL_TIMEOUT_SECONDS", 10.0),
        max_tool_steps=_env_int("MAX_TOOL_STEPS", 8),
        context_token_budget=_env_int("CONTEXT_TOKEN_BUDGET", 6000),
        max_history_messages=_env_int("MAX_HISTORY_MESSAGES", 40),
        keep_recent_messages=_env_int("KEEP_RECENT_MESSAGES", 8),
        summary_max_chars=_env_int("SUMMARY_MAX_CHARS", 800),
        data_dir=data_dir,
    )
