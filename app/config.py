"""
应用配置模块。

这个模块负责三件事：
1. 从项目根目录的 `.env` 文件读取配置（本地开发时不用手动 export 环境变量）；
2. 把环境变量里的字符串转换成正确的类型（整数、浮点数、路径）；
3. 把所有配置放进一个不可变的 `Settings` 对象，供其他模块共享。

为什么用 `get_settings()` + `lru_cache`？
- 环境变量在程序启动后一般不会变化，只需要读取一次；
- 缓存之后，所有模块拿到的是同一个 `Settings` 实例，避免重复解析。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# `__file__` 指向 app/config.py，所以 parent 是 app/，再 parent 才是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 读取项目根目录下的 .env；如果文件不存在，这一步不会报错，只会静默跳过。
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str = "") -> str:
    """读取字符串环境变量，并去掉首尾空格。"""
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量；没配置就用默认值，配置错了就尽早报错。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        # 配置错误应该在启动阶段暴露，而不是等到运行中途才出现奇怪的问题。
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    """读取浮点数环境变量，规则和 _env_int 相同。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """
    全部运行时配置。

    `frozen=True` 表示创建后不能修改，避免程序运行到一半时配置被意外改掉。
    字段名和 .env 里的变量名一一对应，便于对照排查。
    """

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_proxy: str
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
        """三个必填项都齐全时，才认为 LLM 可用。"""
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)

    @property
    def sessions_dir(self) -> Path:
        """会话文件存放目录：<data_dir>/sessions。"""
        return self.data_dir / "sessions"

    @property
    def traces_dir(self) -> Path:
        """Trace 日志存放目录：<data_dir>/traces。"""
        return self.data_dir / "traces"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取环境变量并构造 Settings；同一个进程内只会真正执行一次。"""
    data_dir = Path(_env_str("DATA_DIR", "./data"))
    # 相对路径统一相对于项目根目录解析，避免受启动时工作目录影响。
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    return Settings(
        llm_base_url=_env_str("LLM_BASE_URL"),
        llm_api_key=_env_str("LLM_API_KEY"),
        llm_model=_env_str("LLM_MODEL"),
        llm_proxy=_env_str("LLM_PROXY"),
        llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
        tool_timeout_seconds=_env_float("TOOL_TIMEOUT_SECONDS", 10.0),
        max_tool_steps=_env_int("MAX_TOOL_STEPS", 8),
        context_token_budget=_env_int("CONTEXT_TOKEN_BUDGET", 6000),
        max_history_messages=_env_int("MAX_HISTORY_MESSAGES", 40),
        keep_recent_messages=_env_int("KEEP_RECENT_MESSAGES", 8),
        summary_max_chars=_env_int("SUMMARY_MAX_CHARS", 800),
        data_dir=data_dir,
    )
