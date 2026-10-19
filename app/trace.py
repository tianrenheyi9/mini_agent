"""
Trace 记录器。

Trace 用来回答“这一轮 Agent 到底做了什么”：
- 每次 LLM 调用：耗时、token、原始输出、解析结果；
- 每次工具调用：工具名、参数、状态、耗时、结果摘要；
- 每次压缩：压缩前后 token、合并了多少消息、是否降级；
- 解析失败：错误原因和原始输出；
- 轮次结束：总耗时和分类统计。

存储格式是 JSONL：一行一个 JSON 事件，方便追加、grep 和逐行读取。
文件路径：<data_dir>/traces/<session_id>.jsonl

为什么不用数据库？
- MVP 阶段事件量小，JSONL 零依赖、可直接用文本工具排查；
- 追加写不需要事务，单行写入失败最多影响最后一条；
- 后续要查询/聚合时，再把事件导入 SQLite 即可。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from app.agent.context import CompressionResult
from app.agent.llm import LLMResponse
from app.agent.parser import FinalAnswer, ParseError, ParseResult, ToolCall
from app.session.models import TurnStats
from app.tools.base import ToolResult

logger = logging.getLogger(__name__)

# 和 SessionStore 使用同一套 id 规则，防止通过 ../ 写任意文件。
_SESSION_ID_RE = re.compile(r"^s_[A-Za-z0-9_-]{1,64}$")


class TraceLogger:
    """按会话追加 JSONL 事件，并支持读取最近事件。"""

    def __init__(
        self,
        traces_dir: Path,
        *,
        max_field_chars: int = 2000,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        # traces_dir 可由调用方注入，便于测试使用临时目录。
        self._dir = Path(traces_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # 单个字段最多保留多少字符；防止 raw_output / 工具结果撑爆日志。
        self._max_field_chars = max_field_chars
        # 时钟可注入，测试中能断言固定时间戳。
        self._clock = clock or (lambda: datetime.now().astimezone())

    # ------------------------------------------------------------------
    # 写入事件
    # ------------------------------------------------------------------

    def log_turn_start(self, *, session_id: str, turn_id: str) -> None:
        """
        记录一轮开始，便于计算整轮耗时和串联后续事件。

        所有后续事件都带同一个 turn_id，
        UI/Trace 可以据此把 LLM、工具、压缩事件归到同一轮。
        """
        self._write(
            session_id,
            {
                "turn_id": turn_id,
                "event": "turn_start",
            },
        )

    def log_llm_call(
        self,
        *,
        session_id: str,
        turn_id: str,
        step: int,
        response: LLMResponse,
        parsed: ParseResult | None = None,
        prompt_tokens_est: int | None = None,
    ) -> None:
        """
        记录一次 LLM 调用及其解析结果。

        字段说明：
        - latency_ms：这一次模型请求的耗时；
        - prompt_tokens / completion_tokens：API 返回值，部分兼容服务可能为 null；
        - prompt_tokens_est：本地估算值，用于和 API 用量对账；
        - raw_output：模型原始输出，截断后保存；
        - parsed：解析摘要（工具调用 / final / 协议错误）。
        """
        self._write(
            session_id,
            {
                "turn_id": turn_id,
                "step": step,
                "event": "llm_call",
                "model": response.model,
                "latency_ms": response.latency_ms,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "finish_reason": response.finish_reason,
                "prompt_tokens_est": prompt_tokens_est,
                # 原始输出可能很长，统一截断后再落盘。
                "raw_output": self._short(response.content),
                "parsed": self._parsed_summary(parsed),
            },
        )

    def log_tool_call(
        self,
        *,
        session_id: str,
        turn_id: str,
        step: int,
        tool: str,
        tool_input: Mapping[str, Any],
        result: ToolResult,
    ) -> None:
        """
        记录一次工具执行。成功和失败都写同一种事件。

        status 用 ok/error，而不是布尔值：
        日志里一眼就能看出结果，不需要额外解释 true/false。
        """
        event: dict[str, Any] = {
            "turn_id": turn_id,
            "step": step,
            "event": "tool_call",
            "tool": tool,
            "input": self._short(tool_input, 500),
            "status": "ok" if result.ok else "error",
            "duration_ms": result.duration_ms,
        }
        if result.ok:
            event["result_preview"] = self._short(result.data)
        else:
            event["error"] = self._short(result.error)

        if not result.ok:
            logger.warning(
                "Tool %s failed in session %s turn %s: %s",
                tool,
                session_id,
                turn_id,
                result.error,
            )
        self._write(session_id, event)

    def log_parse_error(
        self,
        *,
        session_id: str,
        turn_id: str,
        step: int,
        reason: str,
        raw_output: str,
    ) -> None:
        """记录模型输出不符合协议的情况。"""
        # 同时记录 reason 和原始输出：
        # reason 用于统计哪类错误最多，原始输出用于复现。
        self._write(
            session_id,
            {
                "turn_id": turn_id,
                "step": step,
                "event": "parse_error",
                "reason": reason,
                "raw_output": self._short(raw_output),
            },
        )

    def log_compression(
        self,
        *,
        session_id: str,
        turn_id: str,
        result: CompressionResult,
    ) -> None:
        """记录一次上下文压缩的效果。"""
        # before/after 都是估算值，用于观察压缩是否真的降低了 Prompt 体积。
        self._write(
            session_id,
            {
                "turn_id": turn_id,
                "event": "compress",
                "compressed": result.compressed,
                "summarized_messages": result.summarized_messages,
                "summary_until_msg_id": result.summary_until_msg_id,
                "before_tokens_est": result.before_tokens,
                "after_tokens_est": result.after_tokens,
                "fallback": result.fallback,
            },
        )

    def log_turn_end(
        self,
        *,
        session_id: str,
        turn_id: str,
        status: str,
        stats: TurnStats,
        latency_ms: int,
        reply_preview: str | None = None,
    ) -> None:
        """记录一轮结束和分类统计。"""
        # stats 来自 TurnStats.to_dict()，
        # LLM、工具、压缩的调用次数与 token 都分开记录。
        event: dict[str, Any] = {
            "turn_id": turn_id,
            "event": "turn_end",
            "status": status,
            "latency_ms": latency_ms,
            "stats": stats.to_dict(),
        }
        if reply_preview is not None:
            event["reply_preview"] = self._short(reply_preview)
        self._write(session_id, event)

    # ------------------------------------------------------------------
    # 读取事件
    # ------------------------------------------------------------------

    def read_events(
        self,
        session_id: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        读取最近 N 条事件。

        单行损坏时跳过并记录警告，不影响其他事件。
        limit <= 0 时返回空列表。
        """
        if limit <= 0:
            return []

        path = self._path_for(session_id)
        if not path.exists():
            return []

        events: list[dict[str, Any]] = []
        # 逐行读取，而不是一次性 json.load：
        # JSONL 的每一行都是独立事件，损坏一行不影响其它行。
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skip corrupt trace line in %s", session_id
                    )
                    continue
                if isinstance(event, dict):
                    events.append(event)

        # 只返回最近的 limit 条，避免 API 一次返回整个文件。
        return events[-limit:]

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _write(self, session_id: str, event: dict[str, Any]) -> None:
        """
        追加一行 JSON。

        使用 "a" 模式打开，写入一行后立即 flush：
        - 进程崩溃时最多丢最后几行；
        - 不需要维护长期打开的文件句柄。
        """
        path = self._path_for(session_id)
        # 统一补上时间戳和 session_id，调用方不需要重复传。
        # **event 放在后面：如果事件里显式带 ts/session_id，会被保留。
        payload = {
            "ts": self._clock().isoformat(timespec="milliseconds"),
            "session_id": session_id,
            **event,
        }
        # separators 去掉多余空格，一行更紧凑；ensure_ascii=False 保留中文。
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            # Trace 是“尽力而为”的可观测性数据：
            # flush 保证进入操作系统缓冲区，但不做 fsync，避免拖慢主流程。
            handle.flush()

        logger.info(
            "trace session=%s turn=%s event=%s",
            session_id,
            event.get("turn_id"),
            event.get("event"),
        )

    def _path_for(self, session_id: str) -> Path:
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(f"非法的会话 id：{session_id!r}")
        return self._dir / f"{session_id}.jsonl"

    def _short(self, value: Any, limit: int | None = None) -> str:
        """
        把任意值转成短字符串，避免 Trace 被超长内容撑爆。

        规则：
        - None -> 空字符串；
        - 字符串原样处理；
        - 字典/列表转 JSON；
        - 其他类型用 str() 兜底。
        """
        if value is None:
            return ""
        if isinstance(value, str):
            text = value
        elif isinstance(value, Mapping) or isinstance(value, (list, tuple)):
            # default=str 保证遇到不可序列化对象时也不会抛异常。
            text = json.dumps(value, ensure_ascii=False, default=str)
        else:
            text = str(value)

        max_chars = limit or self._max_field_chars
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "…"

    @staticmethod
    def _parsed_summary(parsed: ParseResult | None) -> dict[str, Any] | None:
        """
        把解析结果压缩成适合落盘的摘要。

        这里不保存完整 thought：
        - thought 只对当前轮决策有价值；
        - 完整保存会让 Trace 迅速变大。
        """
        if parsed is None:
            return None
        if isinstance(parsed, ToolCall):
            return {
                "type": "tool_call",
                "action": parsed.action,
                "input": parsed.input,
            }
        if isinstance(parsed, FinalAnswer):
            return {
                "type": "final",
                "final": parsed.final[:500],
            }
        if isinstance(parsed, ParseError):
            return {
                "type": "parse_error",
                "reason": parsed.reason,
            }
        return None
