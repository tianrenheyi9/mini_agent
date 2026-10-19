"""
Agent Loop：整个 Runtime 的核心。

一轮用户输入的完整流程：

1. 取得会话锁，追加 user 消息；
2. 需要时压缩历史（由 ContextManager 决定）；
3. 循环：
   - 组装上下文；
   - 调用 LLM；
   - 解析 thought / action / final；
   - final：结束本轮；
   - action：执行工具，把 observation 以 user 角色回填，继续循环；
   - 解析失败：追加纠错消息，有限次重试；
4. 触达任一上限时返回兜底回复；
5. 统一写 final、累加统计、记录 Trace。

重要的消息格式约定：
- 模型的工具调用是一条 assistant 消息，内容是它原始输出的 JSON；
- 工具结果是一条 user 消息，带 [工具执行结果] 标记；
- 不使用 role="tool"，因为自定义协议没有原生 tool_call_id。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.agent.context import (
    CompressionResult,
    ContextBudgetExceeded,
    ContextManager,
)
from app.agent.llm import LLMClient, LLMConfigError, LLMError
from app.agent.parser import FinalAnswer, ParseError, ToolCall, parse
from app.agent.prompt import build_repair_message
from app.config import Settings, get_settings
from app.session.manager import SessionManager
from app.session.models import Session, TurnStats, now_iso
from app.tools import build_default_registry
from app.tools.base import ToolContext, ToolResult
from app.tools.registry import ToolRegistry
from app.trace import TraceLogger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnResult:
    """
    一轮执行的返回结果。

    steps 是给 UI 展示的简化时间线：
    - {"type": "parse_error", ...}
    - {"type": "tool", "tool": "...", "status": "ok/error", ...}
    """

    session_id: str
    turn_id: str
    reply: str
    thought: str
    steps: list[dict[str, Any]]
    stats: dict[str, int]
    status: str
    # ok / max_tool_calls / max_llm_calls / parse_error / llm_error / context_overflow
    compression: dict[str, Any] | None = None


class AgentLoop:
    """负责一轮对话的编排，不关心 HTTP / UI。"""

    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMClient,
        registry: ToolRegistry,
        sessions: SessionManager,
        context: ContextManager,
        trace: TraceLogger,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        # 依赖全部注入：生产环境由 build_agent_loop 装配，
        # 测试可以注入 FakeLLM、临时目录和固定时钟。
        self._settings = settings
        self._llm = llm
        self._registry = registry
        self._sessions = sessions
        self._context = context
        self._trace = trace
        self._clock = clock or (lambda: datetime.now().astimezone())

    def _now(self) -> datetime:
        """当前时间，用于 System Prompt 和 ToolContext。"""
        return self._clock()

    @property
    def sessions(self) -> SessionManager:
        """API 层通过它做会话 CRUD，保证和 Loop 使用同一个实例。"""
        return self._sessions

    @property
    def trace(self) -> TraceLogger:
        """API 层通过它读取 Trace。"""
        return self._trace

    async def aclose(self) -> None:
        """关闭底层 LLM 客户端；供 FastAPI lifespan 调用。"""
        await self._llm.aclose()

    def _now_iso(self) -> str:
        """当前时间的 ISO 字符串，用于消息时间戳。"""
        return now_iso(self._clock())

    async def run_turn(self, session_id: str, user_text: str) -> TurnResult:
        """
        执行一轮完整的 Agent 对话。

        调用方通常是 API 层；这里不处理 HTTP 状态码，
        只抛出 ValueError（空消息）或 SessionNotFound（会话不存在）。
        """
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("消息不能为空")
        user_text = user_text.strip()

        turn_id = f"u_{uuid4().hex[:12]}"
        turn_stats = TurnStats()
        steps: list[dict[str, Any]] = []
        started = time.perf_counter()
        repairs = 0
        tool_limit_notified = False
        compression: CompressionResult | None = None

        # 同一会话的轮次串行执行；不同会话互不阻塞。
        async with self._sessions.locked(session_id) as session:
            self._trace.log_turn_start(
                session_id=session.id,
                turn_id=turn_id,
            )
            session.append_user(
                user_text,
                turn_id,
                created_at=self._now_iso(),
            )

            # ---------- 1. 可选的上下文压缩 ----------
            try:
                compression = await self._context.maybe_compress(
                    session,
                    turn_id=turn_id,
                    tools=self._registry.tools(),
                    turn_stats=turn_stats,
                    now=self._now(),
                )
            except Exception:
                # 压缩是优化项，失败不应该中断对话；
                # 后续 build() 仍会做硬截断兜底。
                logger.exception("Context compression failed; continuing")
                compression = None

            if compression is not None and compression.compressed:
                self._trace.log_compression(
                    session_id=session.id,
                    turn_id=turn_id,
                    result=compression,
                )

            # ---------- 2. 主循环 ----------
            # 循环上限是“总 LLM 调用次数”，不是工具次数。
            # 工具次数由 turn_stats.tool_calls 单独限制。
            for step in range(self._settings.max_llm_calls_per_turn):
                # 第二道预算：协议纠错和工具步骤都会消耗 LLM 调用。
                if (
                    turn_stats.llm_calls
                    >= self._settings.max_llm_calls_per_turn
                ):
                    return self._finish(
                        session=session,
                        turn_id=turn_id,
                        reply="本轮 LLM 调用次数已达上限，先给你阶段性结果。",
                        thought="",
                        steps=steps,
                        turn_stats=turn_stats,
                        started=started,
                        status="max_llm_calls",
                        compression=compression,
                    )

                # build() 在“System + 当前轮”已经超出预算时会抛出异常。
                # 这是明确的可恢复错误：不要继续请求一个注定超限的 Prompt。
                try:
                    messages = self._context.build(
                        session,
                        turn_id=turn_id,
                        tools=self._registry.tools(),
                        now=self._now(),
                    )
                except ContextBudgetExceeded:
                    return self._finish(
                        session=session,
                        turn_id=turn_id,
                        reply=(
                            "本轮上下文超出预算（可能来自较长的工具结果），"
                            "请重试或缩小任务范围。"
                        ),
                        thought="",
                        steps=steps,
                        turn_stats=turn_stats,
                        started=started,
                        status="context_overflow",
                        compression=compression,
                    )

                prompt_tokens_est = self._context.estimate_tokens(messages)

                # 先计数再调用：失败的调用也占用了预算和一次尝试机会。
                turn_stats.llm_calls += 1
                try:
                    response = await self._llm.chat(messages)
                except LLMError as exc:
                    # 模型不可用是预期内的外部故障：给用户可读回复，
                    # 完整异常只进日志，不把堆栈/密钥泄漏到对话里。
                    logger.warning("LLM call failed: %s", exc)
                    if isinstance(exc, LLMConfigError):
                        reply = (
                            "模型未配置，请检查 LLM_BASE_URL、"
                            "LLM_API_KEY 和 LLM_MODEL。"
                        )
                    else:
                        reply = "模型服务暂时不可用，请稍后重试。"
                    return self._finish(
                        session=session,
                        turn_id=turn_id,
                        reply=reply,
                        thought="",
                        steps=steps,
                        turn_stats=turn_stats,
                        started=started,
                        status="llm_error",
                        compression=compression,
                    )

                turn_stats.prompt_tokens += response.prompt_tokens or 0
                turn_stats.completion_tokens += response.completion_tokens or 0

                parsed = parse(response.content)
                self._trace.log_llm_call(
                    session_id=session.id,
                    turn_id=turn_id,
                    step=step,
                    response=response,
                    parsed=parsed,
                    prompt_tokens_est=prompt_tokens_est,
                )

                # 分支 A：模型决定直接回复。
                if isinstance(parsed, FinalAnswer):
                    return self._finish(
                        session=session,
                        turn_id=turn_id,
                        reply=parsed.final,
                        thought=parsed.thought,
                        steps=steps,
                        turn_stats=turn_stats,
                        started=started,
                        status="ok",
                        compression=compression,
                    )

                # 分支 B：模型输出不符合协议，有限次纠错。
                if isinstance(parsed, ParseError):
                    steps.append(
                        {
                            "type": "parse_error",
                            "step": step,
                            "reason": parsed.reason,
                        }
                    )
                    self._trace.log_parse_error(
                        session_id=session.id,
                        turn_id=turn_id,
                        step=step,
                        reason=parsed.reason,
                        raw_output=response.content,
                    )

                    if repairs >= self._settings.max_parse_repairs:
                        return self._finish(
                            session=session,
                            turn_id=turn_id,
                            reply="模型输出格式异常，请重试。",
                            thought="",
                            steps=steps,
                            turn_stats=turn_stats,
                            started=started,
                            status="parse_error",
                            compression=compression,
                        )

                    repairs += 1
                    # 保留原始输出，再追加明确的纠错提示。
                    session.append_assistant_raw(
                        response.content,
                        turn_id,
                        created_at=self._now_iso(),
                    )
                    session.append_user(
                        build_repair_message(
                            response.content,
                            parsed.reason,
                        ),
                        turn_id,
                        created_at=self._now_iso(),
                        kind="repair",
                    )
                    continue

                # 分支 C：调用工具。
                # 类型上走到这里只剩 ToolCall。
                tool_call: ToolCall = parsed

                # 工具次数是独立预算：超过上限后不再执行，
                # 而是给模型一次“基于已有信息收尾”的机会。
                if turn_stats.tool_calls >= self._settings.max_tool_calls:
                    if tool_limit_notified:
                        # 已经提醒过一次仍然继续要工具，直接兜底结束。
                        return self._finish(
                            session=session,
                            turn_id=turn_id,
                            reply=(
                                "已达到单轮最大工具调用次数"
                                f"（{self._settings.max_tool_calls}），"
                                "先给你阶段性结果。"
                            ),
                            thought="",
                            steps=steps,
                            turn_stats=turn_stats,
                            started=started,
                            status="max_tool_calls",
                            compression=compression,
                        )

                    tool_limit_notified = True
                    steps.append(
                        {
                            "type": "tool_limit",
                            "step": step,
                            "limit": self._settings.max_tool_calls,
                        }
                    )
                    session.append_user(
                        "工具调用次数已达上限"
                        f"（{self._settings.max_tool_calls} 次）。"
                        "请不要再调用工具，直接基于已有信息输出 final。",
                        turn_id,
                        created_at=self._now_iso(),
                        kind="repair",
                    )
                    continue

                session.append_assistant_tool_call(
                    thought=tool_call.thought,
                    action=tool_call.action,
                    tool_input=tool_call.input,
                    raw_content=response.content,
                    turn_id=turn_id,
                    created_at=self._now_iso(),
                )

                result = await self._registry.execute(
                    tool_call.action,
                    tool_call.input,
                    ToolContext(
                        session_id=session.id,
                        todos=session.todos,
                        now=self._now(),
                    ),
                )
                turn_stats.tool_calls += 1

                # observation 以 user 角色回填，不使用 role="tool"。
                # 这样既符合自定义协议，又不需要伪造 tool_call_id。
                observation = format_tool_observation(
                    tool_call.action,
                    tool_call.input,
                    result,
                    max_chars=self._settings.tool_observation_max_chars,
                )
                session.append_tool_observation(
                    tool=tool_call.action,
                    status="ok" if result.ok else "error",
                    content=observation,
                    turn_id=turn_id,
                    created_at=self._now_iso(),
                )

                steps.append(
                    {
                        "type": "tool",
                        "step": step,
                        "tool": tool_call.action,
                        "input": tool_call.input,
                        "status": "ok" if result.ok else "error",
                        "result": result.data if result.ok else None,
                        "error": result.error,
                        "duration_ms": result.duration_ms,
                        "thought": tool_call.thought,
                    }
                )
                self._trace.log_tool_call(
                    session_id=session.id,
                    turn_id=turn_id,
                    step=step,
                    tool=tool_call.action,
                    tool_input=tool_call.input,
                    result=result,
                )

                # 工具结果可能很长；执行后再次检查旧历史是否需要压缩。
                # 当前轮本身不会被摘要，避免丢掉正在推理的 observation。
                try:
                    mid_compression = await self._context.maybe_compress(
                        session,
                        turn_id=turn_id,
                        tools=self._registry.tools(),
                        turn_stats=turn_stats,
                        now=self._now(),
                    )
                except Exception:
                    logger.exception("Mid-turn compression failed; continuing")
                    mid_compression = None

                if mid_compression is not None and mid_compression.compressed:
                    compression = mid_compression
                    self._trace.log_compression(
                        session_id=session.id,
                        turn_id=turn_id,
                        result=mid_compression,
                    )

            # ---------- 3. 达到 LLM 调用上限 ----------
            return self._finish(
                session=session,
                turn_id=turn_id,
                reply="本轮 LLM 调用次数已达上限，先给你阶段性结果。",
                thought="",
                steps=steps,
                turn_stats=turn_stats,
                started=started,
                status="max_llm_calls",
                compression=compression,
            )

    def _finish(
        self,
        *,
        session: Session,
        turn_id: str,
        reply: str,
        thought: str,
        steps: list[dict[str, Any]],
        turn_stats: TurnStats,
        started: float,
        status: str,
        compression: CompressionResult | None,
    ) -> TurnResult:
        """
        统一收尾：
        1. 写入 final 消息；
        2. 把本轮统计累加到 Session；
        3. 写 turn_end Trace；
        4. 返回结构化结果。

        所有结束分支都走这里，避免漏写 final 或漏记统计。
        """
        session.append_assistant_final(
            thought=thought,
            final=reply,
            turn_id=turn_id,
            created_at=self._now_iso(),
        )
        session.stats.add_turn(turn_stats)

        latency_ms = int((time.perf_counter() - started) * 1000)
        self._trace.log_turn_end(
            session_id=session.id,
            turn_id=turn_id,
            status=status,
            stats=turn_stats,
            latency_ms=latency_ms,
            reply_preview=reply,
        )

        return TurnResult(
            session_id=session.id,
            turn_id=turn_id,
            reply=reply,
            thought=thought,
            steps=steps,
            stats=turn_stats.to_dict(),
            status=status,
            compression=_compression_dict(compression),
        )


def format_tool_observation(
    tool: str,
    tool_input: dict[str, Any],
    result: ToolResult,
    *,
    max_chars: int = 2000,
) -> str:
    """
    构造回填给模型的 observation 文本。

    显式的 [工具执行结果] 标记很重要：
    模型需要区分“这是工具返回值”，而不是新的用户需求。
    """
    input_text = _preview(tool_input, max_chars)

    if result.ok:
        return (
            "[工具执行结果]\n"
            f"tool: {tool}\n"
            "status: ok\n"
            f"input: {input_text}\n"
            f"result: {_preview(result.data, max_chars)}\n\n"
            "请继续：需要更多信息就调用工具，否则输出 final。"
        )

    return (
        "[工具执行结果]\n"
        f"tool: {tool}\n"
        "status: error\n"
        f"input: {input_text}\n"
        f"error: {result.error or '未知错误'}\n"
        "请修正参数后重试，或向用户说明原因。"
    )


def _preview(value: Any, max_chars: int) -> str:
    """把工具输入/结果转成短 JSON，避免 observation 无限膨胀。"""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _compression_dict(
    result: CompressionResult | None,
) -> dict[str, Any] | None:
    """把压缩结果转成 API/UI 易用的字典。"""
    if result is None:
        return None
    return {
        "compressed": result.compressed,
        "summarized_messages": result.summarized_messages,
        "summary_until_msg_id": result.summary_until_msg_id,
        "before_tokens": result.before_tokens,
        "after_tokens": result.after_tokens,
        "fallback": result.fallback,
    }


def build_agent_loop(settings: Settings | None = None) -> AgentLoop:
    """
    生产环境的默认装配入口。

    API 层只需要调用一次，然后复用同一个 AgentLoop 实例。
    """
    settings = settings or get_settings()
    llm = LLMClient(settings)
    return AgentLoop(
        settings=settings,
        llm=llm,
        registry=build_default_registry(
            timeout_seconds=settings.tool_timeout_seconds
        ),
        sessions=SessionManager(settings),
        context=ContextManager(settings, llm),
        trace=TraceLogger(
            settings.traces_dir,
            max_field_chars=settings.trace_max_chars,
        ),
    )
