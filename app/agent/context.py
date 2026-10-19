"""
上下文构建与压缩。

每轮调用 LLM 前，ContextManager 负责回答三个问题：

1. 哪些消息应该进入 Prompt？
   - System Prompt（协议 + 工具目录 + 摘要 + 待办 + 当前时间）
   - 最近 K 条已完成轮次（只保留 user 原文和 assistant final）
   - 当前轮的完整中间步骤（tool_call / tool_observation / repair）

2. Prompt 会不会太长？
   - 用字符加权估算 token，再乘安全系数；
   - 超过预算时先压缩旧消息，仍然超就硬截断最旧的历史。

3. 压缩哪些消息？
   - 只压缩 summary_until_msg_id 之后、最近窗口之前的新消息；
   - 游标只前进不后退，避免重复摘要。
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.agent.llm import LLMClient, LLMError
from app.agent.prompt import ToolLike, build_system_prompt
from app.config import Settings
from app.session.models import Message, Session, TurnStats, now_iso

logger = logging.getLogger(__name__)

# 中日韩字符（含中文、日文假名、韩文）通常接近 1 token/字；
# 英文和其他字符大约 4 字符/token。
_CJK_RE = re.compile(
    r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]"
)


@dataclass(frozen=True)
class CompressionResult:
    """一次压缩调用的结果，供 Trace、UI 和测试使用。"""

    compressed: bool  # 是否真的执行了压缩
    summarized_messages: int = 0  # 本次合并了多少条消息
    summary_until_msg_id: str | None = None  # 压缩后的游标
    before_tokens: int = 0  # 压缩前估算 token
    after_tokens: int = 0  # 压缩后估算 token
    fallback: bool = False  # 是否走了“摘要失败”的降级路径


class ContextManager:
    """上下文组装、token 估算与增量摘要。"""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._clock = clock or (lambda: datetime.now().astimezone())

    # ------------------------------------------------------------------
    # token 估算
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_text_tokens(text: str) -> int:
        """
        估算一段文本的 token 数。

        这是近似值，不追求和某个厂商 tokenizer 完全一致：
        - 中日韩字符按 1 token/字；
        - 其他字符按 4 字符/token；
        - 向上取整，宁可略微高估。
        """
        if not text:
            return 0
        cjk_count = len(_CJK_RE.findall(text))
        other_count = len(text) - cjk_count
        return math.ceil(cjk_count + other_count / 4)

    def estimate_messages_tokens(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> int:
        """估算一组 OpenAI messages 的 token 数。"""
        total = 0
        for message in messages:
            content = str(message.get("content", ""))
            # 每条消息还有 role、分隔符等固定开销，粗略按 4 计。
            total += self.estimate_text_tokens(content) + 4
        return total

    def estimate_tokens(self, messages: Sequence[dict[str, Any]]) -> int:
        """估算并乘以安全系数，用于和上下文预算比较。"""
        raw = self.estimate_messages_tokens(messages)
        return math.ceil(raw * self._settings.token_safety_factor)

    # ------------------------------------------------------------------
    # 上下文组装
    # ------------------------------------------------------------------

    def build(
        self,
        session: Session,
        *,
        turn_id: str,
        tools: Sequence[ToolLike],
        now: datetime | None = None,
        trim: bool = True,
    ) -> list[dict[str, str]]:
        """
        组装本轮发送给 LLM 的 messages。

        `trim=True` 时，如果仍然超过预算，会从最旧的历史消息开始丢弃。
        这只影响本次 Prompt，不修改 Session 中的消息。
        """
        current_time = now or self._clock()

        # 1. System Prompt：协议、工具目录、摘要、待办、时间。
        system_prompt = build_system_prompt(
            tools,
            summary=session.summary,
            todos=session.todos,
            now=current_time,
        )

        # 2. 最近 K 条已完成对话（只含 user / final）。
        history = self._recent_history(session, turn_id)

        # 3. 当前轮的全部消息：
        #    用户输入、工具调用、observation、协议纠错都要保留，
        #    否则模型无法完成多步工具串联。
        current_turn = [
            message
            for message in session.messages
            if message.turn_id == turn_id
        ]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(message.to_llm_message() for message in history)
        messages.extend(message.to_llm_message() for message in current_turn)

        # 4. 最后一道保险：仍然超预算就从最旧历史开始删。
        if trim:
            messages = self._trim_history_to_budget(
                messages,
                history_count=len(history),
            )
        return messages

    def _recent_history(
        self,
        session: Session,
        turn_id: str,
    ) -> list[Message]:
        """
        选择最近 K 条已完成消息。

        规则：
        - 只考虑摘要游标之后、且不是当前轮的消息；
        - 只保留 user / final，历史 thought 和工具结果不进入上下文；
        - 如果切片以 final 开头，把配对的 user 消息补回来，避免上下文断裂。
        """
        keep = max(0, self._settings.keep_recent_messages)
        if keep == 0:
            # 配置为 0 表示不保留任何历史，只发送当前轮。
            return []

        # 先从“游标之后”取，避免已压缩的消息重新进入窗口。
        visible = [
            message
            for message in session.messages_after_cursor()
            if message.turn_id != turn_id and message.visible_in_history
        ]
        recent = visible[-keep:]

        # 如果切片刚好从 final 开始，它对应的 user 问题会被截掉。
        # 这里把前一条补回来，保持问答成对。
        if recent and recent[0].kind == "final":
            index = next(
                (
                    i
                    for i, message in enumerate(visible)
                    if message.msg_id == recent[0].msg_id
                ),
                0,
            )
            if index > 0:
                recent = visible[index - 1 :]
        return recent

    def _trim_history_to_budget(
        self,
        messages: list[dict[str, str]],
        *,
        history_count: int,
    ) -> list[dict[str, str]]:
        """
        硬截断兜底：从最旧的历史消息开始删，直到回到预算内。

        messages[0] 是 System Prompt，必须保留；
        当前轮消息在数组尾部，也不删除。
        """
        while (
            history_count > 0
            and self.estimate_tokens(messages)
            > self._settings.context_token_budget
        ):
            # messages[0] 是 System，永远保留；
            # messages[1] 是当前最旧的历史消息。
            del messages[1]
            history_count -= 1
        return messages

    def needs_compression(
        self,
        session: Session,
        *,
        turn_id: str,
        tools: Sequence[ToolLike],
        now: datetime | None = None,
    ) -> bool:
        """判断是否需要压缩：token 超预算，或未压缩消息条数超上限。"""
        # 用 trim=False 估算真实压力，避免硬截断把“需要压缩”的信号藏起来。
        messages = self.build(
            session,
            turn_id=turn_id,
            tools=tools,
            now=now,
            trim=False,
        )
        if self.estimate_tokens(messages) > self._settings.context_token_budget:
            return True

        # 条数上限只看“可压缩的旧消息”，最近窗口本身是有界的。
        uncompressed = self._compression_candidates(session, turn_id)
        return len(uncompressed) > self._settings.max_history_messages

    # ------------------------------------------------------------------
    # 压缩
    # ------------------------------------------------------------------

    async def maybe_compress(
        self,
        session: Session,
        *,
        turn_id: str,
        tools: Sequence[ToolLike],
        turn_stats: TurnStats,
        now: datetime | None = None,
    ) -> CompressionResult:
        """需要时才压缩；否则返回 compressed=False。"""
        # 决策（needs_compression）和执行（compress）分开：
        # 循环可以在需要时调用，测试也能单独测其中一半。
        if not self.needs_compression(
            session,
            turn_id=turn_id,
            tools=tools,
            now=now,
        ):
            return CompressionResult(compressed=False)

        return await self.compress(
            session,
            turn_id=turn_id,
            tools=tools,
            turn_stats=turn_stats,
            now=now,
        )

    async def compress(
        self,
        session: Session,
        *,
        turn_id: str,
        tools: Sequence[ToolLike],
        turn_stats: TurnStats,
        now: datetime | None = None,
    ) -> CompressionResult:
        """
        把游标之后、最近窗口之前的消息合并进摘要。

        失败时降级为“旧摘要 + 最后两条可见消息”，不中断对话。
        """
        # 每轮压缩次数有限，防止异常情况下反复调用 LLM。
        if (
            turn_stats.compression_calls
            >= self._settings.max_compression_calls_per_turn
        ):
            return CompressionResult(compressed=False)

        # 第 1 步：找出本次真正要合并的旧消息。
        candidates = self._compression_candidates(session, turn_id)

        # 记录压缩前的估算值，方便 Trace 对比效果。
        before_tokens = self.estimate_tokens(
            self.build(
                session,
                turn_id=turn_id,
                tools=tools,
                now=now,
                trim=False,
            )
        )

        # 没有可压缩的新消息：交给 build 的硬截断兜底。
        if not candidates:
            return CompressionResult(
                compressed=False,
                before_tokens=before_tokens,
                after_tokens=before_tokens,
            )

        # 第 2 步：把候选消息格式化成紧凑文本，并拼出摘要 Prompt。
        formatted = self._format_for_summary(candidates)
        prompt = self._summary_prompt(session.summary, formatted)
        fallback = False

        # 第 3 步：调用 LLM 生成新摘要。
        try:
            response = await self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self._settings.summary_max_tokens,
            )
            summary = response.content.strip()
            turn_stats.compression_prompt_tokens += (
                response.prompt_tokens
                if response.prompt_tokens is not None
                else self.estimate_text_tokens(prompt)
            )
            turn_stats.compression_completion_tokens += (
                response.completion_tokens
                if response.completion_tokens is not None
                else self.estimate_text_tokens(summary)
            )
        except LLMError as exc:
            # 摘要失败不能影响主对话：降级为本地拼接的兜底摘要。
            logger.warning("Summary LLM failed, using fallback: %s", exc)
            summary = self._fallback_summary(session.summary, candidates)
            fallback = True
            turn_stats.compression_prompt_tokens += self.estimate_text_tokens(
                prompt
            )

        # 第 4 步：截断到配置上限，写回摘要并推进游标。
        summary = _truncate(summary, self._settings.summary_max_chars)
        session.summary = summary
        session.mark_summarized_until(
            candidates[-1].msg_id,
            updated_at=now_iso(now or self._clock()),
        )
        turn_stats.compression_calls += 1

        # 第 5 步：重新估算压缩后的 Prompt，记录效果。
        after_tokens = self.estimate_tokens(
            self.build(
                session,
                turn_id=turn_id,
                tools=tools,
                now=now,
                trim=False,
            )
        )

        return CompressionResult(
            compressed=True,
            summarized_messages=len(candidates),
            summary_until_msg_id=session.summary_until_msg_id,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            fallback=fallback,
        )

    def _compression_candidates(
        self,
        session: Session,
        turn_id: str,
    ) -> list[Message]:
        """
        选出可以进入摘要的消息。

        必须是：
        - 摘要游标之后；
        - 不在最近窗口内；
        - 不属于当前轮。
        """
        recent = self._recent_history(session, turn_id)
        # 用 turn_id 判断边界，而不是 msg_id：
        # 同一轮里可能还有 tool_call / observation 等不可见消息，
        # 它们也应该和这一轮一起被保留，不进入本次摘要。
        recent_turn_ids = {message.turn_id for message in recent}

        candidates: list[Message] = []
        for message in session.messages_after_cursor():
            if message.turn_id == turn_id:
                # 当前轮正在执行，绝对不能摘要。
                continue
            # 遇到最近窗口所属的轮次就停止：
            # 这些轮次整体保留，不进入摘要。
            if message.turn_id in recent_turn_ids:
                break
            candidates.append(message)
        return candidates

    @staticmethod
    def _format_for_summary(messages: Sequence[Message]) -> str:
        """把消息压缩成适合放进摘要 Prompt 的紧凑文本。"""
        lines: list[str] = []
        for message in messages:
            # 只保留对“长期事实”有价值的内容；
            # 原始 JSON 和超长工具输出会做截断。
            if message.kind == "user":
                lines.append(f"用户：{message.content}")
            elif message.kind == "final":
                lines.append(f"助手：{message.content}")
            elif message.kind == "tool_call":
                tool_input = json.dumps(
                    message.tool_input or {},
                    ensure_ascii=False,
                )
                lines.append(f"调用工具 {message.tool}：{tool_input}")
            elif message.kind == "tool_observation":
                lines.append(
                    f"工具 {message.tool} 结果（{message.status}）："
                    f"{_truncate(message.content, 300)}"
                )
            elif message.kind == "repair":
                lines.append("（发生了一次协议纠错）")
            elif message.kind == "raw":
                lines.append(f"（无效输出）：{_truncate(message.content, 100)}")
        return "\n".join(lines)

    def _summary_prompt(self, old_summary: str, formatted: str) -> str:
        """
        构造摘要 Prompt。

        摘要是“记忆压缩”，不是对话，所以这里要求模型只输出正文，
        不走 JSON 协议，避免多一层解析失败风险。
        """
        return f"""你是一个对话摘要器。请把下面的对话压缩成不超过 {self._settings.summary_max_chars} 字的中文摘要。

必须保留：
1. 用户目标与偏好
2. 已确认的关键事实、数字、日期
3. 已执行的工具操作及其结论
4. 当前待办与未决问题

不要编造信息；如果旧摘要和新对话冲突，以新对话为准。
只输出摘要正文，不要输出标题或解释。

【旧摘要】
{old_summary.strip() or "（无）"}

【需要合并的新对话】
{formatted}
"""

    @staticmethod
    def _fallback_summary(
        old_summary: str,
        messages: Sequence[Message],
    ) -> str:
        """摘要 LLM 失败时，用旧摘要 + 最后两条可见消息兜底。"""
        lines: list[str] = []
        if old_summary.strip():
            # 旧摘要已经包含更早的历史，必须保留。
            lines.append(f"旧摘要：{old_summary.strip()}")

        # 取最后两条 user/final，尽量保留最近事实。
        anchors = [
            message for message in messages if message.visible_in_history
        ][-2:]
        for message in anchors:
            role = "用户" if message.kind == "user" else "助手"
            lines.append(f"{role}：{message.content}")

        return "\n".join(lines) or "（摘要失败，早期历史已截断）"


def _truncate(text: str, limit: int) -> str:
    """按字符数硬截断，超出时加省略号。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"
