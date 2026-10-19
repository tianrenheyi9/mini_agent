"""
会话数据模型。

一个 Session 包含：

- 基本信息：id、标题、创建/更新时间；
- 记忆：summary + summary_until_msg_id（压缩游标）；
- 状态：todos（待办）；
- 历史：messages（完整消息，供 UI / Trace / 上下文构建）；
- 统计：stats（LLM/工具/压缩次数与 token）。

消息使用 `msg_id = m_1, m_2, ...` 单调递增。
压缩游标依赖它来判断“哪些消息已经被摘要覆盖”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

DEFAULT_TITLE = "新会话"

# 历史轮次进入上下文时只保留这两类消息：
# - user：用户原话；
# - final：Agent 的最终回答。
# tool_call / tool_observation / repair / raw 只用于当前轮推理、UI 和 Trace。
HISTORY_KINDS = frozenset({"user", "final"})


def now_iso(now: datetime | None = None) -> str:
    """返回带时区的 ISO 时间字符串，精确到秒。"""
    return (now or datetime.now().astimezone()).isoformat(timespec="seconds")


@dataclass
class TurnStats:
    """单轮对话的统计，避免和 Session 的累计值混在一起。"""

    llm_calls: int = 0
    tool_calls: int = 0
    compression_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    compression_prompt_tokens: int = 0
    compression_completion_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "compression_calls": self.compression_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "compression_prompt_tokens": self.compression_prompt_tokens,
            "compression_completion_tokens": self.compression_completion_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TurnStats":
        if not isinstance(data, dict):
            return cls()
        return cls(
            llm_calls=_int(data.get("llm_calls")),
            tool_calls=_int(data.get("tool_calls")),
            compression_calls=_int(data.get("compression_calls")),
            prompt_tokens=_int(data.get("prompt_tokens")),
            completion_tokens=_int(data.get("completion_tokens")),
            compression_prompt_tokens=_int(
                data.get("compression_prompt_tokens")
            ),
            compression_completion_tokens=_int(
                data.get("compression_completion_tokens")
            ),
        )


@dataclass
class SessionStats(TurnStats):
    """Session 生命周期内的累计统计。"""

    def add_turn(self, turn: TurnStats) -> None:
        self.llm_calls += turn.llm_calls
        self.tool_calls += turn.tool_calls
        self.compression_calls += turn.compression_calls
        self.prompt_tokens += turn.prompt_tokens
        self.completion_tokens += turn.completion_tokens
        self.compression_prompt_tokens += turn.compression_prompt_tokens
        self.compression_completion_tokens += turn.compression_completion_tokens


@dataclass
class Message:
    """一条会话消息。"""

    msg_id: str
    role: str  # user / assistant
    kind: str  # user / final / tool_call / tool_observation / repair / raw
    content: str
    turn_id: str
    created_at: str
    thought: str = ""
    tool: str | None = None
    tool_input: dict[str, Any] | None = None
    status: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def visible_in_history(self) -> bool:
        """是否应该出现在后续轮次的历史窗口中。"""
        return self.kind in HISTORY_KINDS

    def to_llm_message(self) -> dict[str, str]:
        """转换成 OpenAI messages 数组中的一项。"""
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "msg_id": self.msg_id,
            "role": self.role,
            "kind": self.kind,
            "content": self.content,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
        }
        if self.thought:
            data["thought"] = self.thought
        if self.tool is not None:
            data["tool"] = self.tool
        if self.tool_input is not None:
            data["tool_input"] = self.tool_input
        if self.status is not None:
            data["status"] = self.status
        if self.meta:
            data["meta"] = self.meta
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            msg_id=str(data.get("msg_id", "")),
            role=str(data.get("role", "user")),
            kind=str(data.get("kind", "user")),
            content=str(data.get("content", "")),
            turn_id=str(data.get("turn_id", "")),
            created_at=str(data.get("created_at", "")),
            thought=str(data.get("thought", "") or ""),
            tool=data.get("tool"),
            tool_input=data.get("tool_input"),
            status=data.get("status"),
            meta=data.get("meta") if isinstance(data.get("meta"), dict) else {},
        )


@dataclass
class Session:
    """一个独立会话。"""

    id: str
    title: str = DEFAULT_TITLE
    created_at: str = ""
    updated_at: str = ""
    summary: str = ""
    summary_until_msg_id: str | None = None
    todos: list[dict[str, Any]] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    next_msg_seq: int = 1
    stats: SessionStats = field(default_factory=SessionStats)

    def append_message(
        self,
        *,
        role: str,
        kind: str,
        content: str,
        turn_id: str,
        created_at: str | None = None,
        thought: str = "",
        tool: str | None = None,
        tool_input: dict[str, Any] | None = None,
        status: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Message:
        """
        追加一条消息，并分配单调递增的 msg_id。

        msg_id 不复用，保证压缩游标不会指错消息。
        """
        timestamp = created_at or now_iso()
        if not self.created_at:
            self.created_at = timestamp

        message = Message(
            msg_id=f"m_{self.next_msg_seq}",
            role=role,
            kind=kind,
            content=content,
            turn_id=turn_id,
            created_at=timestamp,
            thought=thought,
            tool=tool,
            tool_input=tool_input,
            status=status,
            meta=meta or {},
        )
        self.next_msg_seq += 1
        self.messages.append(message)
        self.updated_at = timestamp

        # 第一条真正的用户消息用来生成标题；
        # repair 消息虽然也是 user 角色，但不参与命名。
        if kind == "user" and self.title == DEFAULT_TITLE:
            self.title = _title_from(content)

        return message

    def append_user(
        self,
        content: str,
        turn_id: str,
        *,
        created_at: str | None = None,
        kind: str = "user",
    ) -> Message:
        return self.append_message(
            role="user",
            kind=kind,
            content=content,
            turn_id=turn_id,
            created_at=created_at,
        )

    def append_assistant_raw(
        self,
        raw_content: str,
        turn_id: str,
        *,
        created_at: str | None = None,
    ) -> Message:
        """保存模型未通过协议校验的原始输出，供纠错与排障。"""
        return self.append_message(
            role="assistant",
            kind="raw",
            content=raw_content,
            turn_id=turn_id,
            created_at=created_at,
        )

    def append_assistant_tool_call(
        self,
        *,
        thought: str,
        action: str,
        tool_input: dict[str, Any],
        raw_content: str,
        turn_id: str,
        created_at: str | None = None,
    ) -> Message:
        """保存模型的工具调用决策（原始 JSON + 解析后的字段）。"""
        return self.append_message(
            role="assistant",
            kind="tool_call",
            content=raw_content,
            turn_id=turn_id,
            created_at=created_at,
            thought=thought,
            tool=action,
            tool_input=tool_input,
        )

    def append_tool_observation(
        self,
        *,
        tool: str,
        status: str,
        content: str,
        turn_id: str,
        created_at: str | None = None,
    ) -> Message:
        """
        保存工具执行结果。

        注意：role 是 user，不是 tool。
        自定义协议没有原生 tool_call_id，不能伪造 role="tool" 消息。
        """
        return self.append_message(
            role="user",
            kind="tool_observation",
            content=content,
            turn_id=turn_id,
            created_at=created_at,
            tool=tool,
            status=status,
        )

    def append_assistant_final(
        self,
        *,
        thought: str,
        final: str,
        turn_id: str,
        created_at: str | None = None,
    ) -> Message:
        return self.append_message(
            role="assistant",
            kind="final",
            content=final,
            turn_id=turn_id,
            created_at=created_at,
            thought=thought,
        )

    def mark_summarized_until(
        self,
        msg_id: str,
        *,
        updated_at: str | None = None,
    ) -> None:
        """推进摘要游标；游标只能指向真实存在的消息。"""
        if self.message_index(msg_id) is None:
            raise ValueError(f"消息不存在：{msg_id!r}")
        self.summary_until_msg_id = msg_id
        self.updated_at = updated_at or now_iso()

    def message_index(self, msg_id: str) -> int | None:
        """返回 msg_id 在 messages 中的下标；找不到返回 None。"""
        for index, message in enumerate(self.messages):
            if message.msg_id == msg_id:
                return index
        return None

    def messages_after_cursor(self) -> list[Message]:
        """返回摘要游标之后的所有消息。"""
        if self.summary_until_msg_id is None:
            return list(self.messages)
        index = self.message_index(self.summary_until_msg_id)
        if index is None:
            return list(self.messages)
        return self.messages[index + 1 :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "summary_until_msg_id": self.summary_until_msg_id,
            "stats": self.stats.to_dict(),
            "todos": self.todos,
            "messages": [message.to_dict() for message in self.messages],
            "next_msg_seq": self.next_msg_seq,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        messages = [
            Message.from_dict(item)
            for item in data.get("messages", [])
            if isinstance(item, dict)
        ]
        next_msg_seq = data.get("next_msg_seq")
        if not isinstance(next_msg_seq, int) or next_msg_seq < 1:
            next_msg_seq = _next_seq_from_messages(messages)

        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title") or DEFAULT_TITLE),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            summary=str(data.get("summary", "") or ""),
            summary_until_msg_id=data.get("summary_until_msg_id"),
            todos=[
                todo
                for todo in data.get("todos", [])
                if isinstance(todo, dict)
            ],
            messages=messages,
            next_msg_seq=next_msg_seq,
            stats=SessionStats.from_dict(data.get("stats")),
        )


def _title_from(content: str) -> str:
    """取首条用户消息前 20 个字符作为标题。"""
    normalized = " ".join(content.split())
    return normalized[:20] or DEFAULT_TITLE


def _next_seq_from_messages(messages: list[Message]) -> int:
    """兼容旧数据：没有 next_msg_seq 时，从 m_n 推断。"""
    max_seq = 0
    for message in messages:
        if message.msg_id.startswith("m_") and message.msg_id[2:].isdigit():
            max_seq = max(max_seq, int(message.msg_id[2:]))
    return max_seq + 1


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
