"""ContextManager 的 token 估算、上下文组装与压缩测试。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agent.context import ContextManager
from app.agent.llm import LLMError
from app.config import Settings, get_settings
from app.session.models import Session, TurnStats
from tests.llm.fake_llm import FakeLLM

T0 = "2026-10-19T09:00:00+08:00"
T1 = "2026-10-19T09:01:00+08:00"
T2 = "2026-10-19T09:02:00+08:00"

FIXED_NOW = datetime(
    2026,
    10,
    19,
    9,
    0,
    0,
    tzinfo=timezone(timedelta(hours=8)),
)


@dataclass(frozen=True)
class ToolStub:
    name: str
    description: str
    parameters: dict[str, Any]


TOOLS = [
    ToolStub(
        name="weather",
        description="查询天气",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
]


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "context_token_budget": 6000,
        "token_safety_factor": 1.0,
        "summary_max_tokens": 128,
        "summary_max_chars": 200,
        "keep_recent_messages": 1,
        "max_history_messages": 40,
        "max_compression_calls_per_turn": 1,
    }
    values.update(overrides)
    return replace(get_settings(), **values)


def make_context(llm: FakeLLM, **overrides: Any) -> ContextManager:
    return ContextManager(
        make_settings(**overrides),
        llm,  # type: ignore[arg-type]
        clock=lambda: FIXED_NOW,
    )


def make_session() -> Session:
    """构造两轮历史 + 一个当前轮。"""
    session = Session(id="s_test", created_at=T0, updated_at=T0)
    session.summary = "旧摘要"
    session.todos.append(
        {"id": "t_1", "content": "买牛奶", "done": False, "created_at": T0}
    )

    # 第 1 轮：user -> tool_call -> observation -> final
    session.append_user("第一个问题", "u_1", created_at=T0)
    session.append_assistant_tool_call(
        thought="先查天气",
        action="weather",
        tool_input={"city": "北京"},
        raw_content='{"action":"weather","input":{"city":"北京"}}',
        turn_id="u_1",
        created_at=T0,
    )
    session.append_tool_observation(
        tool="weather",
        status="ok",
        content="[工具执行结果] 北京晴",
        turn_id="u_1",
        created_at=T0,
    )
    session.append_assistant_final(
        thought="完成",
        final="第一个回答",
        turn_id="u_1",
        created_at=T0,
    )

    # 第 2 轮：纯对话
    session.append_user("第二个问题", "u_2", created_at=T1)
    session.append_assistant_final(
        thought="完成",
        final="第二个回答",
        turn_id="u_2",
        created_at=T1,
    )

    # 当前轮
    session.append_user("当前问题", "u_3", created_at=T2)
    return session


def test_estimate_text_tokens() -> None:
    """中文按字、英文按 4 字符/token 估算。"""
    assert ContextManager.estimate_text_tokens("") == 0
    assert ContextManager.estimate_text_tokens("你好") == 2
    assert ContextManager.estimate_text_tokens("hello") == 2
    assert ContextManager.estimate_text_tokens("hello world") == 3


def test_estimate_tokens_applies_safety_factor() -> None:
    """安全系数会放大估算值。"""
    context = make_context(FakeLLM([]), token_safety_factor=2.0)
    messages = [{"role": "user", "content": "你好"}]

    # 原始估算：2 个中文字 + 4 固定开销 = 6；乘 2 = 12
    assert context.estimate_tokens(messages) == 12


def test_build_includes_history_current_turn_and_memory() -> None:
    """历史只保留最近一轮的 user/final；当前轮完整；System Prompt 含 memory。"""
    context = make_context(FakeLLM([]), keep_recent_messages=1)
    session = make_session()

    messages = context.build(
        session,
        turn_id="u_3",
        tools=TOOLS,
        now=FIXED_NOW,
    )

    assert messages[0]["role"] == "system"
    assert "旧摘要" in messages[0]["content"]
    assert "买牛奶" in messages[0]["content"]
    assert "weather" in messages[0]["content"]

    contents = [message["content"] for message in messages[1:]]
    assert "第二个问题" in contents
    assert "第二个回答" in contents
    assert "当前问题" in contents

    # 更早的轮次、历史 thought、历史工具结果都不应进入上下文。
    assert "第一个问题" not in contents
    assert "第一个回答" not in contents
    assert "北京晴" not in contents
    assert "先查天气" not in contents


def test_build_extends_final_to_its_user_message() -> None:
    """最近窗口从 final 开始时，要补上配对的 user 消息。"""
    context = make_context(FakeLLM([]), keep_recent_messages=1)
    session = make_session()

    messages = context.build(session, turn_id="u_3", tools=TOOLS)
    contents = [message["content"] for message in messages[1:]]

    assert contents[:2] == ["第二个问题", "第二个回答"]


def test_build_trims_oldest_history_when_over_budget() -> None:
    """预算极小时，保留 System + 当前轮，删掉全部历史。"""
    context = make_context(
        FakeLLM([]),
        context_token_budget=1,
        keep_recent_messages=10,
    )
    session = make_session()

    messages = context.build(session, turn_id="u_3", tools=TOOLS)
    contents = [message["content"] for message in messages[1:]]

    assert contents == ["当前问题"]


def test_needs_compression_by_token_budget() -> None:
    """超过 token 预算时触发压缩。"""
    context = make_context(FakeLLM([]), context_token_budget=1)
    session = make_session()

    assert context.needs_compression(
        session,
        turn_id="u_3",
        tools=TOOLS,
    )


def test_needs_compression_by_message_count() -> None:
    """未压缩消息条数超过上限时触发压缩。"""
    context = make_context(
        FakeLLM([]),
        context_token_budget=100000,
        max_history_messages=2,
    )
    session = make_session()

    assert context.needs_compression(
        session,
        turn_id="u_3",
        tools=TOOLS,
    )


def test_needs_compression_false_when_within_limits() -> None:
    """预算和条数都没超时不需要压缩。"""
    context = make_context(
        FakeLLM([]),
        context_token_budget=100000,
        max_history_messages=100,
    )
    session = make_session()

    assert not context.needs_compression(
        session,
        turn_id="u_3",
        tools=TOOLS,
    )


async def test_compress_updates_summary_cursor_and_stats() -> None:
    """压缩旧轮次：更新摘要、推进游标、记录分类统计。"""
    llm = FakeLLM(["压缩后的摘要"])
    context = make_context(llm)
    session = make_session()
    stats = TurnStats()

    result = await context.compress(
        session,
        turn_id="u_3",
        tools=TOOLS,
        turn_stats=stats,
        now=FIXED_NOW,
    )

    assert result.compressed is True
    assert result.summarized_messages == 4  # m_1 ~ m_4
    assert result.summary_until_msg_id == "m_4"
    assert result.fallback is False
    assert session.summary == "压缩后的摘要"
    assert session.summary_until_msg_id == "m_4"
    assert stats.compression_calls == 1
    assert stats.compression_prompt_tokens > 0
    assert stats.compression_completion_tokens > 0

    # 摘要 Prompt 里应该同时包含旧摘要和新对话。
    prompt = llm.calls[0][0]["content"]
    assert "旧摘要" in prompt
    assert "第一个问题" in prompt


async def test_build_excludes_summarized_messages() -> None:
    """压缩后，游标之前的消息不再进入上下文。"""
    llm = FakeLLM(["压缩后的摘要"])
    context = make_context(llm)
    session = make_session()
    await context.compress(
        session,
        turn_id="u_3",
        tools=TOOLS,
        turn_stats=TurnStats(),
        now=FIXED_NOW,
    )

    messages = context.build(session, turn_id="u_3", tools=TOOLS)
    contents = [message["content"] for message in messages[1:]]

    assert "第一个问题" not in contents
    assert "第二个问题" in contents
    assert "当前问题" in contents


async def test_second_compression_does_not_repeat_cursor() -> None:
    """游标之后没有新的可压缩消息时，不再次调用摘要 LLM。"""
    llm = FakeLLM(["压缩后的摘要"])
    context = make_context(llm)
    session = make_session()
    await context.compress(
        session,
        turn_id="u_3",
        tools=TOOLS,
        turn_stats=TurnStats(),
        now=FIXED_NOW,
    )

    # 第二次调用时脚本已经用完；如果再次调用 LLM 会抛 AssertionError。
    second = await context.compress(
        session,
        turn_id="u_3",
        tools=TOOLS,
        turn_stats=TurnStats(),
        now=FIXED_NOW,
    )

    assert second.compressed is False
    assert len(llm.calls) == 1


async def test_compress_falls_back_when_summary_llm_fails() -> None:
    """摘要 LLM 失败时使用旧摘要 + 最后两条可见消息兜底。"""
    llm = FakeLLM([LLMError("boom")])
    context = make_context(llm)
    session = make_session()
    stats = TurnStats()

    result = await context.compress(
        session,
        turn_id="u_3",
        tools=TOOLS,
        turn_stats=stats,
        now=FIXED_NOW,
    )

    assert result.compressed is True
    assert result.fallback is True
    assert "旧摘要" in session.summary
    assert "第一个回答" in session.summary
    assert session.summary_until_msg_id == "m_4"
    assert stats.compression_calls == 1
    assert stats.compression_prompt_tokens > 0


async def test_compress_truncates_long_summary() -> None:
    """摘要超过字符上限时硬截断。"""
    llm = FakeLLM(["x" * 100])
    context = make_context(llm, summary_max_chars=10)
    session = make_session()

    await context.compress(
        session,
        turn_id="u_3",
        tools=TOOLS,
        turn_stats=TurnStats(),
        now=FIXED_NOW,
    )

    assert session.summary == "x" * 10 + "…"


async def test_compression_budget_is_enforced() -> None:
    """达到单轮压缩次数上限后不再调用 LLM。"""
    llm = FakeLLM([])
    context = make_context(llm)
    session = make_session()
    stats = TurnStats(compression_calls=1)

    result = await context.compress(
        session,
        turn_id="u_3",
        tools=TOOLS,
        turn_stats=stats,
        now=FIXED_NOW,
    )

    assert result.compressed is False
    assert llm.calls == []


async def test_maybe_compress_skips_when_under_budget() -> None:
    """未触发阈值时 maybe_compress 不调用 LLM。"""
    llm = FakeLLM([])
    context = make_context(
        llm,
        context_token_budget=100000,
        max_history_messages=100,
    )
    session = make_session()

    result = await context.maybe_compress(
        session,
        turn_id="u_3",
        tools=TOOLS,
        turn_stats=TurnStats(),
    )

    assert result.compressed is False
    assert llm.calls == []


async def test_maybe_compress_runs_when_threshold_hit() -> None:
    """超过消息条数上限时 maybe_compress 会执行压缩。"""
    llm = FakeLLM(["summary"])
    context = make_context(
        llm,
        context_token_budget=100000,
        max_history_messages=2,
    )
    session = make_session()

    result = await context.maybe_compress(
        session,
        turn_id="u_3",
        tools=TOOLS,
        turn_stats=TurnStats(),
        now=FIXED_NOW,
    )

    assert result.compressed is True
    assert len(llm.calls) == 1
