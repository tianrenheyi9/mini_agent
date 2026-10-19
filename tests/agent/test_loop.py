"""AgentLoop 的核心循环、异常路径与上下文集成测试。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.agent.context import ContextManager
from app.agent.llm import LLMConfigError, LLMRequestError
from app.agent.loop import AgentLoop
from app.config import Settings, get_settings
from app.session.manager import SessionManager
from app.session.store import SessionStore
from app.tools import build_default_registry
from app.tools.base import ToolSpec
from app.tools.registry import ToolRegistry
from app.trace import TraceLogger
from tests.llm.fake_llm import FakeLLM

FIXED_NOW = datetime(
    2026,
    10,
    19,
    9,
    0,
    0,
    tzinfo=timezone(timedelta(hours=8)),
)

FINAL_HELLO = '{"thought": "直接回答", "final": "你好！"}'
CALCULATOR_CALL = (
    '{"thought": "需要计算", "action": "calculator", '
    '"input": {"expression": "2**10"}}'
)


def make_settings(**overrides: Any) -> Settings:
    """给测试一组宽松的默认预算，避免无关的压缩/上限干扰。"""
    values: dict[str, Any] = {
        "max_tool_calls": 8,
        "max_llm_calls_per_turn": 10,
        "max_parse_repairs": 2,
        "max_compression_calls_per_turn": 1,
        "context_token_budget": 100000,
        "token_safety_factor": 1.0,
        "keep_recent_messages": 8,
        "summary_max_chars": 800,
        "summary_max_tokens": 128,
        "max_history_messages": 100,
        "tool_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return replace(get_settings(), **values)


def make_loop(
    tmp_path: Path,
    responses: list[str | Exception],
    *,
    registry: ToolRegistry | None = None,
    **overrides: Any,
) -> tuple[AgentLoop, Any, FakeLLM, TraceLogger, SessionManager]:
    """
    组装一个使用临时目录和 FakeLLM 的 AgentLoop。

    返回 (loop, session, llm, trace, sessions)，方便测试检查
    模型收到的消息、落盘结果和 Trace。
    """
    settings = make_settings(**overrides)
    llm = FakeLLM(responses)
    sessions = SessionManager(
        store=SessionStore(tmp_path / "sessions"),
        traces_dir=tmp_path / "traces",
        clock=lambda: FIXED_NOW,
    )
    trace = TraceLogger(
        tmp_path / "traces",
        max_field_chars=2000,
        clock=lambda: FIXED_NOW,
    )
    context = ContextManager(
        settings,
        llm,  # type: ignore[arg-type]
        clock=lambda: FIXED_NOW,
    )
    loop = AgentLoop(
        settings=settings,
        llm=llm,  # type: ignore[arg-type]
        registry=registry
        or build_default_registry(timeout_seconds=1.0),
        sessions=sessions,
        context=context,
        trace=trace,
        clock=lambda: FIXED_NOW,
    )
    session = sessions.create("测试会话")
    return loop, session, llm, trace, sessions


async def test_direct_answer(tmp_path: Path) -> None:
    """纯对话：一次 LLM 调用直接 final。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path, [FINAL_HELLO]
    )

    result = await loop.run_turn(session.id, "你好")

    assert result.status == "ok"
    assert result.reply == "你好！"
    assert result.thought == "直接回答"
    assert result.steps == []
    assert result.stats["llm_calls"] == 1
    assert result.stats["tool_calls"] == 0
    assert result.compression is not None
    assert result.compression["compressed"] is False

    # 会话落盘后，只有 user + final 两条消息。
    reloaded = sessions.get(session.id)
    assert [message.kind for message in reloaded.messages] == [
        "user",
        "final",
    ]
    assert reloaded.stats.llm_calls == 1

    # Trace 顺序：开始 -> LLM -> 结束。
    events = trace.read_events(session.id)
    assert [event["event"] for event in events] == [
        "turn_start",
        "llm_call",
        "turn_end",
    ]
    assert events[-1]["status"] == "ok"


async def test_single_tool_and_observation_format(tmp_path: Path) -> None:
    """单工具调用：结果以 user observation 回填，不使用 role=tool。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        [
            CALCULATOR_CALL,
            '{"thought": "得到结果", "final": "2 的 10 次方是 1024。"}',
        ],
    )

    result = await loop.run_turn(session.id, "2 的 10 次方是多少？")

    assert result.status == "ok"
    assert "1024" in result.reply
    assert len(result.steps) == 1
    assert result.steps[0]["tool"] == "calculator"
    assert result.steps[0]["status"] == "ok"
    assert result.steps[0]["result"]["result"] == 1024
    assert result.stats["llm_calls"] == 2
    assert result.stats["tool_calls"] == 1

    # 第二次 LLM 调用应该能看到：模型原始决策 + user observation。
    second_messages = llm.calls[1]
    roles = [message["role"] for message in second_messages]
    assert "tool" not in roles

    observation = next(
        message["content"]
        for message in second_messages
        if message["role"] == "user"
        and message["content"].startswith("[工具执行结果]")
    )
    assert "tool: calculator" in observation
    assert "status: ok" in observation
    assert "1024" in observation
    assert any(
        message["role"] == "assistant"
        and '"calculator"' in message["content"]
        for message in second_messages
    )

    # 完整消息链：user -> tool_call(raw) -> observation(user) -> final。
    kinds = [message.kind for message in sessions.get(session.id).messages]
    assert kinds == ["user", "tool_call", "tool_observation", "final"]

    tool_events = [
        event
        for event in trace.read_events(session.id)
        if event["event"] == "tool_call"
    ]
    assert len(tool_events) == 1
    assert tool_events[0]["tool"] == "calculator"
    assert tool_events[0]["status"] == "ok"


async def test_multi_step_tool_chain(tmp_path: Path) -> None:
    """多工具串联：weather -> todo(add) -> final。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        [
            '{"action": "weather", "input": {"city": "北京"}}',
            (
                '{"action": "todo", "input": {"action": "add", '
                '"content": "给妈妈买生日礼物"}}'
            ),
            '{"final": "天气已查，待办已记录。"}',
        ],
    )

    result = await loop.run_turn(
        session.id,
        "北京天气怎么样？另外记一条待办",
    )

    assert result.status == "ok"
    assert [step["tool"] for step in result.steps] == ["weather", "todo"]
    assert all(step["status"] == "ok" for step in result.steps)
    assert result.stats["llm_calls"] == 3
    assert result.stats["tool_calls"] == 2

    reloaded = sessions.get(session.id)
    assert reloaded.todos[0]["content"] == "给妈妈买生日礼物"


async def test_unknown_tool_self_correct(tmp_path: Path) -> None:
    """未知工具：错误作为 observation 回填，模型可改用 final。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        [
            '{"action": "missing_tool", "input": {}}',
            '{"final": "抱歉，这个工具当前不可用。"}',
        ],
    )

    result = await loop.run_turn(session.id, "调用一个不存在的工具")

    assert result.status == "ok"
    assert result.steps[0]["status"] == "error"
    assert "未知工具" in result.steps[0]["error"]

    observation = next(
        message["content"]
        for message in llm.calls[1]
        if message["role"] == "user"
        and message["content"].startswith("[工具执行结果]")
    )
    assert "status: error" in observation
    assert "可用工具" in observation


async def test_invalid_json_repair(tmp_path: Path) -> None:
    """非法 JSON：追加原始输出和纠错提示，然后重试成功。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        ["这不是 JSON", '{"final": "已修复"}'],
    )

    result = await loop.run_turn(session.id, "随便问问")

    assert result.status == "ok"
    assert result.reply == "已修复"
    assert result.steps[0]["type"] == "parse_error"
    assert result.stats["llm_calls"] == 2

    # 第二次调用应该带上原始错误输出和纠错提示。
    second_messages = llm.calls[1]
    assert any(
        message["role"] == "user"
        and "不符合约定协议" in message["content"]
        for message in second_messages
    )
    assert any(
        message["role"] == "assistant" and "这不是 JSON" in message["content"]
        for message in second_messages
    )

    kinds = [message.kind for message in sessions.get(session.id).messages]
    assert kinds == ["user", "raw", "repair", "final"]


async def test_parse_repair_exhausted(tmp_path: Path) -> None:
    """纠错次数耗尽后返回格式异常提示。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        ["bad", "bad", "unused"],
        max_parse_repairs=1,
    )

    result = await loop.run_turn(session.id, "测试格式错误")

    assert result.status == "parse_error"
    assert "格式异常" in result.reply
    assert result.stats["llm_calls"] == 2
    assert len(result.steps) == 2
    assert all(step["type"] == "parse_error" for step in result.steps)

    # 第二次解析失败后不再追加 repair，只写 final。
    kinds = [message.kind for message in sessions.get(session.id).messages]
    assert kinds == ["user", "raw", "repair", "final"]


async def test_tool_limit_keeps_final_chance(tmp_path: Path) -> None:
    """工具次数用完后，仍保留一次生成最终回答的 LLM 调用。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        [
            '{"action": "todo", "input": {"action": "list"}}',
            '{"action": "todo", "input": {"action": "list"}}',
            '{"final": "两次工具结果已经足够。"}',
        ],
        max_tool_calls=2,
        max_llm_calls_per_turn=10,
    )

    result = await loop.run_turn(session.id, "连续调用两次工具")

    assert result.status == "ok"
    assert result.reply == "两次工具结果已经足够。"
    assert result.stats["tool_calls"] == 2
    assert result.stats["llm_calls"] == 3
    assert len(result.steps) == 2


async def test_tool_limit_forces_model_to_final(tmp_path: Path) -> None:
    """超过工具上限时不执行，而是提示模型直接收尾。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        [
            '{"action": "todo", "input": {"action": "list"}}',
            '{"action": "todo", "input": {"action": "list"}}',
            '{"final": "基于已有结果收尾。"}',
        ],
        max_tool_calls=1,
        max_llm_calls_per_turn=10,
    )

    result = await loop.run_turn(session.id, "尝试两次工具")

    assert result.status == "ok"
    assert result.reply == "基于已有结果收尾。"
    assert result.stats["tool_calls"] == 1
    assert any(step["type"] == "tool_limit" for step in result.steps)


async def test_tool_limit_ignored_finalizes(tmp_path: Path) -> None:
    """模型收到上限提示后仍继续要工具，则直接兜底结束。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        [
            '{"action": "todo", "input": {"action": "list"}}',
            '{"action": "todo", "input": {"action": "list"}}',
            '{"action": "todo", "input": {"action": "list"}}',
        ],
        max_tool_calls=1,
        max_llm_calls_per_turn=10,
    )

    result = await loop.run_turn(session.id, "一直调用工具")

    assert result.status == "max_tool_calls"
    assert "最大工具调用次数" in result.reply
    assert result.stats["tool_calls"] == 1


async def test_max_llm_calls(tmp_path: Path) -> None:
    """达到 LLM 调用上限后停止循环。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        ['{"action": "todo", "input": {"action": "list"}}'],
        max_tool_calls=10,
        max_llm_calls_per_turn=1,
    )

    result = await loop.run_turn(session.id, "消耗调用次数")

    assert result.status == "max_llm_calls"
    assert "LLM 调用次数" in result.reply
    assert result.stats["llm_calls"] == 1
    assert result.stats["tool_calls"] == 1
    assert len(result.steps) == 1


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMRequestError("down", status_code=503), "模型服务暂时不可用"),
        (LLMConfigError("missing key"), "模型未配置"),
    ],
)
async def test_llm_errors_produce_readable_reply(
    tmp_path: Path,
    error: Exception,
    expected: str,
) -> None:
    """LLM 故障转换为可读回复，并写入 final 与 Trace。"""
    loop, session, llm, trace, sessions = make_loop(tmp_path, [error])

    result = await loop.run_turn(session.id, "你好")

    assert result.status == "llm_error"
    assert expected in result.reply
    assert result.stats["llm_calls"] == 1

    reloaded = sessions.get(session.id)
    assert reloaded.messages[-1].kind == "final"
    assert reloaded.messages[-1].content == result.reply

    assert trace.read_events(session.id)[-1]["status"] == "llm_error"


async def test_tool_error_continues(tmp_path: Path) -> None:
    """工具业务错误回填后，模型仍可继续给出最终回答。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        [
            (
                '{"action": "todo", "input": {"action": "complete", '
                '"id": "t_99"}}'
            ),
            '{"final": "这条待办不存在。"}',
        ],
    )

    result = await loop.run_turn(session.id, "完成不存在的待办")

    assert result.status == "ok"
    assert result.steps[0]["status"] == "error"
    assert "待办不存在" in result.steps[0]["error"]
    assert result.stats["tool_calls"] == 1


async def test_compression_integration(tmp_path: Path) -> None:
    """压缩在进入主循环前执行，摘要和统计都写入 Session。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        ["压缩摘要", '{"final": "新回答"}'],
        max_history_messages=1,
        keep_recent_messages=1,
    )

    # 先造两轮历史：第一轮会被压缩，第二轮留在最近窗口。
    async with sessions.locked(session.id) as stored:
        stored.append_user("旧问题一", "u_old1")
        stored.append_assistant_final(
            thought="",
            final="旧回答一",
            turn_id="u_old1",
        )
        stored.append_user("旧问题二", "u_old2")
        stored.append_assistant_final(
            thought="",
            final="旧回答二",
            turn_id="u_old2",
        )

    result = await loop.run_turn(session.id, "新问题")

    assert result.status == "ok"
    assert result.compression is not None
    assert result.compression["compressed"] is True
    assert result.compression["summarized_messages"] == 2
    assert result.stats["compression_calls"] == 1
    assert result.stats["llm_calls"] == 1

    reloaded = sessions.get(session.id)
    assert reloaded.summary == "压缩摘要"
    assert reloaded.summary_until_msg_id == "m_2"
    assert reloaded.stats.compression_calls == 1

    # 压缩调用是第一次 LLM 调用；主循环是第二次。
    assert len(llm.calls) == 2
    assert "旧问题一" in llm.calls[0][0]["content"]
    assert "压缩摘要" in llm.calls[1][0]["content"]

    events = trace.read_events(session.id)
    assert any(event["event"] == "compress" for event in events)


async def test_context_overflow_returns_readable_reply(
    tmp_path: Path,
) -> None:
    """System + 当前轮超预算时不调用 LLM，直接返回明确提示。"""
    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        [],
        context_token_budget=1,
    )

    result = await loop.run_turn(session.id, "这条消息本身就很长")

    assert result.status == "context_overflow"
    assert "上下文超出预算" in result.reply
    assert llm.calls == []
    assert sessions.get(session.id).messages[-1].kind == "final"


async def test_large_tool_output_is_compacted(tmp_path: Path) -> None:
    """工具结果超长时压缩 observation，而不是中断当前任务。"""

    async def big_handler(
        input_data: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        return {"text": "X" * 5000}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="big",
            description="返回大段文本",
            parameters={"type": "object", "properties": {}},
            handler=big_handler,
        )
    )

    loop, session, llm, trace, sessions = make_loop(
        tmp_path,
        ['{"action": "big", "input": {}}', '{"final": "已完成"}'],
        registry=registry,
        context_token_budget=1600,
        observation_prompt_chars=100,
        keep_recent_messages=0,
        tool_observation_max_chars=5000,
    )

    result = await loop.run_turn(session.id, "调用大工具")

    assert result.status == "ok"
    assert result.reply == "已完成"

    # 第二次调用看到的 observation 应该已经被压缩。
    observation = next(
        message["content"]
        for message in llm.calls[1]
        if message["role"] == "user"
        and (
            "[工具执行结果]" in message["content"]
            or "工具结果" in message["content"]
        )
    )
    assert len(observation) < 1000
    assert "已截断" in observation or "已省略" in observation


async def test_empty_input_raises(tmp_path: Path) -> None:
    """空消息在进入循环前就被拒绝。"""
    loop, session, llm, trace, sessions = make_loop(tmp_path, [])

    with pytest.raises(ValueError, match="消息不能为空"):
        await loop.run_turn(session.id, "   ")

    assert llm.calls == []
