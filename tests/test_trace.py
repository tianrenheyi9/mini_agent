"""TraceLogger 的 JSONL 写入与读取测试。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.agent.context import CompressionResult
from app.agent.llm import LLMResponse
from app.agent.parser import FinalAnswer, ParseError, ToolCall
from app.session.models import TurnStats
from app.tools.base import ToolResult
from app.trace import TraceLogger

FIXED_NOW = datetime(
    2026,
    10,
    19,
    9,
    0,
    0,
    123000,
    tzinfo=timezone(timedelta(hours=8)),
)


def make_logger(tmp_path: Path) -> TraceLogger:
    return TraceLogger(
        tmp_path / "traces",
        max_field_chars=50,
        clock=lambda: FIXED_NOW,
    )


def read_lines(tmp_path: Path, session_id: str = "s_test") -> list[dict]:
    path = tmp_path / "traces" / f"{session_id}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_log_turn_start(tmp_path: Path) -> None:
    """turn_start 是整轮事件的起点。"""
    logger = make_logger(tmp_path)
    logger.log_turn_start(session_id="s_test", turn_id="u_1")

    event = read_lines(tmp_path)[0]
    assert event["session_id"] == "s_test"
    assert event["turn_id"] == "u_1"
    assert event["event"] == "turn_start"
    assert event["ts"].endswith("+08:00")


def test_log_llm_call(tmp_path: Path) -> None:
    """LLM 事件包含耗时、token、截断后的原始输出和解析摘要。"""
    logger = make_logger(tmp_path)
    response = LLMResponse(
        content="x" * 100,
        model="test-model",
        prompt_tokens=10,
        completion_tokens=5,
        finish_reason="stop",
        latency_ms=123,
    )

    logger.log_llm_call(
        session_id="s_test",
        turn_id="u_1",
        step=0,
        response=response,
        parsed=ToolCall(
            thought="查天气",
            action="weather",
            input={"city": "北京"},
        ),
        prompt_tokens_est=99,
    )

    event = read_lines(tmp_path)[0]
    assert event["event"] == "llm_call"
    assert event["model"] == "test-model"
    assert event["latency_ms"] == 123
    assert event["prompt_tokens"] == 10
    assert event["completion_tokens"] == 5
    assert event["prompt_tokens_est"] == 99
    assert len(event["raw_output"]) == 51  # 50 字符 + 省略号
    assert event["parsed"] == {
        "type": "tool_call",
        "action": "weather",
        "input": {"city": "北京"},
    }


def test_log_final_parse_summary(tmp_path: Path) -> None:
    """final 事件只保存前 500 个字符。"""
    logger = make_logger(tmp_path)
    response = LLMResponse(
        content='{"final":"ok"}',
        model="m",
        prompt_tokens=None,
        completion_tokens=None,
        finish_reason="stop",
        latency_ms=1,
    )

    logger.log_llm_call(
        session_id="s_test",
        turn_id="u_1",
        step=2,
        response=response,
        parsed=FinalAnswer(thought="完成", final="y" * 700),
    )

    parsed = read_lines(tmp_path)[0]["parsed"]
    assert parsed["type"] == "final"
    assert len(parsed["final"]) == 500


def test_log_tool_call_success(tmp_path: Path) -> None:
    """成功工具调用记录 result_preview。"""
    logger = make_logger(tmp_path)
    logger.log_tool_call(
        session_id="s_test",
        turn_id="u_1",
        step=0,
        tool="weather",
        tool_input={"city": "北京"},
        result=ToolResult(
            tool="weather",
            ok=True,
            data={"city": "北京", "condition": "晴"},
            duration_ms=2,
        ),
    )

    event = read_lines(tmp_path)[0]
    assert event["event"] == "tool_call"
    assert event["status"] == "ok"
    assert event["duration_ms"] == 2
    assert "北京" in event["result_preview"]
    assert "error" not in event


def test_log_tool_call_error(tmp_path: Path) -> None:
    """失败工具调用记录 error 字段，不写 result_preview。"""
    logger = make_logger(tmp_path)
    logger.log_tool_call(
        session_id="s_test",
        turn_id="u_1",
        step=1,
        tool="todo",
        tool_input={"action": "complete", "id": "t_99"},
        result=ToolResult(
            tool="todo",
            ok=False,
            error="待办不存在：'t_99'",
            duration_ms=1,
        ),
    )

    event = read_lines(tmp_path)[0]
    assert event["status"] == "error"
    assert "待办不存在" in event["error"]
    assert "result_preview" not in event


def test_log_parse_error(tmp_path: Path) -> None:
    """解析错误保留原因和截断后的原始输出。"""
    logger = make_logger(tmp_path)
    logger.log_parse_error(
        session_id="s_test",
        turn_id="u_1",
        step=0,
        reason="没有找到合法的 JSON 对象",
        raw_output="我不确定" * 50,
    )

    event = read_lines(tmp_path)[0]
    assert event["event"] == "parse_error"
    assert "没有找到" in event["reason"]
    assert len(event["raw_output"]) == 51


def test_log_compression(tmp_path: Path) -> None:
    """压缩事件记录前后 token、合并条数和降级标记。"""
    logger = make_logger(tmp_path)
    logger.log_compression(
        session_id="s_test",
        turn_id="u_1",
        result=CompressionResult(
            compressed=True,
            summarized_messages=4,
            summary_until_msg_id="m_4",
            before_tokens=7200,
            after_tokens=4100,
            fallback=False,
        ),
    )

    event = read_lines(tmp_path)[0]
    assert event["event"] == "compress"
    assert event["summarized_messages"] == 4
    assert event["summary_until_msg_id"] == "m_4"
    assert event["before_tokens_est"] == 7200
    assert event["after_tokens_est"] == 4100
    assert event["fallback"] is False


def test_log_turn_end_with_stats(tmp_path: Path) -> None:
    """轮次结束记录分类统计。"""
    logger = make_logger(tmp_path)
    stats = TurnStats(
        llm_calls=3,
        tool_calls=2,
        compression_calls=1,
        prompt_tokens=100,
        completion_tokens=20,
        compression_prompt_tokens=30,
        compression_completion_tokens=5,
    )

    logger.log_turn_end(
        session_id="s_test",
        turn_id="u_1",
        status="ok",
        stats=stats,
        latency_ms=1800,
        reply_preview="北京今天多云。",
    )

    event = read_lines(tmp_path)[0]
    assert event["event"] == "turn_end"
    assert event["status"] == "ok"
    assert event["latency_ms"] == 1800
    assert event["reply_preview"] == "北京今天多云。"
    assert event["stats"]["llm_calls"] == 3
    assert event["stats"]["tool_calls"] == 2
    assert event["stats"]["compression_calls"] == 1
    assert event["stats"]["compression_prompt_tokens"] == 30


def test_events_are_appended_in_order(tmp_path: Path) -> None:
    """多次写入按顺序追加，不覆盖旧事件。"""
    logger = make_logger(tmp_path)
    logger.log_turn_start(session_id="s_test", turn_id="u_1")
    logger.log_turn_start(session_id="s_test", turn_id="u_2")

    events = read_lines(tmp_path)
    assert [event["turn_id"] for event in events] == ["u_1", "u_2"]


def test_read_events_returns_last_n_and_skips_corrupt(
    tmp_path: Path,
) -> None:
    """读取最近 N 条；损坏行被跳过。"""
    logger = make_logger(tmp_path)
    for turn in ("u_1", "u_2", "u_3"):
        logger.log_turn_start(session_id="s_test", turn_id=turn)

    path = tmp_path / "traces" / "s_test.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{broken json\n")

    events = logger.read_events("s_test", limit=2)
    assert [event["turn_id"] for event in events] == ["u_2", "u_3"]


def test_read_events_missing_file_returns_empty(tmp_path: Path) -> None:
    """文件不存在时返回空列表。"""
    logger = make_logger(tmp_path)
    assert logger.read_events("s_missing") == []


def test_read_events_limit_zero(tmp_path: Path) -> None:
    """limit <= 0 直接返回空列表。"""
    logger = make_logger(tmp_path)
    logger.log_turn_start(session_id="s_test", turn_id="u_1")
    assert logger.read_events("s_test", limit=0) == []


@pytest.mark.parametrize("session_id", ["../evil", "s_", "abc", "s_bad/name"])
def test_invalid_session_id_is_rejected(
    tmp_path: Path,
    session_id: str,
) -> None:
    """非法 id 不能用于写 Trace。"""
    logger = make_logger(tmp_path)

    with pytest.raises(ValueError, match="非法的会话 id"):
        logger.log_turn_start(session_id=session_id, turn_id="u_1")


def test_unicode_is_preserved(tmp_path: Path) -> None:
    """Trace 中的中文不转义，方便直接阅读。"""
    logger = make_logger(tmp_path)
    logger.log_turn_start(session_id="s_test", turn_id="u_1")

    raw = (tmp_path / "traces" / "s_test.jsonl").read_text(encoding="utf-8")
    assert "s_test" in raw
    assert "\\u" not in raw


def test_parse_error_summary(tmp_path: Path) -> None:
    """ParseError 也会被转换成可落盘的摘要。"""
    logger = make_logger(tmp_path)
    response = LLMResponse(
        content="not json",
        model="m",
        prompt_tokens=None,
        completion_tokens=None,
        finish_reason="stop",
        latency_ms=1,
    )

    logger.log_llm_call(
        session_id="s_test",
        turn_id="u_1",
        step=0,
        response=response,
        parsed=ParseError("没有找到合法的 JSON 对象"),
    )

    parsed = read_lines(tmp_path)[0]["parsed"]
    assert parsed == {
        "type": "parse_error",
        "reason": "没有找到合法的 JSON 对象",
    }
