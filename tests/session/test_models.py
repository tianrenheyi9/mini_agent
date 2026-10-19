"""Session / Message 数据模型的单元测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.session.models import (
    Session,
    SessionStats,
    TurnStats,
    now_iso,
)

T0 = "2026-10-19T09:00:00+08:00"
T1 = "2026-10-19T09:01:00+08:00"


def make_session() -> Session:
    return Session(id="s_test", created_at=T0, updated_at=T0)


def test_append_assigns_monotonic_msg_ids_and_title() -> None:
    """msg_id 单调递增；首条用户消息生成标题。"""
    session = make_session()
    first = session.append_user(
        "帮我写一份本周周报，包含三条要点并记录待办",
        "u_1",
        created_at=T0,
    )
    second = session.append_user("再记一条待办", "u_1", created_at=T1)

    assert first.msg_id == "m_1"
    assert second.msg_id == "m_2"
    assert session.next_msg_seq == 3
    assert session.title == "帮我写一份本周周报，包含三条要点并记录待办"[:20]
    assert session.updated_at == T1


def test_title_is_not_overwritten_by_later_messages() -> None:
    """标题只在第一条用户消息时生成。"""
    session = make_session()
    session.append_user("第一条消息", "u_1", created_at=T0)
    session.append_user("第二条消息", "u_1", created_at=T1)

    assert session.title == "第一条消息"


def test_repair_message_does_not_become_title() -> None:
    """协议纠错消息也是 user 角色，但不应参与标题生成。"""
    session = make_session()
    session.append_user("   ", "u_1", created_at=T0, kind="repair")

    assert session.title == "新会话"


def test_message_kinds_and_history_visibility() -> None:
    """只有 user 原文和 final 应该进入后续历史窗口。"""
    session = make_session()
    user = session.append_user("你好", "u_1", created_at=T0)
    raw = session.append_assistant_raw("not json", "u_1", created_at=T0)
    call = session.append_assistant_tool_call(
        thought="查天气",
        action="weather",
        tool_input={"city": "北京"},
        raw_content='{"action":"weather"}',
        turn_id="u_1",
        created_at=T0,
    )
    observation = session.append_tool_observation(
        tool="weather",
        status="ok",
        content="[工具执行结果]",
        turn_id="u_1",
        created_at=T0,
    )
    final = session.append_assistant_final(
        thought="完成",
        final="北京今天多云。",
        turn_id="u_1",
        created_at=T0,
    )

    assert user.visible_in_history is True
    assert final.visible_in_history is True
    assert raw.visible_in_history is False
    assert call.visible_in_history is False
    assert observation.visible_in_history is False

    assert call.tool == "weather"
    assert call.tool_input == {"city": "北京"}
    assert observation.role == "user"
    assert observation.kind == "tool_observation"
    assert observation.status == "ok"
    assert final.to_llm_message() == {
        "role": "assistant",
        "content": "北京今天多云。",
    }


def test_summary_cursor_and_messages_after_cursor() -> None:
    """压缩游标推进后，只返回游标之后的消息。"""
    session = make_session()
    session.append_user("m1", "u_1", created_at=T0)
    session.append_assistant_final(
        thought="", final="r1", turn_id="u_1", created_at=T0
    )
    session.append_user("m2", "u_1", created_at=T1)
    session.append_assistant_final(
        thought="", final="r2", turn_id="u_1", created_at=T1
    )

    session.mark_summarized_until("m_2", updated_at=T1)

    assert session.summary_until_msg_id == "m_2"
    assert [message.content for message in session.messages_after_cursor()] == [
        "m2",
        "r2",
    ]


def test_mark_unknown_msg_id_raises() -> None:
    """游标不能指向不存在的消息。"""
    session = make_session()
    try:
        session.mark_summarized_until("m_99")
    except ValueError as exc:
        assert "消息不存在" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_stats_add_turn() -> None:
    """Session 统计按轮累加。"""
    stats = SessionStats(llm_calls=1, prompt_tokens=10)
    stats.add_turn(TurnStats(llm_calls=2, tool_calls=1, prompt_tokens=5))

    assert stats.llm_calls == 3
    assert stats.tool_calls == 1
    assert stats.prompt_tokens == 15


def test_roundtrip_to_dict_and_back() -> None:
    """序列化再反序列化后，关键字段保持一致。"""
    session = make_session()
    session.append_user("你好", "u_1", created_at=T0)
    session.append_assistant_final(
        thought="完成", final="你好！", turn_id="u_1", created_at=T0
    )
    session.summary = "用户打了招呼"
    session.summary_until_msg_id = "m_2"
    session.todos.append(
        {"id": "t_1", "content": "买牛奶", "done": False, "created_at": T0}
    )
    session.stats.add_turn(TurnStats(llm_calls=1, tool_calls=0))

    restored = Session.from_dict(session.to_dict())

    assert restored.id == session.id
    assert restored.title == session.title
    assert restored.summary == session.summary
    assert restored.summary_until_msg_id == "m_2"
    assert restored.next_msg_seq == 3
    assert restored.todos == session.todos
    assert [message.to_dict() for message in restored.messages] == [
        message.to_dict() for message in session.messages
    ]
    assert restored.stats.to_dict() == session.stats.to_dict()


def test_from_dict_derives_next_seq_for_old_data() -> None:
    """旧数据没有 next_msg_seq 时，从消息 id 推断。"""
    restored = Session.from_dict(
        {
            "id": "s_old",
            "messages": [
                {"msg_id": "m_1", "content": "a"},
                {"msg_id": "m_7", "content": "b"},
            ],
        }
    )

    assert restored.next_msg_seq == 8


def test_now_iso_uses_injected_datetime() -> None:
    """固定时间便于测试断言。"""
    value = datetime(
        2026,
        10,
        19,
        9,
        0,
        0,
        tzinfo=timezone(timedelta(hours=8)),
    )
    assert now_iso(value) == T0
