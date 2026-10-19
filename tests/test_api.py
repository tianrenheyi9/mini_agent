"""FastAPI 路由与 AgentLoop 的集成测试。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.context import ContextManager
from app.agent.loop import AgentLoop
from app.config import get_settings
from app.main import create_app
from app.session.manager import SessionManager
from app.session.store import SessionStore
from app.tools import build_default_registry
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


def make_client(
    tmp_path: Path,
    responses: list[str | Exception],
) -> tuple[TestClient, AgentLoop]:
    """用临时目录和 FakeLLM 装配一个可测试的 API。"""
    settings = replace(
        get_settings(),
        data_dir=tmp_path,
        max_tool_calls=8,
        max_llm_calls_per_turn=10,
        max_parse_repairs=2,
        max_compression_calls_per_turn=1,
        context_token_budget=100000,
        token_safety_factor=1.0,
        keep_recent_messages=8,
        max_history_messages=100,
    )
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
    agent = AgentLoop(
        settings=settings,
        llm=llm,  # type: ignore[arg-type]
        registry=build_default_registry(timeout_seconds=1.0),
        sessions=sessions,
        context=context,
        trace=trace,
        clock=lambda: FIXED_NOW,
    )
    return TestClient(create_app(agent=agent)), agent


def create_session(client: TestClient, title: str = "测试会话") -> str:
    response = client.post("/api/sessions", json={"title": title})
    assert response.status_code == 201
    return response.json()["id"]


def test_health(tmp_path: Path) -> None:
    """健康检查不依赖 Agent 是否已配置。"""
    client, _ = make_client(tmp_path, [])
    with client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_list_get_delete(tmp_path: Path) -> None:
    """会话 CRUD 全流程。"""
    client, agent = make_client(tmp_path, [])

    with client:
        first = create_session(client, "窗口 1")
        second = create_session(client, "窗口 2")

        listed = client.get("/api/sessions")
        assert listed.status_code == 200
        ids = {item["id"] for item in listed.json()["sessions"]}
        assert {first, second} <= ids

        detail = client.get(f"/api/sessions/{first}")
        assert detail.status_code == 200
        assert detail.json()["title"] == "窗口 1"
        assert detail.json()["messages"] == []

        deleted = client.delete(f"/api/sessions/{first}")
        assert deleted.status_code == 200

        assert client.get(f"/api/sessions/{first}").status_code == 404


def test_send_message_direct(tmp_path: Path) -> None:
    """POST messages 真正进入 AgentLoop 并返回 final。"""
    client, agent = make_client(tmp_path, [FINAL_HELLO])

    with client:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "你好"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["reply"] == "你好！"
        assert body["status"] == "ok"
        assert body["steps"] == []
        assert body["stats"]["llm_calls"] == 1

        detail = client.get(f"/api/sessions/{session_id}").json()
        assert [message["kind"] for message in detail["messages"]] == [
            "user",
            "final",
        ]


def test_send_message_tool_call(tmp_path: Path) -> None:
    """API 返回工具调用时间线。"""
    client, agent = make_client(
        tmp_path,
        [
            CALCULATOR_CALL,
            '{"thought": "完成", "final": "结果是 1024。"}',
        ],
    )

    with client:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "2 的 10 次方是多少"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["reply"] == "结果是 1024。"
        assert len(body["steps"]) == 1
        assert body["steps"][0]["tool"] == "calculator"
        assert body["steps"][0]["status"] == "ok"
        assert body["stats"]["tool_calls"] == 1


def test_traces_endpoint(tmp_path: Path) -> None:
    """发送消息后可以读取 Trace。"""
    client, agent = make_client(tmp_path, [FINAL_HELLO])

    with client:
        session_id = create_session(client)
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "你好"},
        )

        response = client.get(
            f"/api/sessions/{session_id}/traces?limit=10"
        )
        assert response.status_code == 200
        events = response.json()["events"]
        assert [event["event"] for event in events] == [
            "turn_start",
            "llm_call",
            "turn_end",
        ]


def test_empty_message_returns_422(tmp_path: Path) -> None:
    """纯空白消息在路由层被拒绝。"""
    client, agent = make_client(tmp_path, [])

    with client:
        session_id = create_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "   "},
        )

    assert response.status_code == 422


def test_missing_session_returns_404(tmp_path: Path) -> None:
    """不存在的会话返回 404。"""
    client, agent = make_client(tmp_path, [])

    with client:
        response = client.post(
            "/api/sessions/s_missing/messages",
            json={"content": "你好"},
        )

    assert response.status_code == 404


def test_delete_removes_trace(tmp_path: Path) -> None:
    """删除会话后 Trace 也不可访问。"""
    client, agent = make_client(tmp_path, [FINAL_HELLO])

    with client:
        session_id = create_session(client)
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "你好"},
        )
        assert client.get(
            f"/api/sessions/{session_id}/traces"
        ).status_code == 200

        assert client.delete(
            f"/api/sessions/{session_id}"
        ).status_code == 200

        assert client.get(
            f"/api/sessions/{session_id}/traces"
        ).status_code == 404


def test_create_session_without_body(tmp_path: Path) -> None:
    """创建会话可以不传 title。"""
    client, agent = make_client(tmp_path, [])

    with client:
        response = client.post("/api/sessions")

    assert response.status_code == 201
    assert response.json()["title"] == "新会话"
