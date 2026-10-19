"""SessionManager 的会话隔离、持久化与并发测试。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.session.manager import SessionManager
from app.session.store import SessionNotFound, SessionStore


class FakeClock:
    """每次调用前进一秒，让 updated_at 有稳定顺序。"""

    def __init__(self) -> None:
        self.current = datetime(
            2026,
            10,
            19,
            9,
            0,
            0,
            tzinfo=timezone(timedelta(hours=8)),
        )

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


def make_manager(tmp_path: Path, clock: FakeClock | None = None) -> SessionManager:
    store = SessionStore(tmp_path / "sessions")
    return SessionManager(
        store=store,
        traces_dir=tmp_path / "traces",
        clock=clock or FakeClock(),
    )


def test_create_get_and_list(tmp_path: Path) -> None:
    """创建后文件立即存在，能按 id 读取并出现在列表中。"""
    manager = make_manager(tmp_path)
    session = manager.create("窗口 1")

    assert session.id.startswith("s_")
    assert (tmp_path / "sessions" / f"{session.id}.json").exists()
    assert manager.get(session.id).title == "窗口 1"

    items = manager.list_sessions()
    assert [item["id"] for item in items] == [session.id]


def test_get_missing_raises(tmp_path: Path) -> None:
    """不存在的会话抛 SessionNotFound。"""
    manager = make_manager(tmp_path)

    with pytest.raises(SessionNotFound):
        manager.get("s_missing")


async def test_locked_persists_changes(tmp_path: Path) -> None:
    """locked 退出时自动保存，新 Manager 能读到。"""
    manager = make_manager(tmp_path)
    session = manager.create()

    async with manager.locked(session.id) as locked:
        locked.append_user("你好", "u_1")

    reloaded = make_manager(tmp_path).get(session.id)
    assert [message.content for message in reloaded.messages] == ["你好"]


async def test_sessions_are_isolated(tmp_path: Path) -> None:
    """两个会话的消息与待办互不影响。"""
    manager = make_manager(tmp_path)
    first = manager.create("窗口 1")
    second = manager.create("窗口 2")

    async with manager.locked(first.id) as session:
        session.append_user("窗口 1 的消息", "u_1")
        session.todos.append({"id": "t_1", "content": "买礼物", "done": False})

    async with manager.locked(second.id) as session:
        session.append_user("窗口 2 的消息", "u_2")
        session.todos.append({"id": "t_1", "content": "写周报", "done": False})

    reloaded_first = manager.get(first.id)
    reloaded_second = manager.get(second.id)

    assert [m.content for m in reloaded_first.messages] == ["窗口 1 的消息"]
    assert [m.content for m in reloaded_second.messages] == ["窗口 2 的消息"]
    assert reloaded_first.todos[0]["content"] == "买礼物"
    assert reloaded_second.todos[0]["content"] == "写周报"


async def test_same_session_lock_serializes(tmp_path: Path) -> None:
    """同一会话的 locked 块必须串行执行。"""
    manager = make_manager(tmp_path)
    session = manager.create()

    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def worker(name: str, wait: bool) -> None:
        async with manager.locked(session.id):
            order.append(f"{name}-enter")
            if wait:
                first_entered.set()
                await release_first.wait()
            order.append(f"{name}-exit")

    first_task = asyncio.create_task(worker("a", True))
    await first_entered.wait()

    second_task = asyncio.create_task(worker("b", False))
    await asyncio.sleep(0.02)
    assert order == ["a-enter"]

    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert order == ["a-enter", "a-exit", "b-enter", "b-exit"]


async def test_save_on_exception(tmp_path: Path) -> None:
    """locked 内抛异常时，已经追加的消息仍然落盘。"""
    manager = make_manager(tmp_path)
    session = manager.create()

    with pytest.raises(RuntimeError):
        async with manager.locked(session.id) as locked:
            locked.append_user("先记下来", "u_1")
            raise RuntimeError("boom")

    reloaded = manager.get(session.id)
    assert [message.content for message in reloaded.messages] == ["先记下来"]


async def test_delete_removes_session_and_trace(tmp_path: Path) -> None:
    """删除会话时同时删除 Trace 文件。"""
    manager = make_manager(tmp_path)
    session = manager.create()
    trace_path = tmp_path / "traces" / f"{session.id}.jsonl"
    trace_path.write_text('{"event":"llm_call"}\n', encoding="utf-8")

    assert await manager.delete(session.id) is True
    assert not (tmp_path / "sessions" / f"{session.id}.json").exists()
    assert not trace_path.exists()
    assert await manager.delete(session.id) is False


def test_clock_controls_timestamps(tmp_path: Path) -> None:
    """注入时钟后，创建时间可预测。"""
    manager = make_manager(tmp_path)
    session = manager.create()

    assert session.created_at == "2026-10-19T09:00:00+08:00"
    assert session.updated_at == "2026-10-19T09:00:00+08:00"
