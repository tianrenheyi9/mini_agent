"""SessionStore 持久化层的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.session.models import Session
from app.session.store import SessionStore, SessionStoreError

T0 = "2026-10-19T09:00:00+08:00"
T1 = "2026-10-19T09:01:00+08:00"


def make_store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def make_session(session_id: str = "s_test", title: str = "测试会话") -> Session:
    return Session(
        id=session_id,
        title=title,
        created_at=T0,
        updated_at=T0,
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """保存后能完整读回，且不残留临时文件。"""
    store = make_store(tmp_path)
    session = make_session()
    session.append_user("你好", "u_1", created_at=T0)

    store.save(session)
    loaded = store.load("s_test")

    assert loaded is not None
    assert loaded.id == "s_test"
    assert loaded.title == "测试会话"
    assert [message.content for message in loaded.messages] == ["你好"]
    assert not list((tmp_path / "sessions").glob("*.tmp"))


def test_load_missing_returns_none(tmp_path: Path) -> None:
    """不存在的会话返回 None。"""
    store = make_store(tmp_path)
    assert store.load("s_missing") is None


def test_list_metadata_sorted_by_updated_at(tmp_path: Path) -> None:
    """列表按 updated_at 倒序，并带上待办数量。"""
    store = make_store(tmp_path)
    older = make_session("s_older", "旧会话")
    newer = make_session("s_newer", "新会话")
    older.updated_at = T0
    newer.updated_at = T1
    newer.todos.append({"id": "t_1", "content": "买牛奶", "done": False})
    store.save(older)
    store.save(newer)

    items = store.list_metadata()

    assert [item["id"] for item in items] == ["s_newer", "s_older"]
    assert items[0]["todo_open_count"] == 1
    assert items[0]["message_count"] == 0


def test_delete(tmp_path: Path) -> None:
    """删除返回 True；再次删除返回 False。"""
    store = make_store(tmp_path)
    store.save(make_session())

    assert store.delete("s_test") is True
    assert store.delete("s_test") is False
    assert store.load("s_test") is None


@pytest.mark.parametrize(
    "session_id",
    ["../evil", "s_../evil", "abc", "s_", "s_bad/name"],
)
def test_invalid_session_id_is_rejected(
    tmp_path: Path,
    session_id: str,
) -> None:
    """非法 id 必须被拒绝，防止路径穿越。"""
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="非法的会话 id"):
        store.path_for(session_id)


def test_corrupt_file_raises_store_error(tmp_path: Path) -> None:
    """损坏的 JSON 不能被当成空会话。"""
    store = make_store(tmp_path)
    path = tmp_path / "sessions" / "s_broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(SessionStoreError, match="读取会话失败"):
        store.load("s_broken")


def test_list_metadata_skips_corrupt_file(tmp_path: Path) -> None:
    """单个损坏文件不应让整个列表失败。"""
    store = make_store(tmp_path)
    store.save(make_session("s_good"))
    (tmp_path / "sessions" / "s_broken.json").write_text(
        "{not json", encoding="utf-8"
    )

    items = store.list_metadata()

    assert [item["id"] for item in items] == ["s_good"]


def test_mismatched_id_is_rejected(tmp_path: Path) -> None:
    """文件内 id 与文件名不一致时视为损坏。"""
    store = make_store(tmp_path)
    path = tmp_path / "sessions" / "s_test.json"
    path.write_text(
        '{"id": "s_other", "messages": []}',
        encoding="utf-8",
    )

    with pytest.raises(SessionStoreError, match="不一致"):
        store.load("s_test")


def test_unicode_is_preserved(tmp_path: Path) -> None:
    """中文内容不能变成 \\uXXXX。"""
    store = make_store(tmp_path)
    session = make_session()
    session.append_user("给妈妈买生日礼物 🎁", "u_1", created_at=T0)
    store.save(session)

    raw = (tmp_path / "sessions" / "s_test.json").read_text(encoding="utf-8")
    assert "给妈妈买生日礼物 🎁" in raw
