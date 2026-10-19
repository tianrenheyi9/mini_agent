"""
会话管理器。

职责：
1. 创建会话、读取会话、列出会话、删除会话；
2. 为每个会话提供一把 asyncio.Lock，保证同一会话的请求串行执行；
3. 在锁内加载会话，退出时统一落盘；
4. 删除会话时同时清理对应的 Trace 文件。

为什么需要会话锁？
同一窗口连续点两次发送，或 API 被并发调用时，
两个请求可能同时读写同一个 JSON 文件，导致消息丢失或顺序错乱。
锁把“加载 → 修改 → 保存”变成一个临界区。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.config import Settings, get_settings
from app.session.models import DEFAULT_TITLE, Session, now_iso
from app.session.store import SessionNotFound, SessionStore

logger = logging.getLogger(__name__)


class SessionManager:
    """会话生命周期与并发控制。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: SessionStore | None = None,
        traces_dir: Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store or SessionStore(self._settings.sessions_dir)
        self._traces_dir = Path(traces_dir or self._settings.traces_dir)
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._locks: dict[str, asyncio.Lock] = {}

    def _now(self) -> str:
        return now_iso(self._clock())

    def create(self, title: str | None = None) -> Session:
        """创建一个新会话并立即落盘。"""
        timestamp = self._now()
        session = Session(
            id=f"s_{uuid4().hex}",
            title=(title or "").strip() or DEFAULT_TITLE,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._store.save(session)
        return session

    def get(self, session_id: str) -> Session:
        """读取会话；不存在时抛 SessionNotFound。"""
        session = self._store.load(session_id)
        if session is None:
            raise SessionNotFound(f"会话不存在：{session_id}")
        return session

    def list_sessions(self) -> list[dict[str, object]]:
        """列出会话元数据，按更新时间倒序。"""
        return self._store.list_metadata()

    def save(self, session: Session) -> None:
        """显式保存会话（通常不需要，locked 退出时会自动保存）。"""
        self._store.save(session)

    def lock_for(self, session_id: str) -> asyncio.Lock:
        """
        获取（或创建）某个会话的锁。

        asyncio.Lock 是协程安全的：
        在同一个事件循环里，setdefault 之间没有 await，不会产生竞态。
        """
        return self._locks.setdefault(session_id, asyncio.Lock())

    @asynccontextmanager
    async def locked(self, session_id: str) -> AsyncIterator[Session]:
        """
        在会话锁内加载会话，退出时保存。

        用法：
            async with manager.locked(session_id) as session:
                session.append_user(...)
                # 做 LLM / 工具调用
        """
        lock = self.lock_for(session_id)
        async with lock:
            session = self._store.load(session_id)
            if session is None:
                raise SessionNotFound(f"会话不存在：{session_id}")
            try:
                yield session
            finally:
                # 即使本轮中途出错，也保存已经发生的用户消息和待办变化。
                self._store.save(session)

    async def delete(self, session_id: str) -> bool:
        """删除会话文件与 Trace 文件。"""
        lock = self.lock_for(session_id)
        async with lock:
            deleted = self._store.delete(session_id)
            self._locks.pop(session_id, None)

            if deleted:
                trace_path = self._traces_dir / f"{session_id}.jsonl"
                try:
                    trace_path.unlink()
                except FileNotFoundError:
                    pass
            return deleted
