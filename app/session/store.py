"""
会话持久化层。

每个会话保存成一个 JSON 文件：

    <data_dir>/sessions/s_xxxx.json

写入采用“临时文件 + os.replace”的原子替换：
- 进程中途崩溃不会留下半写文件；
- 读取方要么看到旧版本，要么看到新版本。

这一层只负责“把 Session 对象安全地读写到磁盘”，
不处理会话锁、并发或标题生成，那些属于 SessionManager。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from app.session.models import Session

logger = logging.getLogger(__name__)

# 会话 id 只允许安全字符，防止通过 ../ 构造任意文件路径。
_SESSION_ID_RE = re.compile(r"^s_[A-Za-z0-9_-]{1,64}$")


class SessionStoreError(Exception):
    """会话文件读写失败或内容损坏。"""


class SessionNotFound(SessionStoreError):
    """请求的会话不存在。"""


class SessionStore:
    """负责 Session 的 JSON 落盘与读取。"""

    def __init__(self, sessions_dir: Path) -> None:
        self._dir = Path(sessions_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_valid_id(session_id: str) -> bool:
        """检查会话 id 是否安全、格式是否正确。"""
        return bool(_SESSION_ID_RE.fullmatch(session_id))

    def path_for(self, session_id: str) -> Path:
        """把会话 id 映射成文件路径。"""
        if not self.is_valid_id(session_id):
            raise ValueError(f"非法的会话 id：{session_id!r}")
        return self._dir / f"{session_id}.json"

    def exists(self, session_id: str) -> bool:
        return self.path_for(session_id).exists()

    def save(self, session: Session) -> None:
        """原子写入会话文件。"""
        target = self.path_for(session.id)
        payload = json.dumps(
            session.to_dict(),
            ensure_ascii=False,
            indent=2,
        )

        # mkstemp 会在目标目录创建唯一临时文件。
        # 临时文件必须和目标文件在同一目录，os.replace 才能保证原子性。
        fd, temp_name = tempfile.mkstemp(
            dir=self._dir,
            prefix=f".{session.id}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                # 尽量确保数据真正落盘，再替换目标文件。
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except BaseException:
            # 任何一步失败都要清理临时文件，避免目录里堆垃圾。
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def load(self, session_id: str) -> Session | None:
        """读取会话；文件不存在返回 None，文件损坏则抛 SessionStoreError。"""
        path = self.path_for(session_id)
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionStoreError(f"读取会话失败：{session_id}") from exc

        if not isinstance(data, dict):
            raise SessionStoreError(f"会话文件格式错误：{session_id}")

        session = Session.from_dict(data)

        # 文件内容里的 id 应该和文件名一致；
        # 旧数据没有 id 时补上，出现冲突则视为损坏。
        if session.id and session.id != session_id:
            raise SessionStoreError(
                f"会话文件 id 与文件名不一致：{session_id}"
            )
        session.id = session_id
        return session

    def delete(self, session_id: str) -> bool:
        """删除会话文件；不存在返回 False。"""
        path = self.path_for(session_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def list_ids(self) -> list[str]:
        """列出所有会话 id。"""
        return sorted(
            path.stem
            for path in self._dir.glob("s_*.json")
            if path.is_file()
        )

    def list_metadata(self) -> list[dict[str, Any]]:
        """
        返回会话列表的轻量信息。

        单个文件损坏不应该让整个列表失效：
        记录警告并跳过该文件。
        """
        items: list[dict[str, Any]] = []
        for session_id in self.list_ids():
            try:
                session = self.load(session_id)
            except SessionStoreError as exc:
                logger.warning("Skip unreadable session %s: %s", session_id, exc)
                continue
            if session is None:
                continue

            items.append(
                {
                    "id": session.id,
                    "title": session.title,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "message_count": len(session.messages),
                    "todo_open_count": sum(
                        1 for todo in session.todos if not todo.get("done")
                    ),
                }
            )

        # 最近更新的会话排在前面。
        items.sort(key=lambda item: item["updated_at"], reverse=True)
        return items
