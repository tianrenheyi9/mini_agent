"""
HTTP 接口层。

路由只负责：
1. 解析/校验请求；
2. 调用 AgentLoop 或 SessionManager；
3. 把领域异常映射成 HTTP 状态码。

Agent 的具体逻辑不放在这里，方便单独测试 Runtime。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.agent.loop import AgentLoop
from app.config import get_settings
from app.session.store import SessionNotFound, SessionStoreError

router = APIRouter()


class CreateSessionRequest(BaseModel):
    title: str | None = None


class SendMessageRequest(BaseModel):
    # 先做基础长度校验；纯空白内容在路由里再检查一次。
    content: str = Field(min_length=1, max_length=10_000)


def get_agent(request: Request) -> AgentLoop:
    """从 app.state 取出 lifespan 中装配好的 AgentLoop。"""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    return agent


@router.get("/health")
async def health() -> dict[str, object]:
    """健康检查：只暴露模型名和是否配置完整，不返回密钥。"""
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.llm_model,
        "llm_configured": settings.llm_configured,
    }


@router.post("/sessions", status_code=201)
def create_session(
    payload: CreateSessionRequest | None = None,
    agent: AgentLoop = Depends(get_agent),
) -> dict[str, Any]:
    """创建一个新会话。"""
    session = agent.sessions.create(payload.title if payload else None)
    return _session_summary(session)


@router.get("/sessions")
def list_sessions(
    agent: AgentLoop = Depends(get_agent),
) -> dict[str, Any]:
    """列出会话元数据，按更新时间倒序。"""
    return {"sessions": agent.sessions.list_sessions()}


@router.get("/sessions/{session_id}")
def get_session(
    session_id: str,
    agent: AgentLoop = Depends(get_agent),
) -> dict[str, Any]:
    """返回会话完整信息：消息、摘要、待办、统计。"""
    try:
        session = agent.sessions.get(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法的会话 id") from exc
    except SessionStoreError as exc:
        raise HTTPException(status_code=500, detail="读取会话失败") from exc
    return session.to_dict()


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    agent: AgentLoop = Depends(get_agent),
) -> dict[str, bool]:
    """删除会话及其 Trace 文件。"""
    try:
        deleted = await agent.sessions.delete(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法的会话 id") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": True}


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    payload: SendMessageRequest,
    agent: AgentLoop = Depends(get_agent),
) -> dict[str, Any]:
    """发送一条消息，执行完整的一轮 Agent Loop。"""
    if not payload.content.strip():
        raise HTTPException(status_code=422, detail="消息不能为空")

    try:
        result = await agent.run_turn(session_id, payload.content)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SessionStoreError as exc:
        raise HTTPException(status_code=500, detail="会话存储失败") from exc

    return {
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "reply": result.reply,
        "thought": result.thought,
        "steps": result.steps,
        "stats": result.stats,
        "status": result.status,
        "compression": result.compression,
    }


@router.get("/sessions/{session_id}/traces")
def get_traces(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    agent: AgentLoop = Depends(get_agent),
) -> dict[str, Any]:
    """返回最近 N 条 Trace 事件。"""
    try:
        agent.sessions.get(session_id)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法的会话 id") from exc
    except SessionStoreError as exc:
        raise HTTPException(status_code=500, detail="读取会话失败") from exc

    return {"events": agent.trace.read_events(session_id, limit=limit)}


def _session_summary(session: Any) -> dict[str, Any]:
    """会话创建/列表使用同一套轻量字段。"""
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": len(session.messages),
        "todo_open_count": sum(
            1 for todo in session.todos if not todo.get("done")
        ),
    }
