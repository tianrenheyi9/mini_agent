"""
HTTP 接口定义。

目前只有健康检查；后续会在这里加入 Session、消息和 Trace 相关接口。
路由文件只负责“接收请求、返回响应”，具体业务逻辑放在 agent/session 模块里。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings

# APIRouter 可以理解为一组接口的集合，最后在 main.py 中统一注册。
router = APIRouter()


@router.get("/health")
async def health() -> dict[str, object]:
    """
    健康检查接口。

    返回服务状态、当前模型名，以及 LLM 是否配置完整。
    注意这里不返回 API Key，避免密钥通过接口泄漏。
    """
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.llm_model,
        "llm_configured": settings.llm_configured,
    }
