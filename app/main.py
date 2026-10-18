"""
FastAPI 应用入口。

启动方式：
    uvicorn app.main:app --reload --port 8000

这个文件只做“组装”工作：
- 创建 FastAPI 应用；
- 注册 API 路由；
- 挂载前端静态文件；
- 在应用启动时创建数据目录并检查配置。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings

# 统一日志格式，方便在控制台看到时间、级别和来源模块。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# 前端文件所在目录：app/web/
WEB_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 的生命周期钩子。

    yield 之前的代码在应用启动时执行，yield 之后的代码在关闭时执行。
    这里在启动阶段创建数据目录，并检查 LLM 是否配置完整。
    """
    settings = get_settings()
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    settings.traces_dir.mkdir(parents=True, exist_ok=True)

    if not settings.llm_configured:
        # 没有配置 Key 时服务仍能启动（方便先看页面），但调用模型会失败。
        logger.warning(
            "LLM is not configured; set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env"
        )
    yield


app = FastAPI(title="Mini Agent", version="0.1.0", lifespan=lifespan)

# API 路由统一放在 /api 前缀下，例如 /api/health。
app.include_router(router, prefix="/api")

# 静态文件挂载在根路径，必须放在 API 路由之后，否则会拦截 /api 请求。
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
