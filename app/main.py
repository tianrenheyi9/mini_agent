"""
FastAPI 应用入口与装配。

`create_app(agent=...)` 允许测试注入一个用 FakeLLM 装配的 AgentLoop；
生产环境不传 agent，由 lifespan 调用 build_agent_loop() 创建。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.agent.loop import AgentLoop, build_agent_loop
from app.api.routes import router
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

WEB_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger(__name__)


def create_app(agent: AgentLoop | None = None) -> FastAPI:
    """创建 FastAPI 应用；测试可传入自己的 AgentLoop。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        # 即使用注入的 agent，也确保数据目录存在（API 侧可能直接读目录）。
        settings.sessions_dir.mkdir(parents=True, exist_ok=True)
        settings.traces_dir.mkdir(parents=True, exist_ok=True)

        owns_agent = agent is None
        if agent is None:
            # 生产路径：真正装配 LLM、会话、工具、上下文和 Trace。
            app.state.agent = build_agent_loop(settings)
        else:
            app.state.agent = agent

        if not settings.llm_configured:
            logger.warning(
                "LLM is not configured; set LLM_BASE_URL, LLM_API_KEY "
                "and LLM_MODEL in .env"
            )

        try:
            yield
        finally:
            # 只关闭自己创建的客户端；测试注入的由测试负责。
            if owns_agent:
                await app.state.agent.aclose()

    app = FastAPI(
        title="Mini Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api")

    # 静态文件必须放在 API 路由之后，避免拦截 /api 请求。
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
