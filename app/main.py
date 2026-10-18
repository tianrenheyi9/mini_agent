from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

WEB_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    settings.traces_dir.mkdir(parents=True, exist_ok=True)
    if not settings.llm_configured:
        logger.warning(
            "LLM is not configured; set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env"
        )
    yield


app = FastAPI(title="Mini Agent", version="0.1.0", lifespan=lifespan)
app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
