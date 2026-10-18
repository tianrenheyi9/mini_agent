"""服务连通性冒烟测试：应用能启动，健康检查接口能正常响应。"""

from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    """TestClient 会触发 FastAPI 的 lifespan，因此启动逻辑也会被测试到。"""
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
