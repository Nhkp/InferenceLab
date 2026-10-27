import sys

import pytest
from httpx import ASGITransport, AsyncClient

from inference_lab.gateway.app import app

pytestmark = pytest.mark.unit


async def test_health_without_gpu_imports() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert not {"torch", "vllm", "triton"}.intersection(sys.modules)
