from __future__ import annotations

from fastapi import APIRouter

from src.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Liveness and dependency health check.
    Always returns HTTP 200 — weaviate field signals dependency status.
    """
    weaviate_status = "not_connected"
    try:
        from src.services.weaviate_client import get_client
        client = get_client()
        weaviate_status = "ok" if client.is_live() else "degraded"
    except Exception:
        weaviate_status = "not_connected"

    return HealthResponse(status="ok", weaviate=weaviate_status)
