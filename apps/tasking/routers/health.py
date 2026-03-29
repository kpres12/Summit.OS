"""Health and metrics endpoints."""
from fastapi import APIRouter

import state

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "tasking"}


# /metrics only registered if PROM_AVAILABLE
if state.PROM_AVAILABLE:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        @router.get("/metrics")
        async def metrics():
            import fastapi.responses
            return fastapi.responses.Response(
                content=generate_latest(), media_type=CONTENT_TYPE_LATEST
            )

    except Exception:
        pass
