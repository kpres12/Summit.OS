import os
from fastapi import APIRouter, HTTPException
import httpx


def register(app, fusion_url: str):
    """Register wildfire mission routes on the API Gateway."""
    router = APIRouter()

    @router.get("/v1/wildfire/ignitions")
    async def get_wildfire_ignitions(limit: int = 50):
        # Convention: wildfire ignitions are observations with class 'fire.ignition'
        params = {"limit": str(limit), "cls": "fire.ignition"}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{fusion_url}/observations", params=params)
                r.raise_for_status()
                return {"ignitions": r.json()}
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Fusion upstream error: {e}")

    app.include_router(router)
