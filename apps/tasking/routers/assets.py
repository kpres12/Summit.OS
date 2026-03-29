"""Asset registry endpoints."""
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

import state
from models import AssetIn, AssetOut
from tables import assets
from helpers import _require_auth, _safe_json

router = APIRouter()


@router.post("/api/v1/assets", response_model=AssetOut)
async def register_asset(asset: AssetIn, request: Request):
    await _require_auth(request)
    assert state.SessionLocal is not None
    now = datetime.now(timezone.utc)
    async with state.SessionLocal() as session:
        # Upsert-like behavior: try update, if 0 rows then insert
        result = await session.execute(
            assets.update()
            .where(assets.c.asset_id == asset.asset_id)
            .values(
                type=asset.type,
                capabilities=asset.capabilities,
                battery=asset.battery,
                link=asset.link,
                constraints=asset.constraints,
                updated_at=now,
            )
        )
        if result.rowcount == 0:
            await session.execute(
                assets.insert().values(
                    asset_id=asset.asset_id,
                    type=asset.type,
                    capabilities=asset.capabilities,
                    battery=asset.battery,
                    link=asset.link,
                    constraints=asset.constraints,
                    updated_at=now,
                )
            )
        await session.commit()

    if state.METRIC_ASSETS_REGISTERED:
        state.METRIC_ASSETS_REGISTERED.inc()

    return AssetOut(**asset.model_dump(), updated_at=now)


@router.get("/api/v1/assets", response_model=List[AssetOut])
async def list_assets():
    assert state.SessionLocal is not None
    async with state.SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT asset_id, type, capabilities, battery, link, constraints, updated_at FROM assets ORDER BY updated_at DESC NULLS LAST"
            )
        )
        rows = result.all()
        out: List[AssetOut] = []
        for r in rows:
            m = dict(r._mapping)
            out.append(
                AssetOut(
                    asset_id=m["asset_id"],
                    type=m.get("type"),
                    capabilities=_safe_json(m.get("capabilities")),
                    battery=m.get("battery"),
                    link=m.get("link"),
                    constraints=_safe_json(m.get("constraints")),
                    updated_at=m.get("updated_at"),
                )
            )
        return out


@router.get("/api/v1/assets/{asset_id}", response_model=AssetOut)
async def get_asset(asset_id: str):
    assert state.SessionLocal is not None
    async with state.SessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT asset_id, type, capabilities, battery, link, constraints, updated_at FROM assets WHERE asset_id = :aid"
            ),
            {"aid": asset_id},
        )
        row = result.first()
        if not row:
            raise HTTPException(status_code=404, detail="Asset not found")
        m = dict(row._mapping)
        return AssetOut(
            asset_id=m["asset_id"],
            type=m.get("type"),
            capabilities=_safe_json(m.get("capabilities")),
            battery=m.get("battery"),
            link=m.get("link"),
            constraints=_safe_json(m.get("constraints")),
            updated_at=m.get("updated_at"),
        )
