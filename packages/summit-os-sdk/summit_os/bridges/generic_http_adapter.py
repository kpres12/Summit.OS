"""
Generic HTTP adapter: receive vendor HTTP webhooks and publish normalized Summit contracts to MQTT.
Usage:
  python -m summit_os.bridges.generic_http_adapter --device-id sensor-001 --port 9009 \
    --map-lat vendor.lat --map-lon vendor.lng --map-alt vendor.alt

If vendor already posts Summit-shaped JSON, omit mappings and just POST to /telemetry or /detection.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .adapter_base import BaseAdapter, AdapterConfig

app: Optional[FastAPI] = None
_adapter: Optional[BaseAdapter] = None
_mappings: Dict[str, str] = {}


def _extract(path: str, data: Dict[str, Any]) -> Any:
    cur: Any = data
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def build_app(adapter: BaseAdapter, mappings: Dict[str, str]) -> FastAPI:
    api = FastAPI(title="Summit Generic HTTP Adapter")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
    )

    @api.get("/health")
    async def health():
        return {"status": "ok", "device_id": adapter.cfg.device_id}

    @api.post("/telemetry")
    async def telemetry(req: Request):
        try:
            payload = await req.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid json")
        # If already normalized, pass through
        if "device_id" in payload and "location" in payload:
            data = payload
        else:
            lat = _extract(mappings.get("lat", ""), payload) if mappings else None
            lon = _extract(mappings.get("lon", ""), payload) if mappings else None
            alt = _extract(mappings.get("alt", ""), payload) if mappings else None
            if lat is None or lon is None:
                raise HTTPException(status_code=400, detail="mapping for lat/lon missing or not found")
            data = {
                "device_id": adapter.cfg.device_id,
                "ts_iso": adapter.now_iso(),
                "location": {"lat": float(lat), "lon": float(lon), "alt": float(alt) if alt is not None else None},
                "status": "ACTIVE",
                "sensors": payload,
            }
        await adapter.publish(f"telemetry/{adapter.cfg.device_id}", data)
        return {"status": "published"}

    @api.post("/detection")
    async def detection(req: Request):
        try:
            payload = await req.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid json")
        cls = payload.get("class") or payload.get("type") or "object"
        conf = float(payload.get("confidence", 0.5))
        lat = payload.get("lat") or _extract(mappings.get("lat", ""), payload)
        lon = payload.get("lon") or _extract(mappings.get("lon", ""), payload)
        out = {
            "class": cls,
            "confidence": conf,
            "ts_iso": adapter.now_iso(),
            "lat": float(lat) if lat is not None else None,
            "lon": float(lon) if lon is not None else None,
            "source": adapter.cfg.device_id,
            "attributes": payload,
        }
        await adapter.publish(f"detections/{adapter.cfg.device_id}", out)
        return {"status": "published"}

    return api


def main(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument("--device-id", required=True)
    p.add_argument("--port", type=int, default=int(os.getenv("GEN_HTTP_PORT", "9009")))
    p.add_argument("--map-lat", default="")
    p.add_argument("--map-lon", default="")
    p.add_argument("--map-alt", default="")
    p.add_argument("--register", action="store_true")
    p.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000"))
    args = p.parse_args(argv)

    cfg = AdapterConfig(device_id=args.device_id)
    adapter = BaseAdapter(cfg)

    if args.register:
        try:
            adapter.register_with_gateway(args.api, node_type="SENSOR", capabilities=["GENERIC_HTTP"], comm=["HTTP","MQTT"])
        except Exception:
            pass

    mappings = {k: v for k, v in {"lat": args.map_lat, "lon": args.map_lon, "alt": args.map_alt}.items() if v}

    global app, _adapter, _mappings
    _adapter = adapter
    _mappings = mappings
    app = build_app(adapter, mappings)

    # Start MQTT in background and run API
    async def runner():
        await adapter._connect_mqtt()
    asyncio.get_event_loop().run_until_complete(runner())

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
