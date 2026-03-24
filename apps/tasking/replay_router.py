"""
Mission Replay API for Summit.OS Tasking Service

Records world-state snapshots at regular intervals during active missions,
then serves a time-indexed playback API so operators can scrub through
what happened.

Snapshot format (stored in Redis key  replay:{mission_id}:{ts_ms}):
  {
    "ts_iso":    "2024-...",
    "mission_id": "...",
    "assignments": [ {asset_id, lat, lon, status, completed_seq} ],
    "events":    [ {type, description, ts_iso} ]
  }

API endpoints:
  GET  /api/v1/missions/{mission_id}/replay/timeline
  GET  /api/v1/missions/{mission_id}/replay/snapshot?ts={iso}
  POST /api/v1/missions/{mission_id}/replay/record   (internal — called by ExecutionMonitor)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("tasking.replay")

FABRIC_URL     = os.getenv("FABRIC_URL", "http://fabric:8001")
REPLAY_MAX_PTS = int(os.getenv("REPLAY_MAX_SNAPSHOTS_PER_MISSION", "3600"))  # 1 h @ 1/s

# In-memory replay store: mission_id → sorted list of snapshot dicts
# (keyed by ts_ms for O(1) range queries)
_replay_store: Dict[str, List[Dict[str, Any]]] = {}


# ── Recording ────────────────────────────────────────────────────────────────

def record_snapshot(mission_id: str, snapshot: Dict[str, Any]):
    """
    Append a snapshot to the in-memory replay store.
    Called internally by ExecutionMonitor every tick while mission is ACTIVE.
    """
    if mission_id not in _replay_store:
        _replay_store[mission_id] = []

    store = _replay_store[mission_id]
    store.append(snapshot)

    # Cap to avoid unbounded growth
    if len(store) > REPLAY_MAX_PTS:
        _replay_store[mission_id] = store[-REPLAY_MAX_PTS:]


def record_event(mission_id: str, event_type: str, description: str):
    """Append a lightweight event to the last snapshot for this mission."""
    store = _replay_store.get(mission_id)
    if store:
        store[-1].setdefault("events", []).append({
            "type":        event_type,
            "description": description,
            "ts_iso":      datetime.now(timezone.utc).isoformat(),
        })


# ── Router ───────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/v1/missions", tags=["replay"])


@router.get("/{mission_id}/replay/timeline")
async def get_timeline(mission_id: str):
    """
    Return the list of snapshot timestamps available for replay.

    Response:
      { "mission_id": "...", "count": N, "timestamps": ["2024-...", ...] }
    """
    store = _replay_store.get(mission_id)
    if store is None:
        raise HTTPException(status_code=404, detail=f"No replay data for mission {mission_id}")

    timestamps = [s["ts_iso"] for s in store]
    return {
        "mission_id": mission_id,
        "count":      len(timestamps),
        "start":      timestamps[0] if timestamps else None,
        "end":        timestamps[-1] if timestamps else None,
        "timestamps": timestamps,
    }


@router.get("/{mission_id}/replay/snapshot")
async def get_snapshot(
    mission_id: str,
    ts: Optional[str] = Query(None, description="ISO timestamp; returns nearest snapshot"),
    index: Optional[int] = Query(None, description="Zero-based snapshot index"),
):
    """
    Return the world-state snapshot nearest to the requested time.

    Provide either ?ts=<iso> or ?index=<n>.
    """
    store = _replay_store.get(mission_id)
    if not store:
        raise HTTPException(status_code=404, detail=f"No replay data for mission {mission_id}")

    if index is not None:
        idx = max(0, min(index, len(store) - 1))
        return store[idx]

    if ts is not None:
        try:
            target = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid ts format (expected ISO 8601)")
        # Linear scan — good enough for ≤3600 points; binary search if needed
        best = store[0]
        best_dt = abs(datetime.fromisoformat(
            best["ts_iso"].replace("Z", "+00:00")).timestamp() - target)
        for snap in store[1:]:
            dt = abs(datetime.fromisoformat(
                snap["ts_iso"].replace("Z", "+00:00")).timestamp() - target)
            if dt < best_dt:
                best, best_dt = snap, dt
        return best

    # Default: latest
    return store[-1]


@router.delete("/{mission_id}/replay")
async def clear_replay(mission_id: str):
    """Free replay memory for a completed/failed mission."""
    if mission_id in _replay_store:
        del _replay_store[mission_id]
    return {"status": "cleared", "mission_id": mission_id}
